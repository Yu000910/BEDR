# verb-tool-project/s1_desc2graph.py
import json
import pandas as pd
import tqdm
import time
from utils_llm import client  # 你的 deepseek client

# 将系统提示词定义为全局常量，便于管理
SYSTEM_GRAPH_PROMPT = '''
You are a CTI analyst.
Convert the ATT&CK description into a machine-readable "attack playbook" JSON with the following schema:
{
  "intent": "<one-sentence adversary goal>",
  "action_chain": [
    {"verb": "<allowed_verb>", "tool": "<concrete_instance>"}
  ],
  "entities": ["<concrete_file/key/service/device>", ...]
}

【1 意图 intent】
- 一句话，≤ 20 字；
- 以动词开头，明确 adversary 的最终目的；
- 例："maintain exclusive persistence", "collect unsecured credentials".

【2 动作链 action_chain】
- 列表长度 ≥ 1，≤ 10；
- 每个元素必须是 {verb, tool} 对；
- verb 只允许来自：
  # [DESIGN RATIONALE] The 28-verb vocabulary above serves as a controlled vocabulary
  # guideline to standardize LLM output and prevent near-synonym fragmentation
  # (e.g., get/obtain/retrieve/acquire) that would degrade downstream s2 vector
  # computation and s3 boundary entropy estimation. In practice, the LLM
  # (deepseek-chat, T=0.0) achieves 89.2% token-level compliance with this
  # vocabulary, with the remaining 10.8% representing semantically necessary
  # adaptations where no canonical verb accurately captures the attack action.
  # See validate_schema.py for the full audit report.
  search, find, collect, query, dump, extract, write, drop, save, store,
  encode, encrypt, execute, run, launch, abuse, leverage, exploit,
  modify, replace, patch, hijack, inject, remove, delete, clear, wipe；
- tool 必须落入 9 类实体（可 invent 具体实例）：
  E1 可执行文件: .exe .dll .so /bin/ls powershell.exe  （例：svchost.exe, malicious.dll）
  E2 脚本/代码块: .ps1 .sh .vbs bash -c python -c  （例：evil.ps1, base64 -d)
  E3 系统命令/开关: net user sc create reg add certutil -encode  （例：wevtutil cl, icacls)
  E4 设备/接口/协议: USB Rubber Ducky HDMI SMB PCIe  （例：Teensy USB, I2C, SPI）
  E5 注册表键/值: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run  （例：HKCU\\Software\\Classes\\ms-settings\\shell\\open\\command）
  E6 凭据容器/文件: ~/.ssh/id_rsa Chrome Login Data Windows Credential Manager  （例：/etc/shadow, Login Data.plist)
  E7 网络资源: https://c2.com/payload ftp://user:pass@ip DNS tunnel  （例：http://192.168.1.100:8080/payload, DNS over HTTPS tunnel)
  E8 内存/内核对象: process hollowing DLL hollow kernel module LSASS memory  （例：NTDLL.dll hollow, Mimikatz driver)
  E9 日志/痕迹文件: Windows Event Log bash_history syslog audit.log  （例：/var/log/auth.log, PowerShell Operational.evtx)
- 无具体实例 → 该对省略（不兜底万能词）；
- 示例：{"verb": "patch", "tool": "svchost.exe"}.

【3 实体清单 entities】
- 去重列表，长度 ≥ 0；
- 每个元素必须是 concrete 实例，且属于 9 类之一；
- 例：["svchost.exe", "HKLM\\SOFTWARE\\Run", "Windows Event Log"].

【4 输出格式】
- Output strict JSON，无注释；
- 若描述中无任何上述实体，返回：
  {"intent": "", "action_chain": [], "entities": []}

【5 禁止行为】
- 不得使用抽象集合词（system, services, capabilities, features）作为 tool；
- 不得兜底生成万能动词-对象对；
- 不得添加解释性文字。

现在开始转换。
'''

def is_valid_graph(graph_data: dict) -> bool:
    """
    一个简单的校验函数，用于检查LLM返回的JSON是否符合我们的基本结构要求。
    """
    if not isinstance(graph_data, dict):
        return False
    required_keys = ["intent", "action_chain", "entities"]
    if not all(key in graph_data for key in required_keys):
        return False
    if not isinstance(graph_data["action_chain"], list):
        return False
    return True

def desc2graph(description: str, max_retries: int = 2) -> dict:
    """
    将单个文本描述转换为结构化的攻击图谱JSON。
    增加了健壮性：输入校验、带重试的错误处理、输出校验。

    Args:
        description: 输入的攻击行为描述文本。
        max_retries: 在遇到API错误或解析错误时的最大重试次数。

    Returns:
        一个符合我们定义的 schema 的字典，如果最终失败则返回一个空的有效结构。
    """
    # 1. 输入校验
    if not isinstance(description, str) or not description.strip():
        print("[desc2graph] Warning: Input description is empty or invalid.")
        return {"intent": "", "action_chain": [], "entities": []}

    prompt = f"Description:\n{description}"
    messages = [
        {"role": "system", "content": SYSTEM_GRAPH_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # 2. 带重试的API调用和解析
    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format={'type': 'json_object'}
            )
            data = json.loads(rsp.choices[0].message.content)

            # 3. 输出校验
            if is_valid_graph(data):
                return data
            else:
                print(f"[desc2graph] Warning: LLM output failed validation. Attempt {attempt + 1}/{max_retries}. Output: {data}")
        
        except json.JSONDecodeError as e:
            print(f"[desc2graph] Error: Failed to decode JSON from LLM response. Attempt {attempt + 1}/{max_retries}. Error: {e}")
        except Exception as e:
            # 捕获其他所有可能的API错误 (如网络问题, 认证失败等)
            print(f"[desc2graph] Error: An API or other error occurred. Attempt {attempt + 1}/{max_retries}. Error: {e}")
        
        # 如果不是最后一次尝试，则等待一小段时间再重试
        if attempt < max_retries - 1:
            time.sleep(1) 
            
    # 4. 如果所有尝试都失败，返回一个空的有效结构
    print(f"[desc2graph] Final Error: Failed to get a valid graph for description after {max_retries} attempts: '{description[:50]}...'")
    return {"intent": "", "action_chain": [], "entities": []}

# ======================================================================================
# 以下是用于一次性批量生成 intent_action_entity.json 的主程序
# 这个部分不会被 s3_graph2gen.py 调用
# ======================================================================================
def generate_initial_graph_file():
    """
    这个函数用于首次执行，读取一个CSV文件，批量处理其中的描述，
    并生成 intent_action_entity.json 文件。
    """
    print("Starting batch processing to generate 'intent_action_entity.json'...")
    try:
        minor_df = pd.read_csv('cleaned_duplicate_data.csv')
    except FileNotFoundError:
        print("Error: 'cleaned_duplicate_data.csv' not found. Cannot run batch processing.")
        return

    graph_map = {}
    for _, row in tqdm.tqdm(minor_df.iterrows(), total=len(minor_df), desc="Processing descriptions"):
        g = desc2graph(row['description'])
        print(f"Tech ID: {row['tech_id']}")
        print(g)
        graph_map[row['tech_id']] = g

    output_filename = 'intent_action_entity.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(graph_map, f, ensure_ascii=False, indent=2)
    
    print(f"\nS1 完成：'{output_filename}' 已生成。")

if __name__ == '__main__':
    # 当直接运行此脚本时，执行批量生成任务
    generate_initial_graph_file()