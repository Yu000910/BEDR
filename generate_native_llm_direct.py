# generate_naive_llm_direct.py (对照实验 - 精确差额版)
import json
import pandas as pd
import tqdm
import numpy as np
import time
import random
import ast
import os

# 只需要LLM客户端
from utils_llm import client 

# ======================================================================================
# 全局配置
# ======================================================================================
# 输入文件
ORIGINAL_DATA_PATH = 'cleaned_duplicate_data.csv'
GRAPH_MAP_PATH = 'intent_action_entity.json'
# 参考文件 (仅用于计算原始数量，不再控制总数)
# FIRST_PASS_AUG_PATH = 'aug_sentences_final_filtered.csv'
# ... (不再需要加载之前的增强文件来确定总数)

# 输出文件
NAIVE_LLM_OUTPUT_PATH = 'aug_sentences_naive_llm_direct.csv'

TARGET_SAMPLE_COUNT = 50
MAX_SAMPLES_PER_SINGLE_CALL = 50 # 设定一个单次请求的上限，以防API崩溃
MAX_LLM_RETRIES_PER_TECH = 3 # 如果单次请求失败，最多重试次数

# ======================================================================================
# 主程序
# ======================================================================================
def main():
    print("--- 步骤 1/4: 加载数据并计算初始样本数 ---")
    start_time = time.time()
    try:
        original_df = pd.read_csv(ORIGINAL_DATA_PATH)
        description_lookup = pd.Series(original_df['description'].values, index=original_df['tech_id']).to_dict()
        print(f"成功加载 {len(original_df)} 条原始技术描述。")

        graph_map = json.load(open(GRAPH_MAP_PATH))
        all_known_techs = list(graph_map.keys())
        print(f"成功加载 {len(all_known_techs)} 个技术类别列表。")

    except FileNotFoundError as e:
        print(f"\n[错误!] 文件未找到: {e}！程序终止。")
        return
    except Exception as e:
        print(f"\n[错误!] 加载文件时出错: {e}！程序终止。")
        return

    # 计算原始样本数 (仅基于 examples)
    initial_counts = pd.Series({row['tech_id']: len(ast.literal_eval(row['examples'])) for _, row in original_df.iterrows() if isinstance(row['examples'], str) and row['examples'].startswith('[')})
    print(f"已计算 {len(initial_counts)} 个类别的原始样本数。")

    # 识别所有样本数不足50的类别
    minority_techs_to_augment = [
        tech_id for tech_id in all_known_techs
        if initial_counts.get(tech_id, 0) < TARGET_SAMPLE_COUNT
    ]
    print(f"共识别出 {len(minority_techs_to_augment)} 个需要进行朴素过采样的小样本技术。\n")
    load_end_time = time.time()
    print(f"--- 步骤 1 完成，耗时: {load_end_time - start_time:.2f} 秒 ---\n")


    # --- 步骤 2/4: 朴素LLM生成循环 (精确差额) ---
    print(f"--- 步骤 2/4: 开始朴素LLM生成 (精确差额版) ---")
    naive_llm_sentences = []
    total_generated_count = 0
    skipped_techs = []
    failed_techs = [] # 记录生成失败的类别

    tech_progress_bar = tqdm.tqdm(minority_techs_to_augment, desc="朴素生成进度 (类别)")

    for tech_id in tech_progress_bar:
        description = description_lookup.get(tech_id)
        if not description or not isinstance(description, str) or len(description) < 20:
            print(f"\n[警告] 类别 {tech_id} 的描述无效或过短，跳过生成。")
            skipped_techs.append(tech_id)
            continue
            
        initial_count_for_tech = initial_counts.get(tech_id, 0)
        needed_for_tech = TARGET_SAMPLE_COUNT - initial_count_for_tech
        
        if needed_for_tech <= 0: continue # 理论上不会发生，因为我们只遍历了minority_techs

        # 确定本次调用实际请求的数量
        request_num = min(needed_for_tech, MAX_SAMPLES_PER_SINGLE_CALL)
        if request_num < needed_for_tech:
            print(f"\n[警告] 类别 {tech_id} 需要 {needed_for_tech} 个样本，但单次请求上限为 {MAX_SAMPLES_PER_SINGLE_CALL}。本次只请求 {request_num} 个。")
        
        # 构建简单Prompt
        prompt = f"""
Based on your understanding of the following ATT&CK technique description, please generate exactly {request_num} diverse, realistic, and concise example sentences illustrating how this technique might be used in practice.

Technique ID: {tech_id}
Description:
{description}

CRITICAL INSTRUCTIONS:
1.  **LANGUAGE:** You MUST generate the output in ENGLISH only.
2.  **FOCUS:** Each sentence should be a specific example based on the description. Avoid overly generic statements.
3.  **FORMAT:** Output ONLY a valid JSON list of strings containing exactly {request_num} elements. Do not add any other text, comments, or explanations.

JSON Output Example (if requesting 3):
["Example sentence 1.", "Another example sentence.", "Third example."]
"""
        
        accepted_this_tech = 0
        for attempt in range(MAX_LLM_RETRIES_PER_TECH):
            print(f"\n[Attempt {attempt + 1}/{MAX_LLM_RETRIES_PER_TECH}] For {tech_id}. Requesting {request_num} samples...")
            
            try:
                rsp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={'type': 'json_object'})
                content = rsp.choices[0].message.content
                candidate_sents = json.loads(content)
                if isinstance(candidate_sents, dict):
                    candidate_sents = next((v for v in candidate_sents.values() if isinstance(v, list)), [])

                if not isinstance(candidate_sents, list):
                    print(f"  [Warning] LLM did not return a list for {tech_id}.")
                    if attempt < MAX_LLM_RETRIES_PER_TECH - 1:
                         print("    Retrying...")
                         time.sleep(2)
                         continue
                    else:
                         print(f"    达到最大重试次数，放弃类别 {tech_id}。")
                         failed_techs.append(tech_id)
                         break # 跳出重试循环

                # 检查返回数量是否大致符合要求 (允许一定误差)
                if not (0.8 * request_num <= len(candidate_sents) <= 1.2 * request_num):
                     print(f"  [Warning] LLM 返回了 {len(candidate_sents)} 个样本，与请求的 {request_num} 个差异较大。")
                     # 可以选择重试或接受当前结果，这里我们选择接受

                # 朴素生成：直接接受所有有效的句子
                accepted_this_attempt = 0
                for sent in candidate_sents:
                    # 不再检查总数上限，只检查本类别是否足够
                    if accepted_this_tech >= needed_for_tech: break 
                    
                    if isinstance(sent, str) and len(sent) >= 15:
                        #print(f"  [NAIVE ACCEPTED] Sentence for {tech_id}: {sent}") # 减少打印量
                        naive_llm_sentences.append({'tech_id': tech_id, 'sentence': sent})
                        total_generated_count += 1
                        accepted_this_tech += 1
                
                print(f"  成功接受 {accepted_this_attempt} 个样本。类别 {tech_id} 当前总计生成 {accepted_this_tech} / {needed_for_tech} 个。")
                
                # 如果本次调用成功获取了样本，就认为成功，跳出重试循环
                break 

            except Exception as e:
                print(f"  [ERROR] An exception occurred during generation for {tech_id}: {e}")
                if attempt < MAX_LLM_RETRIES_PER_TECH - 1:
                    print("    Retrying...")
                    time.sleep(5) # 发生错误时等待更长时间
                    continue
                else:
                    print(f"    达到最大重试次数，放弃类别 {tech_id}。")
                    failed_techs.append(tech_id)
                    break # 跳出重试循环
        
        # 更新总体进度条的后缀信息
        tech_progress_bar.set_postfix({
            'Total Generated': total_generated_count,
            'Failed Techs': len(failed_techs)
        })

    gen_end_time = time.time()
    print(f"\n朴素LLM生成过程结束，总共生成了 {len(naive_llm_sentences)} 个样本。")
    print(f"--- 步骤 2 完成，耗时: {gen_end_time - load_end_time:.2f} 秒 ---\n")
    if skipped_techs:
        print(f"[警告] 以下 {len(skipped_techs)} 个类别因描述无效被跳过: {skipped_techs}\n")
    if failed_techs:
        print(f"[警告] 以下 {len(failed_techs)} 个类别因生成失败或重试超限而被放弃: {failed_techs}\n")

    # --- 步骤 3/4: 保存结果 ---
    print(f"--- 步骤 3/4: 保存朴素LLM生成的样本 ---")
    pd.DataFrame(naive_llm_sentences).to_csv(NAIVE_LLM_OUTPUT_PATH, index=False)
    print(f"所有朴素LLM生成的样本已保存到 '{NAIVE_LLM_OUTPUT_PATH}'。\n")
    
    # --- 步骤 4/4: 最终总结 ---
    final_end_time = time.time()
    print(f"--- 步骤 4/4: 最终总结 ---")
    print(f"目标类别数: {len(minority_techs_to_augment)}")
    print(f"实际生成总数: {len(naive_llm_sentences)}")
    print(f"因描述无效跳过的类别数: {len(skipped_techs)}")
    print(f"因生成失败放弃的类别数: {len(failed_techs)}")
    print(f"总耗时: {final_end_time - start_time:.2f} 秒")
    print("\n处理完成！")

if __name__ == '__main__':
    main()