# s3_third_pass_augmentation.py (智能三阶段补录版)
import json
import pandas as pd
import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import time
import random
import ast

from utils_embedding import embed_sentences
from utils_llm import *

# ======================================================================================
# 核心辅助函数 (与S3最终版完全相同)
# ======================================================================================

def get_embedding_for_new_sentence(tech_id: str, sentence_text: str, graph_map: dict, intent_vector_lookup: dict) -> np.ndarray:
    """
    最终版：为新生成的句子创建其“官方意图+句子向量”的混合向量。
    """
    official_intent_text = graph_map.get(tech_id, {}).get('intent')
    intent_vec = intent_vector_lookup.get(official_intent_text) if official_intent_text else None
    
    sentence_vec_list = embed_sentences([sentence_text])
    if sentence_vec_list.shape[0] == 0:
        return np.zeros(768)
    sentence_vec = sentence_vec_list[0]
    
    if intent_vec is not None and sentence_vec is not None:
        combined_vec = (intent_vec + sentence_vec) / 2.0
    elif sentence_vec is not None:
        combined_vec = sentence_vec
    else:
        return np.zeros(768)

    norm = np.linalg.norm(combined_vec)
    if norm > 0:
        combined_vec /= norm
        
    return combined_vec


def calculate_local_boundary_entropy(target_vec: np.ndarray, all_sample_vectors: np.ndarray, all_sample_labels: np.ndarray, k: int) -> float:
    """
    在所有真实样本构成的空间中，计算局部边界熵。
    """
    distances = cdist(target_vec.reshape(1, -1), all_sample_vectors, metric='cosine')[0]
    # 添加检查确保有足够的邻居
    if len(all_sample_vectors) <= k:
        print(f"    [Warning] Not enough samples ({len(all_sample_vectors)}) to find {k} neighbors. Skipping entropy calculation.")
        return np.inf # 返回一个极大值表示无效

    k_nearest_indices = np.argpartition(distances, k)[:k]
    k_nearest_labels = all_sample_labels[k_nearest_indices]
    
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    print(f"    [DEBUG] Nearest {k} labels distribution: {dict(zip(unique_labels, counts))}")
    
    class_probs = counts / k
    # 增加对概率和的检查，确保数值稳定性
    if not np.isclose(np.sum(class_probs), 1.0):
         print(f"    [Warning] Probabilities do not sum to 1: {class_probs}. Entropy might be inaccurate.")
         
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-9))
    
    return entropy


def main():
    # --- 0. 全局配置 ---
    # 输入文件
    HYBRID_EMBEDDING_NPZ_PATH = 'all_samples_intent_sentence_embed.npz'
    GRAPH_MAP_PATH = 'intent_action_entity.json'
    ORIGINAL_DATA_PATH = 'cleaned_duplicate_data.csv'
    FIRST_PASS_AUG_PATH = 'aug_sentences_final_filtered.csv'
    SECOND_PASS_AUG_PATH = 'aug_sentences_second_pass.csv' # 第二轮的成果
    
    # 输出文件
    THIRD_PASS_OUTPUT_PATH = 'aug_sentences_third_pass.csv' # 本次“第三轮补录”的输出
    FINAL_REPORT_PATH = 'augmentation_summary_report_final.txt' # 更新最终报告文件名
    GOLDEN_SAMPLES_OUTPUT_PATH = 'stubborn_categories_golden_samples_pass3.csv' # 第三轮的黄金样本
    
    TARGET_SAMPLE_COUNT_THIS_PASS = 10 # 本轮的目标是至少达到10个
    
    # 实时自适应动态阈值配置 (可以考虑在本轮稍微放宽上限)
    STRICT_ENTROPY_THRESHOLD = 1.5
    MAX_LENIENT_THRESHOLD = 3.5 # 稍微放宽上限
    SUCCESS_RATE_TRIGGER = 0.10 # 触发条件也可以稍微放宽
    
    SAMPLES_PER_LLM_CALL = 10
    MAX_LLM_RETRIES = 6
    K_NEIGHBORS = 15

    # --- 1. 加载所有需要的数据 ---
    print("--- 步骤 1/5: 加载所有需要的数据 ---")
    try:
        graph_map = json.load(open(GRAPH_MAP_PATH))
        data = np.load(HYBRID_EMBEDDING_NPZ_PATH, allow_pickle=True)
        X_all_samples, y_all_labels = data['vectors'], data['labels']
        original_df = pd.read_csv(ORIGINAL_DATA_PATH)
        first_pass_df = pd.read_csv(FIRST_PASS_AUG_PATH)
        second_pass_df = pd.read_csv(SECOND_PASS_AUG_PATH)
        print("所有数据文件加载成功。\n")
    except FileNotFoundError as e:
        print(f"\n[错误!] 文件未找到: {e}！请确保前两轮的文件都在。程序终止。")
        return
        
    # --- 2. 预计算“官方意图”的嵌入 ---
    print("--- 步骤 2/5: 预计算所有官方意图的嵌入向量 ---")
    official_intents = {tech_id: data.get('intent') for tech_id, data in graph_map.items() if data.get('intent')}
    all_intent_texts = list(set(official_intents.values()))
    intent_vectors = embed_sentences(all_intent_texts)
    intent_vector_lookup = dict(zip(all_intent_texts, intent_vectors))
    print(f"已为 {len(intent_vector_lookup)} 个独特的官方意图创建了向量查找字典。\n")

    # --- 3. 诊断当前状态，识别需要“第三轮补录”的类别 (<10) ---
    print("--- 步骤 3/5: 诊断当前数据集状态，识别需要进行“第三轮补录”的类别 ---")
    initial_counts = pd.Series({row['tech_id']: len(ast.literal_eval(row['examples'])) for _, row in original_df.iterrows() if isinstance(row['examples'], str) and row['examples'].startswith('[')})
    first_pass_counts = first_pass_df['tech_id'].value_counts()
    second_pass_counts = second_pass_df['tech_id'].value_counts()
    # 计算当前总数 = 原始 + 第一轮 + 第二轮
    current_total_counts = initial_counts.add(first_pass_counts, fill_value=0).add(second_pass_counts, fill_value=0)
    
    all_known_techs = list(graph_map.keys())
    # 找出所有当前总数仍 < 10 的类别
    third_pass_targets = [
        tech_id for tech_id in all_known_techs 
        if current_total_counts.get(tech_id, 0) < TARGET_SAMPLE_COUNT_THIS_PASS
    ]
    print(f"诊断完成，共识别出 {len(third_pass_targets)} 个样本数仍 < {TARGET_SAMPLE_COUNT_THIS_PASS} 的类别，将对它们进行第三轮补录。\n")

    # --- 4. 核心生成与过滤循环 (只针对第三轮目标) ---
    print("--- 步骤 4/5: 开始对目标类别进行第三轮生成与过滤 ---")
    third_pass_sentences = []
    golden_pass_sentences_pass3 = []

    for tech_id in tqdm.tqdm(third_pass_targets, desc="第三轮补录"):
        g = graph_map.get(tech_id)
        if not g: continue
        
        # 重新计算当前总数和本轮需求数
        current_sample_count = current_total_counts.get(tech_id, 0)
        needed_samples = TARGET_SAMPLE_COUNT_THIS_PASS - int(current_sample_count)
        if needed_samples <= 0: continue # 如果因为某种原因已经达标，则跳过

        action_chain_str = "\n".join([f"- {d['verb']}: {d['tool']}" for d in g.get('action_chain', [])[:2]])
        prompt = f"""
As a cybersecurity expert, your task is to generate {SAMPLES_PER_LLM_CALL} diverse, realistic, and concise example sentences for the ATT&CK technique {tech_id}.
**Key Objective (Intent):**
{g.get('intent', 'N/A')}
**Generate sentences that clearly describe one of these core actions:**
{action_chain_str}
**CRITICAL INSTRUCTIONS:**
1.  **LANGUAGE:** You MUST generate the output in ENGLISH only.
2.  **FOCUS:** Each sentence must be a clear example of ONE specific action, not a long story or a combination of many actions.
3.  **FORMAT:** Output ONLY a valid JSON list of strings.
**Example of a good, focused sentence:**
"The malware used 'vssadmin delete shadows' to erase volume shadow copies and hinder system recovery."
**JSON Output Example:**
["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5", "sentence 6", "sentence 7", "sentence 8", "sentence 9", "sentence 10"]
"""
        
        generated_for_this_tech = []
        golden_for_this_tech = []
        total_candidates_generated = 0
        rejected_entropies = []
        
        current_threshold = STRICT_ENTROPY_THRESHOLD
        threshold_reason = "Strict (Initial)"

        for retry_count in range(MAX_LLM_RETRIES):
            if len(generated_for_this_tech) >= needed_samples: break
            
            # 实时自适应动态阈值逻辑 (与上一版相同，但阈值上限可能已调整)
            if retry_count > 0:
                success_rate = len(generated_for_this_tech) / total_candidates_generated if total_candidates_generated > 0 else 0
                if success_rate < SUCCESS_RATE_TRIGGER and rejected_entropies:
                    median_rejected_entropy = np.median(rejected_entropies)
                    adaptive_suggestion = current_threshold + (median_rejected_entropy - current_threshold) * 0.5
                    if adaptive_suggestion > current_threshold:
                        new_threshold = min(MAX_LENIENT_THRESHOLD, adaptive_suggestion)
                        threshold_reason = f"Adaptive (SR {success_rate:.1%} low, raising from {current_threshold:.2f})"
                        current_threshold = new_threshold
                else:
                    threshold_reason = f"Strict (SR {success_rate:.1%} OK, holding at {current_threshold:.2f})"

            print(f"\n[Attempt {retry_count + 1}/{MAX_LLM_RETRIES}] For {tech_id} (Need {needed_samples - len(generated_for_this_tech)} more). Threshold: {current_threshold:.2f} ({threshold_reason})")
            
            try:
                rsp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={'type': 'json_object'})
                content = rsp.choices[0].message.content
                candidate_sents = json.loads(content)
                if isinstance(candidate_sents, dict):
                    candidate_sents = next((v for v in candidate_sents.values() if isinstance(v, list)), [])

                if not isinstance(candidate_sents, list):
                    print(f"  [Warning] LLM did not return a list.")
                    continue
                
                total_candidates_generated += len(candidate_sents)

                for sent in candidate_sents:
                    if len(generated_for_this_tech) >= needed_samples: break
                    if not isinstance(sent, str) or len(sent) < 15: continue
                    
                    hybrid_vec = get_embedding_for_new_sentence(tech_id, sent, graph_map, intent_vector_lookup)
                    local_entropy = calculate_local_boundary_entropy(hybrid_vec, X_all_samples, y_all_labels, k=K_NEIGHBORS)
                    
                    if np.isinf(local_entropy): # 跳过无法计算熵的情况
                        print(f"  [SKIPPED] Cannot calculate entropy for: {sent}")
                        continue

                    if local_entropy < STRICT_ENTROPY_THRESHOLD:
                        print(f"  [GOLDEN ACCEPTED] Entropy: {local_entropy:.4f} (< {STRICT_ENTROPY_THRESHOLD:.2f}) | Sentence: {sent}")
                        golden_for_this_tech.append({'tech_id': tech_id, 'sentence': sent, 'entropy': local_entropy})
                        generated_for_this_tech.append({'tech_id': tech_id, 'sentence': sent})
                    elif local_entropy < current_threshold:
                        print(f"  [ACCEPTED] Entropy: {local_entropy:.4f} (< {current_threshold:.2f}) | Sentence: {sent}")
                        generated_for_this_tech.append({'tech_id': tech_id, 'sentence': sent})
                    else:
                        print(f"  [REJECTED] Entropy: {local_entropy:.4f} (>= {current_threshold:.2f}) | Sentence: {sent}")
                        rejected_entropies.append(local_entropy)
            except Exception as e:
                print(f"  [ERROR] An exception occurred: {e}")
                time.sleep(2)
                continue
        third_pass_sentences.extend(generated_for_this_tech)
        golden_pass_sentences_pass3.extend(golden_for_this_tech)

    # --- 5. 保存并生成最终报告 ---
    print(f"\n--- 步骤 5/5: 保存结果并生成最终分析报告 ---")
    pd.DataFrame(third_pass_sentences).to_csv(THIRD_PASS_OUTPUT_PATH, index=False)
    print(f"第三轮（补录）过程完成，所有被接纳的 {len(third_pass_sentences)} 个样本已保存到 '{THIRD_PASS_OUTPUT_PATH}'。")
    
    pd.DataFrame(golden_pass_sentences_pass3).to_csv(GOLDEN_SAMPLES_OUTPUT_PATH, index=False)
    print(f"在第三轮补录过程中，共发现 {len(golden_pass_sentences_pass3)} 个'黄金样本'（熵<1.0），已单独保存到 '{GOLDEN_SAMPLES_OUTPUT_PATH}'。")

    print(f"正在生成最终的统计报告 '{FINAL_REPORT_PATH}'...")
    third_pass_counts = pd.Series(third_pass_sentences).apply(lambda x: x['tech_id']).value_counts()
    # 最终总数 = 原始 + Pass1 + Pass2 + Pass3
    final_total_counts = current_total_counts.add(third_pass_counts, fill_value=0)
    
    report_data = []
    for tech_id in sorted(graph_map.keys()):
        initial = int(initial_counts.get(tech_id, 0))
        pass1_gen = int(first_pass_counts.get(tech_id, 0))
        pass2_gen = int(second_pass_counts.get(tech_id, 0))
        pass3_gen = int(third_pass_counts.get(tech_id, 0)) # 新增第三轮计数
        final_total = int(final_total_counts.get(tech_id, 0))
        
        status = "已达标 (>=10)" if final_total >= TARGET_SAMPLE_COUNT_THIS_PASS else "未达标 (<10)"

        report_data.append({
            "TechID": tech_id,
            "Initial": initial,
            "Pass1_Gen": pass1_gen,
            "Pass2_Gen": pass2_gen,
            "Pass3_Gen": pass3_gen, # 新增列
            "Final_Total": final_total,
            "Status": status
        })

    report_df = pd.DataFrame(report_data)
    
    with open(FINAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("                         数据集增强最终状态分析报告 (三轮后)\n")
        f.write("="*90 + "\n\n")
        
        summary = (
            f"第一轮目标样本数/类别: 50\n"
            f"第三轮目标样本数/类别: {TARGET_SAMPLE_COUNT_THIS_PASS}\n"
            f"总技术类别数: {len(report_df)}\n"
            f"最终达标的类别数 (>=10): {len(report_df[report_df['Status'] == '已达标 (>=10)'])}\n"
            f"最终未达标的类别数 (<10): {len(report_df[report_df['Status'] == '未达标 (<10)'])}\n\n"
        )
        f.write("--- 总体概览 ---\n")
        f.write(summary)
        
        f.write("\n--- 所有类别详细列表 ---\n")
        # 调整列宽以适应新列
        pd.set_option('display.width', 120)
        report_string = report_df.to_string(index=False)
        f.write(report_string)
        
    print(f"最终报告已成功保存到 '{FINAL_REPORT_PATH}'。")
    print("\n第三轮补录完成！")

if __name__ == '__main__':
    main()