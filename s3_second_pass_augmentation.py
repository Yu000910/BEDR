# s3_second_pass_augmentation.py (最终修复版 - 修复动态阈值重置问题)
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
# 核心辅助函数 (已最终确定)
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
    k_nearest_indices = np.argpartition(distances, k)[:k]
    k_nearest_labels = all_sample_labels[k_nearest_indices]
    
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    print(f"    [DEBUG] Nearest {k} labels distribution: {dict(zip(unique_labels, counts))}")
    
    class_probs = counts / k
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-9))
    
    return entropy


def main():
    # --- 0. 全局配置 ---
    HYBRID_EMBEDDING_NPZ_PATH = 'all_samples_intent_sentence_embed.npz'
    GRAPH_MAP_PATH = 'intent_action_entity.json'
    ORIGINAL_DATA_PATH = 'cleaned_duplicate_data.csv'
    FIRST_PASS_AUG_PATH = 'aug_sentences_final_filtered.csv'
    
    SECOND_PASS_OUTPUT_PATH = 'aug_sentences_second_pass.csv'
    GOLDEN_SAMPLES_OUTPUT_PATH = 'stubborn_categories_golden_samples.csv'
    FINAL_REPORT_PATH = 'augmentation_summary_report.txt'
    
    TARGET_SAMPLE_COUNT = 50
    # 实时自适应动态阈值配置
    STRICT_ENTROPY_THRESHOLD = 1.0
    MAX_LENIENT_THRESHOLD = 3.0
    SUCCESS_RATE_TRIGGER = 0.20 # 20%的成功率及格线
    
    SAMPLES_PER_LLM_CALL = 10
    MAX_LLM_RETRIES = 5
    K_NEIGHBORS = 15

    # --- 1. 加载所有需要的数据 ---
    print("--- 步骤 1/5: 加载所有需要的数据 ---")
    try:
        graph_map = json.load(open(GRAPH_MAP_PATH))
        data = np.load(HYBRID_EMBEDDING_NPZ_PATH, allow_pickle=True)
        X_all_samples, y_all_labels = data['vectors'], data['labels']
        original_df = pd.read_csv(ORIGINAL_DATA_PATH)
        first_pass_df = pd.read_csv(FIRST_PASS_AUG_PATH)
        print("所有数据文件加载成功。\n")
    except FileNotFoundError as e:
        print(f"\n[错误!] 文件未找到: {e}！程序终止。")
        return
        
    # --- 2. 预计算“官方意图”的嵌入 ---
    print("--- 步骤 2/5: 预计算所有官方意图的嵌入向量 ---")
    official_intents = {tech_id: data.get('intent') for tech_id, data in graph_map.items() if data.get('intent')}
    all_intent_texts = list(set(official_intents.values()))
    intent_vectors = embed_sentences(all_intent_texts)
    intent_vector_lookup = dict(zip(all_intent_texts, intent_vectors))
    print(f"已为 {len(intent_vector_lookup)} 个独特的官方意图创建了向量查找字典。\n")

    # --- 3. 诊断当前状态，识别需要“补录”的类别 ---
    print("--- 步骤 3/5: 诊断当前数据集状态，识别需要进行“补录”的类别 ---")
    initial_counts = pd.Series({row['tech_id']: len(ast.literal_eval(row['examples'])) for _, row in original_df.iterrows() if isinstance(row['examples'], str) and row['examples'].startswith('[')})
    first_pass_counts = first_pass_df['tech_id'].value_counts()
    current_total_counts = initial_counts.add(first_pass_counts, fill_value=0)
    
    all_known_techs = list(graph_map.keys())
    second_pass_targets = [tech_id for tech_id in all_known_techs if current_total_counts.get(tech_id, 0) < TARGET_SAMPLE_COUNT]
    print(f"诊断完成，共识别出 {len(second_pass_targets)} 个样本数仍未达标的类别，将对它们进行补录。\n")

    # --- 4. 核心生成与过滤循环 (已修复动态阈值逻辑) ---
    print("--- 步骤 4/5: 开始对目标类别进行第二轮生成与过滤 ---")
    second_pass_sentences = []
    golden_pass_sentences = []
    
    for tech_id in tqdm.tqdm(second_pass_targets, desc="第二轮补录"):
        g = graph_map.get(tech_id)
        if not g: continue
        
        current_sample_count = current_total_counts.get(tech_id, 0)
        needed_samples = TARGET_SAMPLE_COUNT - int(current_sample_count)
        if needed_samples <= 0: continue

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
        
        for retry_count in range(MAX_LLM_RETRIES):
            if len(generated_for_this_tech) >= needed_samples: break
            
            # =================================================================
            # 核心逻辑修正：确保阈值只升不降
            # =================================================================
            threshold_reason = f"Strict (Current Base: {current_threshold:.2f})"
            if retry_count > 0: # 从第二次尝试开始，每次都重新评估
                success_rate = len(generated_for_this_tech) / total_candidates_generated if total_candidates_generated > 0 else 0
                
                if success_rate < SUCCESS_RATE_TRIGGER and rejected_entropies:
                    median_rejected_entropy = np.median(rejected_entropies)
                    # 建议值基于严格阈值和中位数的中间点
                    adaptive_suggestion = current_threshold + (median_rejected_entropy - current_threshold) * 0.5
                    
                    # 只有当建议值比当前阈值更高时，才更新
                    if adaptive_suggestion > current_threshold:
                        new_threshold = min(MAX_LENIENT_THRESHOLD, adaptive_suggestion)
                        threshold_reason = f"Adaptive (Success Rate {success_rate:.1%} low, raising from {current_threshold:.2f})"
                        current_threshold = new_threshold
                else:
                    threshold_reason = f"Strict (Success Rate {success_rate:.1%} OK, holding at {current_threshold:.2f})"
            # =================================================================

            print(f"\n[Attempt {retry_count + 1}/{MAX_LLM_RETRIES}] For {tech_id} (Need {needed_samples - len(generated_for_this_tech)} more). Threshold: {current_threshold:.2f} ({threshold_reason})")
            
            try:
                rsp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={'type': 'json_object'})
                content = rsp.choices[0].message.content
                candidate_sents = json.loads(content)
                if isinstance(candidate_sents, dict):
                    candidate_sents = next((v for v in candidate_sents.values() if isinstance(v, list)), [])

                if not isinstance(candidate_sents, list):
                    print(f"  [Warning] LLM did not return a list. Skipping this attempt.")
                    continue
                
                total_candidates_generated += len(candidate_sents)

                for sent in candidate_sents:
                    if len(generated_for_this_tech) >= needed_samples: break
                    if not isinstance(sent, str) or len(sent) < 15: continue
                    
                    hybrid_vec = get_embedding_for_new_sentence(tech_id, sent, graph_map, intent_vector_lookup)
                    local_entropy = calculate_local_boundary_entropy(hybrid_vec, X_all_samples, y_all_labels, k=K_NEIGHBORS)
                    
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
                print(f"  [ERROR] An exception occurred during generation or processing: {e}")
                time.sleep(2)
                continue
        second_pass_sentences.extend(generated_for_this_tech)
        golden_pass_sentences.extend(golden_for_this_tech)

    # --- 5. 保存并生成最终报告 ---
    print(f"\n--- 步骤 5/5: 保存结果并生成最终分析报告 ---")
    pd.DataFrame(second_pass_sentences).to_csv(SECOND_PASS_OUTPUT_PATH, index=False)
    print(f"第二轮（补录）过程完成，所有被接纳的 {len(second_pass_sentences)} 个样本已保存到 '{SECOND_PASS_OUTPUT_PATH}'。")
    
    pd.DataFrame(golden_pass_sentences).to_csv(GOLDEN_SAMPLES_OUTPUT_PATH, index=False)
    print(f"在补录过程中，共发现 {len(golden_pass_sentences)} 个'黄金样本'（熵<1.0），已单独保存到 '{GOLDEN_SAMPLES_OUTPUT_PATH}'。")

    print(f"正在生成最终的统计报告 '{FINAL_REPORT_PATH}'...")
    second_pass_counts = pd.Series(second_pass_sentences).apply(lambda x: x['tech_id']).value_counts()
    final_total_counts = current_total_counts.add(second_pass_counts, fill_value=0)
    
    report_data = []
    for tech_id in sorted(graph_map.keys()):
        initial = int(initial_counts.get(tech_id, 0))
        pass1_gen = int(first_pass_counts.get(tech_id, 0))
        pass2_gen = int(second_pass_counts.get(tech_id, 0))
        final_total = int(final_total_counts.get(tech_id, 0))
        
        status = "已达标" if final_total >= TARGET_SAMPLE_COUNT else "未达标"

        report_data.append({
            "TechID": tech_id,
            "Initial": initial,
            "Pass1_Generated": pass1_gen,
            "Pass2_Generated": pass2_gen,
            "Final_Total": final_total,
            "Status": status
        })

    report_df = pd.DataFrame(report_data)
    
    with open(FINAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                  数据集增强最终状态分析报告\n")
        f.write("="*80 + "\n\n")
        
        summary = (
            f"目标样本数/类别: {TARGET_SAMPLE_COUNT}\n"
            f"总技术类别数: {len(report_df)}\n"
            f"最终达标的类别数: {len(report_df[report_df['Status'] == '已达标'])}\n"
            f"最终未达标的类别数: {len(report_df[report_df['Status'] == '未达标'])}\n\n"
        )
        f.write("--- 总体概览 ---\n")
        f.write(summary)
        
        f.write("\n--- 所有类别详细列表 ---\n")
        report_string = report_df.to_string(index=False)
        f.write(report_string)
        
    print(f"最终报告已成功保存到 '{FINAL_REPORT_PATH}'。")
    print("\nS3 process completed.")


if __name__ == '__main__':
    main()