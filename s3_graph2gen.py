# verb-tool-project/s3_graph2gen.py (最终正确版 - 已修复tqdm报错)
import json
import pandas as pd
import tqdm # 保持库的导入
import numpy as np
from scipy.spatial.distance import cdist
import time
import random

# 导入我们需要的核心函数
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
    
    # 为新生成的句子原文动态生成向量
    sentence_vec = embed_sentences([sentence_text])[0]
    
    # “拼接”向量
    if intent_vec is not None and sentence_vec is not None:
        combined_vec = (intent_vec + sentence_vec) / 2.0
    elif sentence_vec is not None:
        combined_vec = sentence_vec
    else:
        return np.zeros(768) # 如果句子向量也生成失败，返回零向量

    # 归一化
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
    print(f"    [DEBUG] Nearest {k} labels: {dict(zip(unique_labels, counts))}")
    
    class_probs = counts / k
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-8))
    
    return entropy


def main():
    # --- 0. 全局配置 ---
    HYBRID_EMBEDDING_NPZ_PATH = 'all_samples_intent_sentence_embed.npz'
    GRAPH_MAP_PATH = 'intent_action_entity.json'
    TARGET_SAMPLE_COUNT = 50
    ENTROPY_THRESHOLD = 1.0 
    SAMPLES_PER_LLM_CALL = 5
    MAX_LLM_RETRIES = 10
    K_NEIGHBORS = 15

    # --- 1. 加载所有需要的数据 ---
    print("--- 步骤 1/4: 加载所有需要的数据 ---")
    try:
        graph_map = json.load(open(GRAPH_MAP_PATH))
        data = np.load(HYBRID_EMBEDDING_NPZ_PATH, allow_pickle=True)
        X_all_samples = data['vectors']
        y_all_labels = data['labels']
        print(f"成功加载！参照空间包含 {len(y_all_labels)} 个最终的混合向量。\n")
    except FileNotFoundError as e:
        print(f"\n[错误!] 文件未找到: {e}！请确保所有必需文件都在目录中。程序终止。")
        return
        
    # --- 2. 预计算“官方意图”的嵌入 ---
    print("--- 步骤 2/4: 预计算所有官方意图的嵌入向量 ---")
    official_intents = {tech_id: data.get('intent') for tech_id, data in graph_map.items() if data.get('intent')}
    all_intent_texts = list(set(official_intents.values()))
    intent_vectors = embed_sentences(all_intent_texts)
    intent_vector_lookup = dict(zip(all_intent_texts, intent_vectors))
    print(f"已为 {len(intent_vector_lookup)} 个独特的官方意图创建了向量查找字典。\n")

    # --- 3. 统计样本数并识别小样本 ---
    print("--- 步骤 3/4: 识别需要增强的小样本技术 ---")
    all_tech_counts = pd.Series(y_all_labels).value_counts()
    all_known_techs = list(graph_map.keys())
    minority_techs = [tech_id for tech_id in all_known_techs if all_tech_counts.get(tech_id, 0) < TARGET_SAMPLE_COUNT]
    print(f"已根据参照空间完成样本计数，共识别出 {len(minority_techs)} 个需要进行过采样的小样本技术。\n")

    # --- 4. 核心生成与过滤循环 ---
    print("--- 步骤 4/4: 开始核心生成与过滤循环 ---")
    final_augmented_sentences = []
    
    # 核心修正：修复了此处的 tqdm 调用错误
    for tech_id in tqdm.tqdm(minority_techs, desc="扩充技术样本"):
        g = graph_map.get(tech_id)
        if not g: continue
        current_sample_count = all_tech_counts.get(tech_id, 0)
        needed_samples = TARGET_SAMPLE_COUNT - current_sample_count
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
["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5"]
"""
        
        generated_for_this_tech = []
        for retry_count in range(MAX_LLM_RETRIES):
            if len(generated_for_this_tech) >= needed_samples: break
            print(f"\n[Attempt {retry_count + 1}/{MAX_LLM_RETRIES}] Generating for {tech_id} (Need {needed_samples - len(generated_for_this_tech)} more)...")
            try:
                rsp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={'type': 'json_object'})
                content = rsp.choices[0].message.content
                candidate_sents_data = json.loads(content)
                candidate_sents = next((v for v in candidate_sents_data.values() if isinstance(v, list)), []) if isinstance(candidate_sents_data, dict) else candidate_sents_data

                if not isinstance(candidate_sents, list):
                    continue

                for sent in candidate_sents:
                    if not isinstance(sent, str) or len(sent) < 15:
                        continue
                    
                    # 关键步骤：使用与参照空间完全一致的方法为新句子生成向量
                    hybrid_vec = get_embedding_for_new_sentence(tech_id, sent, graph_map, intent_vector_lookup)
                    
                    # 在完美的参照空间中计算边界熵
                    local_entropy = calculate_local_boundary_entropy(hybrid_vec, X_all_samples, y_all_labels, k=K_NEIGHBORS)
                    
                    if local_entropy < ENTROPY_THRESHOLD:
                        print(f"  [ACCEPTED] Entropy: {local_entropy:.4f} (< {ENTROPY_THRESHOLD}) | Sentence: {sent}")
                        generated_for_this_tech.append({'tech_id': tech_id, 'sentence': sent})
                        if len(generated_for_this_tech) >= needed_samples:
                            break
                    else:
                        print(f"  [REJECTED] Entropy: {local_entropy:.4f} (>= {ENTROPY_THRESHOLD}) | Sentence: {sent}")

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(2)
                continue
        final_augmented_sentences.extend(generated_for_this_tech)

    pd.DataFrame(final_augmented_sentences).to_csv('aug_sentences_final_filtered.csv', index=False)
    print('\nS3 process completed.')

if __name__ == '__main__':
    main()