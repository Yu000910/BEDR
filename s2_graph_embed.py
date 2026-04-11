# s2_graph_embed.py (修正版)
import json
import numpy as np
import networkx as nx
import tqdm
from utils_embedding import embed_sentences

print("正在加载结构化图谱 'intent_action_entity.json'...")
graph_map = json.load(open('intent_action_entity.json'))

# --- 步骤 1: 建立完整的图结构 ---
print("正在构建 NetworkX 图...")
G = nx.Graph()
for tech, g in graph_map.items():
    # 添加节点并标记类型
    G.add_node(tech, type='tech')
    if g.get('intent'):
        G.add_node(g['intent'], type='intent')
        G.add_edge(tech, g['intent'])
    for act in g.get('action_chain', []):
        if act.get('tool'):
            G.add_node(act['tool'], type='tool')
            G.add_edge(tech, act['tool'])
    for ent in g.get('entities', []):
        G.add_node(ent, type='entity')
        G.add_edge(tech, ent)

# --- 步骤 2: 为非技术节点（意图、工具、实体）生成嵌入 ---
print("正在为意图、工具、实体节点生成嵌入向量...")
node_embed = {}
non_tech_nodes = [n for n, d in G.nodes(data=True) if d['type'] != 'tech']

# 使用分批嵌入以提高效率
node_vectors = embed_sentences(non_tech_nodes)

for node, vec in zip(non_tech_nodes, node_vectors):
    node_embed[node] = vec

# --- 步骤 3: 计算技术节点（tech_id）的嵌入 ---
# 技术节点的向量 = 其所有邻居节点向量的平均值
print("正在通过邻居节点聚合，计算技术节点的嵌入向量...")
tech_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'tech']

for tech_node in tqdm.tqdm(tech_nodes, desc="聚合技术向量"):
    neighbor_vectors = []
    for neighbor in G.neighbors(tech_node):
        if neighbor in node_embed: # 确保邻居节点已经有向量了
            neighbor_vectors.append(node_embed[neighbor])
    
    if neighbor_vectors:
        # 计算平均向量作为技术节点的向量
        avg_vector = np.mean(np.array(neighbor_vectors), axis=0)
        node_embed[tech_node] = avg_vector
    else:
        # 如果一个技术没有任何有效的邻居，则用零向量填充（或用其自身ID嵌入作为备选）
        print(f"警告: 技术节点 {tech_node} 没有任何可用的邻居节点来生成向量。")
        node_embed[tech_node] = np.zeros(768) # 假设嵌入维度为768

# --- 步骤 4: 保存所有节点的嵌入 ---
np.save('graph_embed.npy', node_embed)
print(f"S2 完成：包含 {len(node_embed)} 个节点的 graph_embed.npy 已生成。")