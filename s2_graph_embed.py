# s2_graph_embed.py
import json
import numpy as np
import networkx as nx
import pandas as pd
import ast
import tqdm
from utils_embedding import embed_sentences

print("Loading structured graph 'intent_action_entity.json'...")
graph_map = json.load(open('intent_action_entity.json'))

# --- Step 1: Build full graph structure ---
print("Building NetworkX graph...")
G = nx.Graph()
for tech, g in graph_map.items():
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

# --- Step 2: Embed non-tech nodes (intents, tools, entities) ---
print("Embedding intent, tool, and entity nodes...")
node_embed = {}
non_tech_nodes = [n for n, d in G.nodes(data=True) if d['type'] != 'tech']

node_vectors = embed_sentences(non_tech_nodes)
for node, vec in zip(non_tech_nodes, node_vectors):
    node_embed[node] = vec

# --- Step 3: Compute tech node embeddings via neighbor aggregation ---
print("Computing tech node embeddings via neighbor aggregation...")
tech_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'tech']

for tech_node in tqdm.tqdm(tech_nodes, desc="Aggregating tech vectors"):
    neighbor_vectors = []
    for neighbor in G.neighbors(tech_node):
        if neighbor in node_embed:
            neighbor_vectors.append(node_embed[neighbor])

    if neighbor_vectors:
        avg_vector = np.mean(np.array(neighbor_vectors), axis=0)
        node_embed[tech_node] = avg_vector
    else:
        print(f"Warning: tech node {tech_node} has no embeddable neighbors.")
        node_embed[tech_node] = np.zeros(768)

# --- Step 4: Embed all training samples from cleaned_duplicate_data.csv ---
# This generates the 'vectors' and 'labels' keys that s3 expects for entropy filtering.
print("Embedding training samples for entropy reference vectors...")
original_df = pd.read_csv('cleaned_duplicate_data.csv')
all_sample_texts = []
all_sample_labels = []

for _, row in original_df.iterrows():
    try:
        exs = ast.literal_eval(row['examples'])
        if isinstance(exs, list):
            for e in exs:
                if isinstance(e, str) and e.strip():
                    all_sample_texts.append(e.strip())
                    all_sample_labels.append(row['tech_id'])
    except:
        pass

print(f"  Total training samples: {len(all_sample_texts)}")
sample_vectors = embed_sentences(all_sample_texts)
vectors_array = np.array(sample_vectors)
labels_array = np.array(all_sample_labels)

# --- Step 5: Save combined .npz ---
# This file serves dual purpose:
#   - 'vectors' + 'labels': used by s3 for boundary entropy filtering
#   - Per-node embeddings: used for graph analysis and intent lookup
save_dict = {
    'vectors': vectors_array,
    'labels': labels_array
}
# Merge graph node embeddings (using ** unpacking)
save_dict.update(node_embed)

np.savez('all_samples_intent_sentence_embed.npz', **save_dict)
np.save('graph_embed.npy', node_embed)
print(f"S2 complete: all_samples_intent_sentence_embed.npz generated "
      f"({len(node_embed)} nodes, {len(all_sample_texts)} training samples).")
