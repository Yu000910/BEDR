# BEDR: Boundary Entropy-Driven Resampling for CTI TTP Classification

This repository contains the official implementation of the BEDR (Boundary Entropy-Driven Resampling) framework. BEDR is also used as the training data construction pipeline for the **Deep-AttacKG** system (see [Yu000910/deep_attackg](https://github.com/Yu000910/deep_attackg)).

## Directory Structure

| File | Description |
|------|-------------|
| `s1_desc2graph.py` | Convert ATT&CK text descriptions into structured graphs (Intent/Action/Entity) |
| `s2_graph_embed.py` | Build hybrid vector space, compute and save reference vectors for entropy calculation |
| `s3_fourth_pass_augmentation.py` | Core BEDR program: multi-round adaptive oversampling with boundary entropy filtering |
| `organize_and_generate_datasets.py` | Boundary entropy-driven undersampling, clean majority classes, generate final training sets |
| `generate_native_llm_direct.py` | Baseline: direct LLM generation without boundary entropy filtering |
| `utils_llm.py` | OpenAI/DeepSeek API wrapper for LLM calls |
| `utils_embedding.py` | Embedding API wrapper for vector computation |

## Intermediate Data Files

| File | Size | Source |
|------|------|--------|
| `cleaned_duplicate_data.csv` | 2.1 MB | Input: cleaned CTI dataset |
| `intent_action_entity.json` | — | Output of `s1_desc2graph.py` |
| `all_samples_intent_sentence_embed.npz` | 44 MB | Output of `s2_graph_embed.py` |
| `aug_sentences_final_filtered.csv` | 808 KB | Output of `s3` (first pass) |
| `aug_sentences_second_pass.csv` | 526 KB | Output of `s3` (second pass) |
| `aug_sentences_third_pass.csv` | 27 KB | Output of `s3` (third pass) |
| `aug_sentences_fourth_pass.csv` | <1 KB | Output of `s3` (fourth pass) |
| `final_train_set_resampled.csv` | 1.8 MB | Output of `organize_and_generate_datasets.py` |
| `final_test_set_resampled.csv` | 464 KB | Output of `organize_and_generate_datasets.py` |
| `BEDR_resampled_dataset.csv` | 2.7 MB | Final BEDR resampled dataset |

## Environment Setup

```bash
conda create -n bedr python=3.10
conda activate bedr
pip install -r requirements.txt
```

## API Configuration

Edit `utils_llm.py` and `utils_embedding.py` to configure your API keys:

```python
# utils_llm.py
client = OpenAI(api_key="your-deepseek-api-key", base_url="https://api.deepseek.com")

# utils_embedding.py
api_key = "your-embedding-api-key"
```

**Models used:**
- LLM: `deepseek-chat` (temperature=0.0)
- Embedding: DashScope text-embedding or OpenAI-compatible embedding API

## Execution Order

### Step 1: Build Structured Graphs
```bash
python s1_desc2graph.py
```
- **Input:** `cleaned_duplicate_data.csv`
- **Output:** `intent_action_entity.json`
- **Runtime:** ~10 minutes

### Step 2: Build Hybrid Vector Space
```bash
python s2_graph_embed.py
```
- **Input:** `intent_action_entity.json`
- **Output:** `all_samples_intent_sentence_embed.npz`
- **Runtime:** ~30 minutes (depends on embedding API rate limits)

### Step 3: Multi-Round Adaptive Oversampling (BEDR Core)
```bash
python s3_fourth_pass_augmentation.py
```
- **Input:** `cleaned_duplicate_data.csv`, `intent_action_entity.json`, `all_samples_intent_sentence_embed.npz`
- **Output:** `aug_sentences_final_filtered.csv`, `aug_sentences_second_pass.csv`, `aug_sentences_third_pass.csv`, `aug_sentences_fourth_pass.csv`
- **Runtime:** ~2-4 hours (LLM generation + entropy computation)

### Step 4: Organize Final Datasets
```bash
python organize_and_generate_datasets.py
```
- **Input:** All augmentation pass CSVs, `cleaned_duplicate_data.csv`, `intent_action_entity.json`
- **Output:** `BEDR_resampled_dataset.csv`, `D_BEDR.npz`, train/test split files
- **Runtime:** ~15 minutes

### Baseline (Optional)
```bash
python generate_native_llm_direct.py
```
- Direct LLM augmentation without boundary entropy filtering (control experiment)

## Expected Final Outputs

After running all steps, the following key outputs are generated:
- `BEDR_resampled_dataset.csv` — Final resampled training dataset (21,453 samples, 679 classes)
- `D_BEDR.npz` — Vectorized training data for deep learning models
- `D_test.npz` — Vectorized test set

## Citation

If you use BEDR in your research, please cite the corresponding paper.

## Related Repositories

- [Yu000910/deep_attackg](https://github.com/Yu000910/deep_attackg) — Deep-AttacKG: Zero-Shot CTI Identification via Semantic Manifold Alignment (uses BEDR for training data construction)

## License

MIT License — see `LICENSE` file for details.
