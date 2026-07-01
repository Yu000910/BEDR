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
api_key = "your-dashscope-api-key"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## API Reproducibility Notes

### LLM (All Pipeline Stages)

| Parameter | Value |
|-----------|-------|
| Provider | DeepSeek |
| Model | `deepseek-chat` (DeepSeek-V3) |
| Base URL | `https://api.deepseek.com` |
| Temperature | 0.0 (deterministic decoding) |
| Generation Period | December 2025 – April 2026 |

### Embedding (Step 2: s2_graph_embed.py)

| Parameter | Value |
|-----------|-------|
| Provider | Aliyun DashScope |
| Model | `text-embedding-v4` |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Dimensions | 768 |
| Generation Period | December 2025 – April 2026 |

### Prompts

All LLM prompts are preserved in their entirety within the released Python scripts (not redacted or summarized):

| Script | Prompt Purpose |
|--------|---------------|
| `s1_desc2graph.py` | System prompt for extracting Intent/Action/Entity graphs from technique descriptions |
| `s3_fourth_pass_augmentation.py` | Prompt for generating diverse technique example sentences with boundary constraints |
| `utils_llm.py` | Paraphrase batch prompt for semantic variation |
| `generate_native_llm_direct.py` | Direct LLM generation prompt (baseline, no boundary entropy filtering) |

### Random Seeds

All scripts use fixed random seeds (`random_state=42`, `seed=42`) for data splits and sampling. These are explicitly set in every script that involves randomness.

### On API-Based Reproducibility

The BEDR pipeline depends on external API services (DeepSeek for LLM augmentation, DashScope for embedding). To address this:

- **Intermediate outputs are provided.** All augmentation CSV files (`aug_sentences_*.csv`) serve as cached LLM outputs, allowing Step 4 (dataset organization) and all downstream Deep-AttacKG experiments to execute without re-calling the LLM API.
- **The embedding file** (`all_samples_intent_sentence_embed.npz`, 44 MB) exceeds GitHub's file size limit. Its generation command is documented (Step 2: `s2_graph_embed.py`) with the exact model and parameters specified.
- **Decoding is deterministic.** Temperature is set to 0.0 for all LLM calls. Given the same model version and input, outputs are reproducible.
- **Model version evolution is expected.** API providers may update underlying model weights over time (e.g., `deepseek-chat` may point to a newer checkpoint). This is a natural characteristic of API-based research and typically improves capability. The fixed methodology (prompts, parameters, seeds, intermediate data) ensures the scientific findings remain verifiable independent of the specific API model version.

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
