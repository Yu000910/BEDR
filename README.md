# BEDR: Boundary Entropy-Driven Resampling for CTI TTP Classification

This repository contains the official implementation of the BEDR (Boundary Entropy-Driven Resampling) framework. BEDR is also used as the training data construction pipeline for the **Deep-AttacKG** system (see [Yu000910/deep_attackg](https://github.com/Yu000910/deep_attackg)).

**Release:** v2.0 | **Commit:** `d19f506`

## Directory Structure

| File | Description |
|------|-------------|
| `s1_desc2graph.py` | Convert ATT&CK text descriptions into structured graphs (Intent/Action/Entity) |
| `s2_graph_embed.py` | Build hybrid vector space, compute and save reference vectors for entropy calculation |
| `s3_graph2gen.py` | First-pass augmentation: graph-to-text generation with boundary entropy scoring |
| `s3_second_pass_augmentation.py` | Second-pass augmentation: targeted resampling for under-represented classes |
| `s3_third_pass_augmentation.py` | Third-pass augmentation: focused generation for stubborn categories |
| `s3_fourth_pass_augmentation.py` | Fourth-pass augmentation: final round of adaptive oversampling with boundary entropy filtering |
| `organize_and_generate_datasets.py` | Boundary entropy-driven undersampling, clean majority classes, generate final training sets |
| `validate_schema.py` | Schema validation & verb distribution analysis for `intent_action_entity.json` |
| `generate_native_llm_direct.py` | Baseline: direct LLM generation without boundary entropy filtering |
| `utils_llm.py` | OpenAI/DeepSeek API wrapper for LLM calls |
| `utils_embedding.py` | Embedding API wrapper for vector computation |

## Intermediate Data Files

| File | Size | Source |
|------|------|--------|
| `cleaned_duplicate_data.csv` | 2.1 MB | Input: cleaned CTI dataset |
| `intent_action_entity.json` | — | Output of `s1_desc2graph.py` |
| `all_samples_intent_sentence_embed.npz` | 44 MB | Output of `s2_graph_embed.py` (also in [Release](https://github.com/Yu000910/BEDR/releases/tag/v2.0-npz-cache)) |
| `aug_sentences_final_filtered.csv` | 808 KB | Output of `s3_graph2gen.py` (first pass) |
| `aug_sentences_second_pass.csv` | 526 KB | Output of `s3_second_pass_augmentation.py` (second pass) |
| `aug_sentences_third_pass.csv` | 27 KB | Output of `s3_third_pass_augmentation.py` (third pass) |
| `aug_sentences_fourth_pass.csv` | <1 KB | Output of `s3_fourth_pass_augmentation.py` (fourth pass) |
| `aug_sentences_naive_llm_direct.csv` | 3.0 MB | Output of `generate_native_llm_direct.py` (baseline) |
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
| Model | `deepseek-chat` (service alias; pointed to DeepSeek-V3 during experiments) |
| Base URL | `https://api.deepseek.com` |
| Temperature | 0.0 for graph extraction (`s1_desc2graph.py`); API default for augmentation (`s3`, `utils_llm.py`, `generate_native_llm_direct.py`) |
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
| `s3_graph2gen.py` | Prompt for first-pass graph-to-text generation with boundary constraints |
| `s3_second_pass_augmentation.py` | Prompt for targeted second-pass augmentation of under-represented classes |
| `s3_third_pass_augmentation.py` | Prompt for focused third-pass generation targeting stubborn categories |
| `s3_fourth_pass_augmentation.py` | Prompt for final-pass diverse technique example generation |
| `utils_llm.py` | Paraphrase batch prompt for semantic variation |
| `generate_native_llm_direct.py` | Direct LLM generation prompt (baseline, no boundary entropy filtering) |

### Random Seeds

All scripts use fixed random seeds (`random_state=42`, `seed=42`) for data splits and sampling. These are explicitly set in every script that involves randomness.

### On API-Based Reproducibility

The BEDR pipeline depends on external API services (DeepSeek for LLM augmentation, DashScope for embedding). To address this:

- **Intermediate outputs are provided.** All augmentation CSV files (`aug_sentences_*.csv`) serve as cached LLM outputs, allowing Step 4 (dataset organization) and all downstream Deep-AttacKG experiments to execute without re-calling the LLM API.
- **The embedding file** (`all_samples_intent_sentence_embed.npz`, 44 MB) is generated by Step 2 (`s2_graph_embed.py`) using the DashScope API (`text-embedding-v4`, 768-dimensional). It is also available for download from the [v2.0-npz-cache GitHub Release](https://github.com/Yu000910/BEDR/releases/tag/v2.0-npz-cache).
- **Decoding is deterministic where appropriate.** Temperature is set to 0.0 for graph extraction (`s1_desc2graph.py`). For data augmentation (`s3`, `utils_llm.py`, `generate_native_llm_direct.py`), the API default temperature is used to ensure sufficient diversity in generated training samples. Given the same model version and input, the graph extraction outputs are reproducible; augmentation outputs will exhibit natural variation.
- **Model version evolution is expected.** API providers may update underlying model weights over time (e.g., `deepseek-chat` may point to a newer checkpoint). This is a natural characteristic of API-based research and typically improves capability. The fixed methodology (prompts, parameters, seeds, intermediate data) ensures the scientific findings remain verifiable independent of the specific API model version.

## Schema Validation

To verify the structural integrity and verb distribution of `intent_action_entity.json`:

```bash
python validate_schema.py
```

This script reports: (i) structural schema validation (all entries conforming to `{intent, action_chain, entities}`), and (ii) verb distribution relative to the canonical vocabulary (percentage of tokens within/outside the recommended verb list). See the script's docstring for details.

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
- **Output:** `all_samples_intent_sentence_embed.npz` (also available for download from the [v2.0-npz-cache Release](https://github.com/Yu000910/BEDR/releases/tag/v2.0-npz-cache))
- **Runtime:** ~30 minutes (depends on embedding API rate limits)

### Step 3: Multi-Round Adaptive Oversampling (BEDR Core)

The BEDR augmentation pipeline runs in four sequential passes, each targeting progressively harder-to-augment classes. Cached outputs from Passes 1–3 are provided in the repository, so only the fourth pass needs to be executed:

```bash
python s3_fourth_pass_augmentation.py
```
- **Input:** `cleaned_duplicate_data.csv`, `intent_action_entity.json`, `all_samples_intent_sentence_embed.npz`, plus cached outputs from earlier passes (`aug_sentences_final_filtered.csv`, `aug_sentences_second_pass.csv`, `aug_sentences_third_pass.csv`)
- **Output:** `aug_sentences_fourth_pass.csv`
- **Runtime:** ~2-4 hours (LLM generation + entropy computation)

To reproduce the full pipeline from scratch (re-running all four passes):
```bash
python s3_graph2gen.py                    # Pass 1: graph-to-text generation
python s3_second_pass_augmentation.py     # Pass 2: targeted resampling
python s3_third_pass_augmentation.py      # Pass 3: stubborn categories
python s3_fourth_pass_augmentation.py     # Pass 4: final adaptive round
```

### Step 4: Organize Final Datasets
```bash
python organize_and_generate_datasets.py
```
- **Input:** All augmentation pass CSVs, `cleaned_duplicate_data.csv`, `intent_action_entity.json`, `aug_sentences_naive_llm_direct.csv`, and the following **cached vector files** (pre-computed via the DashScope embedding API during the original experiment run):
  - `all_samples_plain_embeddings_final.npz` — BEDR sample embeddings
  - `all_plain_sentence_embeddings.npz` — naive baseline embeddings
  - `final_train_resampled_vectors.npz` — training set vectors with encoded labels
  - `final_test_resampled_vectors.npz` — test set vectors with encoded labels
- **Output:** `BEDR_resampled_dataset.csv`, `D_BEDR.npz`, `D_test.npz`, `D_train_BEDR.csv`, train/test split files
- **Runtime:** ~15 minutes
- **Note:** The cached `.npz` files are available from the [v2.0-npz-cache GitHub Release](https://github.com/Yu000910/BEDR/releases/tag/v2.0-npz-cache). Download all five `.npz` files and place them in the project root. If they are missing, the script will report which files are needed and exit with a clear error message. To regenerate them instead, use the embedding API parameters documented in the [API Configuration](#api-configuration) section.

### Baseline (Optional)
```bash
python generate_native_llm_direct.py
```
- Direct LLM augmentation without boundary entropy filtering (control experiment)
- **Output:** `aug_sentences_naive_llm_direct.csv` (also provided as cached output)

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
