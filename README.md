---
title: AdaptAssist Environment
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# AdaptAssist — Data Directory

The large JSON files are not committed to keep the Space size small.
Run the commands below to regenerate them locally or in Colab.

---

## Quick regenerate (run in order)

```bash
# Step 1: Generate expert trajectories from the live environment (~80 samples)
python data/generate_sft_data.py \
    --n 80 \
    --output data/sft_dataset.json \
    --difficulties 1 2 \
    --seed 42

# Step 2: Build combined dataset from 5 sources (~7500 samples total)
python data/dataset_loader.py \
    --env_sft_path data/sft_dataset.json \
    --output data/combined_sft.json \
    --max_glaive 3000 \
    --max_helpful 2000 \
    --max_xlam 2000 \
    --n_drift_augmented 500
```

Both commands take ~5 minutes on CPU. No GPU needed.

---

## Dataset Card

| Source | Samples | HuggingFace ID | Purpose |
|--------|---------|----------------|---------|
| Glaive function calling | 3,000 | `glaiveai/glaive-function-calling-v2` | Teaches JSON tool-call format |
| Helpful instructions | 2,000 | `HuggingFaceH4/helpful_instructions` | Personal task instruction following |
| xLAM function calling | 2,000 | `Salesforce/xlam-function-calling-60k` | Structured API call variety |
| Drift-augmented (synthetic) | 500 | generated | Fail → detect → adapt sequences |
| Environment expert trajectories | ~80 | generated | AdaptAssist-specific tool names |
| **Total** | **~7,580** | | |

---

## What makes the drift-augmented samples unique

No public dataset contains `detect_schema_change` → field-rename recovery sequences.
These 500 samples are generated synthetically by `dataset_loader.py` and are
the only training signal for schema adaptation before RL starts.

Example sample structure:
```json
{
  "source": "drift_augmented",
  "messages": [
    {"role": "system", "content": "You are a personal assistant..."},
    {"role": "user",   "content": "Book a restaurant. Warning: drift active."},
    {"role": "assistant", "content": "{\"tool\": \"detect_schema_change\", \"params\": {\"module\": \"restaurant\"}, \"thought\": \"Drift warning is active. Let me check what changed before acting.\"}"},
    {"role": "user",   "content": "{\"drift_detected\": true, \"description\": \"available_slots renamed to open_times\"}"},
    {"role": "assistant", "content": "{\"tool\": \"book_restaurant\", \"params\": {\"restaurant_id\": \"rst_0\", \"time_slot\": \"19:00\"}}"}
  ],
  "reward": 0.95
}
```

---

## Files generated (not committed)

```
data/
    sft_dataset.json          # ~1.3 MB — expert env trajectories
    combined_sft.json         # ~1.4 MB — all 5 sources merged
    combined_sft_trl.json     # ~1.3 MB — TRL-formatted version
    sft_dataset_hf.json       # ~2.4 MB — HuggingFace Dataset format
```
