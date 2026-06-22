# Outfit Generator — Co-occurrence Recommendation System

A machine learning system that builds cohesive outfits from individual clothing items. Given one item (e.g. *Skinny Jeans*), it recommends the pieces that complete the look, learned from ~94,000 real outfits in the [Marqo/Polyvore](https://huggingface.co/datasets/Marqo/polyvore) dataset.

> **Accessibility motivation:** recommendations are driven by item *identity* and learned pairing patterns rather than color, so the system works for colorblind users who can't rely on color matching to coordinate an outfit.

*Built as a National Student Data Corps (NSDC) team project — Keyan Saguibo, Hana Tanisaka, Aarnav Dharia, Itzel Valdez.*

---

## What it does

Type or pass in a clothing item, and the model fills the remaining outfit slots with compatible pieces pulled from the dataset:

```
Input:  Skinny Jeans
Output: Shirt  + Jacket + Shoes   (real items with images)
```

It learns these pairings purely from how items co-occur in real outfits — a count-based collaborative-filtering approach.

## Approach

The system is a **co-occurrence recommender**: it counts how often labels appear together in the same outfit, then recommends the labels most frequently paired with the user's input. Two versions were built to show the progression from a simple baseline to a more detailed model.

| | Model 1 | Model 2 |
|---|---|---|
| **Learns on** | Broad categories (`Jeans`, `Shoes`) | Style sub-genres (`Skinny Jeans`, `Platform Shoes`) |
| **Key feature** | Simple, interpretable baseline | Blocks same-category items so it never recommends pants when you already have a dress |
| **Output detail** | "Add a shoe" | "Add platform shoes" |

Sub-genres are derived by scanning each item's text description for style keywords (`Skinny`, `Floral`, `Leather`, …) and combining them with the category.

## Results

Evaluated with **leave-one-out** prediction: hide one item from each test outfit and check whether the model recovers its category. Reported against a **popularity baseline** (always guess the most common categories) so the model's actual lift is visible, not just a raw accuracy number.

| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Popularity baseline | *see notebook* | *see notebook* | *see notebook* |
| Model 1 (category) | *see notebook* | *see notebook* | *see notebook* |
| Model 2 (sub-genre) | *see notebook* | *see notebook* | *see notebook* |

> Run `Outfit_Generator_Clean.ipynb` end-to-end to reproduce the table and the comparison chart. (Replace the placeholders above with your numbers once you've run it.)

The headline figure from the original presentation was **87.2% Top-5 category accuracy** on 2,820 test outfits. The cleaned notebook reframes this with top-1/top-3/top-5 and a baseline, which is a fairer picture: top-5 over a handful of categories is an easy bar, so the baseline-relative lift matters more than the raw number.

## Methodology highlights

- **Leak-free evaluation.** The train/test split is done by **outfit**, not by row, so items from the same outfit never appear in both sets. This is the single most important correctness decision in the project — splitting by row would let the model "see the answer" and inflate accuracy.
- **Reproducible.** The notebook loads directly from the public HuggingFace dataset and runs top-to-bottom — no manual downloads, no hardcoded file paths. A single random seed controls the split, the hidden-item choice, and sampling.
- **Honest baseline.** Every result is reported next to a popularity baseline so the model's contribution is measurable.

## Data cleaning

The raw Polyvore data mixes fashion with furniture, makeup, and home decor. Cleaning involved:

1. **Removing** ~260 non-clothing / out-of-scope categories.
2. **Consolidating** fine-grained labels into broad categories (`Boyfriend Jeans` → `Jeans`, `Sandals` → `Shoes`).
3. **Deriving** `outfit_id` from each item's ID and dropping single-item outfits (no pairing to learn).

## Repo structure

```
.
├── Outfit_Generator_Clean.ipynb   # End-to-end: load → clean → model → evaluate → demo
├── README.md
└── (optional) src/                # If you later split helpers out of the notebook
```

## Running it

Open the notebook in Google Colab or Jupyter and **Run all**. The first cell installs the one extra dependency (`datasets`); everything else uses standard libraries (`pandas`, `scikit-learn`, `matplotlib`, `Pillow`).

## Limitations

- **Popularity bias.** Co-occurrence favors common items — `Shirt` and `Shoes` get recommended regardless of true compatibility. PMI / lift weighting would surface more distinctive pairings.
- **Keyword-based sub-genres.** Sub-genres come from string matching, not learned text understanding, so coverage is uneven.
- **Category-level metric.** The accuracy measure rewards predicting the right *type* of item, not the right *style*, so it under-credits Model 2's finer detail.
- **No visual/color modeling.** The system doesn't yet reason about visual style or color harmony.

## Next steps

- **PMI-weighted scoring** to recommend distinctive rather than merely popular pairings.
- **Recall@k at the sub-genre level** to properly measure Model 2's added detail.
- **CLIP image embeddings** for visual compatibility — the natural "Model 3."
- **Hosted demo** (Streamlit / Gradio on HuggingFace Spaces) so anyone can try it from a link.

## Tech stack

Python · pandas · scikit-learn · matplotlib · Pillow · HuggingFace `datasets`
