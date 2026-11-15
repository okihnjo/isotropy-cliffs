# Isotropy Cliffs: The Geometric Signature of Decision-Making in Large Language Models

**Okan S. Coskun, Florian Rottach, Carsten Eickhoff, William Rudman**
University of Tübingen · The University of Texas at Austin

🌐 **Project page:** https://okihnjo.github.io/isotropy-cliffs/
📄 **Paper:** [arXiv:2608.00828](https://arxiv.org/abs/2608.00828)

## About

We study the geometry of decision-making in multiple-choice question answering through the lens of isotropy. Across five open-weight LLMs (Qwen2.5-3B, Qwen3-8B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B-v0.3) and four benchmarks (ARC-Easy, OpenBookQA, Head-QA, MMLU-Pro), we find:

- **Isotropy cliffs.** Isotropy rises through the early layers, peaks, and then drops sharply at a critical layer. Task-relevant clusters and a major representational change (CKA) appear at the same layer.
- **A strong predictor of accuracy.** The steepness of the cliff correlates with downstream accuracy (Spearman ρ = −0.84 overall, up to −0.92 per model) and outperforms intrinsic dimension.
- **Robust.** The cliff persists across four prompt formats and 0/2/4-shot in-context learning.
- **Causal.** Preventing the cliff at layer L+1 destroys the clusters and cuts accuracy by up to 38 points. Qwen3-8B is the exception: it loses its clusters and then rebuilds them a few layers later.

The project page has interactive demos of all of this.

## Code

A step-by-step notebook walks through the core analysis on a small scale: it runs **Qwen3-8B** on **OpenBookQA**, captures the hidden states of every layer, and computes **IsoScore**, **k-NN purity** and **CKA** to locate the isotropy cliff. Model, dataset, prompt format, number of shots and sample size are all set in a single config cell.

### Repository contents

| File | Description |
| --- | --- |
| `isotropy_cliffs_tutorial.ipynb` | The guide: markdown explanations alternating with runnable code |
| `geometric_metrics.py` | IsoScore, k-NN purity, CKA, TwoNN intrinsic dimension, cliff finder and plotting helpers |
| `index.html`, `assets/` | The project page |

### Quick start

Clone or download this repository, install the dependencies, and open the notebook:

```bash
pip install torch transformers accelerate datasets scikit-learn scikit-dimension \
            matplotlib seaborn tqdm scipy pandas bitsandbytes jupyter
jupyter notebook isotropy_cliffs_tutorial.ipynb
```

You need a GPU with about 10 GB of memory for Qwen3-8B in 8-bit (the default), or about 16 GB in float16. For a quick test on a small GPU or CPU, set `MODEL_NAME = "Qwen/Qwen2.5-3B"` and `SAMPLE_SIZE = 2100`. Gated models such as Llama need a Hugging Face login (`huggingface_hub.login()`).

### What the notebook does

1. **Data and prompts.** Loads OpenBookQA and builds multiple-choice prompts in one of the four formats from the paper's appendix, optionally with in-context examples.
2. **Forward pass.** Runs the model and stores the last-token hidden state at every layer as an array of shape `(n_samples, n_layers, hidden_dim)`, together with the model's predictions and accuracy. The embeddings are saved to `results/` so the forward pass only has to run once.
3. **Metrics per layer.** IsoScore (how uniformly variance is spread), k-NN purity (do points with the same answer cluster?) and CKA between neighbouring layers (how much does the representation change?).
4. **Find the cliff.** Reports the transition L → L+1 with the sharpest IsoScore drop, k-NN jump and CKA dip, plus the cliff steepness (ΔLog IsoScore), and draws an overview plot. A t-SNE view shows the layers just before and after the cliff.

### Configuration

All settings are in one cell near the top of the notebook:

| Setting | Meaning |
| --- | --- |
| `MODEL_NAME` | Hugging Face model, e.g. `Qwen/Qwen3-8B` |
| `LOAD_IN_8BIT` | Load in 8-bit (default, as in our scripts). If `False`, the model is loaded in float16 |
| `DATASET` | `openbookqa` or `arc_easy`; other multiple-choice datasets can be added to the `DATASETS` dictionary |
| `SAMPLE_SIZE` | Number of evaluated questions (see the notes below) |
| `PROMPT_FORMAT` | `0`–`3`, the four formats from the paper's appendix |
| `NUM_SHOTS` | Number of in-context examples (0 = zero-shot) |
| `KNN_K` | Number of neighbours for k-NN purity |

### Metrics (`geometric_metrics.py`)

All functions take embeddings of shape `(n_samples, n_layers, hidden_dim)`.

| Function | Returns |
| --- | --- |
| `isoscore_per_layer` | IsoScore of every layer |
| `knn_purity_per_layer` | k-NN purity of every layer (needs ground-truth labels) |
| `cka_neighboring_layers` | Linear CKA between layer `l` and `l+1` (one value fewer than layers) |
| `intrinsic_dim_per_layer` | TwoNN intrinsic dimension per layer (optional, needs `scikit-dimension`) |
| `find_cliff` | Layer transitions with the largest IsoScore drop, k-NN jump and CKA dip |
| `delta_log_iso` | Change in log IsoScore across a transition (the cliff steepness) |
| `plot_layerwise`, `plot_cliff_overview`, `plot_tsne_pair` | Plots |

### Notes

- **Use at least as many samples as hidden dimensions for IsoScore** (4096 for Qwen3-8B, 2048 for Qwen2.5-3B; we use about 5,850 OpenBookQA samples). With fewer samples the PCA has fewer components than dimensions, a warning is shown, and the values are not comparable to the paper.
- **`find_cliff` is a simple heuristic** (it ignores the last two transitions). Always check the plot, and set the cliff layer by hand if needed.


## Citation

```bibtex
@article{coskun2026isotropy,
  title   = {Isotropy Cliffs: The Geometric Signature of Decision-Making in Large Language Models},
  author  = {Coskun, Okan S. and Rottach, Florian and Eickhoff, Carsten and Rudman, William},
  journal = {arXiv preprint arXiv:2608.00828},
  year    = {2026}
}
```

## License

The code in this repository is released under the MIT License.
