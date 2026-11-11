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

🚧 **Coming soon.** Code and instructions to reproduce the experiments will be released here.

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

The project page code is released under the MIT License.
