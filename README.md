# ducb_ogd

JAX implementation of discounted-UCB online gradient descent on a synthetic multi-source convex problem ([arXiv:2610.06329](https://arxiv.org/abs/2610.06329)).

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

## GPU smoke

Job 758191 (1x H200, 15 min) was submitted at 2026-10-07T17:06:14Z. All 32 H200s were allocated. Slurm's backfill start was 2026-10-07T22:36:00Z, after this session, so the job was cancelled and did not run. CPU tests passed.

## What does not match the paper

There is no LLM, and no SFT, DPO, or GRPO. Each source is a quadratic `0.5 ||theta - c_k||^2 + b_k`, and the loss is deterministic, so the paper's stochastic mini-batch is the identity. The sampler and the OGD step follow Algorithm 1: the first K steps visit each source, then the index is `mu_hat + r` with `r = 2 C sqrt(2 alpha log H(beta) / N)`, and `theta <- theta - eta grad`. `beta` and `eta` are the Theorem 2 choices. Double-EMA and the enforced-exploration mixture are implemented and off by default. The sampling rule uses a Gauss-Hermite approximation of the Gaussian argmax probabilities from Appendix B. The OGD step is unprojected.
