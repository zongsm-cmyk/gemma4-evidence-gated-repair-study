# Evidence-Gated Repository Repair — Gemma 4 Paper Track

**Research by Mhamed Askoure, 2026-11-01.**

This repository provides reproducible research assets for a paper investigating **conditional code-search fallback and bounded verification in Gemma 4 coding agents**. The Kaggle writeup is a separate artifact and is not officially submitted merely by publishing this repository.

## Empirical method

An original **injected retrieval failure** microbenchmark on real Python repository symbol definitions. Graph candidates are simulated (not the official Kaggle code graph) with seeded failures and sometimes misleading high confidence. Three code-localization policies are contrasted: graph-only (minimal tools), unconditional graph+exact symbol search, and **confidence-gated fallback**. The data are derived only from real symbol definitions observed in a supplied local repository; noisy candidate responses are controlled injections.

The experiment reports exact localization accuracy, average tool operations, error rates by failure intensity, and the conditional policy's trade-off. It **does not** evaluate a Gemma model, patch correctness, hidden validation tests, or official competition score.

Run after cloning:
```sh
python studies/symbol_fallback_ablation.py --repo /path/to/repository --out results/symbol_fallback_receipt.json
```

Read the generated JSON with the exact input repository commit, sample size, scenario settings and measured outcomes. The script needs only the Python standard library. Its `--repo` argument must point to a real existing Python source repository with enough parsable methods and functions.

## Related Gemma agent versions (prior artifacts)

B22 is semantic-first; B22.1 uses one bounded textual fallback; B23 adds targeted verification; B26 uses conditional graph routing and a budget-aware submission phase. These are **agent prompt-policy proposals**, and archive validity is not accepted patch quality. Official hidden test comparisons are pending.

## Validity and transparency

Fault-injection outcomes characterize control-policy robustness in a *simulated retrieval channel*; an observed gain here would not imply a gain on Google/Kaggle's real Graph Search or SWE-bench tasks. This repository is meant to make the research verifiable, not to invent an agent benchmark score.

Source license: MIT for original benchmark code and documentation. Data extracted from third-party repositories retain the original repository licenses; source repository and commit identifiers are preserved in experiment receipts.

## Paper track

https://www.kaggle.com/competitions/gemma-4-developer-agent-paper

Official Kaggle Writeup submission status must be verified independently.
