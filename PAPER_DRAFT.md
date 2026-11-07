# Evidence-Gated Repository Repair: Tool-Call Efficiency Under Noisy Code Localization

**Mhamed Askoure** — Original research working paper, 1 November 2026

**Target:** [Google Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper). **Current status (verified November 7, 2026):** [official Kaggle Writeup submitted](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/evidence-gated-repository-repair-with-gemma-4-a-b). This longer manuscript is the open reproducibility companion; submission does not mean a prize or cash was awarded.

## Abstract

Software-repair agents must choose between fast, potentially unreliable code-graph lookup and repeated text searches that consume a fixed tool budget. We study a conditional retrieval policy that performs one graph lookup and falls back to exact lexical search only when the candidate confidence is below a threshold. To isolate the controller independently of language-model quality, we built a reproducible failure-injection microbenchmark using 48 actual Python function symbols per corpus, extracted from two independently versioned repositories. We executed 4,800 deterministic controller decisions across 0%, 10%, 25%, 50%, and 75% simulated graph-failure conditions, allowing one quarter of injected faults to have misleading high confidence. At a 50% injected-failure rate, graph-only localization accuracy was 48.96%, conditional fallback reached 88.96%, and unconditional fallback reached 100%. Conditional fallback used 1.40 retrieval operations per query, versus 2.00 for unconditional search (30% fewer). These results establish a resource/robustness trade-off for the **controlled retrieval-failure simulator only**. They do **not** demonstrate improved patch acceptance, model-level performance, or the Gemma 4 competition score.

## 1. Introduction

An agent repairing a large repository has limited time to identify a production symbol, inspect relevant code, edit, verify and submit a patch. Calling every available retrieval tool incurs latency; trusting every first-ranked result risks editing an unrelated file. This is an allocation problem under a fixed tool-call budget.

We investigate whether a confidence gate can reduce redundant lexical searches without losing much localization reliability. The design connects to a family of prompt policies developed for Gemma 4: B22 prioritizes semantic code-graph lookup, B22.1 adds bounded fallback, B23 adds single-test verification, and B26 makes graph lookup conditional on evidence from the issue. The present experiment isolates **one component**: graph-to-text fallback. It does not make claims about the untested verification component.

## 2. Method and reproducibility

### 2.1 Source repositories

We extracted public function definitions from two actual Python source trees using `ast.parse`, then kept only non-private function names appearing **exactly once** in their respective repositories. Each condition sampled 48 such symbols using seed `20261101`.

| Repository | Source commit | Python source files scanned | Unique symbols |
| --- | --- | ---: | ---: |
| [Kaggle PostgreSQL challenge source](https://github.com/igorsiecz/kaggle_postgre_challenge) | `22c704c3e745c1f73eaa616bc91af2b91aed8e11` — owner's benchmarked local fork commit based on organizer source | 21 | 432 |
| [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) | `577e21c17ca2016e4d803561ce9a4ed69ec119bd` — observed local source revision | 815 | 3,350 |

The selected symbol labels and line numbers, and measured aggregate outcomes, appear in the public JSON receipts. Because exact function names are present in the proxy question, the lexical fallback uses a deliberately favorable unambiguous lookup. This is not intended to mimic ambiguous natural-language bug reports.

### 2.2 Three policies

**Graph-only (G):** one graph result is accepted, even if wrong.

**Unconditional (U):** one graph lookup plus one exact symbol lookup on every query.

**Confidence-gated (C):** one graph lookup; if returned confidence is below 0.70, call exact symbol lookup; otherwise accept the graph result.

A controlled fault replaces a correct graph candidate with the wrong symbol at one of five fixed rates. All correct graph results have simulated confidence 0.92. Faulty candidates have confidence 0.15, except that 25% of injected faults are assigned *misleading* confidence 0.85. This tests the gate under imperfect calibration instead of assuming every wrong result is detectable.

We performed ten seeded repetitions per failure rate per corpus. Each corpus supplied `48 × 10 × 5 = 2,400` decisions, giving **4,800** decisions in total. Each simulated call has equal unit cost; real network and inference latency were not measured.

### 2.3 Metrics

Localization accuracy is the proportion of selected targets correctly identified by the policy. Retrieval efficiency is mean graph + lexical tool calls per decision. We record Wilson 95% intervals for the aggregate decisions but caution that repeated trials share symbols and are not independent source tasks; the intervals are descriptive, not a causal generalization guarantee.

## 3. Results

Because graph noise was generated from identical seeds and source-independent injection settings in both corpora, the policy-level accuracy numbers below were the same across corpora. This is **not independent validation of a learned retriever**: the real repositories provide distinct ground-truth symbol labels, while the controlled fault generator determines the difference between retrieval policies.

| Injected graph-failure rate | Graph-only accuracy | Confidence-gated accuracy | Unconditional accuracy | Confidence-gated mean calls |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 100.00% | 100.00% | 100.00% | 1.000 |
| 10% | 90.00% | 97.50% | 100.00% | 1.075 |
| 25% | 75.83% | 92.92% | 100.00% | 1.171 |
| 50% | 48.96% | 88.96% | 100.00% | 1.400 |
| 75% | 22.92% | 80.63% | 100.00% | 1.577 |

At the 50% condition the conditional gate saved `(2.0 − 1.4) / 2.0 = 30%` of calls relative to unconditional search, but lost 11.04 accuracy percentage points because some wrong graph results were confidently presented. At the 10% condition the gate saved 46.25% of calls and lost 2.5 accuracy points. This establishes a tunable trade-off rather than proof of unconditional superiority.

Both public receipts contain the complete 48 selected ground-truth functions per source tree, counts, policy-specific Wilson intervals and all five tested conditions:

- [PostgreSQL-source experiment](results/postgres_repo_proxy.json)
- [Kaggle CLI experiment](results/kaggle_cli_proxy.json)

### 3.1 Interpretation for a Gemma 4 agent

The result supports a bounded control **hypothesis**: avoid redundant lexical search when a retrieval result is credible, but preserve exactly one fallback path when the graph result is weak. The 25% high-confidence fault condition indicates that calibration failure remains a critical weak point: a confident but irrelevant target prevents fallback. A production implementation should add a second independent signal (for example, actual source-file existence and issue-symbol agreement) before trusting graph confidence.

This is a design insight, not a score. The local B22/B23/B26 prompts express different routing and verification policies, but official head-to-head Gemma 4 evaluator outcomes have not been obtained. Prompt archive construction, ZIP validation, and a Kaggle notebook running successfully are not equivalent to resolving software bugs. We do not claim the hidden repair benchmark was improved.

### 3.2 Structural audit of Gemma variants

A separate audit of three existing, locally generated agent archives confirmed that **B22, B23 and B26 all reference the same `gemma-4-31b-it-qat-w4a16-ct` model**, but they expose different numbers of tools (6, 8 and 9 respectively). Archive SHA-256 digests and package sizes are captured in [the structural audit](results/gemma_agent_archives_audit.json). This is a reproducibility check, not task-level evaluation. Because available tools differ, future performance comparisons must explicitly control or stratify the tool set before attributing any effect to prompt policy. B23 and B26 remain unscored on the official repair benchmark.

### 3.3 Independent-seed stress test (additional simulation)

To test whether the identical cross-corpus numbers in Section 3 were merely an artifact of reusing the same pseudo-random fault schedule, we reran the **same simulator** with different fault seeds on the two source trees: `20261111` for the PostgreSQL repository and `20261112` for Kaggle CLI. Each new run reused 48 unique public AST symbols and 10 repetitions at five fault levels, adding **4,800 simulated decisions** (9,600 across initial and follow-up runs). This is a sensitivity analysis of the random fault schedule, not a held-out Gemma model or a new retrieval model.

| Fault rate | PostgreSQL graph-only | PostgreSQL gated | Kaggle CLI graph-only | Kaggle CLI gated |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 89.58% | 97.29% | 89.58% | 96.46% |
| 25% | 75.42% | 92.08% | 77.92% | 94.79% |
| 50% | 49.17% | 88.12% | 49.79% | 85.42% |
| 75% | 23.33% | 78.75% | 24.58% | 80.42% |

At 50% injected graph faults, confidence-gated search used 1.3896 calls/query on PostgreSQL and 1.3562 on Kaggle CLI, versus exactly 2.0000 for the unconditional policy. The advantage over graph-only and the residual vulnerability to confident-wrong results remain visible across these seeds. The differing numbers demonstrate random-schedule sensitivity; because faults are synthetically injected and the symbol tasks are reused, **neither the additional decisions nor the two source trees establish external validation or real software-repair gains**.

Machine-readable follow-up receipts: [PostgreSQL, seed 20261111](results/postgres_holdout_seed_20261111.json) and [Kaggle CLI, seed 20261112](results/kaggle_cli_holdout_seed_20261112.json). Receipt arithmetic and provenance checks are supplied in [the test suite](studies/test_research_receipts.py).

### 3.4 Official competition status and non-comparability (updated 7 November 2026)

The authenticated Kaggle submission page separately reports **Gemma4 B24 E4B Budgeted Agent, version 1: Succeeded, public score 0.03**. This is an official score for the owner's B24 E4B entry, **not** an outcome of the simulated retrieval study above and not evidence that the confidence gate improves patch acceptance. The **Gemma4 B25 Apache2 Baseline Reproduction, version 1** is now officially marked **Succeeded**, with **public score 0.15** as verified on 3 November 2026 from the owner's authenticated Kaggle submissions page. This is the owner's official score for an attributed Apache-2.0 reference reproduction, not a result of the simulator or the original reference author's reported score. B25 reproduces and credits a separate Apache-2.0 reference, and the reference author's score must not be attributed to this account. The B26 conditional-graph ablation has a completed private packaging notebook but **no scored competition submission**. Historical entries remain exception/timeout cases.

These are not matched experiments: B24 uses the E4B model, B25 uses a 31B model, and their tool budgets/prompts differ. Therefore no causal effect of retrieval gating, verification, or model size can be inferred from their current leaderboard statuses. Official page: https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions . The separate Paper Track Writeup reached **6/6 complete and was officially submitted November 7, 2026**. The latest submitted version contains the reproducible retrieval-fault results and their explicit validity limits; this does not imply a winning score or paper-track prize. This section records provenance and prevents the older paper-track draft's statement that no successful public score exists from being reused as current evidence.

## 4. Threats to validity

1. **Synthetic faults:** the graph candidates and confidences are simulated. Real model-generated retrieval may have very different error distributions.
2. **Easy unique-symbol tasks:** exact function names and unique AST definitions make text fallback especially effective; natural issues are harder and often refer to behavior instead of symbols.
3. **Operation count is not runtime:** one lexical search and one graph lookup may have unequal latency and costs.
4. **Reused task labels:** ten repetitions of the same 48 functions are correlated; reported intervals cannot establish generalization.
5. **No Gemma inference or repairs:** this is a deterministic control-policy proxy study. No actual patch quality, hidden tests, or leaderboard result was measured.
6. **Source and licensing:** Python source files belong to their original upstream repositories and retain those licenses. We publish metadata only, not their code.

## 5. Planned model-based extension

A follow-up experiment should fix the official Gemma-4-31B model, seed, sampling parameters and time budget, then compare B22, B22.1, B23 and B26 on the **same real software issue set** with captured tool logs. Report, per issue, first-candidate correctness, fallback trigger, changed production file, test outcome, tool calls, wall-clock runtime and official acceptance. Without matched real issue outcomes, no ranking among these agent prompt variants can be asserted.

## 6. Conclusion

A conditional fallback controller can improve localization accuracy over graph-only decisions in a controlled, reproducible error-injection setting while requiring fewer tool calls than unconditional search. Its reliability is bounded by confidence calibration; confidently wrong retrieval results evade the fallback. The present work contributes an open source reproducibility harness and transparent negative control for future Gemma 4 software-repair agent experiments, rather than a claim of hidden benchmark superiority.

## References

- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR 2024, https://www.swebench.com/.
- Google DeepMind and Kaggle, *Gemma 4 Developer Agent Competition*, https://www.kaggle.com/competitions/gemma-4-developer-agent.
- Google DeepMind and Kaggle, *Gemma 4 Developer Agent Paper Track*, https://www.kaggle.com/competitions/gemma-4-developer-agent-paper.
- Kaggle CLI open source code, https://github.com/Kaggle/kaggle-cli.
- Organizer's PostgreSQL challenge source, https://github.com/igorsiecz/kaggle_postgre_challenge.

**Status (November 7, 2026):** the [official Kaggle Paper Track Writeup is submitted](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/evidence-gated-repository-repair-with-gemma-4-a-b). This public manuscript is a more detailed supporting study. **No paper-track award, prize or cash receipt has been confirmed.**
