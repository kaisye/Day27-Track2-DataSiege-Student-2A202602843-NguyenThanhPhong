# Reflection

**Hardest faults.** The subtle `feature_skew`, `corpus_staleness` and `embedding_drift` instances. Their metrics sit inside the published mean ± 3σ baseline, so the bare baseline thresholds miss them. The baselines also have false-alarm cases: a few clean feature batches landed past the "max" (0.41–0.47σ), while real skew was ≥2.3σ. On the data side, subtle null-rate, freshness and volume faults hide in normal variance, and `std_amount` has no baseline at all. Lineage and contract faults were easy once I read the structured fields (upstream set, downstream count, `violations`) instead of only the numbers.

**What I built.** Exactly one metered call per event, so coverage is full and cost stays near budget. The calls are never repeated or skipped, because a missed fault costs about 4x more score than a false alarm or a slight overage. Each handler combines signals:
- Two-sided z-scores on row count and mean amount, using σ recovered from the baseline range, with the alert line at 2.4σ instead of 3σ.
- Alert lines at a fraction of the published max for one-sided metrics (null rate, staleness, contract delay, runtime, embedding shift and age).
- A learned upstream-edge set per job, to catch missing edges, plus a check for orphaned outputs.
- An online spread check on `std_amount` built from batches that looked clean.
- A wider feature-skew margin (0.75σ) than the baseline, which removed false positives seen on public.

**Results.** Practice: TPR 1.0, FPR about 0.01. Public: TPR 1.0, FPR about 0.02. Full coverage costs about 1.5 credits per event, so public runs about 9% over budget (the 220-credit budget on 160 events). Skipping events to save that would cost more TPR than it gains.

**Private.** TPR 0.63, FPR 0.01, no cost overage, score 31.07. Loosening the thresholds from the first pass (31.48, FPR 0) added false alarms and no catches, so the missed faults are very close to normal variance. I did not tune further against private feedback, to avoid overfitting.

**What I'd change.** Subtle faults are where the remaining score is, and the fractions and margins above were tuned on only about 80 clean events. With another pass I would (1) use a cheaper triage on `ctx.state`, such as repeated-batch or rolling-trend checks, to avoid some calls, (2) combine weak signals, for example 1.5σ on two metrics at once, instead of OR-ing single thresholds, and (3) build per-metric rolling baselines online so the thresholds adapt instead of being fixed constants.
