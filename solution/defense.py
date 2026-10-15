"""
Data Siege defense: one metered call per event, then combine several signals
per event. Published baselines are mean +/- 3 sigma of the clean stream, so
we recover (center, sigma) from them and alert a little *inside* the 3-sigma
edge (Z_ALERT) to catch subtle faults, while requiring a clear excess so clean
variance does not become false alarms.
"""
from api import Verdict

Z_EMBED = 2.5        # one-sided z for embedding metrics
FEATURE_SHIFT_MIN = 0.75  # sigma; min mean-shift to call feature skew
Z_ALERT = 2.4        # two-sided z threshold on baseline-derived sigma


# One-sided metrics only publish a max (mean + 3 sigma). Faults are worth ~4x
# a false alarm in the score, so alert at this fraction of the published max,
# which sits near mean + 2.3..2.6 sigma on clean streams seen in practice/public.
MAX_FRACTION = {"null_rate": 0.87, "staleness": 0.9, "freshness": 0.85, "duration": 0.96}


def _err(r):
    return not isinstance(r, dict) or "error" in r


def _band(b, lo_key, hi_key):
    lo, hi = b[lo_key], b[hi_key]
    return (lo + hi) / 2.0, max((hi - lo) / 6.0, 1e-9)


def _result(alert, pillar, reasons, score=1.0):
    return Verdict(alert=bool(alert), confidence=min(1.0, score),
                   reason="; ".join(reasons), pillar=pillar)


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    b = ctx.baseline
    p = ctx.tools.batch_profile(payload["batch_id"])
    if _err(p):
        return Verdict(alert=False, reason="profile unavailable", pillar="checks")
    reasons = []

    c, s = _band(b, "row_count_min", "row_count_max")
    z = (p["row_count"] - c) / s
    if abs(z) > Z_ALERT:
        reasons.append(f"row_count z={z:.1f}")

    c, s = _band(b, "mean_amount_min", "mean_amount_max")
    z = (p["mean_amount"] - c) / s
    if abs(z) > Z_ALERT:
        reasons.append(f"mean_amount z={z:.1f}")

    # null rate and staleness are one-sided: only the published max applies.
    nulls = max(p["null_rate"].values()) if p.get("null_rate") else 0.0
    if nulls > b["null_rate_max"] * MAX_FRACTION["null_rate"]:
        reasons.append(f"null_rate {nulls:.4f}")
    if p["staleness_min"] > b["staleness_min_max"] * MAX_FRACTION["staleness"]:
        reasons.append(f"staleness {p['staleness_min']:.1f}m")

    # Online sanity check on spread: std_amount has no published baseline, so
    # learn it from batches that looked clean.
    hist = ctx.state.setdefault("std_hist", [])
    std = p.get("std_amount")
    if std is not None and len(hist) >= 8 and not reasons:
        m = sum(hist) / len(hist)
        sd = (sum((x - m) ** 2 for x in hist) / len(hist)) ** 0.5 or 1e-9
        zs = (std - m) / sd
        if abs(zs) > 4.0:
            reasons.append(f"std_amount z={zs:.1f}")
    if std is not None and not reasons:
        hist.append(std)

    return _result(reasons, "checks", reasons)


def check_contract_checkpoint(payload, ctx):
    b = ctx.baseline
    d = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if _err(d):
        return Verdict(alert=False, reason="diff unavailable", pillar="contracts")
    reasons = list(d.get("violations") or [])
    delay = d.get("freshness_delay_min", 0.0)
    sla = (payload.get("declared_sla") or {}).get("freshness_min")
    limit = b["freshness_delay_max_min"] * MAX_FRACTION["freshness"]
    if sla is not None:
        limit = min(limit, sla)
    if delay > limit:
        reasons.append(f"freshness_delay {delay:.1f}m > {limit:.1f}m")
    return _result(reasons, "contracts", reasons)


def check_lineage_run(payload, ctx):
    b = ctx.baseline
    g = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _err(g):
        return Verdict(alert=False, reason="lineage unavailable", pillar="lineage")
    reasons = []

    actual = set(g.get("actual_upstream") or [])
    declared = {i["name"] for i in payload.get("inputs", []) if "name" in i}
    if declared - actual:
        reasons.append(f"declared inputs missing: {sorted(declared - actual)}")

    # Learn which upstream edges this job normally has; a run lacking one that
    # most previous runs had is a missing edge.
    seen = ctx.state.setdefault("upstream_counts", {})
    runs = ctx.state.get("lineage_runs", 0)
    if runs >= 3:
        expected = {k for k, n in seen.items() if n / runs >= 0.6}
        if expected - actual:
            reasons.append(f"missing upstream: {sorted(expected - actual)}")
    ctx.state["lineage_runs"] = runs + 1
    for name in actual:
        seen[name] = seen.get(name, 0) + 1

    if g.get("actual_downstream_count", 1) < 1:
        reasons.append("orphaned output (no downstream)")
    if g.get("duration_ms", 0.0) > b["lineage_duration_ms_max"] * MAX_FRACTION["duration"]:
        reasons.append(f"duration {g['duration_ms']:.0f}ms")
    return _result(reasons, "lineage", reasons)


def check_feature_materialization(payload, ctx):
    b = ctx.baseline
    f = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _err(f):
        return Verdict(alert=False, reason="drift unavailable", pillar="ai_infra")
    reasons = []
    # Published max is mean+3sigma of a half-normal and clean batches were seen
    # slightly past it; real skew starts well above, so use a wider margin.
    limit = max(b["feature_mean_shift_sigma_max"], FEATURE_SHIFT_MIN)
    if f["mean_shift_sigma"] > limit:
        reasons.append(f"mean_shift {f['mean_shift_sigma']:.2f} sigma")
    return _result(reasons, "ai_infra", reasons)


def check_embedding_batch(payload, ctx):
    b = ctx.baseline
    e = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _err(e):
        return Verdict(alert=False, reason="drift unavailable", pillar="ai_infra")
    reasons = []
    # Max-only baselines: assume clean mean ~ max/2, sigma ~ (max-mean)/3 and
    # alert at Z_EMBED, i.e. inside the published 3-sigma edge, for subtle drift.
    for name, key in (("centroid_shift", "embedding_centroid_shift_max"),
                      ("avg_doc_age_days", "corpus_avg_doc_age_days_max")):
        edge = b[key]
        mean, sigma = edge / 2.0, edge / 6.0
        if e[name] > mean + Z_EMBED * sigma:
            reasons.append(f"{name} {e[name]:.3f}")
    return _result(reasons, "ai_infra", reasons)
