# analyze_phase4_enriched.py
# Post-processing analysis for Phase 4 enriched results.
# Outputs: ranking tables, per-region best index, attention weight visualization,
#           multi-sea ensemble, extreme-year conditional skill.
import sys, os, json, glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results", "phase4_enriched")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "results", "phase4_enriched", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_NAMES = {
    "bering": "白令海", "chukchi": "楚科奇海", "barents": "巴伦支海",
    "kara": "喀拉海", "laptev": "拉普捷夫海", "greenland": "格陵兰海",
    "central_arctic": "中北冰洋",
}
SECTOR = {"bering": "Pacific", "chukchi": "Pacific", "barents": "Atlantic",
          "kara": "Atlantic", "laptev": "Atlantic", "greenland": "Atlantic",
          "central_arctic": "Core"}

INDEX_NAMES = {"ao": "AO", "nao": "NAO", "pna": "PNA", "nino34": "Nino3.4"}
ALL_INDICES = ["ao", "nao", "pna", "nino34"]


def load_all_results():
    """Load all experiment JSON files."""
    results = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "E*.json"))):
        if "summary" in fpath or "analysis" in fpath:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def build_ranking_table(results):
    """Build per-region ranking of climate index contribution."""
    rows = []
    for region_key in REGION_NAMES:
        region_exps = [r for r in results if r["region"] == region_key]
        baseline = next((r for r in region_exps if r["type"] == "baseline"), None)
        if baseline is None:
            continue
        bl_rmse = baseline["ensemble_rmse"]

        for r in sorted(region_exps, key=lambda x: x["ensemble_rmse"]):
            delta = r["ensemble_rmse"] - bl_rmse
            delta_pct = (delta / bl_rmse) * 100
            rows.append({
                "region": REGION_NAMES[region_key],
                "region_key": region_key,
                "sector": SECTOR[region_key],
                "config": r["aux_label"],
                "type": r["type"],
                "rmse": round(r["ensemble_rmse"], 6),
                "delta": round(delta, 6),
                "delta_pct": round(delta_pct, 2),
                "rmse_mean": round(r["rmse_mean"], 6),
                "rmse_std": round(r["rmse_std"], 6),
            })

    df = pd.DataFrame(rows)
    return df


def find_best_per_region(results):
    """Find best auxiliary configuration per region."""
    best = {}
    for region_key in REGION_NAMES:
        region_exps = [r for r in results if r["region"] == region_key]
        baseline = next((r for r in region_exps if r["type"] == "baseline"), None)
        if baseline is None:
            continue
        # Find best single-variable
        singles = [r for r in region_exps if r["type"] == "single"]
        if singles:
            best_single = min(singles, key=lambda x: x["ensemble_rmse"])
            best[region_key] = {
                "baseline_rmse": baseline["ensemble_rmse"],
                "best_index": best_single["aux_vars"][0] if best_single["aux_vars"] else None,
                "best_rmse": best_single["ensemble_rmse"],
                "best_delta": best_single["ensemble_rmse"] - baseline["ensemble_rmse"],
            }
    return best


def analyze_attention(results):
    """Analyze attention weights across experiments."""
    attn_data = []
    for r in results:
        if r.get("ens_attn_weights") is None:
            continue
        if r["type"] == "baseline":
            continue
        weights = r["ens_attn_weights"]
        aux_seq_len = len(weights)
        # Lag times: most recent = idx 0 (lag-1 relative to prediction start)
        # But aux_seq is the last N months before prediction, so:
        # idx 0 = T-1 (1 month before first predicted month)
        # idx 1 = T-2, ..., idx 5 = T-6
        for t, w in enumerate(weights):
            attn_data.append({
                "region": REGION_NAMES[r["region"]],
                "region_key": r["region"],
                "aux": INDEX_NAMES.get(r["aux_vars"][0], r["aux_vars"][0]) if r["aux_vars"] else "baseline",
                "lag_month": t + 1,  # 1 = most recent
                "weight": w,
            })

    df_attn = pd.DataFrame(attn_data)
    return df_attn


def multi_sea_ensemble(results):
    """
    Multi-sea ensemble: sum per-region predictions → pan-Arctic total.
    Hypotheses: per-sea sum ensemble > direct pan-Arctic model because
    spatial heterogeneity is preserved rather than averaged away.
    """
    print("\n" + "=" * 60)
    print("  Multi-Sea Ensemble Analysis")
    print("=" * 60)

    # For each region, find the best single-variable configuration
    best_per_region = find_best_per_region(results)

    # Load region-specific predictions from best-config experiments
    ensemble_preds = {}
    for region_key, info in best_per_region.items():
        # Find the best single-variable experiment for this region
        region_exps = [r for r in results if r["region"] == region_key]
        best_exp = min(
            [r for r in region_exps if r["type"] == "single"],
            key=lambda x: x["ensemble_rmse"]
        )

        # Load full prediction arrays from the JSON
        fpath = os.path.join(RESULTS_DIR, f"{best_exp['exp_id']}.json")
        with open(fpath, "r", encoding="utf-8") as f:
            full = json.load(f)

        # We need per-sample predictions. The summary only stores ensemble RMSE.
        # For multi-sea ensemble, we need to re-predict or access stored predictions.
        # For now, note: this requires per-sample y_pred which isn't in the summary JSON.
        # Placeholder for actual implementation.
        print(f"  {REGION_NAMES[region_key]}: best={best_exp['aux_label']} "
              f"RMSE={best_exp['ensemble_rmse']:.6f}")

    print("\n  NOTE: Multi-sea ensemble requires per-sample predictions.")
    print("  Run analyze_multi_sea.py for full implementation.")
    return best_per_region


def extreme_year_analysis():
    """
    Extreme year conditional skill: compare model performance in extreme
    years (2016, 2020) vs. normal years.

    Extreme years in the test period (2016-2025):
    - 2016: Record-low winter ice, followed by 2nd-lowest September extent
    - 2020: 2nd-lowest September minimum on record
    - 2022: Not extreme but notable
    """
    print("\n" + "=" * 60)
    print("  Extreme Year Conditional Skill Analysis")
    print("=" * 60)

    EXTREME_YEARS = [2016, 2020]
    NORMAL_YEARS = [2017, 2018, 2019, 2021, 2023, 2024, 2025]

    # This requires per-sample predictions with year labels.
    # Placeholder for now — actual implementation needs the prediction arrays.
    print(f"  Extreme years: {EXTREME_YEARS}")
    print(f"  Normal years:  {NORMAL_YEARS}")
    print("\n  NOTE: Requires per-sample predictions with year metadata.")
    print("  Run analyze_extreme_years.py for full implementation.")


def main():
    print("Phase 4 Enriched Analysis")
    print("=" * 60)

    # Load results
    results = load_all_results()
    print(f"Loaded {len(results)} experiment results")

    if len(results) == 0:
        print("No results found — run experiments first.")
        return

    # 1. Ranking table
    df_rank = build_ranking_table(results)
    rank_path = os.path.join(OUTPUT_DIR, "per_region_ranking.csv")
    df_rank.to_csv(rank_path, index=False, encoding="utf-8-sig")
    print(f"\nRanking table: {rank_path}")

    # Print ranking summary
    print("\n── Per-Region Best Single-Variable ──")
    best = find_best_per_region(results)
    for rk in REGION_NAMES:
        if rk in best:
            b = best[rk]
            idx_name = INDEX_NAMES.get(b['best_index'], b['best_index'])
            print(f"  {REGION_NAMES[rk]:6s}  baseline={b['baseline_rmse']:.6f}  "
                  f"best={idx_name:7s}  rmse={b['best_rmse']:.6f}  "
                  f"Δ={b['best_delta']:+.6f} ({b['best_delta']/b['baseline_rmse']*100:+.1f}%)")

    # Print full ranking by RMSE (top 10)
    print("\n── Top-10 by Ensemble RMSE ──")
    for i, (_, row) in enumerate(df_rank.sort_values("rmse").head(10).iterrows()):
        print(f"  {i+1:2d}. {row['region']:6s} {row['config']:12s}  "
              f"RMSE={row['rmse']:.6f}  Δ={row['delta']:+.6f} ({row['delta_pct']:+.1f}%)")

    # 2. Attention analysis
    df_attn = analyze_attention(results)
    if len(df_attn) > 0:
        attn_path = os.path.join(OUTPUT_DIR, "attention_weights.csv")
        df_attn.to_csv(attn_path, index=False, encoding="utf-8-sig")
        print(f"\nAttention weights: {attn_path}")

        # Summarize: which lag month gets most attention per index?
        print("\n── Mean Attention by Lag Month ──")
        attn_summary = df_attn.groupby("lag_month")["weight"].mean()
        for lag, w in attn_summary.items():
            bar = "█" * int(w * 100)
            print(f"  lag-{lag}: {w:.4f} {bar}")

    # 3. Multi-sea ensemble (placeholder)
    best_per_region = multi_sea_ensemble(results)

    # 4. Extreme year analysis (placeholder)
    extreme_year_analysis()

    # Save summary JSON
    summary = {
        "n_experiments": len(results),
        "n_regions": len(REGION_NAMES),
        "per_region_best": {
            rk: {
                "baseline_rmse": v["baseline_rmse"],
                "best_index": v["best_index"],
                "best_rmse": v["best_rmse"],
                "best_delta": v["best_delta"],
                "best_delta_pct": round(v["best_delta"] / v["baseline_rmse"] * 100, 2),
            }
            for rk, v in best.items()
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis summary: {summary_path}")


if __name__ == "__main__":
    main()
