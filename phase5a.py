# =============================================================================
# PHASE 5A · MASTER FEATURE ENGINEERING (POLARS)
# Objective:
#   - Flatten [T x N] scenario matrices into master tabular datasets
#   - Strictly prevent leakage by shifting all dynamic inputs to t-1
#   - Build deployable Sim2Real proxies and privileged Oracle latents
#   - Export scenario-specific parquet files for Phase 5B / 5C
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import polars as pl

# -----------------------------------------------------------------------------
# 1) Paths and configuration
# -----------------------------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/"
SCENARIO_DIR = os.path.join(BASE_DIR, "scenarios/")


LISTINGS_PATH = os.path.join(BASE_DIR, "ABM_listings_1.csv")
WEIGHTS_PATH   = os.path.join(BASE_DIR, "ABM_spatial_weights.csv")
PARAMS_PATH    = os.path.join(BASE_DIR, "ABM_params.json")
PROCESSED_DIR  = os.path.join(BASE_DIR, "ml_ready/")

os.makedirs(PROCESSED_DIR, exist_ok=True)

BURN_IN_DAYS = 30
RANDOM_SEED = 42

SCENARIOS = [
    "tourism_boom",
    "economic_crash",
    "localized_contagion",
    "baseline_no_shock",   # optional diagnostic reference
]

TRAIN_SCENARIOS = [
    "tourism_boom",
    "economic_crash",
]

TEST_SCENARIOS = [
    "localized_contagion",
]

# -----------------------------------------------------------------------------
# 2) Load metadata and parameters
# -----------------------------------------------------------------------------
with open(PARAMS_PATH, "r") as f:
    abm_params = json.load(f)

D_DIFF = float(abm_params["D_diff"])

W_df = pd.read_csv(WEIGHTS_PATH, index_col=0)
W = W_df.values.astype(float)
districts = W_df.index.astype(str).tolist()
D = len(districts)

meta_pd = pd.read_csv(LISTINGS_PATH, dtype={"id": str})
meta_pd["id"] = meta_pd["id"].astype(str)
meta_pd["neighbourhood_cleansed"] = meta_pd["neighbourhood_cleansed"].astype(str)

print("=" * 72)
print("PHASE 5A: MASTER FEATURE ENGINEERING")
print("Objective: build leakage-safe datasets for Sim2Real + Oracle models")
print("=" * 72)

# -----------------------------------------------------------------------------
# 3) Feature specifications for later phases
# -----------------------------------------------------------------------------
FEATURES_SIM2REAL = [
    "velocity_1d",
    "velocity_3d",
    "velocity_7d",
    "rolling_7d_sigma",
    "diffusion_proxy",
    "ceiling_proximity",
    "floor_proximity",
]

FEATURES_ORACLE = FEATURES_SIM2REAL + [
    "diffusion_real_t_1",
    "reaction_real_t_1",
]

FEATURES_ORACLE_FULL = FEATURES_ORACLE + [
    "demand_real_t_1",
    "bounds_real_t_1",
    "noise_real_t_1",
]

# -----------------------------------------------------------------------------
# 4) Helpers
# -----------------------------------------------------------------------------
def load_component_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    # Ensure string-like column labels for agent ids
    df.columns = df.columns.astype(str)
    return df


def validate_component_alignment(component_dict: dict, scenario_label: str):
    """
    Make sure all scenario matrices have:
      - same shape
      - same agent ordering
    """
    ref_cols = list(component_dict["price"].columns)
    ref_shape = component_dict["price"].shape

    for name, df in component_dict.items():
        if list(df.columns) != ref_cols:
            if set(df.columns) != set(ref_cols):
                missing = sorted(list(set(ref_cols) - set(df.columns)))
                extra = sorted(list(set(df.columns) - set(ref_cols)))
                raise ValueError(
                    f"[{scenario_label}] Column mismatch in '{name}'.\n"
                    f"Missing columns: {missing[:10]}\n"
                    f"Extra columns: {extra[:10]}"
                )
            # Same set, different order -> reorder
            component_dict[name] = df[ref_cols]

        if df.shape != ref_shape:
            raise ValueError(
                f"[{scenario_label}] Shape mismatch in '{name}'. "
                f"Expected {ref_shape}, got {df.shape}"
            )

    return component_dict


def build_district_matrices(meta_aligned: pd.DataFrame):
    """
    Build the coarse-grained mapping matrices:
      M_mean   : agent -> district averaging
      M_expand : district -> agent expansion
    """
    N = len(meta_aligned)
    M_mean = np.zeros((N, D), dtype=float)
    M_expand = np.zeros((N, D), dtype=float)

    districts_series = meta_aligned["neighbourhood_cleansed"].astype(str).values

    for d_idx, dist in enumerate(districts):
        mask = (districts_series == dist)
        count = int(mask.sum())
        if count > 0:
            M_mean[mask, d_idx] = 1.0 / count
            M_expand[mask, d_idx] = 1.0

    return M_mean, M_expand


def build_spatial_proxy(price_t1_np: np.ndarray, M_mean: np.ndarray, M_expand: np.ndarray):
    """
    Approximate the observable spatial lag proxy from t-1 prices:
      1) average within district
      2) diffuse across district matrix W
      3) map back to agents

    This is a coarse-grained proxy, not a strict full N x N Laplacian.
    """
    # Replace NaNs only for the calculation step
    safe_price = np.nan_to_num(price_t1_np, nan=0.0)

    district_prices = safe_price @ M_mean          # [T x D]
    diffused_district_prices = district_prices @ W.T
    spatial_proxy = diffused_district_prices @ M_expand.T  # [T x N]

    # Restore NaN on rows where shifted price is undefined
    spatial_proxy[np.isnan(price_t1_np)] = np.nan
    return spatial_proxy


def process_scenario_master(scenario_label: str):
    """
    Build one master dataset for a scenario:
      - price_t_1, target price, target delta
      - observable proxy features
      - privileged latent physics from Phase 4
      - static metadata
      - leakage-safe burn-in filtering
    """
    print(f"\nProcessing scenario: {scenario_label}")

    # -------------------------------------------------------------------------
    # Load scenario matrices exported by Phase 4
    # -------------------------------------------------------------------------
    scenario_files = {
        "price": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_price.parquet"),
        "demand": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_demand.parquet"),
        "diffusion": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_diffusion.parquet"),
        "reaction": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_reaction.parquet"),
        "bounds": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_bounds.parquet"),
        "noise": os.path.join(SCENARIO_DIR, f"synthetic_{scenario_label}_noise.parquet"),
    }

    missing = [name for name, path in scenario_files.items() if not os.path.exists(path)]
    if missing:
        print(f"  ⚠ Skip {scenario_label}: missing files -> {missing}")
        return None

    price_df = load_component_parquet(scenario_files["price"])
    demand_df = load_component_parquet(scenario_files["demand"])
    diff_df = load_component_parquet(scenario_files["diffusion"])
    react_df = load_component_parquet(scenario_files["reaction"])
    bounds_df = load_component_parquet(scenario_files["bounds"])
    noise_df = load_component_parquet(scenario_files["noise"])

    components = {
        "price": price_df,
        "demand": demand_df,
        "diffusion": diff_df,
        "reaction": react_df,
        "bounds": bounds_df,
        "noise": noise_df,
    }
    components = validate_component_alignment(components, scenario_label)

    # -------------------------------------------------------------------------
    # Align agents to metadata
    # -------------------------------------------------------------------------
    agents = list(price_df.columns)
    N = len(agents)
    T = len(price_df)

    meta_indexed = meta_pd.set_index("id")
    try:
        meta_aligned = meta_indexed.loc[agents].reset_index()
    except KeyError as e:
        raise ValueError(
            f"[{scenario_label}] Some agent ids in scenario files are missing from listings metadata."
        ) from e

    meta_aligned["id"] = meta_aligned["id"].astype(str)
    meta_aligned["neighbourhood_cleansed"] = meta_aligned["neighbourhood_cleansed"].astype(str)

    # Static metadata (kept for analysis and conditioning)
    meta_static = pl.DataFrame({
        "agent_id": meta_aligned["id"].astype(str).values,
        "neighbourhood_cleansed": meta_aligned["neighbourhood_cleansed"].values,
        "local_p_min": meta_aligned["local_p_min"].astype(float).values,
        "local_p_max": meta_aligned["local_p_max"].astype(float).values,
        "local_sigma": meta_aligned["local_sigma"].astype(float).values,
    })

    # -------------------------------------------------------------------------
    # Build coarse-grained observable proxy from t-1 prices
    # -------------------------------------------------------------------------
    M_mean, M_expand = build_district_matrices(meta_aligned)

    price_t1_df = price_df.shift(1)
    demand_t1_df = demand_df.shift(1)
    diff_t1_df = diff_df.shift(1)
    react_t1_df = react_df.shift(1)
    bounds_t1_df = bounds_df.shift(1)
    noise_t1_df = noise_df.shift(1)

    price_t1_np = price_t1_df.to_numpy(dtype=float)
    spatial_proxy_np = build_spatial_proxy(price_t1_np, M_mean, M_expand)

    # -------------------------------------------------------------------------
    # Build base long-format table
    # -------------------------------------------------------------------------
    base_df = pl.DataFrame({
        "time_step": np.repeat(np.arange(T), N),
        "agent_id": np.tile(np.array(agents, dtype=str), T),
        "y_target_price": price_df.to_numpy(dtype=float).reshape(-1),
        "price_t_1": price_t1_np.reshape(-1),
        "spatial_lag_proxy_t_1": spatial_proxy_np.reshape(-1),
        "demand_real_t_1": demand_t1_df.to_numpy(dtype=float).reshape(-1),
        "diffusion_real_t_1": diff_t1_df.to_numpy(dtype=float).reshape(-1),
        "reaction_real_t_1": react_t1_df.to_numpy(dtype=float).reshape(-1),
        "bounds_real_t_1": bounds_t1_df.to_numpy(dtype=float).reshape(-1),
        "noise_real_t_1": noise_t1_df.to_numpy(dtype=float).reshape(-1),
    })
    base_df = base_df.with_columns([
        (pl.col("y_target_price") - pl.col("price_t_1")).alias("y_target_delta")
    ])
    # Merge static metadata
    base_df = base_df.join(meta_static, on="agent_id", how="left")

    # -------------------------------------------------------------------------
    # Temporal feature engineering
    # -------------------------------------------------------------------------


    # --- 4. Feature Engineering ---
        # --- 4. Feature Engineering ---
    final_df = (
        base_df
        .sort(["agent_id", "time_step"])
        .with_columns([
            pl.col("price_t_1").shift(1).over("agent_id").alias("price_t_2"),
            pl.col("price_t_1").shift(3).over("agent_id").alias("price_t_4"),
            pl.col("price_t_1").shift(7).over("agent_id").alias("price_t_8"),
            pl.col("price_t_1").rolling_std(window_size=7).over("agent_id").alias("rolling_7d_sigma"),
        ])
        .with_columns([
            # Observable Proxy: Price Velocities
            (pl.col("price_t_1") - pl.col("price_t_2")).alias("velocity_1d"),
            (pl.col("price_t_1") - pl.col("price_t_4")).alias("velocity_3d"),
            (pl.col("price_t_1") - pl.col("price_t_8")).alias("velocity_7d"),

            # Properly scaled proxy
            (pl.lit(D_DIFF) * (pl.col("spatial_lag_proxy_t_1") - pl.col("price_t_1"))).alias("diffusion_proxy"),

            # Bounding proxies
            (pl.col("local_p_max") - pl.col("price_t_1")).alias("ceiling_proximity"),
            (pl.col("price_t_1") - pl.col("local_p_min")).alias("floor_proximity"),

            pl.lit(scenario_label).alias("scenario_regime"),
        ])
    )
    # final_df = (
    #     base_df
    #     .sort(["agent_id", "time_step"])
    #     .with_columns([
    #         # Historical lags on price_t_1
    #         pl.col("price_t_1").shift(1).over("agent_id").alias("price_t_2"),
    #         pl.col("price_t_1").shift(3).over("agent_id").alias("price_t_4"),
    #         pl.col("price_t_1").shift(7).over("agent_id").alias("price_t_8"),
    #         pl.col("price_t_1").rolling_std(window_size=7).over("agent_id").alias("rolling_7d_sigma"),

    #         # Target delta
    #         (pl.col("y_target_price") - pl.col("price_t_1")).alias("y_target_delta"),

    #         # Observable velocities
    #         (pl.col("price_t_1") - pl.col("price_t_2")).alias("velocity_1d"),
    #         (pl.col("price_t_1") - pl.col("price_t_4")).alias("velocity_3d"),
    #         (pl.col("price_t_1") - pl.col("price_t_8")).alias("velocity_7d"),

    #         # Observable proxy for diffusion
    #         (pl.lit(D_DIFF) * (pl.col("spatial_lag_proxy_t_1") - pl.col("price_t_1"))).alias("diffusion_proxy"),

    #         # Bounding proxies
    #         (pl.col("local_p_max") - pl.col("price_t_1")).alias("ceiling_proximity"),
    #         (pl.col("price_t_1") - pl.col("local_p_min")).alias("floor_proximity"),

    #         # Scenario tag
    #         pl.lit(scenario_label).alias("scenario_regime"),
    #     ])
    # )

    # -------------------------------------------------------------------------
    # Leakage-safe filtering
    # -------------------------------------------------------------------------
    # Remove the initialization period. Shock starts at day 30, so we keep only
    # the post-burn-in dynamics that are relevant for Phase 5.
    final_df = final_df.filter(pl.col("time_step") >= BURN_IN_DAYS)

    # Remove any null rows created by temporal shifts / rolling windows
    final_df = final_df.drop_nulls()

    # Keep a clean column order for downstream phases
    preferred_order = [
        "time_step",
        "agent_id",
        "neighbourhood_cleansed",
        "scenario_regime",
        "y_target_price",
        "y_target_delta",
        "price_t_1",
        "price_t_2",
        "price_t_4",
        "price_t_8",
        "rolling_7d_sigma",
        "spatial_lag_proxy_t_1",
        "diffusion_proxy",
        "ceiling_proximity",
        "floor_proximity",
        "velocity_1d",
        "velocity_3d",
        "velocity_7d",
        "demand_real_t_1",
        "diffusion_real_t_1",
        "reaction_real_t_1",
        "bounds_real_t_1",
        "noise_real_t_1",
        "local_p_min",
        "local_p_max",
        "local_sigma",
    ]
    remaining_cols = [c for c in final_df.columns if c not in preferred_order]
    final_df = final_df.select(preferred_order + remaining_cols)

    # -------------------------------------------------------------------------
    # Save scenario dataset
    # -------------------------------------------------------------------------
    out_path = os.path.join(PROCESSED_DIR, f"master_xgb_ready_{scenario_label}.parquet")
    final_df.write_parquet(out_path)

    print(f"  ✓ Saved: {out_path}")
    print(f"    Rows: {final_df.height:,} | Cols: {final_df.width}")

    return final_df


# -----------------------------------------------------------------------------
# 5) Generate scenario-specific datasets
# -----------------------------------------------------------------------------
scenario_frames = {}
scenario_manifest = {}

for scenario in SCENARIOS:
    df_out = process_scenario_master(scenario)
    if df_out is not None:
        scenario_frames[scenario] = df_out
        scenario_manifest[scenario] = {
            "output_parquet": os.path.join(PROCESSED_DIR, f"master_xgb_ready_{scenario}.parquet"),
            "rows": int(df_out.height),
            "cols": int(df_out.width),
        }

# -----------------------------------------------------------------------------
# 6) Build combined train / test / all datasets for later phases
# -----------------------------------------------------------------------------
combined_paths = {}

if "tourism_boom" in scenario_frames and "economic_crash" in scenario_frames:
    train_master = pl.concat(
        [scenario_frames["tourism_boom"], scenario_frames["economic_crash"]],
        how="vertical_relaxed"
    )
    train_path = os.path.join(PROCESSED_DIR, "phase5_train_master.parquet")
    train_master.write_parquet(train_path)
    combined_paths["train_master"] = train_path
    print(f"\n✓ Saved combined train master -> {train_path} ({train_master.height:,} rows)")

if "localized_contagion" in scenario_frames:
    test_master = scenario_frames["localized_contagion"]
    test_path = os.path.join(PROCESSED_DIR, "phase5_test_master.parquet")
    test_master.write_parquet(test_path)
    combined_paths["test_master"] = test_path
    print(f"✓ Saved combined test master  -> {test_path} ({test_master.height:,} rows)")

if len(scenario_frames) > 0:
    all_master = pl.concat(list(scenario_frames.values()), how="vertical_relaxed")
    all_path = os.path.join(PROCESSED_DIR, "phase5_all_master.parquet")
    all_master.write_parquet(all_path)
    combined_paths["all_master"] = all_path
    print(f"✓ Saved combined all master   -> {all_path} ({all_master.height:,} rows)")

# -----------------------------------------------------------------------------
# 7) Save feature specification and manifest
# -----------------------------------------------------------------------------
feature_spec = {
    "burn_in_days": BURN_IN_DAYS,
    "shock_start_day": 30,
    "features_sim2real": FEATURES_SIM2REAL,
    "features_oracle": FEATURES_ORACLE,
    "features_oracle_full": FEATURES_ORACLE_FULL,
    "target_price": "y_target_price",
    "target_delta": "y_target_delta",
    "notes": [
        "Sim2Real features are deployable and do not use privileged latent physics.",
        "Oracle features include true latent physics exported from Phase 4.",
        "All dynamic features are shifted to t-1 to avoid leakage.",
        "diffusion_proxy is a coarse-grained observable proxy, not a strict Laplacian."
    ],
}

feature_spec_path = os.path.join(PROCESSED_DIR, "phase5_feature_spec.json")
with open(feature_spec_path, "w") as f:
    json.dump(feature_spec, f, indent=2)

manifest = {
    "scenario_manifest": scenario_manifest,
    "combined_paths": combined_paths,
    "feature_spec_path": feature_spec_path,
}
manifest_path = os.path.join(PROCESSED_DIR, "phase5_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

# -----------------------------------------------------------------------------
# 8) Final report
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PHASE 5A COMPLETE")
print("=" * 72)
print("Deployable Sim2Real features:")
for f in FEATURES_SIM2REAL:
    print(f"  - {f}")

print("\nPrivileged Oracle features:")
for f in FEATURES_ORACLE:
    print(f"  - {f}")

print("\nExtra Oracle-only latent columns available in the master dataset:")
for f in ["demand_real_t_1", "bounds_real_t_1", "noise_real_t_1"]:
    print(f"  - {f}")

print(f"\nFeature spec saved to: {feature_spec_path}")
print(f"Manifest saved to    : {manifest_path}")
print("=" * 72)