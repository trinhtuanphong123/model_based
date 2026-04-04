# =============================================================================
# PHASE 5B · DUAL-MODEL TRAINING & BASELINE COMPARISON
# Objective:
#   - Establish Naive Baseline (ΔP = 0)
#   - Train Sim2Real (Deployable) & Oracle (Privileged) Models
#   - Evaluate information loss under unseen localized contagion
#   - Export predictions and fitted models for Phase 5C
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 0) Reproducibility
# -----------------------------------------------------------------------------
np.random.seed(42)
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# 1) Paths
# -----------------------------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/"
PROCESSED_DIR = os.path.join(BASE_DIR, "ml_ready/")

TRAIN_MASTER_PATH = os.path.join(PROCESSED_DIR, "phase5_train_master.parquet")
TEST_MASTER_PATH  = os.path.join(PROCESSED_DIR, "phase5_test_master.parquet")
ALL_MASTER_PATH   = os.path.join(PROCESSED_DIR, "phase5_all_master.parquet")

FEATURE_SPEC_PATH = os.path.join(PROCESSED_DIR, "phase5_feature_spec.json")
OUTPUT_PRED_PATH  = os.path.join(PROCESSED_DIR, "phase5b_predictions.parquet")
MODEL_SIM_PATH    = os.path.join(PROCESSED_DIR, "model_sim2real.json")
MODEL_ORC_PATH    = os.path.join(PROCESSED_DIR, "model_oracle.json")
REPORT_PATH       = os.path.join(PROCESSED_DIR, "phase5b_report.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 72)
print("PHASE 5B: RIGOROUS DUAL-MODEL TRAINING")
print("=" * 72)

# -----------------------------------------------------------------------------
# 2) Load feature spec and datasets
# -----------------------------------------------------------------------------
if not os.path.exists(FEATURE_SPEC_PATH):
    raise FileNotFoundError(
        f"Missing feature spec: {FEATURE_SPEC_PATH}\n"
        f"Run Phase 5A first."
    )

with open(FEATURE_SPEC_PATH, "r") as f:
    feature_spec = json.load(f)

FEATURES_SIM2REAL = feature_spec["features_sim2real"]
FEATURES_ORACLE = feature_spec["features_oracle"]
TARGET = feature_spec["target_delta"]
TARGET_PRICE = feature_spec["target_price"]

def load_master_dataset(preferred_path: str, fallback_paths: list[str]) -> pl.DataFrame:
    if os.path.exists(preferred_path):
        return pl.read_parquet(preferred_path)
    for p in fallback_paths:
        if os.path.exists(p):
            return pl.read_parquet(p)
    raise FileNotFoundError(
        f"None of the expected dataset files were found.\n"
        f"Checked: {[preferred_path] + fallback_paths}"
    )

print("\n[1/6] Loading datasets...")

train_pl = load_master_dataset(
    TRAIN_MASTER_PATH,
    [
        os.path.join(PROCESSED_DIR, "master_xgb_ready_tourism_boom.parquet"),
        os.path.join(PROCESSED_DIR, "master_xgb_ready_economic_crash.parquet"),
    ],
)

test_pl = load_master_dataset(
    TEST_MASTER_PATH,
    [
        os.path.join(PROCESSED_DIR, "master_xgb_ready_localized_contagion.parquet"),
    ],
)

train_df = train_pl.to_pandas()
test_df = test_pl.to_pandas()

print(f"  -> Training rows: {len(train_df):,}")
print(f"  -> Testing rows : {len(test_df):,}")

# -----------------------------------------------------------------------------
# 3) Data integrity checks
# -----------------------------------------------------------------------------
print("\n[2/6] Running data integrity checks...")

def check_required_columns(df: pd.DataFrame, required_cols: list[str], name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def check_integrity(df: pd.DataFrame, name: str):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        raise ValueError(f"{name} has no numeric columns to validate.")
    if num_df.isna().any().any():
        bad_cols = num_df.columns[num_df.isna().any()].tolist()
        raise ValueError(f"CRITICAL: NaN values found in {name}. Columns: {bad_cols}")
    if np.isinf(num_df.to_numpy()).any():
        raise ValueError(f"CRITICAL: Infinite values found in {name}.")

required_sim_cols = ["y_target_delta"] + FEATURES_SIM2REAL
required_orc_cols = ["y_target_delta"] + FEATURES_ORACLE
required_common_cols = ["price_t_1", "y_target_price", "agent_id", "time_step", "scenario_regime"]

check_required_columns(train_df, required_sim_cols + required_common_cols, "train_df")
check_required_columns(test_df, required_orc_cols + required_common_cols, "test_df")

check_integrity(train_df[required_sim_cols + ["price_t_1", "y_target_price"]], "train_df (selected)")
check_integrity(test_df[required_orc_cols + ["price_t_1", "y_target_price"]], "test_df (selected)")

print("  ✓ All matrices passed NaN/Inf checks.")

# -----------------------------------------------------------------------------
# 4) Split features / target
# -----------------------------------------------------------------------------
X_train_sim = train_df[FEATURES_SIM2REAL].copy()
X_train_orc = train_df[FEATURES_ORACLE].copy()
y_train = train_df[TARGET].to_numpy(dtype=float).ravel()

X_test_sim = test_df[FEATURES_SIM2REAL].copy()
X_test_orc = test_df[FEATURES_ORACLE].copy()
y_test_delta = test_df[TARGET].to_numpy(dtype=float).ravel()

base_price_test = test_df["price_t_1"].to_numpy(dtype=float)
actual_price_test = test_df[TARGET_PRICE].to_numpy(dtype=float)

# -----------------------------------------------------------------------------
# 5) Naive baseline
# -----------------------------------------------------------------------------
# ΔP = 0  =>  P_t_hat = P_{t-1}
pred_delta_naive = np.zeros_like(y_test_delta, dtype=float)
pred_price_naive = base_price_test + pred_delta_naive

# -----------------------------------------------------------------------------
# 6) Model configuration
# -----------------------------------------------------------------------------
def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

print("\n[3/6] Training models...")

model_sim = build_model()
model_sim.fit(X_train_sim, y_train)
print("  ✓ Sim2Real model trained.")

model_orc = build_model()
model_orc.fit(X_train_orc, y_train)
print("  ✓ Oracle model trained.")

# -----------------------------------------------------------------------------
# 7) Predictions and reconstruction
# -----------------------------------------------------------------------------
print("\n[4/6] Running zero-shot predictions on unseen contagion...")

pred_delta_sim = model_sim.predict(X_test_sim)
pred_delta_orc = model_orc.predict(X_test_orc)

pred_price_sim = base_price_test + pred_delta_sim
pred_price_orc = base_price_test + pred_delta_orc

# -----------------------------------------------------------------------------
# 8) Evaluation helpers
# -----------------------------------------------------------------------------
def eval_price(name: str, pred_price: np.ndarray):
    mae = mean_absolute_error(actual_price_test, pred_price)
    rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price))
    r2 = r2_score(actual_price_test, pred_price)
    mean_level = float(np.mean(actual_price_test))
    mae_pct = (mae / mean_level) * 100 if mean_level != 0 else np.nan
    print(f"  {name:<18} | MAE = {mae:.2f} | RMSE = {rmse:.2f} | R² = {r2:.4f} | MAE% = {mae_pct:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_pct": mae_pct}

def eval_delta(name: str, pred_delta: np.ndarray):
    mae = mean_absolute_error(y_test_delta, pred_delta)
    rmse = np.sqrt(mean_squared_error(y_test_delta, pred_delta))
    r2 = r2_score(y_test_delta, pred_delta)
    print(f"  {name:<18} | Δ MAE = {mae:.6f} | Δ RMSE = {rmse:.6f} | Δ R² = {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

print("\n[5/6] Final zero-shot evaluation (localized contagion)...")
print("-" * 72)

metrics_naive = eval_price("Naive Baseline", pred_price_naive)
metrics_sim   = eval_price("Sim2Real Model", pred_price_sim)
metrics_orc   = eval_price("Oracle Model", pred_price_orc)

print("-" * 72)
print("Δ-space sanity check")
metrics_delta_sim = eval_delta("Sim2Real Model", pred_delta_sim)
metrics_delta_orc = eval_delta("Oracle Model", pred_delta_orc)
print("-" * 72)

# -----------------------------------------------------------------------------
# 9) Gap analysis
# -----------------------------------------------------------------------------
uplift_vs_naive = ((metrics_naive["mae"] - metrics_sim["mae"]) / metrics_naive["mae"]) * 100 if metrics_naive["mae"] != 0 else np.nan
gap_pct = ((metrics_sim["mae"] - metrics_orc["mae"]) / metrics_orc["mae"]) * 100 if metrics_orc["mae"] != 0 else np.nan

print("\n" + "=" * 72)
print("SCIENTIFIC GAP ANALYSIS")
print("=" * 72)
print(f"Predictive value     : Sim2Real reduces MAE by {uplift_vs_naive:.2f}% vs Naive Baseline.")
print(f"Information-loss gap : {gap_pct:.2f}% between Oracle and Sim2Real.")
print("\nInterpretation:")
print("The gap is an upper bound of performance loss caused by restricted access")
print("to privileged latent physics. It includes proxy error, missing latent")
print("signals, and model bias. It is not a pure measure of diffusion loss.")
print("=" * 72)

# -----------------------------------------------------------------------------
# 10) Export predictions and model assets
# -----------------------------------------------------------------------------
print("\n[6/6] Exporting Phase 5B assets...")

output_df = pd.DataFrame({
    "agent_id": test_df["agent_id"].astype(str).values,
    "time_step": test_df["time_step"].values,
    "scenario_regime": test_df["scenario_regime"].astype(str).values,
    "spatial_lag_proxy_t_1": test_df["spatial_lag_proxy_t_1"].values,
    "price_t_1": base_price_test,
    "actual_price": actual_price_test,
    "actual_delta": y_test_delta,
    "pred_delta_naive": pred_delta_naive,
    "pred_delta_sim": pred_delta_sim,
    "pred_delta_orc": pred_delta_orc,
    "pred_price_naive": pred_price_naive,
    "pred_price_sim": pred_price_sim,
    "pred_price_orc": pred_price_orc,
})

output_df.to_parquet(OUTPUT_PRED_PATH, index=False)
model_sim.save_model(MODEL_SIM_PATH)
model_orc.save_model(MODEL_ORC_PATH)

print(f"  ✓ Predictions saved to: {OUTPUT_PRED_PATH}")
print(f"  ✓ Sim2Real model saved to: {MODEL_SIM_PATH}")
print(f"  ✓ Oracle model saved to   : {MODEL_ORC_PATH}")

# -----------------------------------------------------------------------------
# 11) Exploratory feature importance plots
# -----------------------------------------------------------------------------
# This is only descriptive. The actual interpretation step happens in Phase 5C.
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

def plot_importance(model, features, ax, title, color):
    imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)
    ax.barh(imp["Feature"], imp["Importance"], color=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Gain-based importance")
    ax.grid(True, axis="x", alpha=0.2)

plot_importance(model_sim, FEATURES_SIM2REAL, axes[0], "Sim2Real Feature Importance", "#2980b9")
plot_importance(model_orc, FEATURES_ORACLE, axes[1], "Oracle Feature Importance", "#8e44ad")

fig.suptitle(
    "Exploratory Gain Importance Only\nDo not use for causal claims. Use Phase 5C for ablation and SHAP.",
    color="red",
    fontweight="bold",
    fontsize=12
)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 12) Save report
# -----------------------------------------------------------------------------
report = {
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "naive": metrics_naive,
    "sim2real": metrics_sim,
    "oracle": metrics_orc,
    "delta_sim2real": metrics_delta_sim,
    "delta_oracle": metrics_delta_orc,
    "uplift_vs_naive_pct": float(uplift_vs_naive),
    "gap_oracle_vs_sim2real_pct": float(gap_pct),
    "feature_spec_path": FEATURE_SPEC_PATH,
    "predictions_path": OUTPUT_PRED_PATH,
    "model_sim_path": MODEL_SIM_PATH,
    "model_oracle_path": MODEL_ORC_PATH,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 72)
print("PHASE 5B COMPLETE")
print("=" * 72)
print(f"Report saved to: {REPORT_PATH}")
print("System is ready for Phase 5C: ablation, SHAP, and error conditioning.")
print("=" * 72)