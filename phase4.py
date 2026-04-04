# =============================================================================
# PHASE 4 · THE SCENARIO MATRIX (SYNTHESIZED DATA GENERATION)
# Objective:
#   - Inject sustained extreme F(t) shocks
#   - Extract full PDE latent states from Phase 2
#   - Export all matrices as Parquet for Phase 5 (ML) and validation
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1) Setup paths and global configuration
# -----------------------------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/"
EXPORT_DIR = os.path.join(BASE_DIR, "scenarios/")
os.makedirs(EXPORT_DIR, exist_ok=True)

LISTINGS_PATH = os.path.join(BASE_DIR, "ABM_listings_1.csv")
WEIGHTS_PATH  = os.path.join(BASE_DIR, "ABM_spatial_weights.csv")
PARAMS_PATH   = os.path.join(BASE_DIR, "ABM_params.json")

SIMULATION_STEPS = 180
SHOCK_START_DAY = 30
RAMP_END_DAY = 60
RANDOM_SEED = 42

MACRO_PRESSURE_MULTIPLIER = 20.0

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

print("=" * 72)
print("PHASE 4: SCENARIO MATRIX GENERATION & LATENT EXPORT")
print("=" * 72)

# -----------------------------------------------------------------------------
# 2) Stress-test specifications
# -----------------------------------------------------------------------------
STRESS_PARAMS = {
    "macro_boom_multiplier": 3.0,    # +300% of baseline demand as extra shock
    "macro_crash_fraction": -0.80,   # -80% of baseline demand as extra shock
    "local_contagion_spike": 5.0     # +500% of baseline demand in Yau Tsim Mong
}

# -----------------------------------------------------------------------------
# 3) Scenario functions
# -----------------------------------------------------------------------------
def scenario_zero_shock(time_step, df):
    """
    No exogenous shock.
    Used for baseline comparison and sanity checks.
    """
    return np.zeros(len(df), dtype=float)


def scenario_tourism_boom(time_step, df):
    base_demand = df["monthly_bookings_proxy"].values.astype(float)
    n_agents = len(df)

    if time_step < SHOCK_START_DAY:
        return np.zeros(n_agents, dtype=float)

    if SHOCK_START_DAY <= time_step < RAMP_END_DAY:
        ramp_factor = (time_step - SHOCK_START_DAY) / float(RAMP_END_DAY - SHOCK_START_DAY)
        return (
            STRESS_PARAMS["macro_boom_multiplier"]
            * base_demand
            * ramp_factor
            * MACRO_PRESSURE_MULTIPLIER
        )

    return (
        STRESS_PARAMS["macro_boom_multiplier"]
        * base_demand
        * MACRO_PRESSURE_MULTIPLIER
    )

def scenario_economic_crash(time_step, df):
    base_demand = df["monthly_bookings_proxy"].values.astype(float)
    n_agents = len(df)

    if time_step < SHOCK_START_DAY:
        return np.zeros(n_agents, dtype=float)

    return (
        STRESS_PARAMS["macro_crash_fraction"]
        * base_demand
        * MACRO_PRESSURE_MULTIPLIER
    )

def scenario_localized_contagion(time_step, df):
    base_demand = df["monthly_bookings_proxy"].values.astype(float)
    n_agents = len(df)

    is_ytm = (df["neighbourhood_cleansed"] == "Yau Tsim Mong").values
    F_t = np.zeros(n_agents, dtype=float)

    if time_step >= SHOCK_START_DAY:
        F_t[is_ytm] = (
            STRESS_PARAMS["local_contagion_spike"]
            * base_demand[is_ytm]
            * MACRO_PRESSURE_MULTIPLIER
        )

    return F_t

# -----------------------------------------------------------------------------
# 4) Helpers
# -----------------------------------------------------------------------------
def export_latent_states(scenario_name: str, latent_states: dict, export_dir: str):
    """
    Save every latent matrix to Parquet using a stable naming convention:
        synthetic_<scenario>_<component>.parquet

    Expected components:
        price, demand, diffusion, reaction, bounds, noise
    """
    saved_paths = {}

    for component_name, matrix_df in latent_states.items():
        out_path = os.path.join(export_dir, f"synthetic_{scenario_name}_{component_name}.parquet")
        df_out = matrix_df.copy()

        # Ensure agent IDs are strings for parquet compatibility and later joining
        df_out.columns = df_out.columns.astype(str)

        df_out.to_parquet(out_path, index=False)
        saved_paths[component_name] = out_path

        print(f"     ✓ Saved {component_name:<9} -> {df_out.shape[0]} x {df_out.shape[1]}")

    return saved_paths

def run_and_export_scenario(
    scenario_name: str,
    scenario_fn,
    steps: int = SIMULATION_STEPS,
    seed: int = RANDOM_SEED,
    export_dir: str = EXPORT_DIR
):
    """
    Run one scenario from a fresh ABM simulator, export all latent states,
    and return the price DataFrame for visualization.
    """
    print(f"\nRunning Scenario: {scenario_name}")

    # Reproducibility
    np.random.seed(seed)

    # Fresh simulator for clean initial state
    sim = ABMSimulator(LISTINGS_PATH, WEIGHTS_PATH, PARAMS_PATH)
    sim.run_simulation(steps=steps, scenario_function=scenario_fn)

    # Extract all latent states from Phase 2
    latent_states = sim.extract_time_series()

    # Export everything to Parquet
    print(f"  -> Exporting latent matrices to: {export_dir}")
    export_latent_states(scenario_name, latent_states, export_dir)

    # Return price matrix for plotting
    return latent_states["price"], latent_states

def summarize_scenario(scenario_name: str, latent_states: dict):
    """
    Build a compact summary for later inspection.
    """
    price_df = latent_states["price"]
    diffusion_df = latent_states["diffusion"]
    reaction_df = latent_states["reaction"]

    mean_price = price_df.mean(axis=1)
    start_mean = float(mean_price.iloc[0])
    end_mean = float(mean_price.iloc[-1])
    drift_pct = ((end_mean - start_mean) / start_mean) * 100 if start_mean != 0 else np.nan

    return {
        "scenario": scenario_name,
        "start_mean_price": start_mean,
        "end_mean_price": end_mean,
        "drift_pct": drift_pct,
        "final_mean_abs_diffusion": float(diffusion_df.abs().mean(axis=1).iloc[-1]),
        "final_mean_abs_reaction": float(reaction_df.abs().mean(axis=1).iloc[-1]),
    }

# -----------------------------------------------------------------------------
# 5) Run all scenarios
# -----------------------------------------------------------------------------
scenario_summaries = []

ts_boom, latent_boom = run_and_export_scenario(
    scenario_name="tourism_boom",
    scenario_fn=scenario_tourism_boom,
    steps=SIMULATION_STEPS,
    seed=RANDOM_SEED
)
scenario_summaries.append(summarize_scenario("tourism_boom", latent_boom))

ts_crash, latent_crash = run_and_export_scenario(
    scenario_name="economic_crash",
    scenario_fn=scenario_economic_crash,
    steps=SIMULATION_STEPS,
    seed=RANDOM_SEED
)
scenario_summaries.append(summarize_scenario("economic_crash", latent_crash))

ts_contagion, latent_contagion = run_and_export_scenario(
    scenario_name="localized_contagion",
    scenario_fn=scenario_localized_contagion,
    steps=SIMULATION_STEPS,
    seed=RANDOM_SEED
)
scenario_summaries.append(summarize_scenario("localized_contagion", latent_contagion))

print("\nAll scenarios generated and latent states exported successfully.")

# Optional baseline export for diagnostics and comparison
print("\nRunning baseline (no shock) for comparison...")
ts_base, latent_base = run_and_export_scenario(
    scenario_name="baseline_no_shock",
    scenario_fn=scenario_zero_shock,
    steps=SIMULATION_STEPS,
    seed=RANDOM_SEED
)
scenario_summaries.append(summarize_scenario("baseline_no_shock", latent_base))

# -----------------------------------------------------------------------------
# 6) Save summary table
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame(scenario_summaries)
summary_path = os.path.join(EXPORT_DIR, "phase4_scenario_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Saved scenario summary -> {summary_path}")

# -----------------------------------------------------------------------------
# 7) Macro visualization of the shocks
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 8))

plt.plot(
    ts_base.mean(axis=1),
    label="Baseline (No Shock)",
    color="black",
    linewidth=2,
    linestyle="--"
)

plt.plot(
    ts_boom.mean(axis=1),
    label="Tourism Boom (+300% sustained, ramped)",
    color="green",
    linewidth=2
)

plt.plot(
    ts_crash.mean(axis=1),
    label="Economic Crash (-80% sustained)",
    color="red",
    linewidth=2
)

plt.plot(
    ts_contagion.mean(axis=1),
    label="Localized Contagion (Yau Tsim Mong +500% sustained)",
    color="purple",
    linewidth=2
)

plt.title(
    "Phase 4: Market Mean Price Reaction to Extreme Scenarios\n(Endogenous Demand + Exogenous F(t) Shocks)",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Time Step (Days)", fontsize=12)
plt.ylabel("Mean Market Price (HKD)", fontsize=12)
plt.axvline(x=SHOCK_START_DAY, color="grey", linestyle=":", label=f"Shock Initiation (Day {SHOCK_START_DAY})")
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

print("\n" + "=" * 72)
print("PHASE 4 COMPLETE")
print("Export convention ready for Phase 5A:")
print("  synthetic_<scenario>_<component>.parquet")
print("Components: price, demand, diffusion, reaction, bounds, noise")
print("=" * 72)