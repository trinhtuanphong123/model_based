# =============================================================================
# PHASE 3 · EMPIRICAL DIAGNOSTIC TESTS
# Objective:
#   - Empirically verify convergence, smoothing, and equilibrium behavior
#   - Validate the physics engine by isolating forces from Phase 2
#   - Use the exported latent states from Phase 2 for additional diagnostics
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# If Phase 2 classes are already defined in the notebook, this script uses them directly:
#   - ABMSimulator
#   - HongKongEnvironment
#   - HostAgent

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)

# -----------------------------------------------------------------------------
# 1) Paths and simulation settings
# -----------------------------------------------------------------------------
LISTINGS_PATH = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_listings_1.csv"
WEIGHTS_PATH  = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_spatial_weights.csv"
PARAMS_PATH   = "/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_params.json"

SIMULATION_STEPS = 100

# Diagnostic-only override.
# This is not a model claim; it is a numerical tuning choice for stable baseline checks.
BASELINE_GAMMA = 1e-5

print("=" * 72)
print("PHASE 3: EMPIRICAL DIAGNOSTIC TESTS")
print("Objective: Validate the Phase 2 physics engine under isolated force settings")
print("=" * 72)

# -----------------------------------------------------------------------------
# 2) Shock function for the baseline test
# -----------------------------------------------------------------------------
def baseline_zero_shock(time_step, df):
    """
    F(t) = 0 for all time steps.
    Used to test the endogenous PDE dynamics without exogenous shocks.
    """
    return np.zeros(len(df), dtype=float)

# -----------------------------------------------------------------------------
# 3) Helpers
# -----------------------------------------------------------------------------
def configure_simulator(
    sim,
    gamma_override=BASELINE_GAMMA,
    alpha_override=None,
    D_diff_override=None,
    sigma_override=None,
    dt_override=None
):
    """
    Apply diagnostic overrides to the environment.

    These are used only for empirical testing:
      - gamma_override: demand sensitivity
      - alpha_override: reaction strength
      - D_diff_override: spatial diffusion strength
      - sigma_override: local noise strength
      - dt_override: Euler time step
    """
    env = sim.environment

    if gamma_override is not None:
        env.gamma = float(gamma_override)
    if alpha_override is not None:
        env.alpha = float(alpha_override)
    if D_diff_override is not None:
        env.D_diff = float(D_diff_override)
    if dt_override is not None:
        env.dt = float(dt_override)

    if sigma_override is not None:
        env.sigma_vec[:] = float(sigma_override)

    return sim

def run_diagnostic_test(
    name,
    gamma_override=BASELINE_GAMMA,
    alpha_override=None,
    D_diff_override=None,
    sigma_override=None,
    dt_override=None,
    steps=SIMULATION_STEPS
):
    """
    Create a fresh simulator, apply overrides, run the zero-shock baseline,
    and return all extracted time series.
    """
    print(f"\n[{name}] Initializing simulator...")
    sim = ABMSimulator(LISTINGS_PATH, WEIGHTS_PATH, PARAMS_PATH)
    sim = configure_simulator(
        sim,
        gamma_override=gamma_override,
        alpha_override=alpha_override,
        D_diff_override=D_diff_override,
        sigma_override=sigma_override,
        dt_override=dt_override
    )

    np.random.seed(42)
    sim.run_simulation(steps=steps, scenario_function=baseline_zero_shock)

    extracted = sim.extract_time_series()
    return extracted

def summarize_price_trajectory(price_df, label):
    """
    Compute key diagnostics for one run.
    """
    mean_price = price_df.mean(axis=1)
    std_price = price_df.std(axis=1)

    start_mean = float(mean_price.iloc[0])
    end_mean = float(mean_price.iloc[-1])
    drift_pct = ((end_mean - start_mean) / start_mean) * 100 if start_mean != 0 else np.nan

    return {
        "label": label,
        "mean_price": mean_price,
        "std_price": std_price,
        "start_mean": start_mean,
        "end_mean": end_mean,
        "drift_pct": drift_pct
    }

def pick_sample_agents(df, n=3, seed=42):
    """
    Pick a small reproducible sample of agent columns for plotting.
    """
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    n = min(n, len(cols))
    return rng.choice(cols, size=n, replace=False)

# -----------------------------------------------------------------------------
# 4) Test A: Full system under zero shock
# -----------------------------------------------------------------------------
print("\n[Test A] Full system baseline (diffusion + reaction + bounds + noise)")
extracted_A = run_diagnostic_test(
    name="Test A",
    gamma_override=BASELINE_GAMMA,
    alpha_override=None,
    D_diff_override=None,
    sigma_override=None,
    dt_override=None,
    steps=SIMULATION_STEPS
)

price_A = extracted_A["price"]
demand_A = extracted_A["demand"]
diffusion_A = extracted_A["diffusion"]
reaction_A = extracted_A["reaction"]
bounds_A = extracted_A["bounds"]
noise_A = extracted_A["noise"]

diag_A = summarize_price_trajectory(price_A, "Full System")

# -----------------------------------------------------------------------------
# 5) Test B: Pure diffusion
#    - remove reaction
#    - remove noise
#    - keep spatial term only
# -----------------------------------------------------------------------------
print("[Test B] Pure diffusion test (smoothing check)")
extracted_B = run_diagnostic_test(
    name="Test B",
    gamma_override=0.0,     # gamma irrelevant when alpha = 0, but we set it cleanly
    alpha_override=0.0,
    D_diff_override=None,
    sigma_override=0.0,
    dt_override=None,
    steps=SIMULATION_STEPS
)

price_B = extracted_B["price"]
diag_B = summarize_price_trajectory(price_B, "Pure Diffusion")

# -----------------------------------------------------------------------------
# 6) Test C: Pure reaction
#    - remove diffusion
#    - remove noise
#    - keep endogenous demand reaction only
# -----------------------------------------------------------------------------
print("[Test C] Pure reaction test (endogenous equilibrium check)")
extracted_C = run_diagnostic_test(
    name="Test C",
    gamma_override=BASELINE_GAMMA,
    alpha_override=None,
    D_diff_override=0.0,
    sigma_override=0.0,
    dt_override=None,
    steps=SIMULATION_STEPS
)

price_C = extracted_C["price"]
diag_C = summarize_price_trajectory(price_C, "Pure Reaction")

# -----------------------------------------------------------------------------
# 7) Additional latent-state diagnostics from Test A
# -----------------------------------------------------------------------------
mean_abs_diffusion_A = diffusion_A.abs().mean(axis=1)
mean_abs_reaction_A = reaction_A.abs().mean(axis=1)
mean_abs_bounds_A = bounds_A.abs().mean(axis=1)
mean_abs_noise_A = noise_A.abs().mean(axis=1)

# -----------------------------------------------------------------------------
# 8) Diagnostic visualization
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Panel 1: Full system
axes[0, 0].plot(diag_A["mean_price"], color="black", linewidth=3, label="Market Mean Price")
sample_agents = pick_sample_agents(price_A, n=3, seed=42)
for agent_id in sample_agents:
    axes[0, 0].plot(price_A[agent_id], alpha=0.5, linewidth=1.5, label=f"Agent {str(agent_id)[:6]}")
axes[0, 0].set_title("Test A: Full System Dynamics (F(t) = 0)", fontweight="bold")
axes[0, 0].set_ylabel("Price (HKD)")
axes[0, 0].legend(loc="best")

# Panel 2: Volatility comparison
axes[0, 1].plot(diag_A["std_price"], color="red", linewidth=2, label="Full System")
axes[0, 1].plot(diag_B["std_price"], color="blue", linestyle="--", linewidth=2, label="Pure Diffusion")
axes[0, 1].plot(diag_C["std_price"], color="green", linestyle=":", linewidth=2, label="Pure Reaction")
axes[0, 1].set_title("System Volatility Over Time", fontweight="bold")
axes[0, 1].set_ylabel("Standard Deviation")
axes[0, 1].legend(loc="best")

# Panel 3: Pure diffusion
axes[1, 0].plot(diag_B["mean_price"], color="black", linewidth=3, label="Market Mean Price")
for agent_id in sample_agents:
    axes[1, 0].plot(price_B[agent_id], alpha=0.6, linewidth=1.5)
axes[1, 0].set_title("Test B: Pure Diffusion (Smoothing Check)", fontweight="bold")
axes[1, 0].set_xlabel("Time Step (Days)")
axes[1, 0].set_ylabel("Price (HKD)")

# Panel 4: Pure reaction
axes[1, 1].plot(diag_C["mean_price"], color="black", linewidth=3, label="Market Mean Price")
for agent_id in sample_agents:
    axes[1, 1].plot(price_C[agent_id], alpha=0.6, linewidth=1.5)
axes[1, 1].set_title("Test C: Pure Reaction (Endogenous Equilibrium)", fontweight="bold")
axes[1, 1].set_xlabel("Time Step (Days)")
axes[1, 1].set_ylabel("Price (HKD)")

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 9) Latent-force diagnostics figure
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
ax2.plot(mean_abs_diffusion_A, linewidth=2, label="|Diffusion| mean")
ax2.plot(mean_abs_reaction_A, linewidth=2, label="|Reaction| mean")
ax2.plot(mean_abs_bounds_A, linewidth=2, label="|Bounds| mean")
ax2.plot(mean_abs_noise_A, linewidth=2, label="|Noise| mean")
ax2.set_title("Test A: Mean Absolute Magnitude of Latent Physics Terms", fontweight="bold")
ax2.set_xlabel("Time Step (Days)")
ax2.set_ylabel("Mean Absolute Force")
ax2.legend(loc="best")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 10) Empirical report
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EMPIRICAL VALIDATION REPORT")
print("This is a diagnostic test, not a mathematical proof.")
print("=" * 72)

print(f"\n[Test A: Full System]")
print(f" Starting Mean Price : {diag_A['start_mean']:.2f} HKD")
print(f" Ending Mean Price   : {diag_A['end_mean']:.2f} HKD")
print(f" Macro Price Drift   : {diag_A['drift_pct']:+.2f}% over {SIMULATION_STEPS} steps")
print(f" Reaction vs Noise ratio : {mean_abs_reaction_A.iloc[-1] / (mean_abs_noise_A.iloc[-1] + 1e-8):.4f}")

if abs(diag_A["drift_pct"]) < 5.0:
    print("  [✓] Convergence check: drift stays within a moderate range.")
else:
    print("  [!] Warning: full system drift is large. Check gamma, dt, or beta.")

print(f"\n[Test B: Pure Diffusion]")
print(f" Starting Std Dev    : {diag_B['std_price'].iloc[0]:.4f}")
print(f" Ending Std Dev      : {diag_B['std_price'].iloc[-1]:.4f}")

if diag_B["std_price"].iloc[-1] < diag_B["std_price"].iloc[0]:
    print("  [✓] Smoothing check: diffusion reduces dispersion over time.")
else:
    print("  [!] Warning: diffusion does not appear to smooth the market.")

print(f"\n[Test C: Pure Reaction]")
print(f" Starting Mean Price : {diag_C['start_mean']:.2f} HKD")
print(f" Ending Mean Price   : {diag_C['end_mean']:.2f} HKD")
print(f" Macro Price Drift   : {diag_C['drift_pct']:+.2f}%")

# NEW: Micro-equilibrium validation
reaction_C = extracted_C["reaction"]
mean_abs_reaction_C = reaction_C.abs().mean(axis=1)

final_reaction = float(mean_abs_reaction_C.iloc[-1])

print(f" Mean |Reaction| (final) : {final_reaction:.6f}")

if final_reaction < 1e-2:
    print("  [✓] Micro-equilibrium: reaction converges to ~0 per agent.")
else:
    print("  [!] Warning: reaction not converging → check local_kappa or gamma.")

print("\n[Latent Force Snapshot from Test A]")
print(f" Mean |Diffusion|    : {mean_abs_diffusion_A.iloc[-1]:.4f}")
print(f" Mean |Reaction|     : {mean_abs_reaction_A.iloc[-1]:.4f}")
print(f" Mean |Bounds|       : {mean_abs_bounds_A.iloc[-1]:.4f}")
print(f" Mean |Noise|        : {mean_abs_noise_A.iloc[-1]:.4f}")

print("=" * 72)
print("Phase 3 Complete.")
print("=" * 72)