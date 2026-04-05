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
# =============================================================================
# FORCE HIERARCHY GATE — THRESHOLDS
# Phase 4 will not run unless ALL of these pass.
# These are not arbitrary; they derive from the fix plan:
#   - Reaction must dominate noise by at least 5x
#   - Diffusion must be detectable above noise (at least 10% of noise)
#   - Bounds must be negligible under normal conditions
#   - Full system drift must stay within 10% (market is not exploding)
#   - Pure reaction must converge: final |reaction| < 1% of initial mean price
# =============================================================================
GATE_REACTION_TO_NOISE_MIN   = 5.0    # |Reaction| / |Noise| >= 5
GATE_DIFFUSION_TO_NOISE_MIN  = 0.10   # |Diffusion| / |Noise| >= 0.1
GATE_BOUNDS_TO_NOISE_MAX     = 0.05   # |Bounds| / |Noise| <= 0.05 (must be dormant)
GATE_FULL_DRIFT_MAX_PCT      = 10.0   # |drift%| <= 10% over SIMULATION_STEPS
GATE_REACTION_CONVERGENCE    = 0.01   # final |reaction| as fraction of initial mean price

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
# 10) QUANTITATIVE GATE — replaces the old empirical report
# -----------------------------------------------------------------------------

def run_force_hierarchy_gate(
    diag_A: dict,
    mean_abs_diffusion_A,
    mean_abs_reaction_A,
    mean_abs_bounds_A,
    mean_abs_noise_A,
    mean_abs_reaction_C,
    initial_mean_price: float,
) -> bool:
    """
    Quantitative gate that enforces the force hierarchy.
    Returns True only if ALL criteria pass.
    Raises RuntimeError if any critical criterion fails,
    blocking Phase 4 from running.
    """

    print("\n" + "=" * 72)
    print("PHASE 3 GATE: FORCE HIERARCHY VALIDATION")
    print("Phase 4 will not proceed unless all checks pass.")
    print("=" * 72)

    gate_results = {}

    # Snapshot final-step values from Test A
    final_react  = float(mean_abs_reaction_A.iloc[-1])
    final_noise  = float(mean_abs_noise_A.iloc[-1])
    final_diff   = float(mean_abs_diffusion_A.iloc[-1])
    final_bounds = float(mean_abs_bounds_A.iloc[-1])

    eps = 1e-8

    # ------------------------------------------------------------------
    # CHECK 1: Reaction dominates noise
    # ------------------------------------------------------------------
    rn_ratio = final_react / (final_noise + eps)
    passed_1 = rn_ratio >= GATE_REACTION_TO_NOISE_MIN
    gate_results["reaction_to_noise"] = {
        "value": rn_ratio,
        "threshold": GATE_REACTION_TO_NOISE_MIN,
        "passed": passed_1,
        "label": "|Reaction| / |Noise|"
    }

    # ------------------------------------------------------------------
    # CHECK 2: Diffusion is detectable above noise
    # ------------------------------------------------------------------
    dn_ratio = final_diff / (final_noise + eps)
    passed_2 = dn_ratio >= GATE_DIFFUSION_TO_NOISE_MIN
    gate_results["diffusion_to_noise"] = {
        "value": dn_ratio,
        "threshold": GATE_DIFFUSION_TO_NOISE_MIN,
        "passed": passed_2,
        "label": "|Diffusion| / |Noise|"
    }

    # ------------------------------------------------------------------
    # CHECK 3: Bounds are dormant (not interfering with normal dynamics)
    # ------------------------------------------------------------------
    bn_ratio = final_bounds / (final_noise + eps)
    passed_3 = bn_ratio <= GATE_BOUNDS_TO_NOISE_MAX
    gate_results["bounds_dormancy"] = {
        "value": bn_ratio,
        "threshold": GATE_BOUNDS_TO_NOISE_MAX,
        "passed": passed_3,
        "label": "|Bounds| / |Noise| (must be LOW)"
    }

    # ------------------------------------------------------------------
    # CHECK 4: Full system drift is bounded (market not exploding)
    # ------------------------------------------------------------------
    drift_abs = abs(diag_A["drift_pct"])
    passed_4 = drift_abs <= GATE_FULL_DRIFT_MAX_PCT
    gate_results["full_system_drift"] = {
        "value": drift_abs,
        "threshold": GATE_FULL_DRIFT_MAX_PCT,
        "passed": passed_4,
        "label": "|Drift%| over simulation (must be LOW)"
    }

    # ------------------------------------------------------------------
    # CHECK 5: Pure reaction converges toward equilibrium
    # Criterion: final |reaction| < 1% of initial mean price
    # ------------------------------------------------------------------
    final_react_C = float(mean_abs_reaction_C.iloc[-1])
    convergence_threshold = GATE_REACTION_CONVERGENCE * initial_mean_price
    passed_5 = final_react_C < convergence_threshold
    gate_results["reaction_convergence"] = {
        "value": final_react_C,
        "threshold": convergence_threshold,
        "passed": passed_5,
        "label": "Pure reaction final |Reaction| (must converge to ~0)"
    }

    # ------------------------------------------------------------------
    # PRINT GATE REPORT
    # ------------------------------------------------------------------
    all_passed = True
    for key, result in gate_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(
            f"  [{status}] {result['label']:<45} "
            f"value={result['value']:>10.4f}  "
            f"threshold={result['threshold']:>10.4f}"
        )
        if not result["passed"]:
            all_passed = False

    print("-" * 72)

    # ------------------------------------------------------------------
    # DIAGNOSIS: if any check fails, identify likely cause
    # ------------------------------------------------------------------
    if not all_passed:
        print("\n⚠  GATE FAILED — Diagnosing likely causes:\n")

        if not gate_results["reaction_to_noise"]["passed"]:
            print(
                "  → Reaction/Noise ratio too low.\n"
                "    Likely cause: demand_to_hkd too small, or local_sigma too large.\n"
                "    Fix: re-run Phase 1 and verify demand_to_hkd = mean_price / mean_demand.\n"
                "    Verify local_sigma was derived from target reaction magnitude, not price std.\n"
            )

        if not gate_results["diffusion_to_noise"]["passed"]:
            print(
                "  → Diffusion invisible against noise.\n"
                "    Likely cause: D_diff too small, or noise still too large after Fix 2.\n"
                "    Fix: increase D_diff or reduce local_sigma floor in Phase 1.\n"
            )

        if not gate_results["bounds_dormancy"]["passed"]:
            print(
                "  → Bounds force is too active under normal conditions.\n"
                "    Likely cause: beta too large, or local_p_min/local_p_max too tight.\n"
                "    Fix: reduce beta in ABM_PARAMS or widen local bounds quantiles.\n"
            )

        if not gate_results["full_system_drift"]["passed"]:
            print(
                "  → Market is drifting too far from initial conditions.\n"
                "    Likely cause: kappa miscalibrated, or alpha too large.\n"
                "    Fix: verify local_kappa uses district equilibrium price, not listing price.\n"
            )

        if not gate_results["reaction_convergence"]["passed"]:
            print(
                "  → Pure reaction is not converging.\n"
                "    Likely cause: local_kappa equilibrium paralysis still present,\n"
                "    or gamma too small to create meaningful demand decay.\n"
                "    Fix: verify local_kappa denominator is district median, not listing price.\n"
            )

        print("=" * 72)
        raise RuntimeError(
            "Phase 3 gate FAILED. Phase 4 is blocked. "
            "Fix the parameters listed above and re-run Phase 1 → Phase 2 → Phase 3."
        )

    print("\n✓ ALL CHECKS PASSED — System is cleared for Phase 4.")
    print("=" * 72)
    return True

# Extract Test C reaction for gate
# -----------------------------------------------------------------------------
# 10a) Force ratio trajectory — the key diagnostic the old code lacked
# -----------------------------------------------------------------------------
eps = 1e-8
ratio_rn = mean_abs_reaction_A / (mean_abs_noise_A + eps)
ratio_dn = mean_abs_diffusion_A / (mean_abs_noise_A + eps)
ratio_bn = mean_abs_bounds_A / (mean_abs_noise_A + eps)

fig3, axes3 = plt.subplots(1, 2, figsize=(18, 5))

# Left: absolute force magnitudes over time
axes3[0].plot(mean_abs_reaction_A,  linewidth=2, label="|Reaction|",  color="blue")
axes3[0].plot(mean_abs_diffusion_A, linewidth=2, label="|Diffusion|", color="green")
axes3[0].plot(mean_abs_noise_A,     linewidth=2, label="|Noise|",     color="red")
axes3[0].plot(mean_abs_bounds_A,    linewidth=2, label="|Bounds|",    color="orange",
              linestyle="--")
axes3[0].set_title("Force Magnitudes Over Time (Test A)", fontweight="bold")
axes3[0].set_xlabel("Time Step")
axes3[0].set_ylabel("Mean Absolute Force (HKD)")
axes3[0].legend()

# Right: ratios against noise — gate thresholds drawn as horizontal lines
axes3[1].plot(ratio_rn, linewidth=2, label="|Reaction|/|Noise|",  color="blue")
axes3[1].plot(ratio_dn, linewidth=2, label="|Diffusion|/|Noise|", color="green")
axes3[1].plot(ratio_bn, linewidth=2, label="|Bounds|/|Noise|",    color="orange",
              linestyle="--")

axes3[1].axhline(
    y=GATE_REACTION_TO_NOISE_MIN, color="blue", linestyle=":",
    linewidth=1.5, label=f"Reaction gate ({GATE_REACTION_TO_NOISE_MIN}x)"
)
axes3[1].axhline(
    y=GATE_DIFFUSION_TO_NOISE_MIN, color="green", linestyle=":",
    linewidth=1.5, label=f"Diffusion gate ({GATE_DIFFUSION_TO_NOISE_MIN}x)"
)
axes3[1].axhline(
    y=GATE_BOUNDS_TO_NOISE_MAX, color="orange", linestyle=":",
    linewidth=1.5, label=f"Bounds ceiling ({GATE_BOUNDS_TO_NOISE_MAX}x)"
)

axes3[1].set_title("Force Ratios vs Noise — Gate Thresholds", fontweight="bold")
axes3[1].set_xlabel("Time Step")
axes3[1].set_ylabel("Ratio (relative to |Noise|)")
axes3[1].legend(fontsize=9)
axes3[1].set_ylim(bottom=0)

plt.tight_layout()
plt.show()


reaction_C = extracted_C["reaction"]
mean_abs_reaction_C = reaction_C.abs().mean(axis=1)

# Run the gate — raises RuntimeError if any check fails
gate_passed = run_force_hierarchy_gate(
    diag_A=diag_A,
    mean_abs_diffusion_A=mean_abs_diffusion_A,
    mean_abs_reaction_A=mean_abs_reaction_A,
    mean_abs_bounds_A=mean_abs_bounds_A,
    mean_abs_noise_A=mean_abs_noise_A,
    mean_abs_reaction_C=mean_abs_reaction_C,
    initial_mean_price=diag_A["start_mean"],
)