# =============================================================================
# PHASE 2 · HYBRID VECTORIZED ABM STRUCTURE (REVISED & COMPLETE)
# Hong Kong Airbnb — Reaction-Diffusion Price Model
# =============================================================================

import json
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd


class HostAgent:
    """
    Micro-level representation of an Airbnb listing.

    State tracked:
      - price: current listing price
      - demand: current effective demand
      - occupancy: current occupancy proxy
      - histories of all latent PDE components for downstream export
    """

    def __init__(
        self,
        agent_id: str,
        initial_price: float,
        district: str,
        p_min: float,
        p_max: float,
        local_sigma: float,
        base_demand: float,
        base_occupancy: float,
        local_kappa: float,
        demand_to_hkd: float,      # NEW: unit converter from Phase 1


    ):
        self.agent_id = str(agent_id)
        self.district = district

        # State variables
        self.price = float(initial_price)
        self.occupancy = float(base_occupancy)

        # Demand: convert to HKD units
        self.base_demand_raw = float(base_demand)
        self.base_demand = float(base_demand * demand_to_hkd)  # HKD units
        self.demand = self.base_demand
        self.demand_to_hkd = float(demand_to_hkd)

        # Localized parameters
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.local_sigma = float(local_sigma)
        self.local_kappa = float(local_kappa)

        # History (all state variables must be set before this)
        self.price_history = [self.price]
        self.demand_history = [self.demand]
        self.diffusion_history = [0.0]
        self.reaction_history = [0.0]
        self.bounds_history = [0.0]
        self.noise_history = [0.0]


    def update_state(
        self,
        new_price: float,
        new_demand: float,
        diffusion: float,
        reaction: float,
        bounds: float,
        noise: float
    ):
        """
        Updates internal state after the environment calculates the PDE terms.
        """

        # Numerical safety clamp for price
        self.price = float(np.clip(new_price, self.p_min * 0.5, self.p_max * 2.0))
        self.demand = float(max(0.0, new_demand))

        # Log all states and physics components
        self.price_history.append(self.price)
        self.demand_history.append(self.demand)
        self.diffusion_history.append(float(diffusion))
        self.reaction_history.append(float(reaction))
        self.bounds_history.append(float(bounds))
        self.noise_history.append(float(noise))


class HongKongEnvironment:
    """
    Macro-level spatial physics engine.

    This implementation uses a coarse-grained spatial approximation:
      - aggregate listing prices to district level
      - diffuse through district-weight matrix
      - map back to individual agents

    This is intentionally NOT presented as a strict N x N Laplacian.
    """

    def __init__(self, agents: List[HostAgent], w_matrix: pd.DataFrame, params: Dict):
        if not isinstance(w_matrix, pd.DataFrame):
            raise TypeError("w_matrix must be a pandas DataFrame with districts as index/columns.")

        if w_matrix.shape[0] != w_matrix.shape[1]:
            raise ValueError("w_matrix must be square (district x district).")

        if list(w_matrix.index) != list(w_matrix.columns):
            raise ValueError("w_matrix index and columns must contain the same district ordering.")

        self.agents = agents
        self.districts = w_matrix.index.tolist()

        # Raw district adjacency / influence matrix
        self.w_raw = w_matrix.values.astype(float)
        self.w_matrix = self._row_normalize(self.w_raw)

        # Map each agent to a district index
        district_to_idx = {d: i for i, d in enumerate(self.districts)}
        try:
            self.agent_district_idx = np.array([district_to_idx[a.district] for a in self.agents], dtype=int)
        except KeyError as e:
            raise ValueError(f"Agent district not found in w_matrix districts: {e}")

        # PDE / dynamics parameters
        # self.kappa = float(params["kappa"])
        self.kappa_vec = np.array([a.local_kappa for a in self.agents])
        self.alpha = float(params["alpha"])
        self.D_diff = float(params["D_diff"])

        # Endogenous demand sensitivity
        self.gamma = float(params.get("gamma", 1e-5))
       

        # Integration settings
        self.dt = float(params.get("dt", 0.05))
        self.beta = float(params.get("beta", 1e-10))

        # Agent-wise vectors
        self.p_min_vec = np.array([a.p_min for a in self.agents], dtype=float)
        self.p_max_vec = np.array([a.p_max for a in self.agents], dtype=float)
        self.sigma_vec = np.array([a.local_sigma for a in self.agents], dtype=float)

        self.step_count = 0

    @staticmethod
    def _row_normalize(mat: np.ndarray) -> np.ndarray:
        """
        Row-normalize a matrix safely.
        Zero rows remain zero.
        """
        row_sums = mat.sum(axis=1, keepdims=True)
        out = np.zeros_like(mat, dtype=float)
        np.divide(mat, row_sums, out=out, where=(row_sums != 0))
        return out

    def approximate_district_spatial_lag(self, current_prices: np.ndarray) -> np.ndarray:
        """
        Coarse-grained spatial lag approximation.

        Steps:
          1) average prices within district
          2) diffuse across district matrix
          3) map back to agents

        Returns:
          agent-level spatial influence vector
        """
        D = len(self.districts)
        district_prices = np.full(D, np.nan, dtype=float)

        for i in range(D):
            mask = (self.agent_district_idx == i)
            if np.any(mask):
                district_prices[i] = np.mean(current_prices[mask])

        # Fallback for any missing district means, if ever needed
        if np.isnan(district_prices).any():
            fallback = np.nanmean(district_prices)
            if np.isnan(fallback):
                fallback = float(np.mean(current_prices))
            district_prices = np.where(np.isnan(district_prices), fallback, district_prices)

        diffused_district_prices = self.w_matrix @ district_prices
        return diffused_district_prices[self.agent_district_idx]

    def step(self, F_t: np.ndarray):
        """
        Advance the system by one time step.

        Dynamics:
          D_new = base_demand * exp(-gamma * P) + F_t
          diffusion = D_diff * (P_lag - P)
          reaction  = alpha * (D_new - kappa * P)
          bounds    = logistic cubic penalty
          noise     = local stochastic shock
        """
        F_t = np.asarray(F_t, dtype=float).reshape(-1)
        if len(F_t) != len(self.agents):
            raise ValueError(
                f"F_t must have length {len(self.agents)}, but got {len(F_t)}."
            )

        P = np.array([a.price for a in self.agents], dtype=float)
        D_base = np.array([a.base_demand for a in self.agents], dtype=float)

        # Endogenous demand with exogenous shock
        D_new = D_base * np.exp(-self.gamma * P) + F_t
        D_new = np.maximum(D_new, 0.0)

        # 1) Spatial diffusion (coarse-grained approximation)
        P_lag = self.approximate_district_spatial_lag(P)
        diff_raw = P_lag - P
        # diff_raw = np.clip(diff_raw, -500, 500)
        p_range = self.p_max_vec - self.p_min_vec
        diff_raw = np.clip(diff_raw, -p_range, p_range)
        diffusion = self.D_diff * diff_raw

        # 2) Demand reaction
        # reaction = self.alpha * (D_new - self.kappa * P)
        
        raw_reaction = D_new - self.kappa_vec * P
        reaction_cap = 5.0 * np.mean(p_range)
        raw_reaction = np.clip(raw_reaction, -reaction_cap, reaction_cap)

        reaction = self.alpha * raw_reaction
        # 3) Logistic bounding
        bound_force = P * (P - self.p_min_vec) * (P - self.p_max_vec)
        bound_force = np.clip(bound_force, -1e9, 1e9)
        bounds = -self.beta * bound_force
        bounds = np.clip(bounds, -np.mean(p_range), np.mean(p_range))


        # 4) Stochastic noise
        
        # noise_scale = 0.3  # 🔥 giảm noise xuống mức kiểm soát
        # noise = noise_scale * self.sigma_vec * np.random.normal(0.0, 1.0, size=len(self.agents)) * np.sqrt(self.dt)
        noise = self.sigma_vec * np.random.normal(0.0, 1.0, size=len(self.agents)) * np.sqrt(self.dt)

        # 5) Euler step
        dP = (diffusion + reaction + bounds) * self.dt + noise
        P_new = P + dP

        # Update agents and log all physics
        for i, agent in enumerate(self.agents):
            agent.update_state(
                new_price=P_new[i],
                new_demand=D_new[i],
                diffusion=diffusion[i],
                reaction=reaction[i],
                bounds=bounds[i],
                noise=noise[i]
            )

        self.step_count += 1
        if self.step_count % 10 == 0:
            mean_diff   = float(np.mean(np.abs(diffusion)))
            mean_react  = float(np.mean(np.abs(reaction)))
            mean_noise  = float(np.mean(np.abs(noise)))

            ratio_rn = mean_react / (mean_noise + 1e-8)
            ratio_dn = mean_diff  / (mean_noise + 1e-8)

            hierarchy_ok = (ratio_rn >= 2.0) and (ratio_dn >= 0.1)
            flag = "✓" if hierarchy_ok else "⚠ HIERARCHY VIOLATION"

            print(
                f"[Step {self.step_count:>4}] "
                f"|diff|={mean_diff:7.2f}  "
                f"|react|={mean_react:7.2f}  "
                f"|noise|={mean_noise:7.2f}  "
                f"R/N={ratio_rn:.2f}  D/N={ratio_dn:.2f}  {flag}"
            )

class ABMSimulator:
    """
    Orchestrates the simulation, loads data, and extracts time-series outputs.
    """

    REQUIRED_COLUMNS = [
        "id",
        "price",
        "neighbourhood_cleansed",
        "local_p_min",
        "local_p_max",
        "local_sigma",
        "monthly_bookings_proxy",
        "occupancy_rate",
    ]

    def __init__(self, listings_path: str, weights_path: str, params_path: str):
        self.listings_path = listings_path
        self.weights_path = weights_path
        self.params_path = params_path

        self.df = pd.read_csv(listings_path, dtype={"id": str})
        self.w_matrix = pd.read_csv(weights_path, index_col=0)

        with open(params_path, "r") as f:
            self.params = json.load(f)

        self._validate_inputs()

        self.agents: List[HostAgent] = []
        self.environment: Optional[HongKongEnvironment] = None
        self._initialize_agents()

    def _validate_inputs(self):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in listings file: {missing}")

        if self.w_matrix.shape[0] != self.w_matrix.shape[1]:
            raise ValueError("Weights matrix must be square.")

        if list(self.w_matrix.index) != list(self.w_matrix.columns):
            raise ValueError("Weights matrix index and columns must have identical district labels.")

        districts_listings = set(self.df["neighbourhood_cleansed"].astype(str).unique())
        districts_weights = set(self.w_matrix.index.astype(str).tolist())
        if not districts_listings.issubset(districts_weights):
            missing_districts = sorted(list(districts_listings - districts_weights))
            raise ValueError(
                f"Some listing districts are missing from the weight matrix: {missing_districts}"
            )

     
        
        for key in ["kappa", "alpha", "D_diff", "demand_to_hkd"]:
            if key not in self.params:
                raise ValueError(f"Missing required parameter in params.json: '{key}'")
    def _initialize_agents(self):
        demand_to_hkd = float(self.params.get("demand_to_hkd", 1.0))
        for _, row in self.df.iterrows():
            agent = HostAgent(
                agent_id=str(row["id"]),
                initial_price=float(row["price"]),
                district=str(row["neighbourhood_cleansed"]),
                p_min=float(row["local_p_min"]),
                p_max=float(row["local_p_max"]),
                local_sigma=float(row["local_sigma"]),
                base_demand=float(row["monthly_bookings_proxy"]),
                base_occupancy=float(row["occupancy_rate"]),
                local_kappa=float(row["local_kappa"]),
                demand_to_hkd=demand_to_hkd,
            )
            self.agents.append(agent)

        self.environment = HongKongEnvironment(self.agents, self.w_matrix, self.params)

    def run_simulation(self, steps: int, scenario_function: Callable[[int, pd.DataFrame], np.ndarray]):
        """
        Runs the simulation loop.

        scenario_function:
            function(time_step, df) -> F_t vector with length = number of agents
        """
        if not callable(scenario_function):
            raise TypeError("scenario_function must be callable.")

        for t in range(steps):
            F_t = scenario_function(t, self.df)
            self.environment.step(F_t)

    def extract_time_series(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of [T x N] DataFrames:
          - price
          - demand
          - diffusion
          - reaction
          - bounds
          - noise
        """
        return {
            "price": pd.DataFrame({a.agent_id: a.price_history for a in self.agents}),
            "demand": pd.DataFrame({a.agent_id: a.demand_history for a in self.agents}),
            "diffusion": pd.DataFrame({a.agent_id: a.diffusion_history for a in self.agents}),
            "reaction": pd.DataFrame({a.agent_id: a.reaction_history for a in self.agents}),
            "bounds": pd.DataFrame({a.agent_id: a.bounds_history for a in self.agents}),
            "noise": pd.DataFrame({a.agent_id: a.noise_history for a in self.agents}),
        }

    def save_time_series(self, output_dir: str, prefix: str = "phase2") -> Dict[str, str]:
        """
        Saves extracted time series as CSV files.
        Returns a mapping from component name to saved path.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        extracted = self.extract_time_series()

        paths = {}
        for name, df in extracted.items():
            path = os.path.join(output_dir, f"{prefix}_{name}.csv")
            df.to_csv(path, index=False)
            paths[name] = path
        return paths


# =============================================================================
# USAGE TEMPLATE
# =============================================================================
"""
def extreme_tourism_boom_scenario(time_step, df):
    n_agents = len(df)
    shock = np.zeros(n_agents)

    if time_step == 10:
        shock += 5.0  # demand shock applied once

    return shock

sim = ABMSimulator(
    listings_path="ABM_listings.csv",
    weights_path="ABM_spatial_weights.csv",
    params_path="ABM_params.json"
)

sim.run_simulation(
    steps=100,
    scenario_function=extreme_tourism_boom_scenario
)

extracted = sim.extract_time_series()
# extracted["price"], extracted["diffusion"], extracted["reaction"], etc.

# Optional save
# sim.save_time_series(output_dir="./phase2_outputs", prefix="hongkong_abm")
"""
