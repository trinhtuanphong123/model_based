# =============================================================================
# PHASE 1 · DATA PROCESSING, CLEANING & ABM PARAMETER ESTIMATION
# Hong Kong Airbnb — ABM Price Prediction Project
#
# Input  : /content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/main_listings.csv
# Output : /content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_listings_1.csv
#          /content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_spatial_weights.csv
#          /content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong/ABM_params.json
# =============================================================================

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", "{:.4f}".format)

# -----------------------------------------------------------------------------
# 0) PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path("/content/drive/MyDrive/air_bnb_ABM/dataset/hong_kong")
INPUT_PATH = BASE_DIR / "main_listings.csv"

OUTPUT_LISTINGS_PATH = BASE_DIR / "ABM_listings_1.csv"
OUTPUT_WEIGHTS_PATH = BASE_DIR / "ABM_spatial_weights.csv"
OUTPUT_PARAMS_PATH = BASE_DIR / "ABM_params.json"

BASE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  PHASE 1 · DATA PROCESSING & ABM PARAMETER ESTIMATION")
print("  Hong Kong Airbnb — ABM Price Prediction Project")
print("=" * 72)

# -----------------------------------------------------------------------------
# 1) SCHEMA INSPECTION
# -----------------------------------------------------------------------------
print("\n[1/10] Loading raw data & inspecting schema ...")
raw = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"  Raw shape     : {raw.shape[0]:,} rows × {raw.shape[1]} columns")
print(f"  Memory usage  : {raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# -----------------------------------------------------------------------------
# 2) COLUMN SELECTION & TYPE CASTING
# -----------------------------------------------------------------------------
print("\n[2/10] Selecting relevant columns ...")

# Columns needed for downstream phases and robust ABM estimation
KEEP_COLS = [
    "id",
    "price",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "amenities",
    "minimum_nights",
    "instant_bookable",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "last_review",
    "host_is_superhost",
    "calculated_host_listings_count",
    "host_since",
]

available_cols = [c for c in KEEP_COLS if c in raw.columns]
missing_critical = [c for c in ["id", "price", "neighbourhood_cleansed", "latitude", "longitude"] if c not in raw.columns]
if missing_critical:
    raise ValueError(f"Missing critical required columns in raw data: {missing_critical}")

df = raw[available_cols].copy()

# Make sure core string columns exist even if absent in raw
if "room_type" not in df.columns:
    df["room_type"] = "Unknown"
if "property_type" not in df.columns:
    df["property_type"] = "Unknown"

df["id"] = df["id"].astype(str)
df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("Unknown").astype(str)

# -----------------------------------------------------------------------------
# 3) PRICE CLEANING (TARGET VARIABLE)
# -----------------------------------------------------------------------------
print("\n[3/10] Cleaning price (target variable) ...")

df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[\$,\s]", "", regex=True)
    .replace("", np.nan)
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df[df["price"].notna() & (df["price"] > 0)].copy()
df["log_price"] = np.log1p(df["price"])

# -----------------------------------------------------------------------------
# 4) DERIVED FEATURES
# -----------------------------------------------------------------------------
print("\n[4/10] Deriving features from raw columns ...")

def parse_bathrooms(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    if "half" in s:
        return 0.5
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else np.nan

def parse_amenities(s):
    if pd.isna(s) or s == "":
        return 0, 0, 0, 0, 0, 0, 0, 0
    try:
        items = json.loads(s) if isinstance(s, str) and s.startswith("[") else []
    except Exception:
        items = re.findall(r'"([^"]+)"', str(s))

    items_lower = " ".join(items).lower()
    return (
        len(items),
        int("wifi" in items_lower),
        int(any(k in items_lower for k in ["air conditioning", "ac", "a/c", "aircon"])),
        int("pool" in items_lower),
        int(any(k in items_lower for k in ["parking", "car park"])),
        int("kitchen" in items_lower),
        int(any(k in items_lower for k in ["washer", "washing machine"])),
        int(("tv" in items_lower) or ("television" in items_lower)),
    )

def normalize_property_group(text):
    if pd.isna(text):
        return "unknown"
    s = str(text).lower()
    if any(k in s for k in ["apartment", "condo", "flat", "unit"]):
        return "apartment"
    if any(k in s for k in ["house", "home", "villa", "townhouse"]):
        return "house"
    if any(k in s for k in ["hotel", "hostel"]):
        return "hospitality"
    if any(k in s for k in ["guest suite", "guesthouse", "guest house"]):
        return "guest_suite"
    return "other"

# Bathrooms
if "bathrooms_text" in df.columns:
    df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)
    df["is_shared_bath"] = df["bathrooms_text"].str.lower().str.contains("shared", na=False).astype(int)
    df.drop(columns=["bathrooms_text"], inplace=True)
else:
    df["bathrooms"] = np.nan
    df["is_shared_bath"] = 0

# Amenities
if "amenities" in df.columns:
    amen = df["amenities"].apply(parse_amenities)
    (
        df["amenity_count"],
        df["has_wifi"],
        df["has_ac"],
        df["has_pool"],
        df["has_parking"],
        df["has_kitchen"],
        df["has_washer"],
        df["has_tv"],
    ) = zip(*amen)
    df.drop(columns=["amenities"], inplace=True)
else:
    df["amenity_count"] = 0
    df["has_wifi"] = 0
    df["has_ac"] = 0
    df["has_pool"] = 0
    df["has_parking"] = 0
    df["has_kitchen"] = 0
    df["has_washer"] = 0
    df["has_tv"] = 0

# Binary mappings
if "instant_bookable" in df.columns:
    df["instant_bookable"] = (
        df["instant_bookable"]
        .map({"t": 1, "f": 0, True: 1, False: 0})
        .fillna(0)
        .astype(int)
    )
else:
    df["instant_bookable"] = 0

if "host_is_superhost" in df.columns:
    df["host_is_superhost"] = (
        df["host_is_superhost"]
        .map({"t": 1, "f": 0, True: 1, False: 0})
        .fillna(0)
        .astype(int)
    )
else:
    df["host_is_superhost"] = 0

# Date features
if "last_review" in df.columns:
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    snapshot_date = df["last_review"].max()
    if pd.isna(snapshot_date):
        snapshot_date = pd.Timestamp.today().normalize()
    df["days_since_last_review"] = (snapshot_date - df["last_review"]).dt.days.fillna(9999).astype(int)
    df.drop(columns=["last_review"], inplace=True)
else:
    df["days_since_last_review"] = 9999

if "host_since" in df.columns:
    df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
    ref = df["host_since"].max()
    if pd.isna(ref):
        ref = pd.Timestamp.today().normalize()
    df["host_tenure_days"] = (ref - df["host_since"]).dt.days.fillna(0).astype(int)
    df.drop(columns=["host_since"], inplace=True)
else:
    df["host_tenure_days"] = 0

# Categorical mappings
room_map = {
    "Entire home/apt": 3,
    "Private room": 2,
    "Hotel room": 2,
    "Shared room": 1,
}
df["room_type_code"] = df["room_type"].map(room_map).fillna(2).astype(int)
df["property_group"] = df["property_type"].apply(normalize_property_group)

# -----------------------------------------------------------------------------
# 5) MISSING VALUE IMPUTATION
# -----------------------------------------------------------------------------
print("\n[5/10] Imputing missing values ...")

for col in ["latitude", "longitude"]:
    if col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df.groupby("neighbourhood_cleansed")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

numeric_impute = [
    "bedrooms",
    "beds",
    "bathrooms",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "reviews_per_month",
    "host_tenure_days",
    "calculated_host_listings_count",
    "minimum_nights",
    "accommodates",
]

for col in numeric_impute:
    if col in df.columns:
        if df[col].isna().sum() > 0:
            if "neighbourhood_cleansed" in df.columns:
                nbhd_median = df.groupby("neighbourhood_cleansed")[col].transform("median")
                df[col] = df[col].fillna(nbhd_median)
            df[col] = df[col].fillna(df[col].median())

for col in ["number_of_reviews", "number_of_reviews_ltm", "reviews_per_month"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

for col in [c for c in df.columns if c.startswith("availability_")]:
    df[col] = df[col].fillna(0)

for col in ["room_type", "property_type"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

if "neighbourhood_cleansed" in df.columns:
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("Unknown")

# -----------------------------------------------------------------------------
# 6) OUTLIER HANDLING
# -----------------------------------------------------------------------------
print("\n[6/10] Handling outliers ...")

p2 = df["price"].quantile(0.02)
p98 = df["price"].quantile(0.98)
df = df[(df["price"] >= p2) & (df["price"] <= p98)].copy()
df["log_price"] = np.log1p(df["price"])

if "minimum_nights" in df.columns:
    df = df[df["minimum_nights"] <= 365].copy()

if "accommodates" in df.columns:
    df = df[df["accommodates"] > 0].copy()

for col in ["review_scores_rating", "review_scores_location", "review_scores_value"]:
    if col in df.columns:
        df[col] = df[col].clip(0, 5)

# -----------------------------------------------------------------------------
# 7) SPATIAL FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("\n[7/10] Engineering spatial features ...")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

LANDMARKS = {
    "central_hk": (22.2796, 114.1627),
    "tsim_sha_tsui": (22.2988, 114.1722),
    "airport": (22.3080, 113.9185),
}

for name, (lat, lon) in LANDMARKS.items():
    df[f"dist_{name}_km"] = haversine_km(
        df["latitude"].values,
        df["longitude"].values,
        lat,
        lon,
    )

district_centroids = (
    df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]]
    .mean()
    .reset_index()
    .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
)

df = df.merge(district_centroids, on="neighbourhood_cleansed", how="left")
df["dist_to_district_center_km"] = haversine_km(
    df["latitude"].values,
    df["longitude"].values,
    df["centroid_lat"].values,
    df["centroid_lon"].values,
)
df.drop(columns=["centroid_lat", "centroid_lon"], inplace=True)

district_price_rank = (
    df.groupby("neighbourhood_cleansed")["price"]
    .median()
    .rank(ascending=True)
)
df["district_price_rank"] = df["neighbourhood_cleansed"].map(district_price_rank)

# -----------------------------------------------------------------------------
# 8) ABM PARAMETER ESTIMATION
# -----------------------------------------------------------------------------
print("\n[8/10] Estimating ABM / PDE parameters ...")
print("-" * 60)

# 8a. Occupancy rate O(x)
print("  [8a] Computing occupancy rate O(x) ...")
if "availability_90" in df.columns:
    df["raw_occupancy"] = 1 - (df["availability_90"] / 90.0)

    # Dead listing penalty
    dead_mask = (df["raw_occupancy"] > 0.9) & (df["days_since_last_review"] > 180)
    df.loc[dead_mask, "raw_occupancy"] = 0.01

    df["occupancy_rate"] = df["raw_occupancy"].clip(0.01, 0.99)
    df.drop(columns=["raw_occupancy"], inplace=True)
else:
    if "reviews_per_month" in df.columns:
        max_rpm = df["reviews_per_month"].quantile(0.99)
        max_rpm = max(max_rpm, 1e-6)
        df["occupancy_rate"] = (df["reviews_per_month"] / max_rpm).clip(0.01, 0.99)
    else:
        df["occupancy_rate"] = 0.5

district_occupancy = (
    df.groupby("neighbourhood_cleansed")["occupancy_rate"]
    .mean()
    .rename("district_avg_occupancy")
)
df = df.merge(district_occupancy, on="neighbourhood_cleansed", how="left")

# 8b. Demand proxy D(x)
print("  [8b] Computing demand proxy D(x) ...")
if "reviews_per_month" in df.columns and df["reviews_per_month"].notna().any():
    df["monthly_bookings_proxy"] = (df["reviews_per_month"] / 0.5).clip(lower=0)
elif "number_of_reviews_ltm" in df.columns:
    df["monthly_bookings_proxy"] = (df["number_of_reviews_ltm"].fillna(0) / 12.0).clip(lower=0)
else:
    df["monthly_bookings_proxy"] = 0.0

district_demand = (
    df.groupby("neighbourhood_cleansed")["monthly_bookings_proxy"]
    .sum()
    .rename("district_monthly_demand")
)
df = df.merge(district_demand, on="neighbourhood_cleansed", how="left")

# 8c. District-level spatial weight matrix w_ij
print("  [8c] Computing spatial weight matrix w_ij ...")

HK_ISLAND_DISTRICTS = {"Central & Western", "Wan Chai", "Eastern", "Southern"}

district_coords = (
    df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]]
    .mean()
    .reset_index()
    .sort_values("neighbourhood_cleansed")
    .reset_index(drop=True)
)

districts = district_coords["neighbourhood_cleansed"].astype(str).values
n_districts = len(districts)

dist_matrix = np.zeros((n_districts, n_districts), dtype=float)

for i in range(n_districts):
    for j in range(n_districts):
        if i == j:
            continue

        d1_name = districts[i]
        d2_name = districts[j]

        base_dist = haversine_km(
            district_coords.loc[i, "latitude"],
            district_coords.loc[i, "longitude"],
            district_coords.loc[j, "latitude"],
            district_coords.loc[j, "longitude"],
        )

        is_cross_harbor = (d1_name in HK_ISLAND_DISTRICTS) != (d2_name in HK_ISLAND_DISTRICTS)
        penalty_multiplier = 2.0 if is_cross_harbor else 1.0

        dist_matrix[i, j] = base_dist * penalty_multiplier

positive_dists = dist_matrix[dist_matrix > 0]
if positive_dists.size == 0:
    d0 = 1.0
else:
    d0 = float(np.median(positive_dists))

d0 = max(d0, 1e-6)

W = np.exp(-dist_matrix / d0)
np.fill_diagonal(W, 0.0)

row_sums = W.sum(axis=1, keepdims=True)
W_norm = np.divide(W, row_sums, out=np.zeros_like(W), where=(row_sums != 0))

weight_df = pd.DataFrame(W_norm, index=districts, columns=districts)
weight_df.to_csv(OUTPUT_WEIGHTS_PATH)
print(f"  ✓ Saved spatial weights -> {OUTPUT_WEIGHTS_PATH}")
  


# 8d. PDE parameters κ, α, D_diff, σ, p_min, p_max
print("  [8d] Estimating localized PDE parameters ...")

mean_price_global = float(df["price"].mean())
mean_demand_global = float(df["monthly_bookings_proxy"].replace(0, np.nan).mean())
mean_demand_global = max(mean_demand_global, 1e-6)

demand_to_hkd = mean_price_global / mean_demand_global

# ─────────────────────────────────────────
# 1. District-level statistics
# ─────────────────────────────────────────
district_stats = (
    df.groupby("neighbourhood_cleansed")
    .agg(
        median_price=("price", "median"),
        mean_occupancy=("occupancy_rate", "mean"),
        mean_demand=("monthly_bookings_proxy", "mean"),
        local_p_min=("price", lambda x: x.quantile(0.05)),
        local_p_max=("price", lambda x: x.quantile(0.95)),
    )
    .reset_index()
)

# ─────────────────────────────────────────
# 2. Core statistics (MUST come before alpha and sigma)
# ─────────────────────────────────────────
median_price_mean = float(district_stats["median_price"].mean())
mean_demand_mean = float(district_stats["mean_demand"].mean())
median_price_mean = max(median_price_mean, 1e-6)

# ─────────────────────────────────────────
# 3. GLOBAL κ
# ─────────────────────────────────────────
kappa = mean_demand_mean / median_price_mean

# ─────────────────────────────────────────
# 4. Spatial diffusion strength
# ─────────────────────────────────────────
district_price_vec = district_stats.set_index("neighbourhood_cleansed")["median_price"]
price_vals = district_price_vec.reindex(districts).fillna(district_price_vec.mean()).values.astype(float)

spatial_lag = W_norm @ price_vals

if np.var(price_vals) > 0:
    morans_i = float(np.cov(price_vals, spatial_lag)[0, 1] / np.var(price_vals))
else:
    morans_i = 0.0

# ✔ diffusion moderate (not dominating)
D_diff = np.clip(abs(morans_i) * 0.05, 0.01, 0.10)

# ─────────────────────────────────────────
# 5. Reaction strength α (MUST come before sigma block)
# ─────────────────────────────────────────
perturbation_fraction = 0.10
# Target: a 10% demand shock should move price by ~1% of median per dt
target_delta_P = 0.01 * median_price_mean   # HKD per step
reference_D_kappa_P = perturbation_fraction * mean_demand_global * demand_to_hkd

alpha = target_delta_P / (reference_D_kappa_P * 0.05)  # divide by dt
alpha = float(np.clip(alpha, 0.01, 5.0))

# ─────────────────────────────────────────
# 6. Noise σ (uses alpha + median_price_mean — both now defined)
# ─────────────────────────────────────────
# TARGET REACTION MAGNITUDE (per listing, at t=0 with small perturbation)
target_reaction_per_listing = (
    alpha
    * perturbation_fraction
    * mean_demand_global
    * demand_to_hkd
)
target_reaction_per_listing = max(target_reaction_per_listing, 1.0)

# NOISE TARGET: 10% of reaction magnitude
NOISE_TO_REACTION_RATIO = 0.10
noise_scale = 1.0
sqrt_dt = np.sqrt(0.05)
expected_noise_unit = noise_scale * sqrt_dt * np.sqrt(2 / np.pi)

target_sigma_global = (
    NOISE_TO_REACTION_RATIO * target_reaction_per_listing / expected_noise_unit
)

# Apply per-district with relative scaling from median price
district_stats["local_sigma"] = (
    target_sigma_global
    * (district_stats["median_price"] / median_price_mean)
).clip(lower=0.5)

# ─────────────────────────────────────────
# 7. Merge localized bounds into df (AFTER sigma is fully computed)
# ─────────────────────────────────────────
df = df.merge(
    district_stats[["neighbourhood_cleansed", "local_p_min", "local_p_max", "local_sigma"]],
    on="neighbourhood_cleansed",
    how="left",
)

# Fallback: use target_sigma_global (NOT old price.std() * 0.1)
df["local_sigma"] = (
    df["local_sigma"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(target_sigma_global)
    .clip(lower=0.5)
)

# ─────────────────────────────────────────
# 8. Demand sensitivity
# ─────────────────────────────────────────
gamma = 1e-5

# ─────────────────────────────────────────
# 9. Local kappa — calibrated to district equilibrium
#    (old broken formula removed entirely)
# ─────────────────────────────────────────
district_equilibrium = district_stats.set_index(
    "neighbourhood_cleansed"
)["median_price"]

df["district_equilibrium_price"] = df["neighbourhood_cleansed"].map(
    district_equilibrium
)

# kappa_i = D_i_base_converted / P_equilibrium_i
df["demand_hkd"] = df["monthly_bookings_proxy"] * demand_to_hkd

df["local_kappa"] = (
    df["demand_hkd"] * np.exp(-gamma * df["district_equilibrium_price"])
    / df["district_equilibrium_price"]
)

# Fallback: global kappa computed the same way
kappa_global = (
    mean_demand_global * demand_to_hkd
    * np.exp(-gamma * median_price_mean)
    / median_price_mean
)

df["local_kappa"] = (
    df["local_kappa"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(kappa_global)
    .clip(lower=1e-6)
)  

# ─────────────────────────────────────────
# 9. Global bounds
# ─────────────────────────────────────────
p_min_global = float(df["price"].quantile(0.05))
p_max_global = float(df["price"].quantile(0.95))

# ─────────────────────────────────────────
# 10. Save parameters
# ─────────────────────────────────────────
ABM_PARAMS = {
    "kappa": round(float(kappa_global), 6),
    "alpha": round(float(alpha), 6),
    "D_diff": round(float(D_diff), 6),
    "gamma": round(float(gamma), 8),
    "dt": 0.05,
    "beta": 1e-10,

    # NEW — unit bridge
    "demand_to_hkd": round(float(demand_to_hkd), 6),

    # NEW — derived sigma (replaces hardcoded noise_scale)
    "target_sigma_global": round(float(target_sigma_global), 6),
    "noise_to_reaction_ratio": NOISE_TO_REACTION_RATIO,

    "global_sigma": round(float(target_sigma_global), 6),
    "global_p_min": round(float(p_min_global), 6),
    "global_p_max": round(float(p_max_global), 6),

    "morans_I": round(float(morans_i), 6),
    "bandwidth_km": round(float(d0), 6),

    "n_districts": int(n_districts),
    "n_listings": int(len(df)),

    # Diagnostic ratios for Phase 3 gate
    "target_reaction_magnitude": round(float(target_reaction_per_listing), 6),
    "target_noise_magnitude": round(
        float(NOISE_TO_REACTION_RATIO * target_reaction_per_listing), 6
    ),
}

with open(OUTPUT_PARAMS_PATH, "w") as f:
    json.dump(ABM_PARAMS, f, indent=2)

print(f"  ✓ Saved ABM params -> {OUTPUT_PARAMS_PATH}")

# -----------------------------------------------------------------------------
# 9) FINAL DATASET ASSEMBLY & EXPORT
# -----------------------------------------------------------------------------
print("\n[9/10] Assembling and exporting final dataset ...")

df["id"] = df["id"].astype(str)
df["room_type"] = df["room_type"].fillna("Unknown")
df["property_type"] = df["property_type"].fillna("Unknown")

COL_ORDER = [
    "id",
    "price",
    "log_price",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "dist_central_hk_km",
    "dist_tsim_sha_tsui_km",
    "dist_airport_km",
    "dist_to_district_center_km",
    "district_price_rank",
    "room_type",
    "room_type_code",
    "property_type",
    "property_group",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "is_shared_bath",
    "amenity_count",
    "has_wifi",
    "has_ac",
    "has_pool",
    "has_parking",
    "has_kitchen",
    "has_washer",
    "has_tv",
    "minimum_nights",
    "instant_bookable",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "days_since_last_review",
    "host_is_superhost",
    "calculated_host_listings_count",
    "host_tenure_days",
    "occupancy_rate",
    "monthly_bookings_proxy",
    "district_avg_occupancy",
    "district_monthly_demand",
    "local_p_min",
    "local_p_max",
    "local_sigma",
    "local_kappa",
]

final_cols = [c for c in COL_ORDER if c in df.columns]
extra_cols = [c for c in df.columns if c not in final_cols]
df_final = df[final_cols + extra_cols].copy()

df_final.to_csv(OUTPUT_LISTINGS_PATH, index=False)
print(f"  ✓ Saved listings -> {OUTPUT_LISTINGS_PATH}")

# -----------------------------------------------------------------------------
# 10) SUMMARY REPORT
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PHASE 1 COMPLETE — SUMMARY REPORT")
print("=" * 72)
print(f"Listings rows   : {len(df_final):,}")
print(f"Listings cols   : {df_final.shape[1]:,}")
print(f"Districts found  : {n_districts}")
print(f"ABM params file  : {OUTPUT_PARAMS_PATH}")
print(f"Weight matrix    : {OUTPUT_WEIGHTS_PATH}")
print(f"Listings output  : {OUTPUT_LISTINGS_PATH}")
print("=" * 72)