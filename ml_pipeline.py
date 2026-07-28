

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# STEP 1 — DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean_clients(path="clients.csv"):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop full duplicates
    df = df.drop_duplicates(subset="client_id").reset_index(drop=True)

    # Normalize categorical labels
    df["client_type"] = df["client_type"].str.strip().str.title()
    df["gender"] = df["gender"].str.strip().str.upper()
    df["country"] = df["country"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["acquisition_purpose"] = df["acquisition_purpose"].str.strip().str.title()
    df["loan_applied"] = df["loan_applied"].str.strip().str.title()
    df["referral_channel"] = df["referral_channel"].str.strip().str.title()

    # Parse dates — mixed formats
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], format="mixed", errors="coerce")

    # Calculate age
    today = datetime(2024, 1, 1)
    df["age"] = df["date_of_birth"].apply(
        lambda d: (today - d).days // 365 if pd.notnull(d) else np.nan
    )

    # Fill missing numeric with median
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())
    df["age"] = df["age"].fillna(df["age"].median())

    # Fill missing categoricals with mode
    for col in ["client_type", "gender", "country", "region",
                "acquisition_purpose", "loan_applied", "referral_channel"]:
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    return df


def load_and_clean_properties(path="properties.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Clean sale_price: remove $ and commas
    df["sale_price"] = (
        df["sale_price"]
        .astype(str)
        .str.replace("[$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")
    df["listing_status"] = df["listing_status"].str.strip().str.title()
    df["unit_category"] = df["unit_category"].str.strip().str.title()

    return df


def merge_datasets(clients, properties):
    """Aggregate property data per client and merge."""
    sold = properties[properties["listing_status"] == "Sold"].copy()

    agg = sold.groupby("client_ref").agg(
        num_properties=("listing_id", "count"),
        avg_sale_price=("sale_price", "mean"),
        total_investment=("sale_price", "sum"),
        avg_floor_area=("floor_area_sqft", "mean"),
        has_office=("unit_category", lambda x: int("Office" in x.values)),
    ).reset_index().rename(columns={"client_ref": "client_id"})

    merged = clients.merge(agg, on="client_id", how="left")

    # Clients without any property purchase
    merged["num_properties"] = merged["num_properties"].fillna(0)
    merged["avg_sale_price"] = merged["avg_sale_price"].fillna(merged["avg_sale_price"].median())
    merged["total_investment"] = merged["total_investment"].fillna(0)
    merged["avg_floor_area"] = merged["avg_floor_area"].fillna(merged["avg_floor_area"].median())
    merged["has_office"] = merged["has_office"].fillna(0)

    return merged


# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENCODING
#   Per spec: One-Hot Encoding + Label Encoding on
#   client_type, region, acquisition_purpose,
#   referral_channel, country.
#
#   client_type / acquisition_purpose can have more
#   than two categories in real data (e.g. "Company",
#   "Individual", "Trust", "Government"), so they are
#   genuinely one-hot encoded here rather than collapsed
#   into a single binary flag — collapsing would silently
#   discard any category beyond the first two.
#
#   gender / loan_applied are true binary fields
#   (Yes/No, M/F) so a 0/1 flag loses no information —
#   these use simple binary encoding by design.
# ─────────────────────────────────────────────

ONE_HOT_COLS = ["client_type", "region", "acquisition_purpose",
                 "referral_channel", "country"]

BINARY_BASE_COLS = ["loan_applied_enc", "gender_enc"]
NUMERIC_BASE_COLS = ["age", "satisfaction_score", "num_properties",
                      "avg_sale_price", "total_investment", "avg_floor_area",
                      "has_office"]


def encode_features(df, one_hot_cols=ONE_HOT_COLS, max_categories=15):
    """One-hot encode nominal categoricals + binary-encode true binary fields.

    Returns (encoded_df, feature_cols) — feature_cols is built dynamically
    since one-hot encoding produces a variable number of columns depending
    on how many distinct categories exist in the data.
    """
    encode_df = df.copy()

    # True binary fields — 0/1 flag loses no information
    encode_df["loan_applied_enc"] = (encode_df["loan_applied"] == "Yes").astype(int)
    encode_df["gender_enc"] = (encode_df["gender"] == "M").astype(int)

    one_hot_feature_cols = []
    for col in one_hot_cols:
        n_unique = encode_df[col].nunique(dropna=True)
        if n_unique > max_categories:
            # Too many distinct values for one-hot (e.g. a very long tail of
            # countries) — bucket the long tail into "Other" so the feature
            # space doesn't explode, while still one-hot encoding the result.
            top = encode_df[col].value_counts().nlargest(max_categories).index
            grouped = encode_df[col].where(encode_df[col].isin(top), other="Other")
        else:
            grouped = encode_df[col]

        dummies = pd.get_dummies(grouped, prefix=col, dtype=int)
        one_hot_feature_cols.extend(dummies.columns.tolist())
        encode_df = pd.concat([encode_df, dummies], axis=1)

    feature_cols = NUMERIC_BASE_COLS + BINARY_BASE_COLS + one_hot_feature_cols
    return encode_df, feature_cols


# ─────────────────────────────────────────────
# STEP 3 — FEATURE SCALING
# ─────────────────────────────────────────────

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ─────────────────────────────────────────────
# STEP 4 & 5 — CLUSTERING + OPTIMAL K
# ─────────────────────────────────────────────

def compute_elbow(X_scaled, k_range=range(2, 11)):
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_
    return inertias


def compute_silhouette(X_scaled, k_range=range(2, 11)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def run_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_hierarchical(X_scaled, n_clusters=4):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X_scaled)
    return labels


def run_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca


# ─────────────────────────────────────────────
# STEP 6 — CLUSTER LABELING
#
#   Interpreted per spec on: investment purpose,
#   geographic distribution, loan behavior, and
#   customer demographics.
#
#   The four canonical PRD segments are:
#     Corporate Buyers   -> companies purchasing multiple units
#     Luxury Investors   -> high satisfaction, large investments
#     First-Time Buyers  -> younger, loan dependent
#     Global Investors   -> investment-purpose buyers spread
#                            across many countries (no "income"
#                            field exists in the schema, so
#                            geographic spread + investment intent
#                            is used as the closest available proxy)
#
#   Each cluster gets a 0-1 score against each of the four profiles,
#   and scipy's Hungarian algorithm (linear_sum_assignment) finds the
#   globally-optimal one-to-one match — this replaces the old
#   "pick the max, remove it, repeat" elimination logic, which could
#   mislabel a cluster just because of the order it happened to be
#   evaluated in.
#
#   If n_clusters != 4, there's no one-to-one mapping possible, so
#   clusters are instead described by their single strongest trait
#   (still using the same four profiles) with a cluster index suffix
#   to keep labels unique.
# ─────────────────────────────────────────────

PRD_LABELS = ["Corporate Buyers", "Luxury Investors", "First-Time Buyers", "Global Investors"]


def _minmax_norm(series):
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _build_cluster_summary(df):
    summary = df.groupby("cluster").agg(
        avg_investment=("total_investment", "mean"),
        pct_company=("is_company", "mean"),
        avg_num_properties=("num_properties", "mean"),
        avg_age=("age", "mean"),
        pct_loan=("loan_applied_enc", "mean"),
        pct_investment_purpose=("is_investment", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        country_diversity=("country", lambda x: x.nunique() / max(len(x), 1)),
        dominant_country=("country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        dominant_region=("region", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        size=("client_id", "count"),
    )
    return summary


def _score_profiles(summary):
    """Return a (n_clusters x 4) score matrix, columns aligned to PRD_LABELS."""
    corp = _minmax_norm(summary["pct_company"]) * 0.6 + _minmax_norm(summary["avg_num_properties"]) * 0.4
    luxury = _minmax_norm(summary["avg_investment"]) * 0.5 + _minmax_norm(summary["avg_satisfaction"]) * 0.5
    first_time = _minmax_norm(summary["pct_loan"]) * 0.5 + (1 - _minmax_norm(summary["avg_age"])) * 0.5
    global_inv = _minmax_norm(summary["pct_investment_purpose"]) * 0.5 + _minmax_norm(summary["country_diversity"]) * 0.5

    scores = pd.DataFrame({
        "Corporate Buyers": corp,
        "Luxury Investors": luxury,
        "First-Time Buyers": first_time,
        "Global Investors": global_inv,
    }, index=summary.index)
    return scores


def label_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels

    summary = _build_cluster_summary(df)
    scores = _score_profiles(summary)
    n_clusters = len(summary)

    mapping = {}
    if n_clusters == len(PRD_LABELS):
        # Optimal one-to-one assignment (Hungarian algorithm) — maximizes
        # total fit across all clusters simultaneously instead of greedily
        # grabbing the best match one at a time.
        cost = -scores[PRD_LABELS].values  # linear_sum_assignment minimizes
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            cluster_id = scores.index[r]
            mapping[cluster_id] = PRD_LABELS[c]
    else:
        # No clean 1:1 mapping possible — label each cluster by its single
        # strongest matching profile, disambiguating duplicates.
        seen = {}
        for cluster_id in scores.index:
            best_label = scores.loc[cluster_id].idxmax()
            seen[best_label] = seen.get(best_label, 0) + 1
            suffix = f" ({seen[best_label]})" if seen[best_label] > 1 else ""
            mapping[cluster_id] = f"{best_label}{suffix}"

    df["cluster_label"] = df["cluster"].map(mapping)

    # Attach geographic + behavioral summary per cluster, satisfying the
    # "interpret each cluster by investment purpose, geographic distribution,
    # loan behavior, and demographics" requirement with an inspectable table.
    summary_readable = summary.copy()
    summary_readable["cluster_label"] = summary_readable.index.map(mapping)
    summary_readable = summary_readable.reset_index()

    return df, mapping, summary_readable


# ─────────────────────────────────────────────
# MAIN — Build & Cache Pipeline Results
# ─────────────────────────────────────────────

def run_pipeline(clients_path="clients.csv", properties_path="properties.csv", n_clusters=4):
    clients = load_and_clean_clients(clients_path)
    properties = load_and_clean_properties(properties_path)
    merged = merge_datasets(clients, properties)

    # Binary flags needed for both feature encoding and cluster scoring
    merged["is_company"] = (merged["client_type"] == "Company").astype(int)
    merged["is_investment"] = (merged["acquisition_purpose"] == "Investment").astype(int)

    encoded, feature_cols = encode_features(merged)
    X_scaled, scaler = scale_features(encoded, feature_cols)

    elbow = compute_elbow(X_scaled)
    silhouette = compute_silhouette(X_scaled)

    km_labels, km_model = run_kmeans(X_scaled, n_clusters)
    hc_labels = run_hierarchical(X_scaled, n_clusters)
    pca_coords, pca_model = run_pca(X_scaled)

    labeled_df, cluster_mapping, cluster_summary = label_clusters(encoded, km_labels)

    # Attach PCA coords
    labeled_df["pca_x"] = pca_coords[:, 0]
    labeled_df["pca_y"] = pca_coords[:, 1]
    labeled_df["hc_cluster"] = hc_labels

    return {
        "df": labeled_df,
        "properties": properties,
        "elbow": elbow,
        "silhouette": silhouette,
        "km_model": km_model,
        "cluster_mapping": cluster_mapping,
        "cluster_summary": cluster_summary,
        "feature_cols": feature_cols,
        "n_clusters": n_clusters,
        "km_silhouette": silhouette_score(X_scaled, km_labels),
    }


if __name__ == "__main__":
    result = run_pipeline()
    df = result["df"]
    print(f"Pipeline complete. Records: {len(df)}")
    print("\nCluster distribution:")
    print(df["cluster_label"].value_counts())
    print(f"\nOverall Silhouette Score: {result['km_silhouette']:.4f}")
    print("\nCluster summary (incl. geographic distribution):")
    print(result["cluster_summary"])