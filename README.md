# Buyer Segmentation & Investment Profiling — Parcl Co.

Machine learning pipeline and interactive dashboard for segmenting real estate buyers and profiling their investment behavior, built for Parcl Co.'s market-intelligence platform.

Real estate buyers are not one audience — individual home buyers, corporations, first-time buyers, and international investors all behave differently. This project clusters clients using K-Means and Hierarchical clustering, labels the resulting segments against four business-defined buyer personas, and serves the results through a Streamlit dashboard.

## Features

- **Data cleaning** — deduplication, mixed-format date parsing, categorical normalization, median/mode imputation
- **Feature engineering** — one-hot encoding for nominal fields (client type, region, acquisition purpose, referral channel, country), binary encoding for true binary fields (gender, loan status)
- **Clustering** — K-Means and Agglomerative Hierarchical clustering, cross-validated against each other
- **Optimal K selection** — Elbow method and Silhouette score across K = 2–10
- **Cluster labeling** — clusters are scored against four personas (Corporate Buyers, Luxury Investors, First-Time Buyers, Global Investors) on investment behavior, loan usage, demographics, and geographic spread, then matched optimally via the Hungarian algorithm rather than a greedy heuristic
- **Interactive dashboard** — 5 tabs (Segmentation Overview, Investor Behavior, Geographic Analysis, Segment Insights, Model Diagnostics) with live filtering by country, region, acquisition purpose, client type, and segment, plus an adjustable cluster count

## Project Structure

```
.
├── ml_pipeline.py   # Data cleaning, feature engineering, clustering, cluster labeling
├── app.py           # Streamlit dashboard
├── clients.csv       # Client dataset (not included — see Data below)
├── properties.csv     # Property-transaction dataset (not included — see Data below)
└── README.md
```

## Data

The pipeline expects two CSV files in the working directory:

**`clients.csv`**

| Column | Description |
|---|---|
| `client_id` | Unique client identifier |
| `client_type` | Individual / Company / other buyer entity type |
| `gender` | Gender of buyer |
| `country` | Country of residence |
| `region` | Geographic region |
| `date_of_birth` | Used to derive buyer age |
| `acquisition_purpose` | Investment / Personal use / Resale |
| `loan_applied` | Financing indicator (Yes/No) |
| `referral_channel` | Source of customer acquisition |
| `satisfaction_score` | Customer satisfaction rating |

**`properties.csv`**

| Column | Description |
|---|---|
| `listing_id` | Unique listing identifier |
| `client_ref` | Foreign key linking to `client_id` |
| `sale_price` | Transaction sale price |
| `transaction_date` | Date of transaction |
| `listing_status` | Sold / Active / Withdrawn |
| `unit_category` | Residential / Office / Retail / Industrial |
| `floor_area_sqft` | Unit floor area in square feet |

Column names are case/whitespace-insensitive — the pipeline normalizes them automatically.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install directly:

```bash
pip install pandas numpy scikit-learn scipy streamlit plotly
```

### Run the pipeline standalone

```bash
python ml_pipeline.py
```

This prints the cluster distribution, overall Silhouette score, and a cluster interpretation summary (including geographic breakdown) to the console.

### Run the dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## How Cluster Labeling Works

Rather than picking the highest-scoring persona for each cluster one at a time — which can mislabel a cluster depending on evaluation order — the pipeline scores every cluster against all four personas simultaneously and solves the cluster-to-persona assignment as an optimal matching problem (`scipy.optimize.linear_sum_assignment`, the Hungarian algorithm). This only produces a clean 1-to-1 mapping when K = 4; for other values of K, clusters are labeled by their single strongest-matching trait instead.

## Tech Stack

- **Data & ML:** pandas, NumPy, scikit-learn, SciPy
- **Dashboard:** Streamlit, Plotly

## Deliverables

- [x] ML pipeline (`ml_pipeline.py`)
- [x] Streamlit dashboard (`app.py`)
- [x] Research paper — `Buyer_Segmentation_Research_Paper.docx`

## Author

**Bitan Das**
The LNM Institute of Information Technology (LNMIIT), Jaipur
📧 23uec529@lnmiit.ac.in

