# UIDAI Hackathon - Aadhar Data Analysis

This project contains analysis and processing of Aadhar enrollment, demographic, and biometric data.

## How to Clone

```bash
git clone <repository-url>
cd "UIDAI Hackathon"
```


## Dataset

The `dataset/` folder contains three types of Aadhar data:

### 1. Aadhaar Enrolment Dataset
**Location:** `api_data_aadhar_enrolment/` (~1M records)

This dataset provides aggregated information on Aadhaar enrolments across various demographic and geographic levels. It includes variables such as the date of enrollment, state, district, PIN code, and age-wise categories (0–5 years, 5–17 years, and 18 years and above). The dataset captures both temporal and spatial patterns of enrolment activity, enabling detailed descriptive, comparative, and trend analysis.

### 2. Aadhaar Demographic Update Dataset
**Location:** `api_data_aadhar_demographic/` (~2.07M records)

This dataset captures aggregated information related to updates made to residents' demographic data linked to Aadhaar, such as name, address, date of birth, gender, and mobile number. It provides insights into the frequency and distribution of demographic changes across different time periods and geographic levels (state, district, and PIN code).

### 3. Aadhaar Biometric Update Dataset
**Location:** `api_data_aadhar_biometric/` (~1.86M records)

This dataset contains aggregated information on biometric updates (modalities such as fingerprints, iris, and face). It reflects the periodic revalidation or correction of biometric details, especially for children transitioning into adulthood.

## Notebooks Structure

The `notebooks/` directory contains Jupyter notebooks for analysis:

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Initial data exploration, statistics, and data quality assessment |
| `02_data_cleaning.ipynb` | Data cleaning, handling missing values, and removing duplicates |
| `03_feature_engineering.ipynb` | Creating new features and transforming existing ones |
| `03b_optimize_feature_matrix.ipynb` | Memory optimization and efficient storage of feature matrix |
| `04_model_development.ipynb` | Building and training machine learning models for fraud detection |

## Project Structure

```
UIDAI Hackathon/
├── dataset/                          # Raw data files
│   ├── api_data_aadhar_biometric/
│   ├── api_data_aadhar_demographic/
│   └── api_data_aadhar_enrolment/
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 03b_optimize_feature_matrix.ipynb
│   └── 04_model_development.ipynb
├── outputs/                          # Processed data and results
│   └── results/
│       ├── cleaned_*.csv             # Cleaned datasets
│       ├── feature_matrix.csv        # Engineered features
│       └── optimization_report.txt   # Memory optimization details
└── README.md
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install required packages:
``**Cleaned datasets**: `cleaned_biometric.csv`, `cleaned_demographic.csv`, `cleaned_enrolment.csv`
- **Feature matrix**: `feature_matrix.csv` (engineered features for modeling)
- **Reports**: 
  - `data_summary.csv` - Statistical summaries
  - `cleaning_comparison.csv` - Before/after cleaning metrics
  - `optimization_report.txt` - Memory optimization details
  - `feature_matrix_sample.csv` - Sample of engineered features

**Note**: Large output files (`.parquet`, `.csv.gz`) are excluded from version control to comply with GitHub's file size limits.

## Usage

1. Start with `01_data_exploration.ipynb` to understand the data
2. Run `02_data_cleaning.ipynb` to clean and prepare the data
3. Use `03_feature_engineering.ipynb` to create new features
4. Run `03b_optimize_feature_matrix.ipynb` to optimize memory usage
5. Build and train models in `04_model_development.ipynb`

## Outputs

Processed data and results are saved in the `outputs/results/` directory:
- Cleaned datasets (biometric, demographic, enrolment)
- Feature matrices
- Data summaries and comparison reports
