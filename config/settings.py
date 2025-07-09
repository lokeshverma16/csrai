"""
Application Settings Configuration

Centralized configuration for the Customer Analytics & Recommendation System
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# File paths
CUSTOMERS_FILE = DATA_DIR / "customers.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
SEGMENTATION_RESULTS_FILE = DATA_DIR / "customer_segmentation_results.csv"

# Output file paths
CLUSTER_ANALYSIS_FILE = RESULTS_DIR / "cluster_analysis.csv"
RECOMMENDATIONS_FILE = RESULTS_DIR / "recommendations.csv"
RECOMMENDATIONS_DETAILED_FILE = RESULTS_DIR / "recommendations_detailed.json"

# Model parameters
DEFAULT_CLUSTERING_FEATURES = ['recency', 'frequency', 'monetary']
DEFAULT_N_RECOMMENDATIONS = 3
RANDOM_STATE = 42

# Recommendation engine weights
RECOMMENDATION_WEIGHTS = {
    'collaborative_filtering': 0.30,
    'content_based': 0.25,
    'cross_selling': 0.20,
    'temporal': 0.15,
    'price_affinity': 0.10
}

# Business rules
MIN_CONFIDENCE_THRESHOLD = 0.1
DIVERSITY_WEIGHT = 0.3
POPULARITY_BOOST = 0.2
SEASONAL_ADJUSTMENT = 0.1

# Application metadata
APP_NAME = "Customer Analytics & Recommendation System"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Data Science Portfolio Project" 