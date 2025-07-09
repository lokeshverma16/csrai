# Project Structure Documentation

## Overview
The Customer Analytics & Recommendation System (CSARS) has been restructured into a professional, modular architecture following Python best practices.

## Directory Structure

```
inventory/
├── src/                           # Source code modules
│   ├── __init__.py               # Main package initialization
│   ├── models/                   # Machine learning models
│   │   ├── __init__.py
│   │   ├── customer_segmentation.py
│   │   └── kmeans_clustering.py
│   ├── engines/                  # Recommendation engines
│   │   ├── __init__.py
│   │   ├── advanced_recommendation_engine.py
│   │   ├── recommendation_engine.py
│   │   └── hybrid_recommendation_demo.py
│   ├── analytics/                # Analysis and demo scripts
│   │   ├── __init__.py
│   │   ├── advanced_recommendation_demo.py
│   │   ├── kmeans_demo.py
│   │   ├── radar_demo.py
│   │   └── rfm_demo.py
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   ├── visualizations.py
│   │   └── rfm_visualizations.py
│   └── config/                   # Configuration modules
│       ├── __init__.py
│       └── settings.py
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_comprehensive_suite.py
│   ├── test_hybrid_recommendations.py
│   ├── test_kmeans.py
│   └── test_radar_charts.py
├── config/                       # Configuration files
│   ├── settings.py               # Application settings
│   ├── logging_config.py         # Logging configuration
│   ├── model_config.yaml         # Model parameters
│   └── data_config.yaml          # Data configuration
├── data/                         # Data files
│   ├── customers.csv
│   ├── products.csv
│   ├── transactions.csv
│   └── customer_segmentation_results.csv
├── results/                      # Analysis outputs
│   ├── cluster_analysis.csv
│   ├── recommendations.csv
│   └── recommendations_detailed.json
├── reports/                      # Generated reports
│   └── radar_analysis_report.md
├── visualizations/               # Generated charts
│   ├── business_insights.png
│   ├── cluster_radar_chart.png
│   └── interactive_cluster_radar_chart.html
├── logs/                         # Log files
│   └── pipeline_*.log
├── main.py                       # Main application entry point
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── [Documentation files]        # Various .md files
```

## Module Descriptions

### `src/models/`
Contains machine learning models and customer segmentation algorithms:
- `customer_segmentation.py`: RFM analysis and customer segmentation
- `kmeans_clustering.py`: K-means clustering implementation

### `src/engines/`
Contains recommendation system implementations:
- `advanced_recommendation_engine.py`: Advanced hybrid recommendation system
- `recommendation_engine.py`: Basic recommendation engine
- `hybrid_recommendation_demo.py`: Demo script for recommendation engine

### `src/analytics/`
Contains analysis scripts and demonstrations:
- `advanced_recommendation_demo.py`: Advanced recommendation demonstrations
- `kmeans_demo.py`: K-means clustering demonstrations
- `radar_demo.py`: Radar chart demonstrations
- `rfm_demo.py`: RFM analysis demonstrations

### `src/utils/`
Contains utility functions and helpers:
- `data_generator.py`: Synthetic data generation
- `visualizations.py`: Standard visualization functions
- `rfm_visualizations.py`: RFM-specific visualizations

### `src/config/`
Contains configuration management:
- `settings.py`: Configuration integration with main config

### `tests/`
Contains comprehensive test suite:
- `test_comprehensive_suite.py`: Main test suite
- `test_hybrid_recommendations.py`: Recommendation engine tests
- `test_kmeans.py`: Clustering tests
- `test_radar_charts.py`: Visualization tests

### `config/`
Contains application configuration:
- `settings.py`: Main application settings and paths
- `logging_config.py`: Centralized logging configuration
- `model_config.yaml`: Model parameters and hyperparameters
- `data_config.yaml`: Data paths and validation settings

## Configuration Management

The project uses a centralized configuration system:

1. **Python Settings** (`config/settings.py`): Core paths and constants
2. **YAML Configuration** (`config/*.yaml`): Model parameters and data settings
3. **Logging Configuration** (`config/logging_config.py`): Centralized logging setup

## Import Structure

### From External Code
```python
# Import from the restructured modules
from src.utils.data_generator import DataGenerator
from src.models.customer_segmentation import CustomerSegmentation
from src.engines.advanced_recommendation_engine import AdvancedRecommendationEngine
from config.settings import BASE_DIR, DATA_DIR
from config.logging_config import setup_logging
```

### From Main Application
```python
# Main application imports everything needed
from main import CustomerAnalyticsPipeline

pipeline = CustomerAnalyticsPipeline()
pipeline.run_complete_pipeline()
```

## Running the Application

### Basic Usage
```bash
# Run complete pipeline
python3 main.py --run-all

# Generate data only
python3 main.py --generate-data

# Run specific components
python3 main.py --rfm-only
python3 main.py --recommendations-only
```

### Testing
```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 tests/test_comprehensive_suite.py
```

## Benefits of New Structure

1. **Modularity**: Clear separation of concerns
2. **Testability**: Isolated components easy to test
3. **Maintainability**: Logical organization of code
4. **Scalability**: Easy to add new features
5. **Professional**: Industry-standard Python project structure
6. **Configuration Management**: Centralized settings
7. **Logging**: Standardized logging across all modules

## Migration Notes

- All imports have been updated to use the new structure
- Main application (`main.py`) remains the primary entry point
- Test files have been moved to dedicated `tests/` directory
- Configuration is now centralized and YAML-based
- Logging is standardized across all modules

## Future Enhancements

The modular structure supports easy addition of:
- New recommendation algorithms in `src/engines/`
- Additional ML models in `src/models/`
- More analytics tools in `src/analytics/`
- Extended utilities in `src/utils/`
- Enhanced configuration in `config/` 