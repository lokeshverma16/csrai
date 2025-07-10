
# Usage Instructions - Customer Analytics Pipeline

## Command Line Interface

### Basic Operations
```bash
# Run complete pipeline with all components
python main.py --run-all

# Generate fresh synthetic data
python main.py --generate-data

# Run only RFM analysis
python main.py --rfm-only

# Run only clustering analysis  
python main.py --clustering-only

# Run only recommendation system
python main.py --recommendations-only

# Generate only visualizations
python main.py --visualizations-only

# Force regeneration of all data
python main.py --run-all --force-regenerate
```

### Advanced Options
```bash
# Custom data size generation
python main.py --generate-data --customers 2000 --products 1000 --transactions 20000

# Skip specific components
python main.py --run-all --skip-visualizations

# Verbose output
python main.py --run-all --verbose

# Save results to custom directory
python main.py --run-all --output-dir custom_results/
```

## Programmatic Usage

### Basic Pipeline Execution
```python
from main import CustomerAnalyticsPipeline

# Initialize pipeline
pipeline = CustomerAnalyticsPipeline()

# Load and validate data
if pipeline.load_and_validate_data():
    print("Data loaded successfully")

# Execute complete pipeline
pipeline.run_complete_pipeline()
```

### Component-Specific Execution
```python
# RFM Analysis only
pipeline.load_and_validate_data()
pipeline.execute_rfm_analysis()

# Clustering with custom parameters
pipeline.execute_clustering_analysis()

# Generate recommendations
pipeline.execute_recommendation_system()

# Create visualizations
pipeline.generate_all_visualizations()
```

### Custom Configuration
```python
# Initialize with custom directories
pipeline = CustomerAnalyticsPipeline(base_dir="/custom/path/")

# Access results programmatically
results = pipeline.results
rfm_data = results['rfm']
clustering_data = results['clustering']
recommendations = results['recommendations']
```

## Output Interpretation

### Customer Segmentation Results
**File:** `results/customer_segmentation_results.csv`
- `customer_id`: Unique customer identifier
- `recency`: Days since last purchase
- `frequency`: Number of purchases
- `monetary`: Total spending
- `rfm_score`: Combined RFM score (3-15)
- `segment`: Business segment classification

### Cluster Analysis Results  
**File:** `results/cluster_analysis.csv`
- `customer_id`: Customer identifier
- `cluster`: Cluster assignment (0, 1, 2...)
- Combined with RFM data for interpretation

### Recommendation Results
**File:** `results/recommendations.csv`
- `customer_id`: Target customer
- `product_id`: Recommended product
- `confidence_score`: Recommendation confidence (0-1)
- `recommendation_type`: Type (cross_sell, upsell, etc.)
- `explanation`: Human-readable reasoning

## Troubleshooting

### Common Issues

#### Data Loading Errors
```
Error: File not found - customers.csv
Solution: Run with --generate-data flag to create sample data
```

#### Memory Issues
```
Error: Memory allocation failed
Solution: Reduce data size or increase available RAM
```

#### Missing Dependencies
```
Error: Module 'sklearn' not found
Solution: pip install -r requirements.txt
```

### Performance Optimization

#### Large Datasets
- Use data sampling for initial testing
- Consider database backends for production
- Implement data chunking for memory efficiency

#### Speed Improvements
- Use SSD storage for data files
- Increase available RAM
- Consider parallel processing for recommendations

### Validation Failures

#### Data Integrity Issues
- Check for missing values in key columns
- Validate date formats and ranges
- Ensure referential integrity between tables

#### Pipeline Component Failures
- Check log files in `logs/` directory
- Validate intermediate outputs
- Run components individually for isolation

## Best Practices

### Data Preparation
1. **Clean Data:** Remove duplicates and invalid records
2. **Validate Relationships:** Ensure foreign key integrity
3. **Date Formatting:** Use consistent date formats (YYYY-MM-DD)
4. **Price Validation:** Ensure positive values for prices

### Pipeline Execution
1. **Start Small:** Test with sample data first
2. **Monitor Logs:** Check logs for warnings and errors
3. **Validate Outputs:** Review generated files for completeness
4. **Backup Results:** Save important analysis outputs

### Performance Monitoring
1. **Track Execution Time:** Monitor pipeline performance
2. **Memory Usage:** Watch for memory leaks or excessive usage
3. **Output Quality:** Validate recommendation confidence scores
4. **Business Metrics:** Verify business logic in results

## Integration Guidelines

### Database Integration
```python
# Example database connection
import pandas as pd
from sqlalchemy import create_engine

# Load from database
engine = create_engine('postgresql://user:pass@host:port/db')
customers_df = pd.read_sql('SELECT * FROM customers', engine)

# Initialize pipeline with database data
pipeline = CustomerAnalyticsPipeline()
pipeline.customers_df = customers_df
# Continue with pipeline execution...
```

### API Integration
```python
# Example API endpoint integration
from flask import Flask, jsonify
from main import CustomerAnalyticsPipeline

app = Flask(__name__)
pipeline = CustomerAnalyticsPipeline()

@app.route('/recommendations/<customer_id>')
def get_recommendations(customer_id):
    recommendations = pipeline.recommendation_engine.generate_advanced_recommendations(customer_id)
    return jsonify(recommendations)
```

---
*Detailed usage instructions for the Customer Analytics Pipeline system*
