# Customer Segmentation and Recommendation System

## 📊 Project Overview

A comprehensive data science portfolio project that implements customer segmentation using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering, combined with a multi-algorithm recommendation system. This project demonstrates end-to-end data science capabilities from data generation to insights delivery.

## 🎯 Key Features

### Customer Segmentation
- **RFM Analysis**: Customer scoring based on Recency, Frequency, and Monetary value
- **K-means Clustering**: Unsupervised learning for customer grouping
- **Segment Classification**: Automatic customer categorization (Champions, Loyal Customers, At-Risk, etc.)

### Recommendation System
- **Item-based Collaborative Filtering**: Product recommendations based on item similarity
- **User-based Collaborative Filtering**: Recommendations based on similar user behavior
- **Content-based Filtering**: Recommendations using product features
- **Hybrid Approach**: Combined algorithm for enhanced accuracy

### Data Visualization
- **Interactive Dashboards**: Comprehensive business intelligence visualizations
- **3D RFM Plots**: Interactive customer behavior exploration
- **Performance Metrics**: Visual analytics for business insights

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
cd customer-segmentation-system

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis pipeline
python main.py
```

### Custom Parameters
```bash
# Run with custom data sizes
python main.py [num_customers] [num_products] [num_transactions]

# Example: Generate 2000 customers, 1000 products, 20000 transactions
python main.py 2000 1000 20000
```

## 📁 Project Structure

```
├── main.py                          # Main execution pipeline
├── data_generator.py                # Realistic data generation
├── customer_segmentation.py         # RFM analysis & clustering
├── recommendation_engine.py         # Multi-algorithm recommendations
├── visualizations.py               # Comprehensive plotting functions
├── requirements.txt                # Project dependencies
├── data/                           # Generated datasets
│   ├── customers.csv              # Customer master data
│   ├── products.csv               # Product catalog
│   ├── transactions.csv           # Transaction history
│   └── customer_segmentation_results.csv  # Analysis results
└── Generated Visualizations:
    ├── customer_distribution.png   # Customer demographics
    ├── product_analysis.png        # Product performance
    ├── sales_trends.png           # Sales patterns
    ├── rfm_analysis.png           # RFM distributions
    ├── customer_segments.png      # Segment performance
    ├── dashboard_summary.png      # Executive dashboard
    └── interactive_rfm_plot.html  # 3D interactive plot
```

## 🔍 Generated Insights

### Customer Segments Identified
1. **Champions** (23.1%): High value, recent, frequent buyers
2. **Loyal Customers** (11.7%): Regular buyers with good frequency
3. **At-Risk** (8.5%): Previously good customers showing decline
4. **Lost** (40.5%): Customers who haven't purchased recently
5. **New Customers** (0.9%): Recent acquisitions with potential
6. **Need Attention** (5.1%): Customers requiring re-engagement

### Key Business Metrics
- **Total Revenue**: $3,857,087.28
- **Average Order Value**: $428.57
- **Customer Segments**: 9 distinct behavioral groups
- **Recommendation Precision**: Real-time evaluation metrics

### Customer Behavior Patterns
- **Frequent Buyers** (20%): Generate 60% of transactions
- **Seasonal Buyers** (30%): Holiday and promotion-driven purchases
- **One-time Purchasers** (50%): Acquisition and retention opportunities

## 🛠️ Technical Implementation

### Data Generation
- **Realistic Customer Profiles**: Demographics, registration patterns, geographic distribution
- **Product Catalog**: 5 categories with realistic pricing structures
- **Transaction Simulation**: Behavioral patterns including seasonality and customer lifecycle

### Machine Learning Models
- **K-means Clustering**: Optimal cluster selection using silhouette analysis
- **Cosine Similarity**: Item and user similarity calculations
- **TF-IDF Vectorization**: Content-based feature extraction

### Evaluation Metrics
- **Silhouette Score**: Cluster quality assessment
- **Precision@K**: Recommendation accuracy measurement
- **Business KPIs**: Revenue impact and customer lifetime value

## 📈 Business Applications

### Marketing Strategy
- **Targeted Campaigns**: Segment-specific messaging and offers
- **Customer Retention**: At-risk customer re-engagement programs
- **Upselling Opportunities**: Champion and loyal customer targeting

### Inventory Management
- **Demand Forecasting**: Category performance insights
- **Product Recommendations**: Cross-selling optimization
- **Seasonal Planning**: Purchase pattern analysis

### Customer Experience
- **Personalization**: Individual customer recommendations
- **Journey Optimization**: Segment-based experience design
- **Lifetime Value Optimization**: Customer development strategies

## 🔬 Advanced Features

### Scalability
- **Configurable Parameters**: Adjustable dataset sizes
- **Modular Architecture**: Independent component execution
- **Performance Optimization**: Efficient similarity calculations

### Extensibility
- **Additional Algorithms**: Easy integration of new recommendation methods
- **Custom Metrics**: Flexible evaluation framework
- **Data Integration**: Support for real-world data sources

## 📊 Sample Results

### Top Customer Segments by Revenue
1. **Champions**: $2,938,937 (76.2% of total revenue)
2. **Loyal Customers**: $547,986 (14.2% of total revenue)
3. **Need Attention**: $125,168 (3.2% of total revenue)

### Recommendation Performance
- **Item-based CF**: Fast, scalable product recommendations
- **User-based CF**: Personalized based on similar customers
- **Hybrid Model**: Improved accuracy through algorithm combination

## 🚦 Usage Examples

### Individual Customer Analysis
```python
from customer_segmentation import CustomerSegmentation

segmentation = CustomerSegmentation()
segmentation.load_data()
segmentation.calculate_rfm()
segmentation.get_customer_insights('CUST_00001')
```

### Recommendation Generation
```python
from recommendation_engine import RecommendationEngine

rec_engine = RecommendationEngine()
rec_engine.load_data()
recommendations = rec_engine.get_hybrid_recommendations('CUST_00001', 10)
```

### Visualization Creation
```python
from visualizations import CustomerVisualization

viz = CustomerVisualization()
viz.load_data()
viz.generate_all_visualizations()
```

## 🎓 Educational Value

This project demonstrates:
- **End-to-end Data Science Workflow**: From data generation to business insights
- **Machine Learning Applications**: Unsupervised learning and recommendation systems
- **Business Intelligence**: KPI calculation and strategic insights
- **Data Visualization**: Interactive and static plotting techniques
- **Software Engineering**: Modular, maintainable, and scalable code architecture

## 📋 Requirements

- Python 3.9+
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- plotly >= 5.15.0
- faker >= 19.0.0

## 🤝 Contributing

This project is designed as a portfolio demonstration. For educational use or extension:

1. Fork the repository
2. Create feature branches for modifications
3. Implement additional algorithms or visualizations
4. Add comprehensive documentation
5. Include unit tests for new functionality

## 📜 License

This project is open-source and available for educational and portfolio purposes.

## 📞 Contact

Created as a data science portfolio project demonstrating:
- Customer Analytics and Segmentation
- Recommendation System Development
- Business Intelligence and Visualization
- Machine Learning Implementation

---

*This project showcases comprehensive data science capabilities including data generation, customer segmentation, recommendation systems, and business intelligence visualization.* 