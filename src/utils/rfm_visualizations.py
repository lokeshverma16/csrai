import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and context for professional visualizations
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ComprehensiveRFMVisualization:
    def __init__(self):
        """Initialize comprehensive RFM visualization system"""
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple  
            'accent': '#F18F01',       # Orange
            'success': '#1B998B',      # Teal
            'warning': '#FFD23F',      # Yellow
            'danger': '#C73E1D',       # Red
            'light': '#F8F9FA',       # Light gray
            'dark': '#2C3E50'          # Dark blue-gray
        }
        
        # Color palettes
        self.segment_palette = ['#2E86AB', '#A23B72', '#F18F01', '#1B998B', '#C73E1D', 
                               '#FFD23F', '#8E44AD', '#27AE60', '#E67E22', '#95A5A6']
        self.rfm_palette = ['#C73E1D', '#F18F01', '#FFD23F', '#1B998B', '#2E86AB']
        
        # Figure sizes
        self.figsize_small = (10, 6)
        self.figsize_medium = (12, 8) 
        self.figsize_large = (15, 10)
        self.figsize_extra_large = (18, 12)
        
        # Create output directory
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        print("🎨 Comprehensive RFM Visualization System Initialized")

    def load_data(self, customers_path='data/customers.csv', 
                  products_path='data/products.csv', 
                  transactions_path='data/transactions.csv',
                  rfm_path='data/customer_segmentation_results.csv'):
        """Load all required data for visualization"""
        try:
            print("\n" + "="*60)
            print("LOADING DATA FOR RFM VISUALIZATION")
            print("="*60)
            
            # Load datasets
            self.customers_df = pd.read_csv(customers_path)
            self.products_df = pd.read_csv(products_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            # Convert dates
            self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
            self.transactions_df['purchase_date'] = pd.to_datetime(self.transactions_df['purchase_date'])
            
            # Create enriched transaction data
            self.enriched_transactions = self.transactions_df.merge(
                self.customers_df[['customer_id', 'name', 'age', 'registration_date']], 
                on='customer_id', how='left'
            ).merge(
                self.products_df[['product_id', 'product_name', 'category', 'price']], 
                on='product_id', how='left',
                suffixes=('_transaction', '_product')
            )
            
            self.enriched_transactions['transaction_value'] = (
                self.enriched_transactions['quantity'] * 
                self.enriched_transactions['price_transaction']
            )
            
            # Add time features
            self.enriched_transactions['year_month'] = self.enriched_transactions['purchase_date'].dt.to_period('M')
            self.enriched_transactions['month'] = self.enriched_transactions['purchase_date'].dt.month
            
            # Load RFM data
            self.rfm_df = pd.read_csv(rfm_path)
            self.rfm_df['registration_date'] = pd.to_datetime(self.rfm_df['registration_date'])
            
            print(f"✅ Loaded {len(self.customers_df)} customers")
            print(f"✅ Loaded {len(self.products_df)} products")
            print(f"✅ Loaded {len(self.transactions_df)} transactions")
            print(f"✅ Loaded {len(self.rfm_df)} RFM records")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def plot_rfm_distributions(self, save_fig=True):
        """Create comprehensive RFM distribution visualizations"""
        print("📊 Creating RFM Distribution Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize_extra_large)
        
        # Recency Distribution with KDE
        sns.histplot(data=self.rfm_df, x='recency', bins=30, kde=True, 
                    color=self.colors['danger'], alpha=0.7, ax=axes[0, 0])
        axes[0, 0].set_title('Recency Distribution\n(Days Since Last Purchase)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # Add statistical lines
        mean_recency = self.rfm_df['recency'].mean()
        median_recency = self.rfm_df['recency'].median()
        axes[0, 0].axvline(mean_recency, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_recency:.0f}')
        axes[0, 0].axvline(median_recency, color='orange', linestyle='--', alpha=0.8, 
                          label=f'Median: {median_recency:.0f}')
        axes[0, 0].legend()
        
        # Frequency Distribution
        sns.histplot(data=self.rfm_df, x='frequency', bins=30, kde=True,
                    color=self.colors['primary'], alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Frequency Distribution\n(Number of Purchases)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        mean_frequency = self.rfm_df['frequency'].mean()
        median_frequency = self.rfm_df['frequency'].median()
        axes[0, 1].axvline(mean_frequency, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_frequency:.1f}')
        axes[0, 1].axvline(median_frequency, color='orange', linestyle='--', alpha=0.8, 
                          label=f'Median: {median_frequency:.1f}')
        axes[0, 1].legend()
        
        # Monetary Distribution
        sns.histplot(data=self.rfm_df, x='monetary', bins=30, kde=True,
                    color=self.colors['success'], alpha=0.7, ax=axes[0, 2])
        axes[0, 2].set_title('Monetary Distribution\n(Total Spent)', 
                            fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Total Spent ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        mean_monetary = self.rfm_df['monetary'].mean()
        median_monetary = self.rfm_df['monetary'].median()
        axes[0, 2].axvline(mean_monetary, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: ${mean_monetary:.0f}')
        axes[0, 2].axvline(median_monetary, color='orange', linestyle='--', alpha=0.8, 
                          label=f'Median: ${median_monetary:.0f}')
        axes[0, 2].legend()
        
        # Box plots for outlier detection
        sns.boxplot(y=self.rfm_df['recency'], color=self.colors['danger'], ax=axes[1, 0])
        axes[1, 0].set_title('Recency Outliers', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Days Since Last Purchase')
        
        sns.boxplot(y=self.rfm_df['frequency'], color=self.colors['primary'], ax=axes[1, 1])
        axes[1, 1].set_title('Frequency Outliers', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Purchases')
        
        sns.boxplot(y=self.rfm_df['monetary'], color=self.colors['success'], ax=axes[1, 2])
        axes[1, 2].set_title('Monetary Outliers', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Total Spent ($)')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/rfm_distributions.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/rfm_distributions.png")
        plt.close()

    def plot_rfm_correlation_analysis(self, save_fig=True):
        """Create RFM correlation and segment analysis"""
        print("📊 Creating RFM Correlation Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # RFM Correlation Heatmap
        rfm_corr = self.rfm_df[['recency', 'frequency', 'monetary']].corr()
        
        sns.heatmap(rfm_corr, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=axes[0, 0])
        axes[0, 0].set_title('RFM Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        
        # RFM Segment Distribution
        segment_counts = self.rfm_df['segment'].value_counts()
        colors = self.segment_palette[:len(segment_counts)]
        
        wedges, texts, autotexts = axes[0, 1].pie(segment_counts.values, 
                                                 labels=segment_counts.index, 
                                                 autopct='%1.1f%%', 
                                                 colors=colors, 
                                                 startangle=90)
        axes[0, 1].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # Segment Revenue Analysis
        segment_revenue = self.rfm_df.groupby('segment')['monetary'].sum().sort_values()
        
        bars = axes[1, 0].barh(segment_revenue.index, segment_revenue.values, 
                              color=colors[:len(segment_revenue)])
        axes[1, 0].set_title('Total Revenue by Segment', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Total Revenue ($)')
        
        # Add value labels
        for i, (segment, revenue) in enumerate(segment_revenue.items()):
            axes[1, 0].text(revenue + max(segment_revenue) * 0.01, i, 
                           f'${revenue:,.0f}', va='center', fontweight='bold')
        
        # Average Order Value by Segment
        if 'avg_order_value' in self.rfm_df.columns:
            avg_order_by_segment = self.rfm_df.groupby('segment')['avg_order_value'].mean().sort_values()
            
            axes[1, 1].bar(range(len(avg_order_by_segment)), avg_order_by_segment.values, 
                          color=colors[:len(avg_order_by_segment)], alpha=0.8)
            axes[1, 1].set_title('Average Order Value by Segment', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Average Order Value ($)')
            axes[1, 1].set_xticks(range(len(avg_order_by_segment)))
            axes[1, 1].set_xticklabels(avg_order_by_segment.index, rotation=45, ha='right')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/rfm_correlation_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/rfm_correlation_analysis.png")
        plt.close()

    def plot_customer_behavior_analysis(self, save_fig=True):
        """Create customer behavior and lifecycle analysis"""
        print("📊 Creating Customer Behavior Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # Monthly spending trends
        monthly_spending = self.enriched_transactions.groupby('year_month')['transaction_value'].sum()
        monthly_spending.index = monthly_spending.index.to_timestamp()
        
        axes[0, 0].plot(monthly_spending.index, monthly_spending.values, 
                       marker='o', linewidth=2, color=self.colors['primary'])
        axes[0, 0].set_title('Monthly Spending Trends', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Total Spending ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add trend line
        x_numeric = np.arange(len(monthly_spending))
        z = np.polyfit(x_numeric, monthly_spending.values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(monthly_spending.index, p(x_numeric), "--", 
                       color=self.colors['danger'], alpha=0.8, label='Trend')
        axes[0, 0].legend()
        
        # Seasonal analysis
        seasonal_spending = self.enriched_transactions.groupby('month')['transaction_value'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        bars = axes[0, 1].bar(range(1, 13), seasonal_spending.values, 
                             color=self.colors['success'], alpha=0.7)
        axes[0, 1].set_title('Seasonal Spending Patterns', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Total Spending ($)')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names)
        
        # Customer lifecycle analysis (spending vs days since registration)
        if 'days_since_registration' in self.rfm_df.columns:
            axes[1, 0].scatter(self.rfm_df['days_since_registration'], 
                              self.rfm_df['monetary'], 
                              alpha=0.6, color=self.colors['accent'])
            axes[1, 0].set_title('Customer Lifecycle: Spending vs Registration Age', 
                                fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Days Since Registration')
            axes[1, 0].set_ylabel('Total Spent ($)')
            
            # Add correlation coefficient
            corr_coef = self.rfm_df['days_since_registration'].corr(self.rfm_df['monetary'])
            axes[1, 0].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                           transform=axes[1, 0].transAxes, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Category-wise spending distribution
        category_spending = self.enriched_transactions.groupby('category')['transaction_value'].sum()
        
        wedges, texts, autotexts = axes[1, 1].pie(category_spending.values, 
                                                 labels=category_spending.index, 
                                                 autopct='%1.1f%%', 
                                                 colors=self.segment_palette[:len(category_spending)], 
                                                 startangle=90)
        axes[1, 1].set_title('Spending by Product Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/customer_behavior_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/customer_behavior_analysis.png")
        plt.close()

    def plot_outlier_analysis(self, save_fig=True):
        """Create comprehensive outlier analysis with scatter plots"""
        print("📊 Creating Outlier Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize_extra_large)
        
        # Create outlier flags if not exist
        if 'is_recency_outlier' not in self.rfm_df.columns:
            # Calculate outliers using IQR method
            def detect_outliers(data, column):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (data[column] < lower_bound) | (data[column] > upper_bound)
            
            self.rfm_df['is_recency_outlier'] = detect_outliers(self.rfm_df, 'recency')
            self.rfm_df['is_frequency_outlier'] = detect_outliers(self.rfm_df, 'frequency')
            self.rfm_df['is_monetary_outlier'] = detect_outliers(self.rfm_df, 'monetary')
        
        # Recency vs Frequency scatter
        outlier_mask = self.rfm_df['is_recency_outlier'] | self.rfm_df['is_frequency_outlier']
        
        axes[0, 0].scatter(self.rfm_df[~outlier_mask]['recency'], 
                          self.rfm_df[~outlier_mask]['frequency'],
                          alpha=0.6, color=self.colors['primary'], label='Normal', s=30)
        axes[0, 0].scatter(self.rfm_df[outlier_mask]['recency'], 
                          self.rfm_df[outlier_mask]['frequency'],
                          alpha=0.8, color=self.colors['danger'], label='Outlier', s=50)
        axes[0, 0].set_title('Recency vs Frequency\n(Outliers Highlighted)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Recency (Days)')
        axes[0, 0].set_ylabel('Frequency (Purchases)')
        axes[0, 0].legend()
        
        # Recency vs Monetary scatter
        outlier_mask = self.rfm_df['is_recency_outlier'] | self.rfm_df['is_monetary_outlier']
        
        axes[0, 1].scatter(self.rfm_df[~outlier_mask]['recency'], 
                          self.rfm_df[~outlier_mask]['monetary'],
                          alpha=0.6, color=self.colors['primary'], label='Normal', s=30)
        axes[0, 1].scatter(self.rfm_df[outlier_mask]['recency'], 
                          self.rfm_df[outlier_mask]['monetary'],
                          alpha=0.8, color=self.colors['danger'], label='Outlier', s=50)
        axes[0, 1].set_title('Recency vs Monetary\n(Outliers Highlighted)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Recency (Days)')
        axes[0, 1].set_ylabel('Monetary ($)')
        axes[0, 1].legend()
        
        # Frequency vs Monetary scatter
        outlier_mask = self.rfm_df['is_frequency_outlier'] | self.rfm_df['is_monetary_outlier']
        
        axes[0, 2].scatter(self.rfm_df[~outlier_mask]['frequency'], 
                          self.rfm_df[~outlier_mask]['monetary'],
                          alpha=0.6, color=self.colors['primary'], label='Normal', s=30)
        axes[0, 2].scatter(self.rfm_df[outlier_mask]['frequency'], 
                          self.rfm_df[outlier_mask]['monetary'],
                          alpha=0.8, color=self.colors['danger'], label='Outlier', s=50)
        axes[0, 2].set_title('Frequency vs Monetary\n(Outliers Highlighted)', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Frequency (Purchases)')
        axes[0, 2].set_ylabel('Monetary ($)')
        axes[0, 2].legend()
        
        # Outlier counts by metric
        outlier_counts = pd.Series({
            'Recency': self.rfm_df['is_recency_outlier'].sum(),
            'Frequency': self.rfm_df['is_frequency_outlier'].sum(),
            'Monetary': self.rfm_df['is_monetary_outlier'].sum()
        })
        
        bars = axes[1, 0].bar(outlier_counts.index, outlier_counts.values, 
                             color=[self.colors['danger'], self.colors['primary'], self.colors['success']], 
                             alpha=0.7)
        axes[1, 0].set_title('Outlier Count by RFM Metric', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Outliers')
        
        # Add value labels on bars
        for bar, value in zip(bars, outlier_counts.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outlier_counts) * 0.01,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        # Outlier percentage by segment
        outlier_by_segment = self.rfm_df.groupby('segment').agg({
            'is_recency_outlier': 'sum',
            'is_frequency_outlier': 'sum', 
            'is_monetary_outlier': 'sum',
            'customer_id': 'count'
        })
        outlier_by_segment['total_outliers'] = (outlier_by_segment['is_recency_outlier'] + 
                                               outlier_by_segment['is_frequency_outlier'] + 
                                               outlier_by_segment['is_monetary_outlier'])
        outlier_by_segment['outlier_percentage'] = (outlier_by_segment['total_outliers'] / 
                                                   (outlier_by_segment['customer_id'] * 3) * 100)
        
        axes[1, 1].bar(outlier_by_segment.index, outlier_by_segment['outlier_percentage'], 
                      color=self.colors['warning'], alpha=0.7)
        axes[1, 1].set_title('Outlier Percentage by Segment', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Outlier Percentage (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # RFM Score distribution for outliers vs normal
        if 'rfm_score' in self.rfm_df.columns:
            any_outlier = (self.rfm_df['is_recency_outlier'] | 
                          self.rfm_df['is_frequency_outlier'] | 
                          self.rfm_df['is_monetary_outlier'])
            
            normal_scores = self.rfm_df[~any_outlier]['rfm_score'].astype(str)
            outlier_scores = self.rfm_df[any_outlier]['rfm_score'].astype(str)
            
            axes[1, 2].hist([normal_scores.astype(float), outlier_scores.astype(float)], 
                           bins=20, alpha=0.7, label=['Normal', 'Outliers'],
                           color=[self.colors['primary'], self.colors['danger']])
            axes[1, 2].set_title('RFM Score Distribution:\nNormal vs Outliers', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('RFM Score')
            axes[1, 2].set_ylabel('Number of Customers')
            axes[1, 2].legend()
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/outlier_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/outlier_analysis.png")
        plt.close()

    def plot_business_insights(self, save_fig=True):
        """Create business insights visualizations"""
        print("📊 Creating Business Insights Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # Top customers by monetary value
        top_customers = self.rfm_df.nlargest(10, 'monetary')
        
        bars = axes[0, 0].barh(range(len(top_customers)), top_customers['monetary'], 
                              color=self.colors['success'], alpha=0.8)
        axes[0, 0].set_title('Top 10 Customers by Total Spending', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Total Spent ($)')
        axes[0, 0].set_yticks(range(len(top_customers)))
        axes[0, 0].set_yticklabels([f"{name[:15]}..." if len(name) > 15 else name 
                                  for name in top_customers['name']])
        
        # Add value labels
        for i, (idx, customer) in enumerate(top_customers.iterrows()):
            axes[0, 0].text(customer['monetary'] + max(top_customers['monetary']) * 0.01, i,
                           f'${customer["monetary"]:,.0f}', va='center', fontweight='bold')
        
        # Most popular product categories
        category_popularity = self.enriched_transactions['category'].value_counts().head(8)
        
        bars = axes[0, 1].bar(range(len(category_popularity)), category_popularity.values, 
                             color=self.segment_palette[:len(category_popularity)], alpha=0.8)
        axes[0, 1].set_title('Most Popular Product Categories', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Transactions')
        axes[0, 1].set_xticks(range(len(category_popularity)))
        axes[0, 1].set_xticklabels(category_popularity.index, rotation=45, ha='right')
        
        # Revenue trends over time (quarterly)
        self.enriched_transactions['quarter'] = self.enriched_transactions['purchase_date'].dt.to_period('Q')
        quarterly_revenue = self.enriched_transactions.groupby('quarter')['transaction_value'].sum()
        quarterly_revenue.index = quarterly_revenue.index.to_timestamp()
        
        axes[1, 0].plot(quarterly_revenue.index, quarterly_revenue.values, 
                       marker='o', linewidth=3, markersize=8, color=self.colors['primary'])
        axes[1, 0].set_title('Quarterly Revenue Trends', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Revenue ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Fill area under curve
        axes[1, 0].fill_between(quarterly_revenue.index, quarterly_revenue.values, 
                               alpha=0.3, color=self.colors['primary'])
        
        # Customer acquisition over time
        monthly_new_customers = self.customers_df.groupby(
            self.customers_df['registration_date'].dt.to_period('M')
        ).size()
        monthly_new_customers.index = monthly_new_customers.index.to_timestamp()
        
        axes[1, 1].bar(monthly_new_customers.index, monthly_new_customers.values, 
                      color=self.colors['accent'], alpha=0.7, width=20)
        axes[1, 1].set_title('Monthly Customer Acquisition', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('New Customers')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/business_insights.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/business_insights.png")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all RFM visualizations at once"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE RFM VISUALIZATIONS")
        print("="*70)
        
        # Generate all visualization sets
        self.plot_rfm_distributions()
        self.plot_rfm_correlation_analysis() 
        self.plot_customer_behavior_analysis()
        self.plot_outlier_analysis()
        self.plot_business_insights()
        
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("📁 Check the 'visualizations/' directory for all PNG files")
        print("="*70)

# Demonstration function
def demonstrate_rfm_visualizations():
    """Demonstrate comprehensive RFM visualizations"""
    print("🚀 STARTING COMPREHENSIVE RFM VISUALIZATION DEMONSTRATION")
    print("="*70)
    
    # Initialize visualization system
    viz = ComprehensiveRFMVisualization()
    
    # Load data
    if not viz.load_data():
        print("❌ Failed to load data. Please ensure all data files exist.")
        return None
    
    # Generate all visualizations
    viz.generate_all_visualizations()
    
    return viz

if __name__ == "__main__":
    # Run demonstration
    viz_system = demonstrate_rfm_visualizations() 