import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and context for professional visualizations
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ComprehensiveRFMVisualization:
    def __init__(self):
        """Initialize visualization system with professional styling"""
        
        # Define consistent color palettes
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
        
        # Color palettes for different visualizations
        self.segment_palette = ['#2E86AB', '#A23B72', '#F18F01', '#1B998B', '#C73E1D', 
                               '#FFD23F', '#8E44AD', '#27AE60', '#E67E22', '#95A5A6']
        self.rfm_palette = ['#C73E1D', '#F18F01', '#FFD23F', '#1B998B', '#2E86AB']
        self.outlier_palette = ['#2E86AB', '#C73E1D']
        
        # Set figure sizes
        self.figsize_small = (10, 6)
        self.figsize_medium = (12, 8) 
        self.figsize_large = (15, 10)
        self.figsize_extra_large = (18, 12)
        
        # Create output directory
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        print("🎨 Professional RFM Visualization System Initialized")
        print("   📊 Seaborn styling configured")
        print("   🎯 Color palettes defined")
        print("   📁 Output directory created")
        
    def load_data(self, customers_path='data/customers.csv', 
                  products_path='data/products.csv', 
                  transactions_path='data/transactions.csv',
                  segmentation_path='data/customer_segmentation_results.csv'):
        """Load and prepare all data for comprehensive visualization"""
        try:
            print("\n" + "="*60)
            print("LOADING DATA FOR VISUALIZATION")
            print("="*60)
            
            # Load core datasets
            self.customers_df = pd.read_csv(customers_path)
            self.products_df = pd.read_csv(products_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            print(f"✅ Loaded {len(self.customers_df)} customers")
            print(f"✅ Loaded {len(self.products_df)} products")
            print(f"✅ Loaded {len(self.transactions_df)} transactions")
            
            # Convert date columns
            self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
            self.transactions_df['purchase_date'] = pd.to_datetime(self.transactions_df['purchase_date'])
            
            # Create enriched transaction data for analysis
            self.enriched_transactions = self.transactions_df.merge(
                self.customers_df[['customer_id', 'name', 'age', 'registration_date']], 
                on='customer_id', how='left'
            ).merge(
                self.products_df[['product_id', 'product_name', 'category', 'price']], 
                on='product_id', how='left',
                suffixes=('_transaction', '_product')
            )
            
            # Calculate transaction values
            self.enriched_transactions['transaction_value'] = (
                self.enriched_transactions['quantity'] * 
                self.enriched_transactions['price_transaction']
            )
            
            # Add time-based features for analysis
            self.enriched_transactions['year'] = self.enriched_transactions['purchase_date'].dt.year
            self.enriched_transactions['month'] = self.enriched_transactions['purchase_date'].dt.month
            self.enriched_transactions['quarter'] = self.enriched_transactions['purchase_date'].dt.quarter
            self.enriched_transactions['day_of_week'] = self.enriched_transactions['purchase_date'].dt.day_name()
            self.enriched_transactions['year_month'] = self.enriched_transactions['purchase_date'].dt.to_period('M')
            
            print(f"✅ Created enriched transaction dataset with {len(self.enriched_transactions)} records")
            
            # Load RFM segmentation results
            try:
                self.rfm_df = pd.read_csv(segmentation_path)
                self.rfm_df['registration_date'] = pd.to_datetime(self.rfm_df['registration_date'])
                print(f"✅ Loaded RFM segmentation data with {len(self.rfm_df)} customers")
                self.has_rfm_data = True
            except Exception as e:
                self.rfm_df = None
                self.has_rfm_data = False
                print(f"⚠️  RFM segmentation data not found: {e}")
                print("   Run customer segmentation analysis first to get RFM visualizations")
            
            print("="*60)
            print("✅ DATA LOADING COMPLETED")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def plot_rfm_distributions(self, save_fig=True):
        """Create comprehensive RFM distribution visualizations"""
        if not self.has_rfm_data:
            print("❌ RFM data not available. Run customer segmentation first.")
            return
        
        print("📊 Creating RFM Distribution Plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize_extra_large)
        
        # RFM Histograms with statistical annotations
        # Recency Distribution
        sns.histplot(data=self.rfm_df, x='recency', bins=30, kde=True, 
                    color=self.colors['danger'], alpha=0.7, ax=axes[0, 0])
        axes[0, 0].set_title('Recency Distribution\n(Days Since Last Purchase)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # Add statistical annotations
        mean_recency = self.rfm_df['recency'].mean()
        median_recency = self.rfm_df['recency'].median()
        axes[0, 0].axvline(mean_recency, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_recency:.0f}')
        axes[0, 0].axvline(median_recency, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_recency:.0f}')
        axes[0, 0].legend()
        
        # Frequency Distribution
        sns.histplot(data=self.rfm_df, x='frequency', bins=30, kde=True,
                    color=self.colors['primary'], alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Frequency Distribution\n(Number of Purchases)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        mean_frequency = self.rfm_df['frequency'].mean()
        median_frequency = self.rfm_df['frequency'].median()
        axes[0, 1].axvline(mean_frequency, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_frequency:.1f}')
        axes[0, 1].axvline(median_frequency, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_frequency:.1f}')
        axes[0, 1].legend()
        
        # Monetary Distribution
        sns.histplot(data=self.rfm_df, x='monetary', bins=30, kde=True,
                    color=self.colors['success'], alpha=0.7, ax=axes[0, 2])
        axes[0, 2].set_title('Monetary Distribution\n(Total Spent)', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Total Spent ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        mean_monetary = self.rfm_df['monetary'].mean()
        median_monetary = self.rfm_df['monetary'].median()
        axes[0, 2].axvline(mean_monetary, color='red', linestyle='--', alpha=0.8, label=f'Mean: ${mean_monetary:.0f}')
        axes[0, 2].axvline(median_monetary, color='orange', linestyle='--', alpha=0.8, label=f'Median: ${median_monetary:.0f}')
        axes[0, 2].legend()
        
        # RFM Box plots showing outliers
        rfm_data = self.rfm_df[['recency', 'frequency', 'monetary']].copy()
        
        # Recency Box Plot
        sns.boxplot(data=rfm_data, y='recency', color=self.colors['danger'], ax=axes[1, 0])
        axes[1, 0].set_title('Recency Outliers', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Days Since Last Purchase')
        
        # Frequency Box Plot  
        sns.boxplot(data=rfm_data, y='frequency', color=self.colors['primary'], ax=axes[1, 1])
        axes[1, 1].set_title('Frequency Outliers', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Purchases')
        
        # Monetary Box Plot
        sns.boxplot(data=rfm_data, y='monetary', color=self.colors['success'], ax=axes[1, 2])
        axes[1, 2].set_title('Monetary Outliers', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Total Spent ($)')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/rfm_distributions.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/rfm_distributions.png")
        plt.close()
    
    def plot_rfm_correlation_heatmap(self, save_fig=True):
        """Create RFM correlation heatmap and segment analysis"""
        if not self.has_rfm_data:
            print("❌ RFM data not available. Run customer segmentation first.")
            return
        
        print("📊 Creating RFM Correlation and Segment Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # RFM Correlation Heatmap
        rfm_corr = self.rfm_df[['recency', 'frequency', 'monetary']].corr()
        mask = np.triu(np.ones_like(rfm_corr, dtype=bool))
        
        sns.heatmap(rfm_corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=axes[0, 0])
        axes[0, 0].set_title('RFM Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        
        # RFM Segment Distribution
        segment_counts = self.rfm_df['segment'].value_counts()
        colors = self.segment_palette[:len(segment_counts)]
        
        axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # Segment Performance Bar Chart
        segment_performance = self.rfm_df.groupby('segment').agg({
            'monetary': 'sum',
            'customer_id': 'count'
        }).reset_index()
        segment_performance['avg_monetary'] = segment_performance['monetary'] / segment_performance['customer_id']
        segment_performance = segment_performance.sort_values('monetary', ascending=True)
        
        bars = axes[1, 0].barh(segment_performance['segment'], segment_performance['monetary'], 
                              color=colors[:len(segment_performance)])
        axes[1, 0].set_title('Total Revenue by Segment', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Total Revenue ($)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 0].text(width, bar.get_y() + bar.get_height()/2, 
                           f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        # RFM Score Distribution
        if 'rfm_score' in self.rfm_df.columns:
            rfm_score_dist = self.rfm_df['rfm_score'].value_counts().head(15)
            axes[1, 1].bar(range(len(rfm_score_dist)), rfm_score_dist.values, 
                          color=self.colors['accent'], alpha=0.7)
            axes[1, 1].set_title('Top 15 RFM Score Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('RFM Score')
            axes[1, 1].set_ylabel('Number of Customers')
            axes[1, 1].set_xticks(range(len(rfm_score_dist)))
            axes[1, 1].set_xticklabels(rfm_score_dist.index, rotation=45)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/rfm_correlation_segments.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/rfm_correlation_segments.png")
        plt.close()
    
    def plot_sales_trends(self, save_fig=True):
        """Plot sales trends and patterns"""
        # Prepare transaction data
        self.transactions_df['year_month'] = self.transactions_df['purchase_date'].dt.to_period('M')
        self.transactions_df['month'] = self.transactions_df['purchase_date'].dt.month
        self.transactions_df['day_of_week'] = self.transactions_df['purchase_date'].dt.day_name()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # Monthly sales trend
        monthly_sales = self.transactions_df.groupby('year_month').agg({
            'price': 'sum',
            'transaction_id': 'count'
        })
        
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(monthly_sales.index.to_timestamp(), monthly_sales['price'], 
                        color='blue', marker='o', label='Revenue')
        line2 = ax2.plot(monthly_sales.index.to_timestamp(), monthly_sales['transaction_id'], 
                        color='red', marker='s', label='Transactions')
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Revenue ($)', color='blue')
        ax2.set_ylabel('Number of Transactions', color='red')
        ax1.set_title('Monthly Sales Trend')
        ax1.tick_params(axis='x', rotation=45)
        
        # Seasonal pattern
        seasonal_sales = self.transactions_df.groupby('month')['price'].sum()
        axes[0, 1].bar(seasonal_sales.index, seasonal_sales.values, color='green', alpha=0.7)
        axes[0, 1].set_title('Seasonal Sales Pattern')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Total Sales ($)')
        
        # Day of week pattern
        daily_pattern = self.transactions_df.groupby('day_of_week')['transaction_id'].count()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_pattern = daily_pattern.reindex(day_order)
        axes[1, 0].bar(daily_pattern.index, daily_pattern.values, color='coral', alpha=0.7)
        axes[1, 0].set_title('Sales by Day of Week')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Transactions')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Category sales
        category_sales = self.transactions_df.merge(self.products_df, on='product_id').groupby('category')['price'].sum()
        axes[1, 1].pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Sales Distribution by Category')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('sales_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rfm_analysis(self, save_fig=True):
        """Plot RFM analysis visualizations"""
        if self.segmentation_df is None:
            print("Segmentation data not available. Run customer segmentation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # RFM distributions
        axes[0, 0].hist(self.segmentation_df['recency'], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        
        axes[0, 1].hist(self.segmentation_df['frequency'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        axes[0, 2].hist(self.segmentation_df['monetary'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spent ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        # RFM scatter plots
        scatter = axes[1, 0].scatter(self.segmentation_df['recency'], 
                                   self.segmentation_df['frequency'],
                                   c=self.segmentation_df['monetary'], 
                                   cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Recency (Days)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Recency vs Frequency (colored by Monetary)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        axes[1, 1].scatter(self.segmentation_df['frequency'], 
                          self.segmentation_df['monetary'],
                          alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Monetary ($)')
        axes[1, 1].set_title('Frequency vs Monetary')
        
        # Segment distribution
        segment_counts = self.segmentation_df['segment'].value_counts()
        axes[1, 2].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Customer Segments Distribution')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('rfm_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_customer_segments(self, save_fig=True):
        """Plot detailed customer segment analysis"""
        if self.segmentation_df is None:
            print("Segmentation data not available. Run customer segmentation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        
        # Segment performance
        segment_performance = self.segmentation_df.groupby('segment').agg({
            'monetary': ['mean', 'sum'],
            'frequency': 'mean',
            'recency': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        # Average monetary value by segment
        avg_monetary = segment_performance['monetary']['mean'].sort_values(ascending=False)
        axes[0, 0].bar(avg_monetary.index, avg_monetary.values, color='gold', alpha=0.7)
        axes[0, 0].set_title('Average Monetary Value by Segment')
        axes[0, 0].set_ylabel('Average Spending ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total revenue by segment
        total_revenue = segment_performance['monetary']['sum'].sort_values(ascending=False)
        axes[0, 1].bar(total_revenue.index, total_revenue.values, color='lightblue', alpha=0.7)
        axes[0, 1].set_title('Total Revenue by Segment')
        axes[0, 1].set_ylabel('Total Revenue ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Segment size
        segment_size = segment_performance['customer_id']['count'].sort_values(ascending=False)
        axes[1, 0].bar(segment_size.index, segment_size.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Number of Customers by Segment')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Average frequency by segment
        avg_frequency = segment_performance['frequency']['mean'].sort_values(ascending=False)
        axes[1, 1].bar(avg_frequency.index, avg_frequency.values, color='orange', alpha=0.7)
        axes[1, 1].set_title('Average Purchase Frequency by Segment')
        axes[1, 1].set_ylabel('Average Purchases')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_rfm_plot(self):
        """Create interactive 3D RFM plot using Plotly"""
        if self.segmentation_df is None:
            print("Segmentation data not available. Run customer segmentation first.")
            return
        
        fig = px.scatter_3d(
            self.segmentation_df, 
            x='recency', 
            y='frequency', 
            z='monetary',
            color='segment',
            hover_data=['customer_id', 'name'],
            title='Interactive 3D RFM Analysis',
            labels={
                'recency': 'Recency (Days)',
                'frequency': 'Frequency (Purchases)',
                'monetary': 'Monetary ($)',
                'segment': 'Customer Segment'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Recency (Days)',
                yaxis_title='Frequency (Purchases)',
                zaxis_title='Monetary ($)'
            ),
            width=1000,
            height=700
        )
        
        fig.show()
        fig.write_html('interactive_rfm_plot.html')
        print("Interactive RFM plot saved as 'interactive_rfm_plot.html'")
    
    def create_cluster_visualization(self):
        """Create cluster visualization if K-means clustering was performed"""
        if self.segmentation_df is None or 'cluster' not in self.segmentation_df.columns:
            print("Cluster data not available. Run K-means clustering first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize_medium)
        
        # 2D cluster plot (Recency vs Monetary)
        scatter1 = axes[0].scatter(self.segmentation_df['recency'], 
                                  self.segmentation_df['monetary'],
                                  c=self.segmentation_df['cluster'], 
                                  cmap='tab10', 
                                  alpha=0.7)
        axes[0].set_xlabel('Recency (Days)')
        axes[0].set_ylabel('Monetary ($)')
        axes[0].set_title('K-means Clusters (Recency vs Monetary)')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 2D cluster plot (Frequency vs Monetary)
        scatter2 = axes[1].scatter(self.segmentation_df['frequency'], 
                                  self.segmentation_df['monetary'],
                                  c=self.segmentation_df['cluster'], 
                                  cmap='tab10', 
                                  alpha=0.7)
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Monetary ($)')
        axes[1].set_title('K-means Clusters (Frequency vs Monetary)')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_dashboard_summary(self, save_fig=True):
        """Create a comprehensive dashboard summary"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        total_customers = len(self.customers_df)
        ax1.text(0.5, 0.5, f'{total_customers:,}', fontsize=24, ha='center', va='center', weight='bold')
        ax1.text(0.5, 0.2, 'Total Customers', fontsize=12, ha='center', va='center')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        total_revenue = self.transactions_df['price'].sum()
        ax2.text(0.5, 0.5, f'${total_revenue:,.0f}', fontsize=24, ha='center', va='center', weight='bold')
        ax2.text(0.5, 0.2, 'Total Revenue', fontsize=12, ha='center', va='center')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        total_transactions = len(self.transactions_df)
        ax3.text(0.5, 0.5, f'{total_transactions:,}', fontsize=24, ha='center', va='center', weight='bold')
        ax3.text(0.5, 0.2, 'Total Transactions', fontsize=12, ha='center', va='center')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        avg_order_value = self.transactions_df['price'].mean()
        ax4.text(0.5, 0.5, f'${avg_order_value:.2f}', fontsize=24, ha='center', va='center', weight='bold')
        ax4.text(0.5, 0.2, 'Avg Order Value', fontsize=12, ha='center', va='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Sales trend
        ax5 = fig.add_subplot(gs[1, :2])
        monthly_sales = self.transactions_df.groupby(self.transactions_df['purchase_date'].dt.to_period('M'))['price'].sum()
        ax5.plot(monthly_sales.index.to_timestamp(), monthly_sales.values, marker='o', linewidth=2)
        ax5.set_title('Monthly Revenue Trend')
        ax5.set_ylabel('Revenue ($)')
        ax5.tick_params(axis='x', rotation=45)
        
        # Category distribution
        ax6 = fig.add_subplot(gs[1, 2:])
        category_sales = self.transactions_df.merge(self.products_df, on='product_id').groupby('category')['price'].sum()
        ax6.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%', startangle=90)
        ax6.set_title('Revenue by Category')
        
        # Customer age distribution
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.hist(self.customers_df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax7.set_title('Customer Age Distribution')
        ax7.set_xlabel('Age')
        ax7.set_ylabel('Number of Customers')
        
        # Top products
        ax8 = fig.add_subplot(gs[2, 2:])
        top_products = (self.transactions_df.merge(self.products_df, on='product_id')
                       .groupby('product_name')['price'].sum()
                       .sort_values(ascending=False).head(10))
        ax8.barh(range(len(top_products)), top_products.values)
        ax8.set_yticks(range(len(top_products)))
        ax8.set_yticklabels(top_products.index)
        ax8.set_title('Top 10 Products by Revenue')
        ax8.set_xlabel('Revenue ($)')
        
        # Segment analysis (if available)
        if self.segmentation_df is not None:
            ax9 = fig.add_subplot(gs[3, :2])
            segment_counts = self.segmentation_df['segment'].value_counts()
            ax9.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
            ax9.set_title('Customer Segments')
            
            ax10 = fig.add_subplot(gs[3, 2:])
            segment_revenue = self.segmentation_df.groupby('segment')['monetary'].sum().sort_values(ascending=False)
            ax10.bar(segment_revenue.index, segment_revenue.values, color='green', alpha=0.7)
            ax10.set_title('Revenue by Segment')
            ax10.set_ylabel('Total Revenue ($)')
            ax10.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Customer Analytics Dashboard', fontsize=20, y=0.98)
        
        if save_fig:
            plt.savefig('dashboard_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating all visualizations...")
        
        try:
            # Skip missing methods for now - focus on working ones
            print("\n1. Customer Distribution Analysis...")
            try:
                self.plot_customer_distribution()
            except AttributeError:
                print("⚠️  Customer distribution plot not available - skipping")
            
            print("\n2. Product Analysis...")
            try:
                self.plot_product_analysis()
            except AttributeError:
                print("⚠️  Product analysis plot not available - skipping")
            
            print("\n3. Sales Trends...")
            self.plot_sales_trends()
            
            if self.segmentation_df is not None:
                print("\n4. RFM Analysis...")
                self.plot_rfm_analysis()
                
                print("\n5. Customer Segments...")
                self.plot_customer_segments()
                
                print("\n6. Interactive RFM Plot...")
                self.create_interactive_rfm_plot()
                
                if 'cluster' in self.segmentation_df.columns:
                    print("\n7. Cluster Visualization...")
                    try:
                        self.create_cluster_visualization()
                    except AttributeError:
                        print("⚠️  Cluster visualization not available - skipping")
            
            print("\n8. Dashboard Summary...")
            self.create_dashboard_summary()
            
            print("\nAll visualizations completed!")
            return True
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    viz = CustomerVisualization()
    
    if viz.load_data():
        viz.generate_all_visualizations() 