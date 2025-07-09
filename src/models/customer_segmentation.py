import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.rfm_data = None
        self.segments = None
        
    def load_data(self, customers_path='data/customers.csv', 
                  products_path='data/products.csv',
                  transactions_path='data/transactions.csv'):
        """Load customer, product, and transaction data with comprehensive validation"""
        try:
            # Load datasets
            self.customers_df = pd.read_csv(customers_path)
            self.products_df = pd.read_csv(products_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            # Convert date columns
            self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
            self.transactions_df['purchase_date'] = pd.to_datetime(self.transactions_df['purchase_date'])
            
            # Data validation
            print("=" * 60)
            print("DATA LOADING & VALIDATION")
            print("=" * 60)
            print(f"✅ Loaded {len(self.customers_df)} customers")
            print(f"✅ Loaded {len(self.products_df)} products") 
            print(f"✅ Loaded {len(self.transactions_df)} transactions")
            
            # Check for data integrity
            customer_ids_in_trans = set(self.transactions_df['customer_id'].unique())
            customer_ids_in_customers = set(self.customers_df['customer_id'].unique())
            product_ids_in_trans = set(self.transactions_df['product_id'].unique())
            product_ids_in_products = set(self.products_df['product_id'].unique())
            
            orphaned_customers = customer_ids_in_trans - customer_ids_in_customers
            orphaned_products = product_ids_in_trans - product_ids_in_products
            
            if orphaned_customers:
                print(f"⚠️  Warning: {len(orphaned_customers)} orphaned customer IDs in transactions")
            else:
                print("✅ All customer IDs properly linked")
                
            if orphaned_products:
                print(f"⚠️  Warning: {len(orphaned_products)} orphaned product IDs in transactions")
            else:
                print("✅ All product IDs properly linked")
            
            # Create enriched transaction dataset
            self.enriched_transactions = self.transactions_df.merge(
                self.customers_df[['customer_id', 'name', 'registration_date', 'age']], 
                on='customer_id', 
                how='left'
            ).merge(
                self.products_df[['product_id', 'product_name', 'category', 'price']], 
                on='product_id',
                how='left',
                suffixes=('_transaction', '_product')
            )
            
            # Calculate transaction value (quantity × price)
            self.enriched_transactions['transaction_value'] = (
                self.enriched_transactions['quantity'] * 
                self.enriched_transactions['price_transaction']
            )
            
            print(f"✅ Created enriched transaction dataset with {len(self.enriched_transactions)} records")
            print("=" * 60)
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def detect_outliers(self, data, column_name):
        """Detect outliers using IQR method"""
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
        
        print(f"   📊 {column_name} outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
        print(f"      Range: [{data[column_name].min():.2f}, {data[column_name].max():.2f}]")
        print(f"      IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outliers[column_name].index.tolist()

    def calculate_rfm(self, analysis_date=None):
        """Calculate comprehensive RFM (Recency, Frequency, Monetary) analysis"""
        if not hasattr(self, 'enriched_transactions'):
            print("❌ Error: Please load data first using load_data()")
            return None
            
        if analysis_date is None:
            analysis_date = self.enriched_transactions['purchase_date'].max()
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE RFM ANALYSIS")
        print("=" * 60)
        print(f"📅 Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")
        
        # Calculate core RFM metrics
        print("\n🔢 CALCULATING CORE RFM METRICS...")
        
        # Recency: Days since last purchase
        recency_data = self.enriched_transactions.groupby('customer_id')['purchase_date'].max()
        recency_data = (analysis_date - recency_data).dt.days
        
        # Frequency: Total number of transactions  
        frequency_data = self.enriched_transactions.groupby('customer_id')['transaction_id'].count()
        
        # Monetary: Total transaction value (quantity × price)
        monetary_data = self.enriched_transactions.groupby('customer_id')['transaction_value'].sum()
        
        # Create base RFM dataframe
        rfm = pd.DataFrame({
            'recency': recency_data,
            'frequency': frequency_data,
            'monetary': monetary_data
        }).round(2)
        
        # Handle edge cases
        print("\n⚠️  HANDLING EDGE CASES...")
        
        # Check for missing values
        missing_recency = rfm['recency'].isnull().sum()
        missing_frequency = rfm['frequency'].isnull().sum()  
        missing_monetary = rfm['monetary'].isnull().sum()
        
        if missing_recency > 0:
            print(f"   🚨 Found {missing_recency} missing recency values")
        if missing_frequency > 0:
            print(f"   🚨 Found {missing_frequency} missing frequency values")
        if missing_monetary > 0:
            print(f"   🚨 Found {missing_monetary} missing monetary values")
            
        # Fill missing values with appropriate defaults
        rfm['recency'] = rfm['recency'].fillna(rfm['recency'].max())
        rfm['frequency'] = rfm['frequency'].fillna(1)
        rfm['monetary'] = rfm['monetary'].fillna(0)
        
        # Check for single-purchase customers
        single_purchase = (rfm['frequency'] == 1).sum()
        print(f"   📊 Single-purchase customers: {single_purchase} ({single_purchase/len(rfm)*100:.1f}%)")
        
        # Check for very recent customers (< 7 days)
        very_recent = (rfm['recency'] <= 7).sum()
        print(f"   📊 Very recent customers (≤7 days): {very_recent} ({very_recent/len(rfm)*100:.1f}%)")
        
        # Check for zero monetary value (should not happen but good to verify)
        zero_monetary = (rfm['monetary'] <= 0).sum()
        if zero_monetary > 0:
            print(f"   🚨 Customers with zero monetary value: {zero_monetary}")
            # Set minimum monetary value to prevent scoring issues
            rfm.loc[rfm['monetary'] <= 0, 'monetary'] = 0.01
        
        # Outlier detection
        print("\n🔍 OUTLIER DETECTION (IQR METHOD)...")
        recency_outliers = self.detect_outliers(rfm, 'recency')
        frequency_outliers = self.detect_outliers(rfm, 'frequency') 
        monetary_outliers = self.detect_outliers(rfm, 'monetary')
        
        # Flag outliers
        rfm['is_recency_outlier'] = rfm.index.isin(recency_outliers)
        rfm['is_frequency_outlier'] = rfm.index.isin(frequency_outliers)
        rfm['is_monetary_outlier'] = rfm.index.isin(monetary_outliers)
        rfm['outlier_count'] = (rfm['is_recency_outlier'].astype(int) + 
                               rfm['is_frequency_outlier'].astype(int) + 
                               rfm['is_monetary_outlier'].astype(int))
        
        extreme_outliers = (rfm['outlier_count'] == 3).sum()
        print(f"   🎯 Customers that are outliers in all 3 metrics: {extreme_outliers}")
        
        # Add customer information
        rfm = rfm.merge(
            self.customers_df[['customer_id', 'name', 'registration_date', 'age']], 
            on='customer_id', 
            how='left'
        )
        
        # Calculate derived metrics
        print("\n📈 CALCULATING DERIVED METRICS...")
        
        # Days since registration
        rfm['days_since_registration'] = (analysis_date - rfm['registration_date']).dt.days
        
        # Average order value (prevent division by zero)
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        
        # Customer lifetime (days between first and last purchase)
        first_purchase = self.enriched_transactions.groupby('customer_id')['purchase_date'].min()
        last_purchase = self.enriched_transactions.groupby('customer_id')['purchase_date'].max()
        rfm['customer_lifetime_days'] = (last_purchase - first_purchase).dt.days
        rfm['customer_lifetime_days'] = rfm['customer_lifetime_days'].fillna(0)  # Single purchase = 0 days
        
        # Purchase frequency rate (purchases per month since registration)
        rfm['purchase_rate_per_month'] = rfm['frequency'] / (rfm['days_since_registration'] / 30.44)  # 30.44 avg days/month
        rfm['purchase_rate_per_month'] = rfm['purchase_rate_per_month'].replace([np.inf, -np.inf], 0)
        rfm['purchase_rate_per_month'] = rfm['purchase_rate_per_month'].fillna(0)
        
        # RFM Scoring (1-5 scale using quantiles)
        print("\n🏆 CALCULATING RFM SCORES...")
        
        # Define quantile boundaries (20%, 40%, 60%, 80%, 100%)
        quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Recency scoring (lower days = higher score, so reverse)
        try:
            rfm['recency_score'] = pd.qcut(rfm['recency'], 
                                         q=5, 
                                         labels=[5, 4, 3, 2, 1],
                                         duplicates='drop')
        except ValueError:
            # If we can't create 5 distinct quantiles, use manual binning
            rec_bins = np.percentile(rfm['recency'], [0, 20, 40, 60, 80, 100])
            rfm['recency_score'] = pd.cut(rfm['recency'], 
                                        bins=rec_bins, 
                                        labels=[5, 4, 3, 2, 1], 
                                        include_lowest=True,
                                        duplicates='drop')
        
        # Frequency scoring (higher count = higher score)
        try:
            rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 
                                           q=5, 
                                           labels=[1, 2, 3, 4, 5],
                                           duplicates='drop')
        except ValueError:
            freq_bins = np.percentile(rfm['frequency'], [0, 20, 40, 60, 80, 100])
            rfm['frequency_score'] = pd.cut(rfm['frequency'], 
                                          bins=freq_bins, 
                                          labels=[1, 2, 3, 4, 5], 
                                          include_lowest=True,
                                          duplicates='drop')
        
        # Monetary scoring (higher amount = higher score)
        try:
            rfm['monetary_score'] = pd.qcut(rfm['monetary'], 
                                          q=5, 
                                          labels=[1, 2, 3, 4, 5],
                                          duplicates='drop')
        except ValueError:
            mon_bins = np.percentile(rfm['monetary'], [0, 20, 40, 60, 80, 100])
            rfm['monetary_score'] = pd.cut(rfm['monetary'], 
                                         bins=mon_bins, 
                                         labels=[1, 2, 3, 4, 5], 
                                         include_lowest=True,
                                         duplicates='drop')
        
        # Convert scores to numeric
        rfm['recency_score'] = pd.to_numeric(rfm['recency_score'], errors='coerce')
        rfm['frequency_score'] = pd.to_numeric(rfm['frequency_score'], errors='coerce')
        rfm['monetary_score'] = pd.to_numeric(rfm['monetary_score'], errors='coerce')
        
        # Handle any NaN scores (edge case)
        rfm['recency_score'] = rfm['recency_score'].fillna(3)  # Default to middle score
        rfm['frequency_score'] = rfm['frequency_score'].fillna(3)
        rfm['monetary_score'] = rfm['monetary_score'].fillna(3)
        
        # Create combined RFM score string
        rfm['rfm_score'] = (rfm['recency_score'].astype(int).astype(str) + 
                           rfm['frequency_score'].astype(int).astype(str) + 
                           rfm['monetary_score'].astype(int).astype(str))
        
        # Store results
        self.rfm_data = rfm
        
        # Print summary statistics
        print("\n📊 RFM SUMMARY STATISTICS:")
        print(f"   Total customers analyzed: {len(rfm):,}")
        print(f"   Date range: {self.enriched_transactions['purchase_date'].min().strftime('%Y-%m-%d')} to {self.enriched_transactions['purchase_date'].max().strftime('%Y-%m-%d')}")
        
        print(f"\n   Recency (days since last purchase):")
        print(f"      Mean: {rfm['recency'].mean():.1f} | Median: {rfm['recency'].median():.1f}")
        print(f"      Min: {rfm['recency'].min():.0f} | Max: {rfm['recency'].max():.0f}")
        
        print(f"\n   Frequency (number of purchases):")
        print(f"      Mean: {rfm['frequency'].mean():.1f} | Median: {rfm['frequency'].median():.1f}")
        print(f"      Min: {rfm['frequency'].min():.0f} | Max: {rfm['frequency'].max():.0f}")
        
        print(f"\n   Monetary (total spent):")
        print(f"      Mean: ${rfm['monetary'].mean():.2f} | Median: ${rfm['monetary'].median():.2f}")
        print(f"      Min: ${rfm['monetary'].min():.2f} | Max: ${rfm['monetary'].max():.2f}")
        
        print(f"\n   Average Order Value:")
        print(f"      Mean: ${rfm['avg_order_value'].mean():.2f} | Median: ${rfm['avg_order_value'].median():.2f}")
        
        print("=" * 60)
        
        return rfm
    
    def create_rfm_segments(self):
        """Create customer segments based on RFM scores with comprehensive business logic"""
        if self.rfm_data is None:
            print("❌ Error: Please calculate RFM first using calculate_rfm()")
            return None
        
        print("\n" + "=" * 60)
        print("RFM CUSTOMER SEGMENTATION")
        print("=" * 60)
        
        def rfm_segment_logic(row):
            """Advanced RFM segmentation with business-relevant logic"""
            r_score = row['recency_score']
            f_score = row['frequency_score'] 
            m_score = row['monetary_score']
            
            # Champions: Recent, frequent, high-value customers
            if r_score >= 4 and f_score >= 4 and m_score >= 4:
                return 'Champions'
            
            # Loyal Customers: High frequency and monetary, any recency
            elif f_score >= 3 and m_score >= 3:
                return 'Loyal Customers'
            
            # Potential Loyalists: Recent customers with good monetary value
            elif r_score >= 3 and m_score >= 3:
                return 'Potential Loyalists'
            
            # New Customers: Very recent but low frequency/monetary
            elif r_score >= 4 and f_score <= 2:
                return 'New Customers'
            
            # At Risk: Were valuable but haven't purchased recently
            elif r_score <= 2 and f_score >= 3 and m_score >= 3:
                return 'At Risk'
            
            # Cannot Lose Them: High value but very inactive
            elif r_score <= 2 and f_score >= 4 and m_score >= 4:
                return 'Cannot Lose Them'
            
            # Hibernating: Low recency, low frequency, but some monetary value
            elif r_score <= 2 and f_score <= 2 and m_score >= 2:
                return 'Hibernating'
            
            # Price Sensitive: High frequency, low monetary
            elif f_score >= 3 and m_score <= 2:
                return 'Price Sensitive'
            
            # Need Attention: Medium across all metrics  
            elif 2 <= r_score <= 3 and 2 <= f_score <= 3 and 2 <= m_score <= 3:
                return 'Need Attention'
            
            # Lost: Low across all metrics
            elif r_score <= 2 and f_score <= 2 and m_score <= 2:
                return 'Lost'
            
            # Default case
            else:
                return 'Others'
        
        # Apply segmentation logic
        print("🎯 APPLYING SEGMENTATION LOGIC...")
        self.rfm_data['segment'] = self.rfm_data.apply(rfm_segment_logic, axis=1)
        
        # Calculate comprehensive segment statistics
        print("\n📊 CALCULATING SEGMENT STATISTICS...")
        
        segment_stats = self.rfm_data.groupby('segment').agg({
            'recency': ['mean', 'median', 'std'],
            'frequency': ['mean', 'median', 'std'],
            'monetary': ['mean', 'median', 'std', 'sum'],
            'avg_order_value': ['mean', 'median'],
            'customer_lifetime_days': ['mean', 'median'],
            'purchase_rate_per_month': ['mean', 'median'],
            'age': ['mean', 'median'],
            'days_since_registration': ['mean', 'median']
        }).round(2)
        
        # Segment distribution
        segment_counts = self.rfm_data['segment'].value_counts()
        segment_percentages = (segment_counts / len(self.rfm_data) * 100).round(1)
        
        # Display results
        print("\n🏆 SEGMENT DISTRIBUTION:")
        print("=" * 50)
        for segment in segment_counts.index:
            count = segment_counts[segment]
            percentage = segment_percentages[segment]
            print(f"   {segment:<20}: {count:>4} customers ({percentage:>5.1f}%)")
        
        print(f"\n   Total Customers: {len(self.rfm_data):,}")
        
        # Business insights for each segment
        print("\n💡 SEGMENT BUSINESS INSIGHTS:")
        print("=" * 50)
        
        segment_insights = {
            'Champions': 'VIP customers - reward and retain them',
            'Loyal Customers': 'Steady revenue base - upsell opportunities', 
            'Potential Loyalists': 'Recent engagers - nurture for loyalty',
            'New Customers': 'Early adopters - onboard effectively',
            'At Risk': 'Valuable but declining - win-back campaigns',
            'Cannot Lose Them': 'Former champions - urgent intervention',
            'Hibernating': 'Dormant customers - reactivation needed',
            'Price Sensitive': 'Frequent buyers - offer value deals',
            'Need Attention': 'Average customers - engagement campaigns',
            'Lost': 'Inactive customers - consider removal',
            'Others': 'Mixed patterns - individual analysis needed'
        }
        
        for segment in segment_counts.index:
            insight = segment_insights.get(segment, 'Analyze individual patterns')
            avg_monetary = self.rfm_data[self.rfm_data['segment'] == segment]['monetary'].mean()
            avg_frequency = self.rfm_data[self.rfm_data['segment'] == segment]['frequency'].mean()
            avg_recency = self.rfm_data[self.rfm_data['segment'] == segment]['recency'].mean()
            
            print(f"\n   {segment}:")
            print(f"      💰 Avg Spend: ${avg_monetary:.2f}")
            print(f"      🔄 Avg Purchases: {avg_frequency:.1f}")
            print(f"      📅 Avg Recency: {avg_recency:.0f} days")
            print(f"      📝 Strategy: {insight}")
        
        # Calculate business value metrics
        print("\n💵 BUSINESS VALUE ANALYSIS:")
        print("=" * 50)
        
        total_revenue = self.rfm_data['monetary'].sum()
        total_customers = len(self.rfm_data)
        
        for segment in segment_counts.index:
            segment_data = self.rfm_data[self.rfm_data['segment'] == segment]
            segment_revenue = segment_data['monetary'].sum()
            segment_revenue_pct = (segment_revenue / total_revenue * 100)
            segment_customer_pct = (len(segment_data) / total_customers * 100)
            
            # Revenue concentration ratio (revenue % / customer %)
            concentration_ratio = segment_revenue_pct / segment_customer_pct if segment_customer_pct > 0 else 0
            
            print(f"\n   {segment}:")
            print(f"      Revenue: ${segment_revenue:,.2f} ({segment_revenue_pct:.1f}% of total)")
            print(f"      Customers: {len(segment_data)} ({segment_customer_pct:.1f}% of total)")
            print(f"      Value Concentration: {concentration_ratio:.2f}x")
        
        # Create simplified segments for easier interpretation
        def simplified_segment(segment):
            if segment in ['Champions', 'Loyal Customers']:
                return 'High Value'
            elif segment in ['Potential Loyalists', 'New Customers']:
                return 'Medium Value'
            elif segment in ['At Risk', 'Cannot Lose Them', 'Need Attention']:
                return 'At Risk'
            else:
                return 'Low Value'
        
        self.rfm_data['simplified_segment'] = self.rfm_data['segment'].apply(simplified_segment)
        
        # Store segment statistics
        self.segment_stats = segment_stats
        self.segment_distribution = segment_counts
        
        print("\n🔍 SIMPLIFIED SEGMENT DISTRIBUTION:")
        print("=" * 40)
        simplified_counts = self.rfm_data['simplified_segment'].value_counts()
        for segment in simplified_counts.index:
            count = simplified_counts[segment]
            percentage = (count / len(self.rfm_data) * 100)
            print(f"   {segment:<12}: {count:>4} customers ({percentage:>5.1f}%)")
        
        print("\n" + "=" * 60)
        print("✅ RFM SEGMENTATION COMPLETED!")
        print("=" * 60)
        
        return self.rfm_data
    
    def prepare_clustering_data(self, features=['recency', 'frequency', 'monetary']):
        """Prepare RFM data for K-means clustering with standardization"""
        if self.rfm_data is None:
            print("❌ Error: Please calculate RFM first using calculate_rfm()")
            return None
        
        print("\n🔧 PREPARING DATA FOR K-MEANS CLUSTERING")
        print("="*50)
        
        # Extract clustering features
        self.clustering_features = features
        clustering_data = self.rfm_data[features].copy()
        
        print(f"📊 Features selected: {features}")
        print(f"📈 Data shape: {clustering_data.shape}")
        
        # Check for missing values
        missing_values = clustering_data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"⚠️  Missing values found:")
            for feature, count in missing_values.items():
                if count > 0:
                    print(f"   {feature}: {count} missing")
            
            # Handle missing values with mean imputation
            clustering_data = clustering_data.fillna(clustering_data.mean())
            print("✅ Missing values filled with feature means")
        else:
            print("✅ No missing values found")
        
        # Store original data for interpretation
        self.clustering_data_original = clustering_data.copy()
        
        # Standardize features using StandardScaler
        self.clustering_scaler = StandardScaler()
        self.clustering_data_scaled = self.clustering_scaler.fit_transform(clustering_data)
        
        print("\n🎯 Standardizing features completed:")
        for i, feature in enumerate(features):
            mean_scaled = self.clustering_data_scaled[:, i].mean()
            std_scaled = self.clustering_data_scaled[:, i].std()
            print(f"   {feature}: mean={mean_scaled:.3f}, std={std_scaled:.3f}")
        
        print("="*50)
        return self.clustering_data_scaled

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if not hasattr(self, 'clustering_data_scaled') or self.clustering_data_scaled is None:
            print("❌ Error: Please prepare clustering data first")
            return None
        
        print("\n🔍 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*50)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        
        print("🔄 Testing different cluster numbers...")
        
        for k in cluster_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(self.clustering_data_scaled)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            sil_score = silhouette_score(self.clustering_data_scaled, cluster_labels)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            
            print(f"   k={k}: Inertia={inertia:.2f}, Silhouette Score={sil_score:.3f}")
        
        # Find optimal k based on silhouette score
        best_k_silhouette = cluster_range[np.argmax(silhouette_scores)]
        best_silhouette_score = max(silhouette_scores)
        
        # Find elbow point using rate of change
        rate_of_change = []
        for i in range(1, len(inertias)):
            rate_of_change.append(inertias[i-1] - inertias[i])
        
        # Elbow is where rate of change starts decreasing significantly
        elbow_differences = []
        for i in range(1, len(rate_of_change)):
            elbow_differences.append(rate_of_change[i-1] - rate_of_change[i])
        
        if elbow_differences:
            elbow_k = cluster_range[np.argmax(elbow_differences) + 2]  # +2 to account for indexing
        else:
            elbow_k = best_k_silhouette
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   🎯 Best k by Silhouette Score: {best_k_silhouette} (score: {best_silhouette_score:.3f})")
        print(f"   📈 Suggested k by Elbow Method: {elbow_k}")
        
        # Silhouette score interpretation
        if best_silhouette_score > 0.7:
            interpretation = "Excellent clustering structure"
        elif best_silhouette_score > 0.5:
            interpretation = "Good clustering structure"
        elif best_silhouette_score > 0.2:
            interpretation = "Weak but acceptable clustering structure"
        else:
            interpretation = "Poor clustering structure"
        
        print(f"   📋 Silhouette Interpretation: {interpretation}")
        
        # Plot elbow curve and silhouette scores
        self.plot_cluster_validation(cluster_range, inertias, silhouette_scores, 
                                   best_k_silhouette, elbow_k)
        
        # Store results
        self.cluster_validation_results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_silhouette': best_k_silhouette,
            'optimal_k_elbow': elbow_k,
            'best_silhouette_score': best_silhouette_score
        }
        
        print("="*50)
        return self.cluster_validation_results

    def plot_cluster_validation(self, cluster_range, inertias, silhouette_scores, 
                              best_k_silhouette, elbow_k):
        """Plot elbow curve and silhouette scores for cluster validation"""
        print("📊 Creating cluster validation plots...")
        
        import matplotlib.pyplot as plt
        plt.style.use('default')  # Use default style to avoid import issues
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow Curve
        axes[0].plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight elbow point
        if elbow_k in cluster_range:
            elbow_idx = list(cluster_range).index(elbow_k)
            axes[0].plot(elbow_k, inertias[elbow_idx], 
                        'ro', markersize=12, label=f'Elbow at k={elbow_k}')
            axes[0].legend()
        
        # Silhouette Scores
        axes[1].plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Average Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        # Highlight best silhouette score
        best_idx = list(cluster_range).index(best_k_silhouette)
        axes[1].plot(best_k_silhouette, silhouette_scores[best_idx], 
                    'ro', markersize=12, 
                    label=f'Best k={best_k_silhouette} (score={silhouette_scores[best_idx]:.3f})')
        
        # Add interpretation lines
        axes[1].axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label='Excellent (>0.7)')
        axes[1].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label='Good (>0.5)')
        axes[1].axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label='Acceptable (>0.2)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/kmeans_cluster_validation.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: visualizations/kmeans_cluster_validation.png")
        plt.close()

    def perform_comprehensive_kmeans_clustering(self, n_clusters=None):
        """Perform K-means clustering with comprehensive validation and interpretation"""
        if not hasattr(self, 'clustering_data_scaled') or self.clustering_data_scaled is None:
            print("❌ Error: Please prepare clustering data first")
            return None
        
        # Use optimal k if not specified
        if n_clusters is None:
            if hasattr(self, 'cluster_validation_results'):
                n_clusters = self.cluster_validation_results['optimal_k_silhouette']
                print(f"🎯 Using optimal k={n_clusters} from validation analysis")
            else:
                n_clusters = 4
                print(f"⚠️  No validation analysis found, using default k={n_clusters}")
        
        print(f"\n🔄 PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        print("="*50)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        # Fit K-means model
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = self.kmeans_model.fit_predict(self.clustering_data_scaled)
        
        # Add cluster labels to RFM data
        self.rfm_data['cluster'] = cluster_labels
        
        # Calculate silhouette metrics
        overall_silhouette = silhouette_score(self.clustering_data_scaled, cluster_labels)
        individual_silhouettes = silhouette_samples(self.clustering_data_scaled, cluster_labels)
        
        print(f"✅ Clustering completed successfully!")
        print(f"📊 Overall Silhouette Score: {overall_silhouette:.3f}")
        
        # Silhouette interpretation
        if overall_silhouette > 0.7:
            print("🎉 Excellent clustering quality!")
        elif overall_silhouette > 0.5:
            print("👍 Good clustering quality")
        elif overall_silhouette > 0.2:
            print("⚠️  Acceptable clustering quality")
        else:
            print("❌ Poor clustering quality - consider different k or features")
        
        # Store silhouette data
        self.rfm_data['silhouette_score'] = individual_silhouettes
        
        # Comprehensive analysis
        self.analyze_kmeans_cluster_characteristics()
        self.create_kmeans_cluster_business_names()
        self.generate_kmeans_cluster_summary_report()
        
        # Create silhouette plot
        self.plot_silhouette_analysis(n_clusters, cluster_labels, 
                                    individual_silhouettes, overall_silhouette)
        
        print("="*50)
        return cluster_labels, overall_silhouette

    def analyze_kmeans_cluster_characteristics(self):
        """Analyze detailed characteristics of each K-means cluster"""
        print("\n📊 ANALYZING K-MEANS CLUSTER CHARACTERISTICS")
        print("="*50)
        
        n_clusters = len(self.rfm_data['cluster'].unique())
        
        # Calculate cluster centroids in original scale
        centroids_scaled = self.kmeans_model.cluster_centers_
        centroids_original = self.clustering_scaler.inverse_transform(centroids_scaled)
        
        # Create centroids dataframe
        centroids_df = pd.DataFrame(
            centroids_original, 
            columns=self.clustering_features
        )
        centroids_df['cluster'] = range(len(centroids_df))
        
        print("🎯 Cluster Centroids (Original Scale):")
        print(centroids_df.round(2))
        
        # Analyze cluster sizes and silhouette scores
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_data = self.rfm_data[self.rfm_data['cluster'] == cluster_id]
            cluster_silhouettes = cluster_data['silhouette_score']
            
            stats = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.rfm_data) * 100,
                'avg_silhouette': cluster_silhouettes.mean(),
                'min_silhouette': cluster_silhouettes.min(),
                'max_silhouette': cluster_silhouettes.max(),
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'avg_monetary': cluster_data['monetary'].mean(),
                'total_revenue': cluster_data['monetary'].sum()
            }
            cluster_stats.append(stats)
        
        self.kmeans_cluster_stats_df = pd.DataFrame(cluster_stats)
        
        print(f"\n📈 Cluster Statistics:")
        for _, stats in self.kmeans_cluster_stats_df.iterrows():
            print(f"\n   Cluster {int(stats['cluster'])}:")
            print(f"      Size: {int(stats['size'])} customers ({stats['percentage']:.1f}%)")
            print(f"      Avg Silhouette: {stats['avg_silhouette']:.3f}")
            print(f"      RFM Profile: R={stats['avg_recency']:.0f}, F={stats['avg_frequency']:.1f}, M=${stats['avg_monetary']:.0f}")
            print(f"      Total Revenue: ${stats['total_revenue']:,.2f}")
        
        # Store centroids for business naming
        self.kmeans_cluster_centroids = centroids_df

    def create_kmeans_cluster_business_names(self):
        """Assign meaningful business names to K-means clusters based on RFM characteristics"""
        print("\n🏷️  ASSIGNING BUSINESS NAMES TO K-MEANS CLUSTERS")
        print("="*50)
        
        cluster_names = {}
        
        # Get overall dataset percentiles for comparison
        recency_median = self.rfm_data['recency'].median()
        frequency_75th = self.rfm_data['frequency'].quantile(0.75)
        frequency_median = self.rfm_data['frequency'].median()
        monetary_75th = self.rfm_data['monetary'].quantile(0.75)
        monetary_median = self.rfm_data['monetary'].median()
        
        for _, cluster_stats in self.kmeans_cluster_stats_df.iterrows():
            cluster_id = int(cluster_stats['cluster'])
            avg_r = cluster_stats['avg_recency']
            avg_f = cluster_stats['avg_frequency']
            avg_m = cluster_stats['avg_monetary']
            
            # Business logic for naming clusters
            if avg_f >= frequency_75th and avg_m >= monetary_75th:
                if avg_r <= recency_median:
                    name = "Champions"
                    description = "Best customers - high value, frequent, recent"
                else:
                    name = "Loyal High-Value"
                    description = "Valuable customers who need re-engagement"
            elif avg_f >= frequency_median and avg_m >= monetary_median:
                if avg_r <= recency_median:
                    name = "Regular Customers"
                    description = "Steady customers with good potential"
                else:
                    name = "At-Risk Valuable"
                    description = "Previously good customers becoming inactive"
            elif avg_r <= recency_median:
                if avg_m >= monetary_median:
                    name = "New High-Value"
                    description = "Recent customers with high spending potential"
                else:
                    name = "New Customers"
                    description = "Recent customers to nurture"
            elif avg_f <= frequency_median and avg_m <= monetary_median:
                name = "Lost/Inactive"
                description = "Customers requiring reactivation campaigns"
            else:
                name = "Needs Attention"
                description = "Mixed characteristics requiring individual analysis"
            
            cluster_names[cluster_id] = {
                'name': name,
                'description': description
            }
            
            print(f"🏷️  Cluster {cluster_id}: {name}")
            print(f"     📝 {description}")
            print(f"     📊 Profile: R={avg_r:.0f}, F={avg_f:.1f}, M=${avg_m:.0f}")
        
        # Add business names to RFM data
        self.rfm_data['kmeans_cluster_name'] = self.rfm_data['cluster'].map(
            lambda x: cluster_names[x]['name']
        )
        self.rfm_data['kmeans_cluster_description'] = self.rfm_data['cluster'].map(
            lambda x: cluster_names[x]['description']
        )
        
        self.kmeans_cluster_business_names = cluster_names
        print("="*50)

    def plot_silhouette_analysis(self, n_clusters, cluster_labels, 
                                sample_silhouette_values, silhouette_avg):
        """Create comprehensive silhouette plot for K-means clusters"""
        print(f"📊 Creating silhouette analysis plot for k={n_clusters}...")
        
        import matplotlib.pyplot as plt
        plt.style.use('default')
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # The silhouette plot
        y_lower = 10
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = colors[i % len(colors)]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontweight='bold')
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette Coefficient Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cluster Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Silhouette Plot for {n_clusters} Clusters\nAverage Score: {silhouette_avg:.3f}', 
                    fontsize=14, fontweight='bold')
        
        # The vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        
        # Add interpretation regions
        ax.axvline(x=0.7, color="green", linestyle=":", alpha=0.5, label='Excellent (>0.7)')
        ax.axvline(x=0.5, color="orange", linestyle=":", alpha=0.5, label='Good (>0.5)')
        ax.axvline(x=0.2, color="red", linestyle=":", alpha=0.5, label='Acceptable (>0.2)')
        
        ax.legend()
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(self.clustering_data_scaled) + (n_clusters + 1) * 10])
        
        plt.tight_layout()
        plt.savefig(f'visualizations/kmeans_silhouette_analysis_k{n_clusters}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Saved: visualizations/kmeans_silhouette_analysis_k{n_clusters}.png")
        plt.close()

    def generate_kmeans_cluster_summary_report(self):
        """Generate comprehensive K-means cluster summary report"""
        print("\n📋 COMPREHENSIVE K-MEANS CLUSTER SUMMARY REPORT")
        print("="*70)
        
        # Overall clustering summary
        n_clusters = len(self.kmeans_cluster_stats_df)
        total_customers = len(self.rfm_data)
        overall_silhouette = self.rfm_data['silhouette_score'].mean()
        
        print(f"🎯 CLUSTERING OVERVIEW:")
        print(f"   📊 Number of Clusters: {n_clusters}")
        print(f"   👥 Total Customers: {total_customers:,}")
        print(f"   🎯 Overall Silhouette Score: {overall_silhouette:.3f}")
        print(f"   📈 Features Used: {', '.join(self.clustering_features)}")
        
        # Business value analysis
        print(f"\n💰 BUSINESS VALUE ANALYSIS:")
        total_revenue = self.rfm_data['monetary'].sum()
        
        # Sort clusters by revenue contribution
        cluster_revenue = self.kmeans_cluster_stats_df.sort_values('total_revenue', ascending=False)
        
        for _, stats in cluster_revenue.iterrows():
            cluster_id = int(stats['cluster'])
            name = self.kmeans_cluster_business_names[cluster_id]['name']
            revenue_pct = (stats['total_revenue'] / total_revenue) * 100
            
            print(f"\n   🏆 {name} (Cluster {cluster_id}):")
            print(f"      👥 Customers: {int(stats['size'])} ({stats['percentage']:.1f}% of total)")
            print(f"      💰 Revenue: ${stats['total_revenue']:,.2f} ({revenue_pct:.1f}% of total)")
            print(f"      📊 Avg Customer Value: ${stats['avg_monetary']:,.2f}")
            print(f"      🎯 Silhouette Quality: {stats['avg_silhouette']:.3f}")
            
            # Value concentration
            value_concentration = revenue_pct / stats['percentage']
            print(f"      📈 Value Concentration: {value_concentration:.2f}x")
        
        # Cluster quality assessment
        print(f"\n🔍 CLUSTER QUALITY ASSESSMENT:")
        
        # Check for clusters with poor silhouette scores
        poor_clusters = self.kmeans_cluster_stats_df[self.kmeans_cluster_stats_df['avg_silhouette'] < 0.2]
        good_clusters = self.kmeans_cluster_stats_df[self.kmeans_cluster_stats_df['avg_silhouette'] >= 0.5]
        
        print(f"   ✅ Clusters with good silhouette scores (≥0.5): {len(good_clusters)}")
        print(f"   ⚠️  Clusters with poor silhouette scores (<0.2): {len(poor_clusters)}")
        
        if len(poor_clusters) > 0:
            print(f"   📝 Poor quality clusters may need attention:")
            for _, cluster in poor_clusters.iterrows():
                cluster_id = int(cluster['cluster'])
                name = self.kmeans_cluster_business_names[cluster_id]['name']
                print(f"      • Cluster {cluster_id} ({name}): {cluster['avg_silhouette']:.3f}")
        
        # Business recommendations
        print(f"\n💡 BUSINESS RECOMMENDATIONS:")
        
        # Find the most valuable cluster
        top_cluster = cluster_revenue.iloc[0]
        top_cluster_id = int(top_cluster['cluster'])
        top_cluster_name = self.kmeans_cluster_business_names[top_cluster_id]['name']
        
        print(f"   🎯 Focus on '{top_cluster_name}' - highest revenue concentration")
        
        # Find clusters with high frequency but low monetary
        for _, stats in self.kmeans_cluster_stats_df.iterrows():
            cluster_id = int(stats['cluster'])
            name = self.kmeans_cluster_business_names[cluster_id]['name']
            
            if (stats['avg_frequency'] >= self.rfm_data['frequency'].median() and 
                stats['avg_monetary'] < self.rfm_data['monetary'].median()):
                print(f"   💰 '{name}' - upselling opportunity (high frequency, low spend)")
            
            if (stats['avg_recency'] > self.rfm_data['recency'].quantile(0.75) and
                stats['avg_monetary'] >= self.rfm_data['monetary'].median()):
                print(f"   🔄 '{name}' - re-engagement needed (valuable but inactive)")
        
        print("="*70)

    def get_customer_insights(self, customer_id=None):
        """Get insights for a specific customer or all customers"""
        if self.rfm_data is None:
            print("Please calculate RFM first using calculate_rfm()")
            return None
        
        if customer_id:
            customer_data = self.rfm_data[self.rfm_data['customer_id'] == customer_id]
            if customer_data.empty:
                print(f"Customer {customer_id} not found")
                return None
            
            customer = customer_data.iloc[0]
            print(f"\nCustomer Insights for {customer['name']} ({customer_id})")
            print("=" * 50)
            print(f"RFM Segment: {customer['segment']}")
            print(f"Simplified Segment: {customer['simplified_segment']}")
            if 'cluster_name' in customer:
                print(f"Cluster: {customer['cluster_name']}")
            print(f"Recency: {customer['recency']} days ago")
            print(f"Frequency: {customer['frequency']} purchases")
            print(f"Monetary: ${customer['monetary']:.2f}")
            print(f"Average Order Value: ${customer['avg_order_value']:.2f}")
            print(f"RFM Score: {customer['rfm_score']}")
            
            return customer_data
        else:
            # Summary insights for all customers
            print("\nOverall Customer Insights")
            print("=" * 50)
            
            # Top customers by monetary value
            top_monetary = self.rfm_data.nlargest(10, 'monetary')[['customer_id', 'name', 'monetary', 'frequency', 'segment']]
            print("\nTop 10 Customers by Monetary Value:")
            print(top_monetary.to_string(index=False))
            
            # Most frequent customers
            top_frequency = self.rfm_data.nlargest(10, 'frequency')[['customer_id', 'name', 'frequency', 'monetary', 'segment']]
            print("\nTop 10 Most Frequent Customers:")
            print(top_frequency.to_string(index=False))
            
            # Segment performance
            segment_summary = self.rfm_data.groupby('segment').agg({
                'customer_id': 'count',
                'monetary': ['mean', 'sum'],
                'frequency': 'mean',
                'recency': 'mean'
            }).round(2)
            
            print("\nSegment Performance Summary:")
            print(segment_summary)
            
            return self.rfm_data
    
    def get_sample_rfm_analysis(self, sample_size=10):
        """Display sample RFM calculations for verification"""
        if self.rfm_data is None:
            print("❌ Error: Please calculate RFM first using calculate_rfm()")
            return None
        
        print("\n" + "=" * 60)
        print("SAMPLE RFM CALCULATIONS")
        print("=" * 60)
        
        # Get a representative sample from different segments
        sample_data = []
        
        # Try to get samples from each segment
        segments = self.rfm_data['segment'].unique()
        samples_per_segment = max(1, sample_size // len(segments))
        
        for segment in segments:
            segment_data = self.rfm_data[self.rfm_data['segment'] == segment]
            if len(segment_data) > 0:
                sample_count = min(samples_per_segment, len(segment_data))
                segment_sample = segment_data.sample(n=sample_count, random_state=42)
                sample_data.append(segment_sample)
        
        # Combine all samples
        if sample_data:
            sample_df = pd.concat(sample_data).head(sample_size)
        else:
            sample_df = self.rfm_data.head(sample_size)
        
        # Display sample calculations
        print(f"📊 Showing {len(sample_df)} sample customer RFM calculations:")
        print()
        
        for idx, row in sample_df.iterrows():
            print(f"🔍 Customer: {row['name']} (ID: {row['customer_id']})")
            print(f"   Age: {row['age']} | Registration: {row['registration_date'].strftime('%Y-%m-%d')}")
            print(f"   Recency: {row['recency']} days (Score: {row['recency_score']})")
            print(f"   Frequency: {row['frequency']} purchases (Score: {row['frequency_score']})")
            print(f"   Monetary: ${row['monetary']:.2f} (Score: {row['monetary_score']})")
            print(f"   Avg Order Value: ${row['avg_order_value']:.2f}")
            print(f"   Customer Lifetime: {row['customer_lifetime_days']} days")
            print(f"   Purchase Rate: {row['purchase_rate_per_month']:.2f} per month")
            print(f"   RFM Score: {row['rfm_score']} | Segment: {row['segment']}")
            
            # Check for outliers
            outlier_flags = []
            if row['is_recency_outlier']:
                outlier_flags.append('Recency')
            if row['is_frequency_outlier']:
                outlier_flags.append('Frequency')
            if row['is_monetary_outlier']:
                outlier_flags.append('Monetary')
            
            if outlier_flags:
                print(f"   ⚠️  Outlier in: {', '.join(outlier_flags)}")
            
            print("-" * 60)
        
        return sample_df

    def save_results(self, filename='data/customer_segmentation_results.csv'):
        """Save RFM results to CSV file"""
        if self.rfm_data is None:
            print("❌ No RFM data to save. Please run analysis first.")
            return False
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            self.rfm_data.to_csv(filename, index=False)
            print(f"✅ Results saved to {filename}")
            print(f"   📄 {len(self.rfm_data)} customer records saved")
            print(f"   📊 {len(self.rfm_data.columns)} columns included")
            return True
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

    def create_cluster_radar_charts(self, save_interactive=True, save_static=True):
        """Create comprehensive radar charts for cluster centroids visualization"""
        if not hasattr(self, 'kmeans_cluster_stats_df') or self.kmeans_cluster_stats_df is None:
            print("❌ Error: Please perform K-means clustering first")
            return None
        
        print("\n📊 CREATING CLUSTER RADAR CHARTS")
        print("="*50)
        
        # Prepare radar chart data
        radar_data = self.prepare_radar_chart_data()
        
        if save_interactive:
            self.create_interactive_radar_chart(radar_data)
        
        if save_static:
            self.create_static_radar_chart(radar_data)
        
        # Create business insights annotations
        self.create_radar_business_insights(radar_data)
        
        print("="*50)
        return radar_data

    def prepare_radar_chart_data(self):
        """Prepare normalized data for radar charts"""
        print("🔧 Preparing radar chart data...")
        
        # Extract cluster centroids and statistics
        cluster_data = []
        
        for _, stats in self.kmeans_cluster_stats_df.iterrows():
            cluster_id = int(stats['cluster'])
            cluster_name = self.kmeans_cluster_business_names[cluster_id]['name']
            
            # Original RFM values (means for each cluster)
            recency = stats['avg_recency']
            frequency = stats['avg_frequency']
            monetary = stats['avg_monetary']
            
            cluster_data.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'description': self.kmeans_cluster_business_names[cluster_id]['description'],
                'size': int(stats['size']),
                'percentage': stats['percentage'],
                'total_revenue': stats['total_revenue'],
                'avg_silhouette': stats['avg_silhouette'],
                'recency_raw': recency,
                'frequency_raw': frequency,
                'monetary_raw': monetary
            })
        
        # Normalize values for radar chart (0-1 scale)
        # For recency: lower is better, so we invert it
        max_recency = max([c['recency_raw'] for c in cluster_data])
        min_recency = min([c['recency_raw'] for c in cluster_data])
        
        # For frequency and monetary: higher is better
        max_frequency = max([c['frequency_raw'] for c in cluster_data])
        min_frequency = min([c['frequency_raw'] for c in cluster_data])
        max_monetary = max([c['monetary_raw'] for c in cluster_data])
        min_monetary = min([c['monetary_raw'] for c in cluster_data])
        
        print(f"📊 Normalization ranges:")
        print(f"   Recency: {min_recency:.0f} - {max_recency:.0f} days")
        print(f"   Frequency: {min_frequency:.1f} - {max_frequency:.1f} purchases")
        print(f"   Monetary: ${min_monetary:.0f} - ${max_monetary:.0f}")
        
        for cluster in cluster_data:
            # Invert recency (lower recency = higher score)
            if max_recency == min_recency:
                cluster['recency_normalized'] = 0.5
            else:
                cluster['recency_normalized'] = 1 - (cluster['recency_raw'] - min_recency) / (max_recency - min_recency)
            
            # Normalize frequency (higher = better)
            if max_frequency == min_frequency:
                cluster['frequency_normalized'] = 0.5
            else:
                cluster['frequency_normalized'] = (cluster['frequency_raw'] - min_frequency) / (max_frequency - min_frequency)
            
            # Normalize monetary (higher = better)
            if max_monetary == min_monetary:
                cluster['monetary_normalized'] = 0.5
            else:
                cluster['monetary_normalized'] = (cluster['monetary_raw'] - min_monetary) / (max_monetary - min_monetary)
        
        print(f"✅ Prepared data for {len(cluster_data)} clusters")
        return cluster_data

    def create_interactive_radar_chart(self, radar_data):
        """Create interactive radar chart using plotly"""
        print("🎨 Creating interactive radar chart...")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.colors as pc
            
            # Create radar chart
            fig = go.Figure()
            
            # Define colors for clusters
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9F43', '#6C5CE7', '#FD79A8']
            
            # Add each cluster to the radar chart
            for i, cluster in enumerate(radar_data):
                color = colors[i % len(colors)]
                
                # Prepare data for radar chart
                r_values = [
                    cluster['recency_normalized'],
                    cluster['frequency_normalized'], 
                    cluster['monetary_normalized']
                ]
                
                # Close the radar chart by repeating first value
                r_values.append(r_values[0])
                
                theta_labels = ['Recency Score<br>(Recent Activity)', 
                              'Frequency Score<br>(Purchase Count)', 
                              'Monetary Score<br>(Spending Level)',
                              'Recency Score<br>(Recent Activity)']
                
                # Create hover text with detailed information
                hover_text = [
                    f"Recency: {cluster['recency_raw']:.0f} days<br>Score: {cluster['recency_normalized']:.2f}",
                    f"Frequency: {cluster['frequency_raw']:.1f} purchases<br>Score: {cluster['frequency_normalized']:.2f}",
                    f"Monetary: ${cluster['monetary_raw']:,.0f}<br>Score: {cluster['monetary_normalized']:.2f}",
                    f"Recency: {cluster['recency_raw']:.0f} days<br>Score: {cluster['recency_normalized']:.2f}"
                ]
                
                # Add cluster to plot
                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_labels,
                    fill='toself',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    line=dict(color=color, width=3),
                    name=f"{cluster['cluster_name']}<br>({cluster['size']} customers)",
                    text=hover_text,
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                "%{text}<br>" +
                                f"Revenue: ${cluster['total_revenue']:,.0f}<br>" +
                                f"Avg Value: ${cluster['total_revenue']/cluster['size']:,.0f}<br>" +
                                f"Silhouette: {cluster['avg_silhouette']:.3f}" +
                                "<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0, 0.25, 0.5, 0.75, 1],
                        ticktext=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'],
                        tickfont=dict(size=10),
                        gridcolor='lightgray',
                        gridwidth=1
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, color='darkblue'),
                        rotation=90,
                        direction='clockwise'
                    )
                ),
                title=dict(
                    text="<b>Customer Cluster Radar Analysis</b><br>" +
                         "<sub>RFM Profile Comparison Across Segments</sub>",
                    x=0.5,
                    font=dict(size=18, color='darkblue')
                ),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=11)
                ),
                width=900,
                height=700,
                showlegend=True,
                plot_bgcolor='white'
            )
            
            # Add annotations with business insights
            total_customers = sum([c['size'] for c in radar_data])
            total_revenue = sum([c['total_revenue'] for c in radar_data])
            
            annotations = [
                dict(
                    text=f"<b>Analysis Summary</b><br>" +
                         f"Total Customers: {total_customers:,}<br>" +
                         f"Total Revenue: ${total_revenue:,.0f}<br>" +
                         f"Clusters: {len(radar_data)}",
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
            
            fig.update_layout(annotations=annotations)
            
            # Save interactive chart
            filename = 'visualizations/interactive_cluster_radar_chart.html'
            fig.write_html(filename)
            print(f"✅ Saved interactive radar chart: {filename}")
            
            return fig
            
        except ImportError:
            print("⚠️  Plotly not available, skipping interactive radar chart")
            return None
        except Exception as e:
            print(f"❌ Error creating interactive radar chart: {e}")
            return None

    def create_static_radar_chart(self, radar_data):
        """Create static radar chart using matplotlib"""
        print("📊 Creating static radar chart...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from math import pi
            
            # Set up the radar chart
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            # Define the angles for each axis
            categories = ['Recency\nScore', 'Frequency\nScore', 'Monetary\nScore']
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Colors for clusters
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9F43', '#6C5CE7', '#FD79A8']
            
            # Plot each cluster
            for i, cluster in enumerate(radar_data):
                color = colors[i % len(colors)]
                
                # Values for this cluster
                values = [
                    cluster['recency_normalized'],
                    cluster['frequency_normalized'],
                    cluster['monetary_normalized']
                ]
                values += values[:1]  # Complete the circle
                
                # Plot the cluster
                ax.plot(angles, values, 'o-', linewidth=3, 
                       label=f"{cluster['cluster_name']} ({cluster['size']} customers)",
                       color=color)
                ax.fill(angles, values, alpha=0.15, color=color)
                
                # Add value labels on the chart
                for angle, value in zip(angles[:-1], values[:-1]):
                    ax.text(angle, value + 0.05, f'{value:.2f}', 
                           ha='center', va='center', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['Low', 'Below Avg', 'Average', 'Above Avg', 'High'], 
                              fontsize=10, alpha=0.7)
            ax.grid(True, alpha=0.3)
            
            # Add title and legend
            plt.title('Customer Cluster Radar Analysis\nRFM Profile Comparison', 
                     size=16, fontweight='bold', pad=30)
            
            # Position legend outside the plot
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
            
            # Add business insights as text
            insights_text = self.generate_radar_insights_text(radar_data)
            plt.figtext(0.02, 0.02, insights_text, fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
                       verticalalignment='bottom')
            
            # Save the plot
            plt.tight_layout()
            filename = 'visualizations/static_cluster_radar_chart.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved static radar chart: {filename}")
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating static radar chart: {e}")
            return False

    def generate_radar_insights_text(self, radar_data):
        """Generate business insights text for radar chart"""
        insights = ["BUSINESS INSIGHTS:"]
        
        # Find most extreme clusters
        highest_value_cluster = max(radar_data, key=lambda x: x['monetary_normalized'])
        most_frequent_cluster = max(radar_data, key=lambda x: x['frequency_normalized'])
        most_recent_cluster = max(radar_data, key=lambda x: x['recency_normalized'])
        
        insights.append(f"• Highest Value: {highest_value_cluster['cluster_name']}")
        insights.append(f"• Most Frequent: {most_frequent_cluster['cluster_name']}")
        insights.append(f"• Most Recent: {most_recent_cluster['cluster_name']}")
        
        # Revenue concentration
        total_revenue = sum([c['total_revenue'] for c in radar_data])
        for cluster in sorted(radar_data, key=lambda x: x['total_revenue'], reverse=True)[:2]:
            revenue_pct = (cluster['total_revenue'] / total_revenue) * 100
            insights.append(f"• {cluster['cluster_name']}: {revenue_pct:.1f}% of revenue")
        
        return '\n'.join(insights)

    def create_radar_business_insights(self, radar_data):
        """Create detailed business insights and recommendations based on radar analysis"""
        print("💡 Generating radar chart business insights...")
        
        # Analyze cluster characteristics
        insights = {
            'cluster_analysis': [],
            'transition_opportunities': [],
            'business_recommendations': [],
            'value_assessment': []
        }
        
        # Cluster analysis
        for cluster in radar_data:
            profile_strength = []
            
            if cluster['recency_normalized'] > 0.7:
                profile_strength.append("Recent Activity")
            elif cluster['recency_normalized'] < 0.3:
                profile_strength.append("Inactive")
            
            if cluster['frequency_normalized'] > 0.7:
                profile_strength.append("High Frequency")
            elif cluster['frequency_normalized'] < 0.3:
                profile_strength.append("Low Frequency")
            
            if cluster['monetary_normalized'] > 0.7:
                profile_strength.append("High Value")
            elif cluster['monetary_normalized'] < 0.3:
                profile_strength.append("Low Value")
            
            avg_score = (cluster['recency_normalized'] + cluster['frequency_normalized'] + cluster['monetary_normalized']) / 3
            
            insights['cluster_analysis'].append({
                'name': cluster['cluster_name'],
                'size': cluster['size'],
                'strengths': profile_strength,
                'avg_score': avg_score,
                'revenue_share': cluster['total_revenue'] / sum([c['total_revenue'] for c in radar_data]) * 100
            })
        
        # Find transition opportunities
        sorted_clusters = sorted(radar_data, key=lambda x: (x['recency_normalized'] + x['frequency_normalized'] + x['monetary_normalized']))
        
        for i in range(len(sorted_clusters) - 1):
            current = sorted_clusters[i]
            next_tier = sorted_clusters[i + 1]
            
            improvements_needed = []
            if next_tier['recency_normalized'] - current['recency_normalized'] > 0.2:
                improvements_needed.append("increase purchase recency")
            if next_tier['frequency_normalized'] - current['frequency_normalized'] > 0.2:
                improvements_needed.append("increase purchase frequency")
            if next_tier['monetary_normalized'] - current['monetary_normalized'] > 0.2:
                improvements_needed.append("increase spending amount")
            
            if improvements_needed:
                insights['transition_opportunities'].append({
                    'from_cluster': current['cluster_name'],
                    'to_cluster': next_tier['cluster_name'],
                    'improvements': improvements_needed,
                    'potential_customers': current['size']
                })
        
        # Business recommendations
        total_revenue = sum([c['total_revenue'] for c in radar_data])
        
        for cluster in radar_data:
            recommendations = []
            revenue_concentration = (cluster['total_revenue'] / total_revenue) / (cluster['size'] / sum([c['size'] for c in radar_data]))
            
            if cluster['recency_normalized'] > 0.7 and cluster['monetary_normalized'] > 0.7:
                recommendations.append("Priority retention programs")
                recommendations.append("Loyalty rewards and VIP treatment")
            
            elif cluster['recency_normalized'] < 0.3:
                recommendations.append("Re-engagement campaigns")
                recommendations.append("Win-back offers and incentives")
            
            elif cluster['frequency_normalized'] > 0.7 and cluster['monetary_normalized'] < 0.5:
                recommendations.append("Upselling opportunities")
                recommendations.append("Premium product recommendations")
            
            elif cluster['monetary_normalized'] > 0.5 and cluster['frequency_normalized'] < 0.5:
                recommendations.append("Cross-selling campaigns")
                recommendations.append("Frequency-building programs")
            
            insights['business_recommendations'].append({
                'cluster': cluster['cluster_name'],
                'revenue_concentration': revenue_concentration,
                'recommendations': recommendations
            })
        
        # Value assessment
        for cluster in radar_data:
            total_customers = sum([c['size'] for c in radar_data])
            customer_share = cluster['size'] / total_customers * 100
            revenue_share = cluster['total_revenue'] / total_revenue * 100
            value_ratio = revenue_share / customer_share if customer_share > 0 else 0
            
            if value_ratio > 2:
                value_category = "High Value Concentration"
            elif value_ratio > 1:
                value_category = "Above Average Value"
            elif value_ratio > 0.5:
                value_category = "Below Average Value"
            else:
                value_category = "Low Value Concentration"
            
            insights['value_assessment'].append({
                'cluster': cluster['cluster_name'],
                'value_ratio': value_ratio,
                'value_category': value_category,
                'customer_share': customer_share,
                'revenue_share': revenue_share
            })
        
        # Print insights
        print("\n📋 RADAR CHART BUSINESS INSIGHTS:")
        print("-" * 50)
        
        print("\n🎯 CLUSTER ANALYSIS:")
        for analysis in insights['cluster_analysis']:
            print(f"  {analysis['name']}:")
            print(f"    Size: {analysis['size']} customers ({analysis['revenue_share']:.1f}% revenue)")
            print(f"    Strengths: {', '.join(analysis['strengths']) if analysis['strengths'] else 'Balanced profile'}")
            print(f"    Overall Score: {analysis['avg_score']:.2f}/1.0")
        
        print("\n🔄 TRANSITION OPPORTUNITIES:")
        for transition in insights['transition_opportunities']:
            print(f"  {transition['from_cluster']} → {transition['to_cluster']}:")
            print(f"    Potential: {transition['potential_customers']} customers")
            print(f"    Actions: {', '.join(transition['improvements'])}")
        
        print("\n💡 BUSINESS RECOMMENDATIONS:")
        for rec in insights['business_recommendations']:
            if rec['recommendations']:
                print(f"  {rec['cluster']} (Value Concentration: {rec['revenue_concentration']:.2f}x):")
                for recommendation in rec['recommendations']:
                    print(f"    • {recommendation}")
        
        print("\n💰 VALUE ASSESSMENT:")
        for assessment in insights['value_assessment']:
            print(f"  {assessment['cluster']}: {assessment['value_category']}")
            print(f"    {assessment['customer_share']:.1f}% customers → {assessment['revenue_share']:.1f}% revenue (ratio: {assessment['value_ratio']:.2f}x)")
        
        # Store insights for later use
        self.radar_business_insights = insights
        
        return insights

    def create_cluster_comparison_matrix(self, radar_data):
        """Create a comparison matrix showing differences between clusters"""
        print("📊 Creating cluster comparison matrix...")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            # Calculate pairwise distances between clusters
            n_clusters = len(radar_data)
            distance_matrix = np.zeros((n_clusters, n_clusters))
            cluster_names = [c['cluster_name'] for c in radar_data]
            
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j:
                        # Calculate Euclidean distance in normalized RFM space
                        r_diff = radar_data[i]['recency_normalized'] - radar_data[j]['recency_normalized']
                        f_diff = radar_data[i]['frequency_normalized'] - radar_data[j]['frequency_normalized']
                        m_diff = radar_data[i]['monetary_normalized'] - radar_data[j]['monetary_normalized']
                        distance = np.sqrt(r_diff**2 + f_diff**2 + m_diff**2)
                        distance_matrix[i][j] = distance
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(distance_matrix, cmap='RdYlBu_r', vmin=0, vmax=distance_matrix.max())
            
            # Add labels
            ax.set_xticks(np.arange(n_clusters))
            ax.set_yticks(np.arange(n_clusters))
            ax.set_xticklabels(cluster_names, rotation=45, ha='right')
            ax.set_yticklabels(cluster_names)
            
            # Add text annotations
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j:
                        text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
                    else:
                        text = ax.text(j, i, "—", ha="center", va="center", color="gray")
            
            ax.set_title("Cluster Similarity Matrix\n(Lower values = more similar)", fontsize=14, fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('RFM Distance', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('visualizations/cluster_comparison_matrix.png', dpi=300, bbox_inches='tight')
            print("✅ Saved cluster comparison matrix: visualizations/cluster_comparison_matrix.png")
            plt.close()
            
            return distance_matrix
            
        except Exception as e:
            print(f"❌ Error creating comparison matrix: {e}")
            return None

    def export_radar_analysis_report(self):
        """Export comprehensive radar analysis report"""
        if not hasattr(self, 'radar_business_insights'):
            print("❌ No radar insights available. Please create radar charts first.")
            return False
        
        print("📄 Exporting radar analysis report...")
        
        try:
            import os
            os.makedirs('reports', exist_ok=True)
            
            report_content = self.generate_radar_analysis_report()
            
            with open('reports/radar_analysis_report.md', 'w') as f:
                f.write(report_content)
            
            print("✅ Saved radar analysis report: reports/radar_analysis_report.md")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return False

    def generate_radar_analysis_report(self):
        """Generate comprehensive markdown report for radar analysis"""
        from datetime import datetime
        
        report = f"""# Customer Cluster Radar Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive radar chart analysis of customer clusters based on RFM (Recency, Frequency, Monetary) metrics. The analysis provides visual insights into customer segment characteristics and actionable business recommendations.

## Cluster Overview

"""
        
        # Add cluster analysis
        insights = self.radar_business_insights
        total_customers = sum([c['size'] for c in insights['cluster_analysis']])
        total_revenue = sum([c['revenue_share'] for c in insights['cluster_analysis']])
        
        for analysis in insights['cluster_analysis']:
            report += f"""### {analysis['name']}
- **Size**: {analysis['size']:,} customers ({analysis['size']/total_customers*100:.1f}% of total)
- **Revenue Share**: {analysis['revenue_share']:.1f}% of total revenue
- **Key Strengths**: {', '.join(analysis['strengths']) if analysis['strengths'] else 'Balanced profile'}
- **Overall RFM Score**: {analysis['avg_score']:.2f}/1.0

"""
        
        # Add transition opportunities
        report += """## Transition Opportunities

The following opportunities exist for moving customers between segments:

"""
        for transition in insights['transition_opportunities']:
            report += f"""### {transition['from_cluster']} → {transition['to_cluster']}
- **Potential Impact**: {transition['potential_customers']:,} customers
- **Required Actions**: {', '.join(transition['improvements'])}

"""
        
        # Add business recommendations
        report += """## Business Recommendations

### Segment-Specific Strategies

"""
        for rec in insights['business_recommendations']:
            if rec['recommendations']:
                report += f"""#### {rec['cluster']}
**Value Concentration**: {rec['revenue_concentration']:.2f}x average

"""
                for recommendation in rec['recommendations']:
                    report += f"- {recommendation}\n"
                report += "\n"
        
        # Add value assessment
        report += """## Value Assessment

| Cluster | Customer Share | Revenue Share | Value Ratio | Category |
|---------|---------------|---------------|-------------|----------|
"""
        
        for assessment in insights['value_assessment']:
            report += f"| {assessment['cluster']} | {assessment['customer_share']:.1f}% | {assessment['revenue_share']:.1f}% | {assessment['value_ratio']:.2f}x | {assessment['value_category']} |\n"
        
        report += """
## Visualization Files

The following visualization files have been generated:

1. **interactive_cluster_radar_chart.html** - Interactive plotly radar chart
2. **static_cluster_radar_chart.png** - Static matplotlib radar chart  
3. **cluster_comparison_matrix.png** - Cluster similarity heatmap

## Methodology

### RFM Normalization
- **Recency**: Inverted scale (recent activity = higher score)
- **Frequency**: Linear scale (higher frequency = higher score)  
- **Monetary**: Linear scale (higher spending = higher score)
- **Range**: All metrics normalized to 0-1 scale

### Business Insights
- Transition opportunities identified based on RFM score gaps
- Value concentration calculated as revenue share / customer share ratio
- Recommendations tailored to each cluster's RFM profile

---
*Report generated by Customer Analytics System*
"""
        
        return report

# Demonstration function
def demonstrate_rfm_analysis():
    """Demonstrate comprehensive RFM analysis with generated data"""
    print("🚀 STARTING RFM ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    # Initialize the segmentation system
    segmentation = CustomerSegmentation()
    
    # Load the data (assumes data has been generated)
    print("\n1️⃣ LOADING DATA...")
    if not segmentation.load_data():
        print("❌ Failed to load data. Please run data generation first.")
        return None
    
    # Calculate RFM metrics
    print("\n2️⃣ CALCULATING RFM METRICS...")
    rfm_data = segmentation.calculate_rfm()
    
    if rfm_data is None:
        print("❌ Failed to calculate RFM metrics.")
        return None
    
    # Create segments
    print("\n3️⃣ CREATING CUSTOMER SEGMENTS...")
    segmented_data = segmentation.create_rfm_segments()
    
    # Show sample calculations
    print("\n4️⃣ SAMPLE RFM CALCULATIONS...")
    sample_data = segmentation.get_sample_rfm_analysis(sample_size=5)
    
    # Save results
    print("\n5️⃣ SAVING RESULTS...")
    segmentation.save_results()
    
    print("\n" + "=" * 70)
    print("✅ RFM ANALYSIS DEMONSTRATION COMPLETED!")
    print("=" * 70)
    
    return segmentation

if __name__ == "__main__":
    # Run the comprehensive RFM demonstration
    segmentation = demonstrate_rfm_analysis() 