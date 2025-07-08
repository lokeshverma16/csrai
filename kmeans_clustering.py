import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

class ComprehensiveKMeansClustering:
    def __init__(self):
        """Initialize comprehensive K-means clustering system"""
        self.clustering_scaler = StandardScaler()
        self.kmeans_model = None
        self.clustering_data_scaled = None
        self.clustering_data_original = None
        self.cluster_analysis_results = None
        self.cluster_stats_df = None
        self.cluster_business_names = None
        
        print("🎯 Comprehensive K-means Clustering System Initialized")
        
        # Create output directory
        import os
        os.makedirs('visualizations', exist_ok=True)

    def load_rfm_data(self, rfm_path='data/customer_segmentation_results.csv'):
        """Load RFM segmentation data for clustering"""
        try:
            print("\n📊 LOADING RFM DATA FOR CLUSTERING")
            print("="*50)
            
            self.rfm_df = pd.read_csv(rfm_path)
            self.rfm_df['registration_date'] = pd.to_datetime(self.rfm_df['registration_date'])
            
            print(f"✅ Loaded {len(self.rfm_df)} customer records")
            print(f"📊 Columns available: {list(self.rfm_df.columns)}")
            
            # Verify required columns
            required_cols = ['recency', 'frequency', 'monetary']
            missing_cols = [col for col in required_cols if col not in self.rfm_df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False
            
            print("✅ All required RFM columns found")
            print("="*50)
            return True
            
        except Exception as e:
            print(f"❌ Error loading RFM data: {e}")
            return False

    def prepare_clustering_data(self, features=['recency', 'frequency', 'monetary']):
        """Prepare data for K-means clustering with proper preprocessing"""
        if not hasattr(self, 'rfm_df'):
            print("❌ Error: Please load RFM data first")
            return None
        
        print("\n🔧 PREPARING DATA FOR CLUSTERING")
        print("="*50)
        
        # Extract clustering features
        self.clustering_features = features
        clustering_data = self.rfm_df[features].copy()
        
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
        
        # Standardize features
        print("\n🎯 Standardizing features...")
        self.clustering_data_scaled = self.clustering_scaler.fit_transform(clustering_data)
        
        # Print scaling statistics
        print("📊 Standardization completed:")
        for i, feature in enumerate(features):
            mean_scaled = self.clustering_data_scaled[:, i].mean()
            std_scaled = self.clustering_data_scaled[:, i].std()
            print(f"   {feature}: mean={mean_scaled:.3f}, std={std_scaled:.3f}")
        
        print("="*50)
        return self.clustering_data_scaled

    def find_optimal_clusters(self, max_clusters=10, plot_results=True):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if self.clustering_data_scaled is None:
            print("❌ Error: Please prepare clustering data first")
            return None
        
        print("\n🔍 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*50)
        
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        silhouette_details = {}
        
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
            
            # Store silhouette details for later analysis
            silhouette_details[k] = {
                'labels': cluster_labels,
                'score': sil_score,
                'individual_scores': silhouette_samples(self.clustering_data_scaled, cluster_labels)
            }
            
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
        
        # Store results
        self.cluster_analysis_results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'silhouette_details': silhouette_details,
            'optimal_k_silhouette': best_k_silhouette,
            'optimal_k_elbow': elbow_k,
            'best_silhouette_score': best_silhouette_score
        }
        
        # Plot results if requested
        if plot_results:
            self.plot_cluster_validation_curves()
            self.plot_silhouette_analysis(best_k_silhouette)
        
        print("="*50)
        return self.cluster_analysis_results

    def plot_cluster_validation_curves(self, save_fig=True):
        """Plot elbow curve and silhouette scores for cluster validation"""
        if self.cluster_analysis_results is None:
            print("❌ No cluster analysis results available for plotting")
            return
            
        print("📊 Creating cluster validation plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        results = self.cluster_analysis_results
        
        # Elbow Curve
        axes[0].plot(results['cluster_range'], results['inertias'], 
                    'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight elbow point
        elbow_k = results['optimal_k_elbow']
        if elbow_k in results['cluster_range']:
            elbow_idx = results['cluster_range'].index(elbow_k)
            axes[0].plot(elbow_k, results['inertias'][elbow_idx], 
                        'ro', markersize=12, label=f'Elbow at k={elbow_k}')
            axes[0].legend()
        
        # Silhouette Scores
        axes[1].plot(results['cluster_range'], results['silhouette_scores'], 
                    'go-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Average Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        # Highlight best silhouette score
        best_k = results['optimal_k_silhouette']
        best_idx = results['cluster_range'].index(best_k)
        axes[1].plot(best_k, results['silhouette_scores'][best_idx], 
                    'ro', markersize=12, label=f'Best k={best_k} (score={results["best_silhouette_score"]:.3f})')
        
        # Add interpretation lines
        axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.7)')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Good (>0.5)')
        axes[1].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Acceptable (>0.2)')
        axes[1].legend()
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('visualizations/cluster_validation.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: visualizations/cluster_validation.png")
        plt.close()

    def plot_silhouette_analysis(self, n_clusters, save_fig=True):
        """Create comprehensive silhouette plot for specific number of clusters"""
        print(f"📊 Creating silhouette analysis for k={n_clusters}...")
        
        # Get silhouette details for this k
        if (self.cluster_analysis_results is not None and 
            'silhouette_details' in self.cluster_analysis_results and
            n_clusters in self.cluster_analysis_results['silhouette_details']):
            details = self.cluster_analysis_results['silhouette_details'][n_clusters]
            cluster_labels = details['labels']
            sample_silhouette_values = details['individual_scores']
            silhouette_avg = details['score']
        else:
            # Calculate if not available
            if hasattr(self, 'kmeans_model') and self.kmeans_model is not None:
                # Use existing model if available
                cluster_labels = self.kmeans_model.labels_
            else:
                # Create new model
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(self.clustering_data_scaled)
            
            sample_silhouette_values = silhouette_samples(self.clustering_data_scaled, cluster_labels)
            silhouette_avg = silhouette_score(self.clustering_data_scaled, cluster_labels)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # The silhouette plot
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
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
        if save_fig:
            plt.savefig(f'visualizations/silhouette_analysis_k{n_clusters}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: visualizations/silhouette_analysis_k{n_clusters}.png")
        plt.close()

    def perform_kmeans_clustering(self, n_clusters=None):
        """Perform K-means clustering with comprehensive validation"""
        if self.clustering_data_scaled is None:
            print("❌ Error: Please prepare clustering data first")
            return None
        
        # Use optimal k if not specified
        if n_clusters is None:
            if self.cluster_analysis_results:
                n_clusters = self.cluster_analysis_results['optimal_k_silhouette']
                print(f"🎯 Using optimal k={n_clusters} from validation analysis")
            else:
                n_clusters = 4
                print(f"⚠️  No validation analysis found, using default k={n_clusters}")
        
        print(f"\n🔄 PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        print("="*50)
        
        # Fit K-means model
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = self.kmeans_model.fit_predict(self.clustering_data_scaled)
        
        # Add cluster labels to RFM data
        self.rfm_df['cluster'] = cluster_labels
        
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
        self.rfm_df['silhouette_score'] = individual_silhouettes
        
        # Analyze cluster characteristics
        self.analyze_cluster_characteristics()
        self.create_cluster_business_names()
        
        print("="*50)
        return cluster_labels, overall_silhouette

    def analyze_cluster_characteristics(self):
        """Analyze detailed characteristics of each cluster"""
        print("\n📊 ANALYZING CLUSTER CHARACTERISTICS")
        print("="*50)
        
        n_clusters = len(self.rfm_df['cluster'].unique())
        
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
            cluster_data = self.rfm_df[self.rfm_df['cluster'] == cluster_id]
            cluster_silhouettes = cluster_data['silhouette_score']
            
            stats = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.rfm_df) * 100,
                'avg_silhouette': cluster_silhouettes.mean(),
                'min_silhouette': cluster_silhouettes.min(),
                'max_silhouette': cluster_silhouettes.max(),
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'avg_monetary': cluster_data['monetary'].mean(),
                'total_revenue': cluster_data['monetary'].sum()
            }
            cluster_stats.append(stats)
        
        self.cluster_stats_df = pd.DataFrame(cluster_stats)
        
        print(f"\n📈 Cluster Statistics:")
        for _, stats in self.cluster_stats_df.iterrows():
            print(f"\n   Cluster {int(stats['cluster'])}:")
            print(f"      Size: {int(stats['size'])} customers ({stats['percentage']:.1f}%)")
            print(f"      Avg Silhouette: {stats['avg_silhouette']:.3f}")
            print(f"      RFM Profile: R={stats['avg_recency']:.0f}, F={stats['avg_frequency']:.1f}, M=${stats['avg_monetary']:.0f}")
            print(f"      Total Revenue: ${stats['total_revenue']:,.2f}")
        
        self.cluster_centroids = centroids_df

    def create_cluster_business_names(self):
        """Assign meaningful business names to clusters based on RFM characteristics"""
        print("\n🏷️  ASSIGNING BUSINESS NAMES TO CLUSTERS")
        print("="*50)
        
        cluster_names = {}
        
        # Get overall dataset percentiles for comparison
        recency_median = self.rfm_df['recency'].median()
        frequency_75th = self.rfm_df['frequency'].quantile(0.75)
        frequency_median = self.rfm_df['frequency'].median()
        monetary_75th = self.rfm_df['monetary'].quantile(0.75)
        monetary_median = self.rfm_df['monetary'].median()
        
        for _, cluster_stats in self.cluster_stats_df.iterrows():
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
        self.rfm_df['cluster_name'] = self.rfm_df['cluster'].map(
            lambda x: cluster_names[x]['name']
        )
        self.rfm_df['cluster_description'] = self.rfm_df['cluster'].map(
            lambda x: cluster_names[x]['description']
        )
        
        self.cluster_business_names = cluster_names
        print("="*50)

    def generate_cluster_summary_report(self):
        """Generate comprehensive cluster summary report"""
        print("\n📋 COMPREHENSIVE CLUSTER SUMMARY REPORT")
        print("="*70)
        
        # Overall clustering summary
        n_clusters = len(self.cluster_stats_df)
        total_customers = len(self.rfm_df)
        overall_silhouette = self.rfm_df['silhouette_score'].mean()
        
        print(f"🎯 CLUSTERING OVERVIEW:")
        print(f"   📊 Number of Clusters: {n_clusters}")
        print(f"   👥 Total Customers: {total_customers:,}")
        print(f"   🎯 Overall Silhouette Score: {overall_silhouette:.3f}")
        print(f"   📈 Features Used: {', '.join(self.clustering_features)}")
        
        # Business value analysis
        print(f"\n💰 BUSINESS VALUE ANALYSIS:")
        total_revenue = self.rfm_df['monetary'].sum()
        
        # Sort clusters by revenue contribution
        cluster_revenue = self.cluster_stats_df.sort_values('total_revenue', ascending=False)
        
        for _, stats in cluster_revenue.iterrows():
            cluster_id = int(stats['cluster'])
            name = self.cluster_business_names[cluster_id]['name']
            revenue_pct = (stats['total_revenue'] / total_revenue) * 100
            
            print(f"\n   🏆 {name} (Cluster {cluster_id}):")
            print(f"      👥 Customers: {int(stats['size'])} ({stats['percentage']:.1f}% of total)")
            print(f"      💰 Revenue: ${stats['total_revenue']:,.2f} ({revenue_pct:.1f}% of total)")
            print(f"      📊 Avg Customer Value: ${stats['avg_monetary']:,.2f}")
            print(f"      🎯 Silhouette Quality: {stats['avg_silhouette']:.3f}")
            
            # Value concentration
            value_concentration = revenue_pct / stats['percentage']
            print(f"      📈 Value Concentration: {value_concentration:.2f}x")
        
        print("="*70)

    def save_clustering_results(self, filename='data/kmeans_clustering_results.csv'):
        """Save clustering results to CSV file"""
        if self.rfm_df is None:
            print("❌ No clustering results to save")
            return False
        
        try:
            import os
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            self.rfm_df.to_csv(filename, index=False)
            print(f"✅ Clustering results saved to {filename}")
            print(f"   📄 {len(self.rfm_df)} customer records with cluster assignments")
            return True
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

    def run_complete_clustering_analysis(self, features=['recency', 'frequency', 'monetary'], max_k=10):
        """Run complete clustering analysis pipeline"""
        print("\n🚀 STARTING COMPLETE K-MEANS CLUSTERING ANALYSIS")
        print("="*70)
        
        # Load data
        if not self.load_rfm_data():
            return False
        
        # Prepare data
        if self.prepare_clustering_data(features) is None:
            return False
        
        # Find optimal clusters
        if self.find_optimal_clusters(max_k, plot_results=True) is None:
            return False
        
        # Perform clustering
        optimal_k = self.cluster_analysis_results['optimal_k_silhouette']
        if self.perform_kmeans_clustering(optimal_k) is None:
            return False
        
        # Generate reports
        self.generate_cluster_summary_report()
        
        # Save results
        self.save_clustering_results()
        
        print("\n🎉 COMPLETE CLUSTERING ANALYSIS FINISHED!")
        print("="*70)
        return True

# Demonstration function
def demonstrate_kmeans_clustering():
    """Demonstrate comprehensive K-means clustering analysis"""
    print("🚀 STARTING K-MEANS CLUSTERING DEMONSTRATION")
    print("="*70)
    
    # Initialize clustering system
    clustering = ComprehensiveKMeansClustering()
    
    # Run complete analysis
    success = clustering.run_complete_clustering_analysis()
    
    if success:
        print("✅ K-means clustering demonstration completed successfully!")
        return clustering
    else:
        print("❌ K-means clustering demonstration failed")
        return None

if __name__ == "__main__":
    # Run demonstration
    clustering_system = demonstrate_kmeans_clustering() 