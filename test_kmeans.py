#!/usr/bin/env python3
"""
Test script for K-means clustering functionality in customer segmentation.

This script tests the comprehensive K-means clustering implementation
with proper validation and business interpretation.
"""

import sys
import os
from customer_segmentation import CustomerSegmentation

def test_kmeans_clustering():
    """Test the K-means clustering functionality"""
    print("🚀 TESTING K-MEANS CLUSTERING FUNCTIONALITY")
    print("="*60)
    
    try:
        # Initialize customer segmentation system
        print("🎯 Initializing Customer Segmentation System...")
        segmentation = CustomerSegmentation()
        
        # Load and calculate RFM data
        print("\n📊 Loading data and calculating RFM metrics...")
        if not segmentation.load_data():
            print("❌ Failed to load data")
            return False
        
        rfm_data = segmentation.calculate_rfm()
        if rfm_data is None:
            print("❌ Failed to calculate RFM")
            return False
        
        print(f"✅ RFM calculated for {len(rfm_data)} customers")
        
        # Test 1: Prepare clustering data
        print("\n" + "="*60)
        print("TEST 1: PREPARING CLUSTERING DATA")
        print("="*60)
        
        clustering_data = segmentation.prepare_clustering_data(['recency', 'frequency', 'monetary'])
        if clustering_data is None:
            print("❌ Failed to prepare clustering data")
            return False
        
        print(f"✅ Clustering data prepared: {clustering_data.shape}")
        
        # Test 2: Find optimal clusters
        print("\n" + "="*60)
        print("TEST 2: FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60)
        
        cluster_validation = segmentation.find_optimal_clusters(max_clusters=6)
        if cluster_validation is None:
            print("❌ Failed to find optimal clusters")
            return False
        
        optimal_k = cluster_validation['optimal_k_silhouette']
        best_score = cluster_validation['best_silhouette_score']
        print(f"✅ Optimal k found: {optimal_k} (score: {best_score:.3f})")
        
        # Test 3: Perform comprehensive clustering
        print("\n" + "="*60)
        print("TEST 3: PERFORMING COMPREHENSIVE K-MEANS CLUSTERING")
        print("="*60)
        
        cluster_labels, silhouette_score = segmentation.perform_comprehensive_kmeans_clustering()
        
        if cluster_labels is None:
            print("❌ Failed to perform clustering")
            return False
        
        print(f"✅ Clustering completed successfully!")
        print(f"📊 Final silhouette score: {silhouette_score:.3f}")
        print(f"📈 Number of unique clusters: {len(set(cluster_labels))}")
        
        # Test 4: Validate results
        print("\n" + "="*60)
        print("TEST 4: VALIDATING CLUSTERING RESULTS")
        print("="*60)
        
        # Check if cluster assignments are added to RFM data
        if 'cluster' not in segmentation.rfm_data.columns:
            print("❌ Cluster assignments not found in RFM data")
            return False
        
        if 'kmeans_cluster_name' not in segmentation.rfm_data.columns:
            print("❌ Cluster names not found in RFM data")
            return False
        
        if 'silhouette_score' not in segmentation.rfm_data.columns:
            print("❌ Silhouette scores not found in RFM data")
            return False
        
        print("✅ All required columns found in RFM data")
        
        # Show cluster distribution
        cluster_dist = segmentation.rfm_data['kmeans_cluster_name'].value_counts()
        print(f"\n📊 Cluster Distribution:")
        for cluster_name, count in cluster_dist.items():
            percentage = count / len(segmentation.rfm_data) * 100
            print(f"   {cluster_name}: {count} customers ({percentage:.1f}%)")
        
        # Test 5: Check generated files
        print("\n" + "="*60)
        print("TEST 5: CHECKING GENERATED FILES")
        print("="*60)
        
        expected_files = [
            'visualizations/kmeans_cluster_validation.png',
            f'visualizations/kmeans_silhouette_analysis_k{optimal_k}.png'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_path} ({file_size:,} bytes)")
            else:
                print(f"⚠️  {file_path} not found")
        
        # Test 6: Business insights validation
        print("\n" + "="*60)
        print("TEST 6: BUSINESS INSIGHTS VALIDATION")
        print("="*60)
        
        if hasattr(segmentation, 'kmeans_cluster_business_names'):
            print("✅ Business names assigned to clusters:")
            for cluster_id, info in segmentation.kmeans_cluster_business_names.items():
                print(f"   Cluster {cluster_id}: {info['name']}")
                print(f"      Description: {info['description']}")
        else:
            print("❌ Business names not found")
            return False
        
        if hasattr(segmentation, 'kmeans_cluster_stats_df'):
            print(f"\n📊 Cluster statistics available for {len(segmentation.kmeans_cluster_stats_df)} clusters")
            
            # Show revenue concentration
            total_revenue = segmentation.kmeans_cluster_stats_df['total_revenue'].sum()
            top_cluster = segmentation.kmeans_cluster_stats_df.sort_values('total_revenue', ascending=False).iloc[0]
            revenue_concentration = (top_cluster['total_revenue'] / total_revenue) * 100
            
            print(f"💰 Top revenue cluster contributes {revenue_concentration:.1f}% of total revenue")
            print(f"   with only {top_cluster['percentage']:.1f}% of customers")
        else:
            print("❌ Cluster statistics not found")
            return False
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ K-means clustering implementation is working correctly")
        print(f"📊 {len(segmentation.rfm_data)} customers successfully clustered")
        print(f"🎯 {len(set(cluster_labels))} meaningful business segments identified")
        print(f"📈 Silhouette score: {silhouette_score:.3f} (Quality: {'Excellent' if silhouette_score > 0.7 else 'Good' if silhouette_score > 0.5 else 'Acceptable' if silhouette_score > 0.2 else 'Poor'})")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_feature_test():
    """Quick test of specific K-means features"""
    print("🚀 QUICK FEATURE TEST")
    print("="*40)
    
    segmentation = CustomerSegmentation()
    
    # Test with minimal data
    if segmentation.load_data():
        segmentation.calculate_rfm()
        
        # Test different k values
        for k in [2, 3, 4]:
            print(f"\n🔧 Testing k={k}...")
            segmentation.prepare_clustering_data()
            cluster_labels, score = segmentation.perform_comprehensive_kmeans_clustering(n_clusters=k)
            if cluster_labels is not None:
                print(f"✅ k={k}: Score={score:.3f}, Clusters={len(set(cluster_labels))}")
            else:
                print(f"❌ k={k}: Failed")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_feature_test()
    else:
        test_kmeans_clustering() 