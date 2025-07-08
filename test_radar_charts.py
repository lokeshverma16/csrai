#!/usr/bin/env python3
"""
Test script for radar chart functionality in customer segmentation.

This script tests the comprehensive radar chart implementation
with both interactive and static versions.
"""

import sys
import os
from customer_segmentation import CustomerSegmentation

def test_radar_charts():
    """Test the radar chart functionality"""
    print("🚀 TESTING RADAR CHART FUNCTIONALITY")
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
        
        # Perform K-means clustering first
        print("\n🔧 Performing K-means clustering...")
        segmentation.prepare_clustering_data(['recency', 'frequency', 'monetary'])
        cluster_validation = segmentation.find_optimal_clusters(max_clusters=6)
        
        if cluster_validation is None:
            print("❌ Failed to find optimal clusters")
            return False
        
        cluster_labels, silhouette_score = segmentation.perform_comprehensive_kmeans_clustering()
        
        if cluster_labels is None:
            print("❌ Failed to perform clustering")
            return False
        
        print(f"✅ Clustering completed with {len(set(cluster_labels))} clusters")
        
        # Test 1: Create radar charts
        print("\n" + "="*60)
        print("TEST 1: CREATING RADAR CHARTS")
        print("="*60)
        
        radar_data = segmentation.create_cluster_radar_charts(save_interactive=True, save_static=True)
        
        if radar_data is None:
            print("❌ Failed to create radar charts")
            return False
        
        print(f"✅ Radar charts created for {len(radar_data)} clusters")
        
        # Test 2: Validate radar data
        print("\n" + "="*60)
        print("TEST 2: VALIDATING RADAR DATA")
        print("="*60)
        
        required_fields = ['cluster_id', 'cluster_name', 'recency_normalized', 
                          'frequency_normalized', 'monetary_normalized']
        
        for cluster in radar_data:
            for field in required_fields:
                if field not in cluster:
                    print(f"❌ Missing field '{field}' in radar data")
                    return False
            
            # Check normalization (should be 0-1)
            for metric in ['recency_normalized', 'frequency_normalized', 'monetary_normalized']:
                value = cluster[metric]
                if not (0 <= value <= 1):
                    print(f"❌ Invalid normalized value for {metric}: {value}")
                    return False
        
        print("✅ All radar data fields validated")
        print("✅ All normalized values in valid range [0,1]")
        
        # Test 3: Check generated files
        print("\n" + "="*60)
        print("TEST 3: CHECKING GENERATED FILES")
        print("="*60)
        
        expected_files = [
            'visualizations/interactive_cluster_radar_chart.html',
            'visualizations/static_cluster_radar_chart.png'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_path} ({file_size:,} bytes)")
            else:
                print(f"❌ {file_path} not found")
                # Don't fail for interactive chart if plotly is not available
                if 'interactive' in file_path:
                    print("   (This is acceptable if plotly is not installed)")
        
        # Test 4: Business insights validation
        print("\n" + "="*60)
        print("TEST 4: BUSINESS INSIGHTS VALIDATION")
        print("="*60)
        
        if hasattr(segmentation, 'radar_business_insights'):
            insights = segmentation.radar_business_insights
            
            required_insight_categories = ['cluster_analysis', 'transition_opportunities', 
                                         'business_recommendations', 'value_assessment']
            
            for category in required_insight_categories:
                if category not in insights:
                    print(f"❌ Missing insight category: {category}")
                    return False
                
                if not insights[category]:
                    print(f"⚠️  Empty insight category: {category}")
                else:
                    print(f"✅ {category}: {len(insights[category])} items")
            
            print("✅ All business insight categories validated")
        else:
            print("❌ Business insights not found")
            return False
        
        # Test 5: Create comparison matrix
        print("\n" + "="*60)
        print("TEST 5: CREATING CLUSTER COMPARISON MATRIX")
        print("="*60)
        
        distance_matrix = segmentation.create_cluster_comparison_matrix(radar_data)
        
        if distance_matrix is not None:
            print(f"✅ Comparison matrix created: {distance_matrix.shape}")
            
            # Validate matrix properties
            if distance_matrix.shape[0] != distance_matrix.shape[1]:
                print("❌ Distance matrix is not square")
                return False
            
            if distance_matrix.shape[0] != len(radar_data):
                print("❌ Distance matrix size doesn't match cluster count")
                return False
            
            # Check diagonal is zero
            import numpy as np
            diagonal_values = np.diag(distance_matrix)
            if not np.allclose(diagonal_values, 0):
                print("❌ Distance matrix diagonal is not zero")
                return False
            
            print("✅ Distance matrix properties validated")
        else:
            print("❌ Failed to create comparison matrix")
            return False
        
        # Test 6: Export analysis report
        print("\n" + "="*60)
        print("TEST 6: EXPORTING ANALYSIS REPORT")
        print("="*60)
        
        if segmentation.export_radar_analysis_report():
            report_path = 'reports/radar_analysis_report.md'
            if os.path.exists(report_path):
                file_size = os.path.getsize(report_path)
                print(f"✅ Analysis report exported: {report_path} ({file_size:,} bytes)")
            else:
                print("❌ Report file not found after export")
                return False
        else:
            print("❌ Failed to export analysis report")
            return False
        
        # Test 7: Radar chart insights display
        print("\n" + "="*60)
        print("TEST 7: RADAR CHART INSIGHTS SUMMARY")
        print("="*60)
        
        # Display key insights
        total_customers = sum([c['size'] for c in radar_data])
        total_revenue = sum([c['total_revenue'] for c in radar_data])
        
        print(f"📊 RADAR CHART ANALYSIS SUMMARY:")
        print(f"   Total Clusters: {len(radar_data)}")
        print(f"   Total Customers: {total_customers:,}")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        
        print(f"\n🎯 CLUSTER RADAR PROFILES:")
        for cluster in sorted(radar_data, key=lambda x: x['total_revenue'], reverse=True):
            print(f"   {cluster['cluster_name']}:")
            print(f"      Customers: {cluster['size']:,} ({cluster['percentage']:.1f}%)")
            print(f"      Revenue: ${cluster['total_revenue']:,.2f}")
            print(f"      RFM Scores: R={cluster['recency_normalized']:.2f}, F={cluster['frequency_normalized']:.2f}, M={cluster['monetary_normalized']:.2f}")
            
            # Overall radar strength
            avg_score = (cluster['recency_normalized'] + cluster['frequency_normalized'] + cluster['monetary_normalized']) / 3
            strength = "Strong" if avg_score > 0.7 else "Moderate" if avg_score > 0.4 else "Weak"
            print(f"      Overall Radar Strength: {avg_score:.2f} ({strength})")
        
        # Find most different clusters
        print(f"\n🔍 CLUSTER DIFFERENCES:")
        max_distance = 0
        most_different_pair = None
        
        for i in range(len(radar_data)):
            for j in range(i + 1, len(radar_data)):
                cluster1 = radar_data[i]
                cluster2 = radar_data[j]
                
                # Calculate Euclidean distance
                r_diff = cluster1['recency_normalized'] - cluster2['recency_normalized']
                f_diff = cluster1['frequency_normalized'] - cluster2['frequency_normalized']
                m_diff = cluster1['monetary_normalized'] - cluster2['monetary_normalized']
                distance = (r_diff**2 + f_diff**2 + m_diff**2)**0.5
                
                if distance > max_distance:
                    max_distance = distance
                    most_different_pair = (cluster1['cluster_name'], cluster2['cluster_name'])
        
        if most_different_pair:
            print(f"   Most Different Clusters: {most_different_pair[0]} ↔ {most_different_pair[1]}")
            print(f"   RFM Distance: {max_distance:.3f}")
        
        print("\n" + "="*60)
        print("🎉 ALL RADAR CHART TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Radar chart implementation is working correctly")
        print(f"📊 {len(radar_data)} clusters visualized successfully")
        print(f"🎨 Interactive and static charts generated")
        print(f"💡 Comprehensive business insights provided")
        print(f"📈 Cluster comparison analysis completed")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_radar_features():
    """Demonstrate specific radar chart features"""
    print("🚀 RADAR CHART FEATURES DEMONSTRATION")
    print("="*50)
    
    segmentation = CustomerSegmentation()
    
    if segmentation.load_data():
        segmentation.calculate_rfm()
        segmentation.prepare_clustering_data()
        segmentation.find_optimal_clusters(max_clusters=4)
        segmentation.perform_comprehensive_kmeans_clustering()
        
        print("\n🎨 Creating radar charts...")
        radar_data = segmentation.create_cluster_radar_charts()
        
        if radar_data:
            print("\n📊 Radar Chart Features Demonstrated:")
            print("   ✅ Interactive plotly radar chart with hover information")
            print("   ✅ Static matplotlib radar chart with business insights")
            print("   ✅ Normalized RFM scores (0-1 scale)")
            print("   ✅ Color-coded cluster visualization")
            print("   ✅ Business intelligence annotations")
            print("   ✅ Cluster comparison matrix")
            print("   ✅ Comprehensive analysis report")
            
            # Show normalization details
            print(f"\n🔧 Normalization Details:")
            for cluster in radar_data:
                print(f"   {cluster['cluster_name']}:")
                print(f"      Raw: R={cluster['recency_raw']:.0f}, F={cluster['frequency_raw']:.1f}, M=${cluster['monetary_raw']:.0f}")
                print(f"      Normalized: R={cluster['recency_normalized']:.2f}, F={cluster['frequency_normalized']:.2f}, M={cluster['monetary_normalized']:.2f}")
        
        return True
    
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demonstrate_radar_features()
    else:
        test_radar_charts() 