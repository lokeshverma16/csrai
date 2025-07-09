#!/usr/bin/env python3
"""
Radar Chart Demonstration Script

This script demonstrates the comprehensive radar chart functionality
for customer cluster visualization using both interactive and static charts.
"""

import sys
import os
import argparse
from customer_segmentation import CustomerSegmentation

def create_sample_data():
    """Generate sample data for demonstration"""
    print("🎯 Generating sample data for radar demonstration...")
    
    from data_generator import DataGenerator
    
    # Create data generator
    generator = DataGenerator()
    
    # Generate sample data
    customers, products, transactions = generator.generate_complete_dataset(
        num_customers=500,
        num_products=300,
        num_transactions=5000
    )
    
    print(f"✅ Generated {len(customers)} customers, {len(products)} products, {len(transactions)} transactions")
    return True

def demonstrate_radar_charts(interactive=True, static=True, comparison=True, report=True):
    """Comprehensive radar chart demonstration"""
    print("🚀 RADAR CHART DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Initialize system
        segmentation = CustomerSegmentation()
        
        # Load data and perform analysis
        print("\n📊 Setting up analysis pipeline...")
        if not segmentation.load_data():
            print("❌ Failed to load data. Generating sample data...")
            if not create_sample_data():
                print("❌ Failed to generate sample data")
                return False
            if not segmentation.load_data():
                print("❌ Still failed to load data")
                return False
        
        # Calculate RFM
        rfm_data = segmentation.calculate_rfm()
        if rfm_data is None:
            print("❌ Failed to calculate RFM")
            return False
        
        # Perform clustering
        print("\n🔧 Performing customer clustering...")
        segmentation.prepare_clustering_data(['recency', 'frequency', 'monetary'])
        validation_results = segmentation.find_optimal_clusters(max_clusters=5)
        
        if validation_results is None:
            print("❌ Failed to validate clusters")
            return False
        
        cluster_labels, silhouette_score = segmentation.perform_comprehensive_kmeans_clustering()
        
        if cluster_labels is None:
            print("❌ Failed to perform clustering")
            return False
        
        print(f"✅ Clustering completed with {len(set(cluster_labels))} clusters")
        print(f"📊 Silhouette Score: {silhouette_score:.3f}")
        
        # Create radar charts
        print("\n" + "=" * 60)
        print("🎨 CREATING RADAR VISUALIZATIONS")
        print("=" * 60)
        
        radar_data = segmentation.create_cluster_radar_charts(
            save_interactive=interactive,
            save_static=static
        )
        
        if radar_data is None:
            print("❌ Failed to create radar charts")
            return False
        
        # Display radar insights
        print("\n📊 RADAR CHART INSIGHTS:")
        print("-" * 50)
        
        total_customers = sum([c['size'] for c in radar_data])
        total_revenue = sum([c['total_revenue'] for c in radar_data])
        
        print(f"🎯 Analysis Overview:")
        print(f"   Clusters: {len(radar_data)}")
        print(f"   Customers: {total_customers:,}")
        print(f"   Revenue: ${total_revenue:,.2f}")
        
        print(f"\n📈 Cluster Radar Profiles:")
        for i, cluster in enumerate(sorted(radar_data, key=lambda x: x['total_revenue'], reverse=True)):
            print(f"\n   {i+1}. {cluster['cluster_name']}:")
            print(f"      👥 Size: {cluster['size']:,} customers ({cluster['percentage']:.1f}%)")
            print(f"      💰 Revenue: ${cluster['total_revenue']:,.2f}")
            print(f"      🎯 RFM Profile:")
            print(f"         📅 Recency: {cluster['recency_raw']:.0f} days (score: {cluster['recency_normalized']:.2f})")
            print(f"         🔄 Frequency: {cluster['frequency_raw']:.1f} purchases (score: {cluster['frequency_normalized']:.2f})")
            print(f"         💵 Monetary: ${cluster['monetary_raw']:.0f} (score: {cluster['monetary_normalized']:.2f})")
            
            # Calculate overall radar strength
            avg_score = (cluster['recency_normalized'] + cluster['frequency_normalized'] + cluster['monetary_normalized']) / 3
            
            if avg_score >= 0.8:
                strength = "🌟 Excellent"
            elif avg_score >= 0.6:
                strength = "💪 Strong"
            elif avg_score >= 0.4:
                strength = "⚖️ Moderate"
            elif avg_score >= 0.2:
                strength = "⚠️ Weak"
            else:
                strength = "🚨 Critical"
            
            print(f"      🎯 Overall Strength: {avg_score:.2f} ({strength})")
        
        # Business recommendations
        if hasattr(segmentation, 'radar_business_insights'):
            insights = segmentation.radar_business_insights
            
            print(f"\n💡 KEY BUSINESS INSIGHTS:")
            print("-" * 40)
            
            # Top value creators
            value_assessments = insights['value_assessment']
            high_value_clusters = [a for a in value_assessments if a['value_ratio'] > 1.5]
            
            if high_value_clusters:
                print("🏆 High-Value Segments:")
                for cluster in high_value_clusters:
                    print(f"   • {cluster['cluster']}: {cluster['value_ratio']:.1f}x revenue concentration")
            
            # Transition opportunities
            transitions = insights['transition_opportunities']
            if transitions:
                print(f"\n🔄 Growth Opportunities:")
                for transition in transitions[:2]:  # Show top 2
                    print(f"   • Move {transition['potential_customers']} customers from")
                    print(f"     {transition['from_cluster']} → {transition['to_cluster']}")
                    print(f"     Actions: {', '.join(transition['improvements'])}")
        
        # Create comparison matrix if requested
        if comparison:
            print(f"\n🔍 Creating cluster comparison matrix...")
            distance_matrix = segmentation.create_cluster_comparison_matrix(radar_data)
            
            if distance_matrix is not None:
                # Find most similar and different clusters
                import numpy as np
                
                # Mask diagonal
                masked_matrix = distance_matrix.copy()
                np.fill_diagonal(masked_matrix, np.inf)
                
                # Find most similar (minimum distance)
                min_idx = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
                min_distance = distance_matrix[min_idx]
                most_similar = (radar_data[min_idx[0]]['cluster_name'], radar_data[min_idx[1]]['cluster_name'])
                
                # Find most different (maximum distance)
                np.fill_diagonal(masked_matrix, -np.inf)
                max_idx = np.unravel_index(np.argmax(masked_matrix), masked_matrix.shape)
                max_distance = distance_matrix[max_idx]
                most_different = (radar_data[max_idx[0]]['cluster_name'], radar_data[max_idx[1]]['cluster_name'])
                
                print(f"✅ Cluster comparison completed:")
                print(f"   📊 Most Similar: {most_similar[0]} ↔ {most_similar[1]} (distance: {min_distance:.3f})")
                print(f"   📊 Most Different: {most_different[0]} ↔ {most_different[1]} (distance: {max_distance:.3f})")
        
        # Export comprehensive report if requested
        if report:
            print(f"\n📄 Exporting comprehensive analysis report...")
            if segmentation.export_radar_analysis_report():
                print("✅ Analysis report exported successfully")
            else:
                print("❌ Failed to export analysis report")
        
        # Summary of generated files
        print("\n" + "=" * 60)
        print("📁 GENERATED FILES")
        print("=" * 60)
        
        generated_files = []
        
        if static and os.path.exists('visualizations/static_cluster_radar_chart.png'):
            size = os.path.getsize('visualizations/static_cluster_radar_chart.png')
            generated_files.append(f"📊 Static Radar Chart: visualizations/static_cluster_radar_chart.png ({size:,} bytes)")
        
        if interactive and os.path.exists('visualizations/interactive_cluster_radar_chart.html'):
            size = os.path.getsize('visualizations/interactive_cluster_radar_chart.html')
            generated_files.append(f"🎨 Interactive Radar Chart: visualizations/interactive_cluster_radar_chart.html ({size:,} bytes)")
        
        if comparison and os.path.exists('visualizations/cluster_comparison_matrix.png'):
            size = os.path.getsize('visualizations/cluster_comparison_matrix.png')
            generated_files.append(f"🔍 Comparison Matrix: visualizations/cluster_comparison_matrix.png ({size:,} bytes)")
        
        if report and os.path.exists('reports/radar_analysis_report.md'):
            size = os.path.getsize('reports/radar_analysis_report.md')
            generated_files.append(f"📄 Analysis Report: reports/radar_analysis_report.md ({size:,} bytes)")
        
        for file_info in generated_files:
            print(f"✅ {file_info}")
        
        if not generated_files:
            print("⚠️  No files were generated")
        
        print("\n" + "=" * 60)
        print("🎉 RADAR CHART DEMONSTRATION COMPLETED!")
        print("=" * 60)
        print("✅ Radar charts successfully demonstrate:")
        print("   🎯 Cluster centroid visualization")
        print("   📊 Normalized RFM score comparison") 
        print("   🎨 Interactive and static chart options")
        print("   💡 Business intelligence insights")
        print("   🔍 Cluster similarity analysis")
        print("   📄 Comprehensive reporting")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_radar_features():
    """Show radar chart features and capabilities"""
    print("🎯 RADAR CHART FEATURES & CAPABILITIES")
    print("=" * 50)
    
    features = [
        "📊 Cluster Centroid Visualization",
        "   • Uses original (non-standardized) RFM values",
        "   • Calculates mean R, F, M for each cluster", 
        "   • Normalizes values for radar display (0-1 scale)",
        "",
        "🎨 Interactive Radar Chart (Plotly)",
        "   • Each cluster as different colored line/area",
        "   • R, F, M as three axes with proper scaling",
        "   • Hover information with detailed metrics",
        "   • Legend with cluster names and sizes",
        "   • Professional title and annotations",
        "",
        "📈 Static Radar Chart (Matplotlib)",
        "   • Same information in static format",
        "   • Professional styling for reports",
        "   • Embedded business insights",
        "   • High-resolution PNG output",
        "",
        "💡 Business Intelligence",
        "   • Text descriptions for each cluster",
        "   • Key characteristics highlighting",
        "   • Actionable recommendations",
        "   • Value concentration analysis",
        "",
        "🔍 Cluster Comparison Features",
        "   • Shows which clusters are most different",
        "   • Identifies transition opportunities",
        "   • Business value assessment",
        "   • Similarity matrix visualization",
        "",
        "📄 Export Options",
        "   • Interactive version as HTML",
        "   • Static version as PNG",
        "   • Comprehensive markdown report",
        "   • Integration with main visualization suite"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n" + "=" * 50)

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description="Radar Chart Demonstration for Customer Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 radar_demo.py                     # Full demonstration
  python3 radar_demo.py --features          # Show features only
  python3 radar_demo.py --static-only       # Create only static charts
  python3 radar_demo.py --no-comparison     # Skip comparison matrix
  python3 radar_demo.py --no-report         # Skip report generation
        """
    )
    
    parser.add_argument('--features', action='store_true',
                       help='Show radar chart features and capabilities')
    parser.add_argument('--static-only', action='store_true',
                       help='Create only static radar charts')
    parser.add_argument('--interactive-only', action='store_true', 
                       help='Create only interactive radar charts')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip cluster comparison matrix')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip analysis report generation')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new sample data before analysis')
    
    args = parser.parse_args()
    
    if args.features:
        show_radar_features()
        return
    
    # Generate sample data if requested
    if args.generate_data:
        create_sample_data()
    
    # Determine which charts to create
    interactive = not args.static_only
    static = not args.interactive_only
    comparison = not args.no_comparison
    report = not args.no_report
    
    # Run demonstration
    success = demonstrate_radar_charts(
        interactive=interactive,
        static=static,
        comparison=comparison,
        report=report
    )
    
    if success:
        print("\n✅ Demonstration completed successfully!")
    else:
        print("\n❌ Demonstration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 