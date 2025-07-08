#!/usr/bin/env python3
"""
K-means Clustering Demonstration Script

This script demonstrates comprehensive K-means clustering analysis
for customer segmentation using RFM metrics.

Usage:
    python kmeans_demo.py              # Full analysis
    python kmeans_demo.py --quick      # Quick analysis (k=2-6)
    python kmeans_demo.py --k 5        # Specific k value
    python kmeans_demo.py --help       # Show help
"""

import argparse
import sys
import os
from kmeans_clustering import ComprehensiveKMeansClustering

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive K-means Clustering Analysis for Customer Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kmeans_demo.py                    # Full analysis (k=2-10)
  python kmeans_demo.py --quick            # Quick analysis (k=2-6)
  python kmeans_demo.py --k 5              # Cluster with specific k=5
  python kmeans_demo.py --features R F     # Use only Recency and Frequency
  python kmeans_demo.py --no-plots         # Skip visualization plots
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick analysis with fewer k values (2-6)'
    )
    
    parser.add_argument(
        '--k', 
        type=int, 
        metavar='N',
        help='Perform clustering with specific k value (skips optimization)'
    )
    
    parser.add_argument(
        '--max-k', 
        type=int, 
        default=10,
        metavar='N',
        help='Maximum k value for optimization (default: 10)'
    )
    
    parser.add_argument(
        '--features',
        nargs='+',
        choices=['recency', 'frequency', 'monetary', 'R', 'F', 'M'],
        help='Features to use for clustering (default: recency frequency monetary)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    parser.add_argument(
        '--input',
        default='data/customer_segmentation_results.csv',
        help='Input RFM data file (default: data/customer_segmentation_results.csv)'
    )
    
    parser.add_argument(
        '--output',
        default='data/kmeans_clustering_results.csv',
        help='Output clustering results file (default: data/kmeans_clustering_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Process feature mapping
    feature_map = {'R': 'recency', 'F': 'frequency', 'M': 'monetary'}
    if args.features:
        features = [feature_map.get(f, f) for f in args.features]
    else:
        features = ['recency', 'frequency', 'monetary']
    
    # Adjust max_k for quick analysis
    if args.quick:
        max_k = min(6, args.max_k)
    else:
        max_k = args.max_k
    
    print("🎯 K-MEANS CLUSTERING ANALYSIS")
    print("="*50)
    print(f"📊 Features: {features}")
    print(f"📈 Max k for optimization: {max_k}")
    print(f"📄 Input file: {args.input}")
    print(f"💾 Output file: {args.output}")
    print(f"📊 Create plots: {not args.no_plots}")
    
    if args.k:
        print(f"🎯 Direct clustering with k={args.k}")
    
    print("="*50)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        print("💡 Please run the RFM analysis first:")
        print("   python rfm_demo.py")
        return 1
    
    try:
        # Initialize clustering system
        clustering = ComprehensiveKMeansClustering()
        
        # Load RFM data
        if not clustering.load_rfm_data(args.input):
            return 1
        
        # Prepare clustering data
        if clustering.prepare_clustering_data(features) is None:
            return 1
        
        if args.k:
            # Direct clustering with specific k
            print(f"\n🎯 PERFORMING DIRECT CLUSTERING (k={args.k})")
            cluster_labels, silhouette_score = clustering.perform_kmeans_clustering(args.k)
            
            if cluster_labels is not None:
                clustering.generate_cluster_summary_report()
                
                # Create silhouette plot for this k
                if not args.no_plots:
                    clustering.plot_silhouette_analysis(args.k)
            
        else:
            # Find optimal clusters and perform analysis
            results = clustering.find_optimal_clusters(max_k, plot_results=not args.no_plots)
            
            if results is None:
                return 1
            
            # Perform clustering with optimal k
            optimal_k = results['optimal_k_silhouette']
            cluster_labels, silhouette_score = clustering.perform_kmeans_clustering(optimal_k)
            
            if cluster_labels is not None:
                clustering.generate_cluster_summary_report()
        
        # Save results
        if clustering.save_clustering_results(args.output):
            print(f"\n✅ Analysis completed successfully!")
            print(f"📄 Results saved to: {args.output}")
            
            if not args.no_plots:
                print(f"📊 Visualizations saved to: visualizations/")
                print(f"   • cluster_validation.png")
                print(f"   • silhouette_analysis_k*.png")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

def quick_demo():
    """Quick demonstration with minimal output"""
    print("🚀 QUICK K-MEANS CLUSTERING DEMO")
    print("="*40)
    
    clustering = ComprehensiveKMeansClustering()
    
    # Check if data exists
    if not os.path.exists('data/customer_segmentation_results.csv'):
        print("❌ RFM data not found. Running RFM analysis first...")
        # Import and run RFM analysis
        try:
            from customer_segmentation import CustomerSegmentation
            
            # Quick RFM analysis
            segmentation = CustomerSegmentation()
            segmentation.load_data()
            segmentation.calculate_rfm()
            segmentation.save_results()
            print("✅ RFM analysis completed")
        except Exception as e:
            print(f"❌ Failed to generate RFM data: {e}")
            return False
    
    # Run clustering analysis
    if clustering.load_rfm_data():
        if clustering.prepare_clustering_data() is not None:
            results = clustering.find_optimal_clusters(max_clusters=6, plot_results=False)
            if results:
                optimal_k = results['optimal_k_silhouette']
                clustering.perform_kmeans_clustering(optimal_k)
                clustering.save_clustering_results()
                print("✅ Quick clustering demo completed!")
                return True
    
    print("❌ Quick demo failed")
    return False

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show help and run quick demo
        print(__doc__)
        print("\n" + "="*50)
        print("No arguments provided. Running quick demo...")
        print("="*50)
        
        if quick_demo():
            print("\n💡 For more options, run: python kmeans_demo.py --help")
    else:
        sys.exit(main()) 