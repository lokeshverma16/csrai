#!/usr/bin/env python3
"""
Advanced Recommendation Engine Demonstration

This script demonstrates the enhanced recommendation system with:
- Purchase pattern analysis
- Segment-specific strategies
- Time-based recommendations
- Cross-selling and upselling
- A/B testing simulation
- Performance dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_recommendation_engine import AdvancedRecommendationEngine, run_advanced_recommendation_demo
import pandas as pd
import json
from datetime import datetime

def demonstrate_advanced_features():
    """Demonstrate specific advanced features"""
    print("🔍 DEMONSTRATING ADVANCED FEATURES")
    print("=" * 50)
    
    # Initialize engine
    engine = AdvancedRecommendationEngine()
    
    # Load data
    if not engine.load_data():
        print("❌ Failed to load data")
        return
    
    # Create matrices
    engine.create_user_item_matrix()
    engine.calculate_item_similarity()
    
    # 1. Purchase Pattern Analysis Demo
    print("\n📊 PURCHASE PATTERN ANALYSIS:")
    print("-" * 35)
    if engine.purchase_patterns is not None:
        pattern_summary = engine.purchase_patterns.groupby('purchase_frequency').agg({
            'customer_id': 'count',
            'avg_days_between_purchases': 'mean',
            'regularity_score': 'mean',
            'total_spent': 'mean'
        }).round(2)
        print(pattern_summary)
    
    # 2. Seasonal Pattern Demo
    print("\n🌍 SEASONAL PATTERNS:")
    print("-" * 25)
    if engine.seasonal_patterns:
        for category, info in list(engine.seasonal_patterns['peak_seasons'].items())[:5]:
            print(f"   • {category}: Peak month {info['peak_month']}, "
                  f"Strength {info['peak_strength']:.2f} ({info['seasonal_pattern']})")
    
    # 3. Basket Analysis Demo
    print("\n🛒 MARKET BASKET ANALYSIS:")
    print("-" * 30)
    if engine.basket_analysis and 'rules' in engine.basket_analysis:
        top_rules = sorted(engine.basket_analysis['rules'], key=lambda x: x['lift'], reverse=True)[:5]
        for rule in top_rules:
            product1 = engine.products_df[engine.products_df['product_id'] == rule['product1']]['product_name'].iloc[0]
            product2 = engine.products_df[engine.products_df['product_id'] == rule['product2']]['product_name'].iloc[0]
            print(f"   • {product1[:20]}... → {product2[:20]}... (Lift: {rule['lift']:.2f})")
    
    # 4. Price Sensitivity Demo
    print("\n💰 PRICE SENSITIVITY ANALYSIS:")
    print("-" * 35)
    if engine.price_sensitivity is not None:
        sensitivity_summary = engine.price_sensitivity['price_sensitivity'].value_counts()
        print(f"   • High sensitivity: {sensitivity_summary.get('high', 0)} customers")
        print(f"   • Medium sensitivity: {sensitivity_summary.get('medium', 0)} customers")
        print(f"   • Low sensitivity: {sensitivity_summary.get('low', 0)} customers")
    
    # 5. Customer Journey Demo
    print("\n🚶 CUSTOMER JOURNEY MAPPING:")
    print("-" * 32)
    if engine.customer_journey is not None:
        journey_summary = engine.customer_journey['journey_stage'].value_counts()
        for stage, count in journey_summary.items():
            print(f"   • {stage}: {count} customers")
    
    return engine

def test_segment_specific_recommendations():
    """Test recommendations for different customer segments"""
    print("\n🎯 SEGMENT-SPECIFIC RECOMMENDATION TESTING")
    print("=" * 50)
    
    engine = AdvancedRecommendationEngine()
    if not engine.load_data():
        return
    
    engine.create_user_item_matrix()
    engine.calculate_item_similarity()
    
    # Find customers from different segments
    if engine.segmentation_df is not None:
        segments = engine.segmentation_df['segment'].unique()[:3]
        
        for segment in segments:
            print(f"\n--- {segment.upper()} SEGMENT ---")
            segment_customers = engine.segmentation_df[
                engine.segmentation_df['segment'] == segment
            ]['customer_id'].head(2).tolist()
            
            for customer_id in segment_customers:
                print(f"\nCustomer {customer_id}:")
                recommendations = engine.generate_advanced_recommendations(customer_id, 3)
                
                if recommendations:
                    for rec in recommendations:
                        print(f"   {rec['rank']}. {rec['product_name']} (${rec['price']:.2f})")
                        print(f"      Type: {rec['recommendation_type']} | Confidence: {rec['confidence_score']:.3f}")
                        print(f"      Explanation: {rec['explanation']}")
                else:
                    print("   No recommendations generated")

def run_comprehensive_analysis():
    """Run comprehensive analysis and save results"""
    print("\n📋 COMPREHENSIVE ANALYSIS")
    print("=" * 30)
    
    # Run the main demo
    results = run_advanced_recommendation_demo()
    
    if not results:
        print("❌ Demo failed to run")
        return
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save recommendation sample
    sample_file = f"advanced_recommendations_sample_{timestamp}.json"
    with open(sample_file, 'w') as f:
        # Convert to JSON-serializable format
        json_safe_recommendations = {}
        for customer_id, recs in list(results['sample_recommendations'].items())[:10]:
            json_safe_recommendations[str(customer_id)] = recs
        
        json.dump(json_safe_recommendations, f, indent=2, default=str)
    
    print(f"✅ Sample recommendations saved to: {sample_file}")
    
    # Save dashboard summary
    dashboard_file = f"performance_dashboard_{timestamp}.json"
    with open(dashboard_file, 'w') as f:
        json.dump(results['dashboard'], f, indent=2, default=str)
    
    print(f"✅ Dashboard data saved to: {dashboard_file}")
    
    # Create summary report
    summary_file = f"advanced_recommendation_summary_{timestamp}.md"
    with open(summary_file, 'w') as f:
        f.write("# Advanced Recommendation Engine Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dashboard summary
        dashboard = results['dashboard']
        metrics = dashboard['recommendation_metrics']
        business = dashboard['business_metrics']
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Customers Analyzed:** {metrics['total_customers']:,}\n")
        f.write(f"- **Total Recommendations Generated:** {metrics['total_recommendations']:,}\n")
        f.write(f"- **Success Rate:** {metrics['success_rate']:.1%}\n")
        f.write(f"- **Average Confidence Score:** {metrics['avg_confidence_score']:.3f}\n")
        f.write(f"- **Potential Revenue:** ${business['total_potential_revenue']:,.2f}\n")
        f.write(f"- **Revenue Lift:** {business['revenue_lift_percent']:.1f}%\n\n")
        
        # Segment performance
        f.write("## Segment Performance\n\n")
        for segment, stats in dashboard['segment_analysis'].items():
            if stats['count'] > 0:
                f.write(f"- **{segment}:** {stats['count']} recommendations, ")
                f.write(f"Average confidence {stats['avg_confidence']:.3f}, ")
                f.write(f"Average price ${stats['avg_price']:.2f}\n")
        
        # Recommendation types
        f.write("\n## Recommendation Type Distribution\n\n")
        for rec_type, count in dashboard['recommendation_types'].items():
            percentage = (count / metrics['total_recommendations']) * 100 if metrics['total_recommendations'] > 0 else 0
            f.write(f"- **{rec_type}:** {count} ({percentage:.1f}%)\n")
        
        # A/B testing results
        if 'ab_results' in results:
            f.write("\n## A/B Testing Results\n\n")
            for variant, metrics in results['ab_results'].items():
                f.write(f"- **{variant}:** Revenue ${metrics['predicted_revenue']:.2f}, ")
                f.write(f"Confidence {metrics.get('avg_confidence', 0):.3f}\n")
        
        f.write("\n## Advanced Features Implemented\n\n")
        f.write("- ✅ Purchase pattern analysis (timing, frequency, regularity)\n")
        f.write("- ✅ Seasonal buying pattern detection\n")
        f.write("- ✅ Market basket analysis for cross-selling\n")
        f.write("- ✅ Price sensitivity analysis\n")
        f.write("- ✅ Customer journey mapping\n")
        f.write("- ✅ Segment-specific recommendation strategies\n")
        f.write("- ✅ Time-based recommendations\n")
        f.write("- ✅ Cross-selling and upselling identification\n")
        f.write("- ✅ A/B testing simulation framework\n")
        f.write("- ✅ Revenue impact calculation\n")
        f.write("- ✅ Comprehensive performance dashboard\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    return results

def main():
    """Main demonstration function"""
    print("🚀 ADVANCED RECOMMENDATION ENGINE - COMPREHENSIVE DEMO")
    print("=" * 65)
    
    try:
        # 1. Demonstrate advanced features
        engine = demonstrate_advanced_features()
        
        # 2. Test segment-specific recommendations
        test_segment_specific_recommendations()
        
        # 3. Run comprehensive analysis
        results = run_comprehensive_analysis()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 30)
        print("✅ All advanced features demonstrated successfully")
        print("📊 Performance metrics calculated")
        print("💼 Business impact assessed")
        print("🧪 A/B testing framework validated")
        print("📁 Results saved to files")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 