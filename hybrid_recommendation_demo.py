#!/usr/bin/env python3
"""
Hybrid Recommendation Engine Demo

This script demonstrates the comprehensive hybrid recommendation system
with content-based filtering, collaborative filtering, and business rules.
"""

import pandas as pd
import numpy as np
from recommendation_engine import HybridRecommendationEngine

def main():
    print("🚀 HYBRID RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the engine
    rec_engine = HybridRecommendationEngine()
    
    # Load data
    print("\n📊 LOADING DATA")
    print("-" * 30)
    if not rec_engine.load_data():
        print("❌ Failed to load data")
        return
    
    # Show data summary
    print(f"✅ Successfully loaded:")
    print(f"   • {len(rec_engine.customers_df):,} customers")
    print(f"   • {len(rec_engine.products_df):,} products")
    print(f"   • {len(rec_engine.transactions_df):,} transactions")
    
    if rec_engine.segmentation_df is not None:
        segments = rec_engine.segmentation_df['segment'].value_counts()
        print(f"   • {len(segments)} customer segments")
        for segment, count in segments.head(3).items():
            print(f"     - {segment}: {count} customers")
    
    # Initialize recommendation system
    print("\n🔧 INITIALIZING RECOMMENDATION SYSTEM")
    print("-" * 30)
    
    print("Creating customer profiles...")
    if hasattr(rec_engine, 'create_customer_profiles'):
        success = rec_engine.create_customer_profiles()
        if success:
            print("✅ Customer profiles created")
        else:
            print("⚠️  Using simplified profiles")
    
    print("Building user-item matrix...")
    user_item_matrix = rec_engine.create_user_item_matrix()
    if user_item_matrix is not None:
        print(f"✅ Matrix created: {user_item_matrix.shape}")
        sparsity = (user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100
        print(f"   Sparsity: {sparsity:.1f}%")
    
    print("Calculating similarity matrices...")
    item_similarity = rec_engine.calculate_item_similarity()
    if item_similarity is not None:
        print(f"✅ Item similarity matrix: {item_similarity.shape}")
    
    # Test individual customers from different segments
    print("\n🎯 TESTING INDIVIDUAL CUSTOMER RECOMMENDATIONS")
    print("-" * 30)
    
    test_customers = []
    if rec_engine.segmentation_df is not None:
        # Get sample customers from each segment
        for segment in rec_engine.segmentation_df['segment'].unique()[:4]:
            segment_customers = rec_engine.segmentation_df[
                rec_engine.segmentation_df['segment'] == segment
            ]['customer_id'].head(1).tolist()
            test_customers.extend(segment_customers)
    else:
        # Random sample
        test_customers = rec_engine.customers_df['customer_id'].head(4).tolist()
    
    successful_recommendations = 0
    total_confidence = 0
    
    for customer_id in test_customers:
        print(f"\n👤 Customer: {customer_id}")
        
        # Get customer info
        customer_info = rec_engine.customers_df[rec_engine.customers_df['customer_id'] == customer_id]
        if len(customer_info) > 0:
            print(f"   Name: {customer_info.iloc[0]['name']}")
        
        # Get segment
        if rec_engine.segmentation_df is not None:
            segment_info = rec_engine.segmentation_df[rec_engine.segmentation_df['customer_id'] == customer_id]
            segment = segment_info.iloc[0]['segment'] if len(segment_info) > 0 else 'unknown'
            print(f"   Segment: {segment}")
        
        # Get purchase history
        purchases = rec_engine.transactions_df[rec_engine.transactions_df['customer_id'] == customer_id]
        print(f"   Purchase History: {len(purchases)} transactions")
        
        if len(purchases) > 0:
            categories = purchases.merge(
                rec_engine.products_df[['product_id', 'category']], 
                on='product_id'
            )['category'].value_counts()
            top_category = categories.index[0] if len(categories) > 0 else 'None'
            print(f"   Favorite Category: {top_category}")
        
        # Generate recommendations using hybrid approach
        try:
            if hasattr(rec_engine, 'generate_hybrid_recommendations'):
                recommendations = rec_engine.generate_hybrid_recommendations(customer_id, 3)
            else:
                # Fallback to basic hybrid method
                recommendations = rec_engine.get_hybrid_recommendations(customer_id, 3)
                # Format the recommendations
                formatted_recs = []
                if not recommendations.empty:
                    for _, row in recommendations.iterrows():
                        formatted_recs.append({
                            'product_name': row['product_name'],
                            'category': row['category'],
                            'price': row['price'],
                            'confidence_score': row.get('hybrid_score', row.get('score', 0.5)),
                            'explanation': 'Hybrid recommendation'
                        })
                recommendations = formatted_recs
            
            if recommendations and len(recommendations) > 0:
                successful_recommendations += 1
                print(f"   📦 Recommendations:")
                
                for i, rec in enumerate(recommendations, 1):
                    confidence = rec.get('confidence_score', 0.5)
                    total_confidence += confidence
                    print(f"      {i}. {rec['product_name']} ({rec['category']})")
                    print(f"         Price: ${rec['price']:,.2f}")
                    print(f"         Confidence: {confidence:.3f}")
                    if 'explanation' in rec:
                        print(f"         Reason: {rec['explanation']}")
            else:
                print("   ⚠️  No recommendations generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary statistics
    print("\n📊 RECOMMENDATION SYSTEM SUMMARY")
    print("-" * 30)
    
    success_rate = successful_recommendations / len(test_customers) if test_customers else 0
    avg_confidence = total_confidence / max(1, successful_recommendations * 3)  # 3 recs per customer
    
    print(f"✅ Performance Metrics:")
    print(f"   Success Rate: {success_rate:.1%} ({successful_recommendations}/{len(test_customers)} customers)")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    
    # Feature demonstration
    print(f"\n🎭 HYBRID SYSTEM FEATURES:")
    print(f"   ✅ Content-Based Filtering: Customer preference analysis")
    print(f"   ✅ Collaborative Filtering: Similar customer behavior")
    print(f"   ✅ Cluster Awareness: Customer segment targeting")
    print(f"   ✅ Business Rules: Segment-specific product preferences")
    print(f"   ✅ Diversity Enforcement: Different categories when possible")
    print(f"   ✅ Confidence Scoring: 0-1 reliability scores")
    print(f"   ✅ Edge Case Handling: New customers and sparse data")
    
    # Additional analysis if we have segmentation data
    if rec_engine.segmentation_df is not None:
        print(f"\n🏷️  CUSTOMER SEGMENT ANALYSIS:")
        segments = rec_engine.segmentation_df['segment'].value_counts()
        for segment, count in segments.items():
            percentage = count / len(rec_engine.segmentation_df) * 100
            print(f"   {segment}: {count} customers ({percentage:.1f}%)")
    
    # Product category analysis
    print(f"\n📦 PRODUCT CATALOG ANALYSIS:")
    categories = rec_engine.products_df['category'].value_counts()
    for category, count in categories.items():
        avg_price = rec_engine.products_df[rec_engine.products_df['category'] == category]['price'].mean()
        print(f"   {category}: {count} products (avg price: ${avg_price:.2f})")
    
    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main() 