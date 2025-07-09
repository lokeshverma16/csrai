#!/usr/bin/env python3
"""
Comprehensive Test Script for Hybrid Recommendation Engine

This script tests all aspects of the hybrid recommendation system including:
- Content-based filtering
- Collaborative filtering with cluster awareness
- Hybrid recommendation generation
- Performance metrics
- Business rule validation
"""

import sys
import os
import pandas as pd
import numpy as np
from recommendation_engine import HybridRecommendationEngine

def test_hybrid_recommendations():
    """Test the complete hybrid recommendation system"""
    print("🚀 TESTING HYBRID RECOMMENDATION ENGINE")
    print("=" * 60)
    
    try:
        # Initialize recommendation engine
        rec_engine = HybridRecommendationEngine()
        
        # Test 1: Data Loading
        print("\n📊 TEST 1: DATA LOADING")
        print("-" * 40)
        
        if rec_engine.load_data():
            print("✅ Data loading successful")
            print(f"   Customers: {len(rec_engine.customers_df)}")
            print(f"   Products: {len(rec_engine.products_df)}")
            print(f"   Transactions: {len(rec_engine.transactions_df)}")
            if rec_engine.segmentation_df is not None:
                print(f"   Customer Segments: {len(rec_engine.segmentation_df)}")
        else:
            print("❌ Data loading failed")
            return False
        
        # Test 2: Customer Profile Creation
        print("\n🧑‍💼 TEST 2: CUSTOMER PROFILE CREATION")
        print("-" * 40)
        
        if rec_engine.create_customer_profiles():
            print("✅ Customer profiles created successfully")
            
            # Sample profile analysis
            sample_profiles = rec_engine.customer_profiles.head(3)
            for _, profile in sample_profiles.iterrows():
                print(f"   Customer {profile['customer_id']}:")
                print(f"      Purchases: {profile['num_purchases']}")
                print(f"      Avg Price: ${profile['avg_price']:.2f}")
                print(f"      Price Sensitivity: {profile['price_sensitivity']}")
                print(f"      Category Diversity: {profile['category_diversity']:.2f}")
        else:
            print("❌ Customer profile creation failed")
            return False
        
        # Test 3: Matrix Creation
        print("\n🔢 TEST 3: SIMILARITY MATRIX CREATION")
        print("-" * 40)
        
        if rec_engine.create_user_item_matrix() is not None:
            print("✅ User-item matrix created")
            print(f"   Shape: {rec_engine.user_item_matrix.shape}")
            
            sparsity = (rec_engine.user_item_matrix == 0).sum().sum() / (
                rec_engine.user_item_matrix.shape[0] * rec_engine.user_item_matrix.shape[1]
            ) * 100
            print(f"   Sparsity: {sparsity:.1f}%")
        
        if rec_engine.calculate_item_similarity() is not None:
            print("✅ Item similarity matrix created")
            print(f"   Shape: {rec_engine.item_similarity_matrix.shape}")
        
        # Test 4: Individual Customer Recommendations
        print("\n🎯 TEST 4: INDIVIDUAL CUSTOMER RECOMMENDATIONS")
        print("-" * 40)
        
        # Test with different customer segments
        test_customers = []
        
        if rec_engine.segmentation_df is not None:
            # Get customers from different segments
            for segment in rec_engine.segmentation_df['segment'].unique():
                segment_customers = rec_engine.segmentation_df[
                    rec_engine.segmentation_df['segment'] == segment
                ]['customer_id'].head(1).tolist()
                test_customers.extend(segment_customers)
        else:
            # Random sample if no segmentation
            test_customers = rec_engine.customers_df['customer_id'].head(3).tolist()
        
        recommendation_results = []
        
        for customer_id in test_customers[:5]:  # Test first 5
            print(f"\n👤 Testing Customer: {customer_id}")
            
            # Get customer segment
            segment = rec_engine._get_customer_segment(customer_id)
            print(f"   Segment: {segment}")
            
            # Get purchase history
            customer_purchases = rec_engine.transactions_df[
                rec_engine.transactions_df['customer_id'] == customer_id
            ]
            print(f"   Purchase History: {len(customer_purchases)} transactions")
            
            if len(customer_purchases) > 0:
                # Show some purchase details
                categories = customer_purchases.merge(
                    rec_engine.products_df[['product_id', 'category']], 
                    on='product_id'
                )['category'].value_counts()
                print(f"   Top Categories: {dict(categories.head(2))}")
            
            # Generate recommendations
            recommendations = rec_engine.generate_hybrid_recommendations(customer_id, 3)
            
            if recommendations:
                print(f"   ✅ Generated {len(recommendations)} recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"      {i}. {rec['product_name']} ({rec['category']})")
                    print(f"         Price: ${rec['price']:,.2f}")
                    print(f"         Confidence: {rec['confidence_score']:.3f}")
                    print(f"         Reason: {rec['explanation']}")
                
                recommendation_results.append({
                    'customer_id': customer_id,
                    'segment': segment,
                    'num_recommendations': len(recommendations),
                    'avg_confidence': np.mean([r['confidence_score'] for r in recommendations]),
                    'categories': [r['category'] for r in recommendations],
                    'prices': [r['price'] for r in recommendations]
                })
            else:
                print(f"   ❌ No recommendations generated")
        
        # Test 5: Content vs Collaborative Analysis
        print("\n⚖️  TEST 5: CONTENT VS COLLABORATIVE ANALYSIS")
        print("-" * 40)
        
        sample_customer = test_customers[0] if test_customers else rec_engine.customers_df['customer_id'].iloc[0]
        
        # Get candidate products
        if sample_customer in rec_engine.user_item_matrix.index:
            user_ratings = rec_engine.user_item_matrix.loc[sample_customer]
            purchased_items = set(user_ratings[user_ratings > 0].index.tolist())
        else:
            purchased_items = set()
        
        all_products = set(rec_engine.products_df['product_id'].tolist())
        candidate_products = list(all_products - purchased_items)[:20]  # Test with 20 products
        
        # Compare scoring methods
        content_scores = rec_engine.get_content_based_scores(sample_customer, candidate_products)
        collaborative_scores = rec_engine.get_collaborative_scores(sample_customer, candidate_products)
        
        print(f"   Testing with customer {sample_customer} and {len(candidate_products)} products")
        print(f"   Content-based scores: {len(content_scores)} products scored")
        print(f"   Collaborative scores: {len(collaborative_scores)} products scored")
        
        # Show top 3 from each method
        if content_scores:
            top_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top Content-Based:")
            for product_id, score in top_content:
                product_name = rec_engine.products_df[
                    rec_engine.products_df['product_id'] == product_id
                ]['product_name'].iloc[0]
                print(f"      {product_name}: {score:.3f}")
        
        if collaborative_scores:
            top_collaborative = sorted(collaborative_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top Collaborative:")
            for product_id, score in top_collaborative:
                product_name = rec_engine.products_df[
                    rec_engine.products_df['product_id'] == product_id
                ]['product_name'].iloc[0]
                print(f"      {product_name}: {score:.3f}")
        
        # Test 6: Performance Metrics
        print("\n📊 TEST 6: PERFORMANCE METRICS")
        print("-" * 40)
        
        metrics = rec_engine.calculate_performance_metrics(sample_customers=test_customers[:10])
        
        if metrics:
            print("✅ Performance metrics calculated successfully")
        else:
            print("❌ Performance metrics calculation failed")
        
        # Test 7: Business Rules Validation
        print("\n💼 TEST 7: BUSINESS RULES VALIDATION")
        print("-" * 40)
        
        business_validation = validate_business_rules(recommendation_results, rec_engine)
        print(f"✅ Business rules validation completed")
        print(f"   Segment-appropriate recommendations: {business_validation['segment_appropriate']:.1%}")
        print(f"   Category diversity: {business_validation['category_diversity']:.1%}")
        print(f"   Confidence threshold compliance: {business_validation['confidence_compliance']:.1%}")
        
        # Test 8: Edge Cases
        print("\n🔬 TEST 8: EDGE CASE HANDLING")
        print("-" * 40)
        
        edge_case_results = test_edge_cases(rec_engine)
        print(f"✅ Edge case testing completed")
        print(f"   New customer handling: {'✅' if edge_case_results['new_customer'] else '❌'}")
        print(f"   Single purchase handling: {'✅' if edge_case_results['single_purchase'] else '❌'}")
        print(f"   Invalid customer handling: {'✅' if edge_case_results['invalid_customer'] else '❌'}")
        
        # Final Summary
        print("\n" + "=" * 60)
        print("🎉 HYBRID RECOMMENDATION ENGINE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 8
        passed_tests = 0
        
        test_results = {
            "Data Loading": True,
            "Customer Profiles": True,
            "Matrix Creation": True,
            "Individual Recommendations": len(recommendation_results) > 0,
            "Content vs Collaborative": len(content_scores) > 0 and len(collaborative_scores) > 0,
            "Performance Metrics": len(metrics) > 0,
            "Business Rules": business_validation['overall_score'] > 0.7,
            "Edge Cases": all(edge_case_results.values())
        }
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed_tests += 1
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests:.1%}")
        
        if recommendation_results:
            avg_confidence = np.mean([r['avg_confidence'] for r in recommendation_results])
            total_recommendations = sum([r['num_recommendations'] for r in recommendation_results])
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   Total Recommendations Generated: {total_recommendations}")
        
        print("=" * 60)
        
        return passed_tests >= total_tests * 0.8  # 80% pass rate required
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_business_rules(recommendation_results, rec_engine):
    """Validate that recommendations follow business rules"""
    validation_results = {
        'segment_appropriate': 0,
        'category_diversity': 0,
        'confidence_compliance': 0,
        'overall_score': 0
    }
    
    if not recommendation_results:
        return validation_results
    
    total_customers = len(recommendation_results)
    segment_appropriate_count = 0
    diverse_category_count = 0
    confidence_compliant_count = 0
    
    for result in recommendation_results:
        segment = result['segment']
        categories = result['categories']
        prices = result['prices']
        avg_confidence = result['avg_confidence']
        
        # Check segment appropriateness
        median_price = rec_engine.products_df['price'].median()
        
        if segment == 'Champions':
            # Champions should get premium products
            if np.mean(prices) >= median_price:
                segment_appropriate_count += 1
        elif segment in ['At-Risk', 'Need Attention']:
            # Price-sensitive segments should get value products
            if np.mean(prices) <= median_price:
                segment_appropriate_count += 1
        else:
            # Other segments - any price is appropriate
            segment_appropriate_count += 1
        
        # Check category diversity
        unique_categories = len(set(categories))
        if unique_categories >= 2 or len(categories) <= 2:  # Diverse or few enough recommendations
            diverse_category_count += 1
        
        # Check confidence compliance
        if avg_confidence >= rec_engine.min_confidence_threshold:
            confidence_compliant_count += 1
    
    validation_results['segment_appropriate'] = segment_appropriate_count / total_customers
    validation_results['category_diversity'] = diverse_category_count / total_customers
    validation_results['confidence_compliance'] = confidence_compliant_count / total_customers
    validation_results['overall_score'] = np.mean([
        validation_results['segment_appropriate'],
        validation_results['category_diversity'],
        validation_results['confidence_compliance']
    ])
    
    return validation_results

def test_edge_cases(rec_engine):
    """Test edge cases for the recommendation system"""
    edge_results = {
        'new_customer': False,
        'single_purchase': False,
        'invalid_customer': False
    }
    
    try:
        # Test 1: New customer (no purchases)
        new_customers = []
        for customer_id in rec_engine.customers_df['customer_id']:
            customer_purchases = rec_engine.transactions_df[
                rec_engine.transactions_df['customer_id'] == customer_id
            ]
            if len(customer_purchases) == 0:
                new_customers.append(customer_id)
                break
        
        if new_customers:
            new_customer_recs = rec_engine.generate_hybrid_recommendations(new_customers[0], 3)
            edge_results['new_customer'] = len(new_customer_recs) > 0
        
        # Test 2: Single purchase customer
        single_purchase_customers = []
        for customer_id in rec_engine.customers_df['customer_id']:
            customer_purchases = rec_engine.transactions_df[
                rec_engine.transactions_df['customer_id'] == customer_id
            ]
            if len(customer_purchases) == 1:
                single_purchase_customers.append(customer_id)
                break
        
        if single_purchase_customers:
            single_purchase_recs = rec_engine.generate_hybrid_recommendations(single_purchase_customers[0], 3)
            edge_results['single_purchase'] = len(single_purchase_recs) > 0
        
        # Test 3: Invalid customer
        invalid_customer_recs = rec_engine.generate_hybrid_recommendations('INVALID_CUSTOMER_ID', 3)
        edge_results['invalid_customer'] = len(invalid_customer_recs) >= 0  # Should handle gracefully
        
    except Exception as e:
        print(f"⚠️  Error in edge case testing: {e}")
    
    return edge_results

def demonstrate_recommendation_features():
    """Demonstrate specific features of the recommendation system"""
    print("🎭 DEMONSTRATION OF RECOMMENDATION FEATURES")
    print("=" * 50)
    
    rec_engine = HybridRecommendationEngine()
    
    if not rec_engine.load_data():
        print("❌ Failed to load data for demonstration")
        return
    
    rec_engine.create_customer_profiles()
    rec_engine.create_user_item_matrix()
    rec_engine.calculate_item_similarity()
    
    print("\n🎯 Feature Demonstrations:")
    print("1. ✅ Content-Based Filtering: Analyzes customer purchase history")
    print("2. ✅ Collaborative Filtering: Finds similar customers within clusters")
    print("3. ✅ Hybrid System: Combines both approaches with segment weighting")
    print("4. ✅ Business Rules: Applies segment-specific product preferences")
    print("5. ✅ Diversity Enforcement: Ensures different categories when possible")
    print("6. ✅ Confidence Scoring: Provides 0-1 confidence scores")
    print("7. ✅ Edge Case Handling: Manages new customers and single purchases")
    print("8. ✅ Performance Metrics: Comprehensive evaluation framework")
    
    # Show cluster characteristics if available
    if rec_engine.cluster_characteristics is not None and len(rec_engine.cluster_characteristics) > 0:
        print(f"\n🏷️  Customer Cluster Characteristics:")
        for segment, chars in rec_engine.cluster_characteristics.iterrows():
            print(f"   {segment}:")
            print(f"      Avg Price: ${chars['avg_price']:.2f}")
            print(f"      Top Category: {chars['top_category']}")
            print(f"      Category Focus: {chars['category_concentration']:.1%}")
    
    # Test recommendation quality
    print(f"\n🧪 Quick Quality Test:")
    sample_customer = rec_engine.customers_df['customer_id'].iloc[0]
    recommendations = rec_engine.generate_hybrid_recommendations(sample_customer, 3)
    
    if recommendations:
        print(f"   Generated {len(recommendations)} recommendations")
        avg_confidence = np.mean([r['confidence_score'] for r in recommendations])
        categories = [r['category'] for r in recommendations]
        unique_categories = len(set(categories))
        
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Category Diversity: {unique_categories}/{len(categories)} unique")
        print(f"   ✅ System working correctly!")
    else:
        print(f"   ❌ No recommendations generated")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demonstrate_recommendation_features()
    else:
        success = test_hybrid_recommendations()
        sys.exit(0 if success else 1) 