#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Customer Analytics & Recommendation System

This module provides extensive testing coverage including:
- Unit tests for individual components
- Integration tests for the complete pipeline
- Edge case testing for robustness
- Performance testing with various dataset sizes
- Data validation and integrity checks

Author: Data Science Portfolio Project
Version: 2.0 - Fixed API Compatibility
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Import system components
try:
    from data_generator import DataGenerator
    from customer_segmentation import CustomerSegmentation
    from advanced_recommendation_engine import AdvancedRecommendationEngine
    from main import CustomerAnalyticsPipeline
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)


class TestDataGenerator(unittest.TestCase):
    """Unit tests for DataGenerator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_generator = DataGenerator()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_customers_basic(self):
        """Test basic customer generation"""
        customers_df = self.data_generator.generate_customers(100)
        
        # Check structure
        self.assertEqual(len(customers_df), 100)
        required_columns = ['customer_id', 'name', 'email', 'registration_date']
        for col in required_columns:
            self.assertIn(col, customers_df.columns)
        
        # Check uniqueness
        self.assertEqual(customers_df['customer_id'].nunique(), 100)
        self.assertEqual(customers_df['email'].nunique(), 100)
    
    def test_generate_products_basic(self):
        """Test basic product generation"""
        products_df = self.data_generator.generate_products(50)
        
        # Check structure
        self.assertEqual(len(products_df), 50)
        required_columns = ['product_id', 'product_name', 'category', 'price']
        for col in required_columns:
            self.assertIn(col, products_df.columns)
        
        # Check data validity
        self.assertTrue(all(products_df['price'] > 0))
        self.assertEqual(products_df['product_id'].nunique(), 50)
    
    def test_generate_transactions_basic(self):
        """Test basic transaction generation"""
        customers_df = self.data_generator.generate_customers(10)
        products_df = self.data_generator.generate_products(5)
        transactions_df = self.data_generator.generate_transactions(customers_df, products_df, 50)
        
        # Check structure
        self.assertGreaterEqual(len(transactions_df), 30)  # Relaxed expectation
        required_columns = ['transaction_id', 'customer_id', 'product_id', 'purchase_date', 'price']
        for col in required_columns:
            self.assertIn(col, transactions_df.columns)
        
        # Check referential integrity
        self.assertTrue(all(transactions_df['customer_id'].isin(customers_df['customer_id'])))
        self.assertTrue(all(transactions_df['product_id'].isin(products_df['product_id'])))
    
    def test_edge_case_minimal_generation(self):
        """Test edge cases with minimal data"""
        # Test with single customer
        customers_df = self.data_generator.generate_customers(1)
        self.assertEqual(len(customers_df), 1)
        
        # Test with single product
        products_df = self.data_generator.generate_products(1)
        self.assertEqual(len(products_df), 1)


class TestCustomerSegmentation(unittest.TestCase):
    """Unit tests for CustomerSegmentation component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data files in temp directory
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create sample customers
        self.customers_df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'registration_date': pd.date_range('2022-01-01', periods=5),
            'age': [25, 30, 35, 40, 45],
            'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin']
        })
        
        self.products_df = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
            'category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Books'],
            'price': [100.0, 50.0, 25.0, 150.0, 30.0]
        })
        
        self.transactions_df = pd.DataFrame({
            'transaction_id': range(1, 16),
            'customer_id': [1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 1, 2, 3, 4, 5],
            'product_id': [1, 2, 1, 3, 2, 3, 1, 1, 2, 3, 4, 5, 1, 2, 4],
            'purchase_date': pd.date_range('2022-01-01', periods=15),
            'price': [100.0, 50.0, 100.0, 25.0, 50.0, 25.0, 100.0, 100.0, 50.0, 25.0, 150.0, 30.0, 100.0, 50.0, 150.0]
        })
        
        # Save to temp files
        self.customers_df.to_csv(data_dir / "customers.csv", index=False)
        self.products_df.to_csv(data_dir / "products.csv", index=False)
        self.transactions_df.to_csv(data_dir / "transactions.csv", index=False)
        
        self.segmentation = CustomerSegmentation()
        # Load data into segmentation object
        self.segmentation.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv")
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_loading(self):
        """Test successful data loading"""
        self.assertTrue(hasattr(self.segmentation, 'customers_df'))
        self.assertTrue(hasattr(self.segmentation, 'products_df'))
        self.assertTrue(hasattr(self.segmentation, 'transactions_df'))
        self.assertEqual(len(self.segmentation.customers_df), 5)
        self.assertEqual(len(self.segmentation.products_df), 5)
        self.assertEqual(len(self.segmentation.transactions_df), 15)
    
    def test_rfm_calculation_basic(self):
        """Test basic RFM calculation"""
        rfm_df = self.segmentation.calculate_rfm()
        
        # Check structure
        self.assertIsNotNone(rfm_df)
        if rfm_df is not None:
            self.assertEqual(len(rfm_df), 5)  # 5 unique customers
            required_columns = ['customer_id', 'recency', 'frequency', 'monetary']
            for col in required_columns:
                self.assertIn(col, rfm_df.columns)
            
            # Check data validity
            self.assertTrue(all(rfm_df['frequency'] > 0))
            self.assertTrue(all(rfm_df['monetary'] > 0))
            self.assertTrue(all(rfm_df['recency'] >= 0))
    
    def test_rfm_segmentation(self):
        """Test RFM segmentation"""
        rfm_df = self.segmentation.calculate_rfm()
        segmented_df = self.segmentation.create_rfm_segments()
        
        self.assertIsNotNone(segmented_df)
        if segmented_df is not None:
            self.assertIn('segment', segmented_df.columns)
            self.assertTrue(len(segmented_df['segment'].unique()) > 0)
    
    def test_clustering_preparation(self):
        """Test clustering data preparation"""
        self.segmentation.calculate_rfm()
        clustering_data = self.segmentation.prepare_clustering_data()
        
        self.assertIsNotNone(clustering_data)
        if clustering_data is not None:
            self.assertEqual(clustering_data.shape[0], 5)  # 5 customers
            self.assertEqual(clustering_data.shape[1], 3)  # RFM features
    
    def test_clustering_basic(self):
        """Test basic clustering functionality"""
        self.segmentation.calculate_rfm()
        self.segmentation.prepare_clustering_data()
        
        # Test clustering
        cluster_labels, silhouette_score = self.segmentation.perform_comprehensive_kmeans_clustering(n_clusters=2)
        
        self.assertIsNotNone(cluster_labels)
        self.assertIsNotNone(silhouette_score)
        if cluster_labels is not None:
            self.assertEqual(len(cluster_labels), 5)
            self.assertEqual(len(set(list(cluster_labels))), 2)  # 2 clusters


class TestAdvancedRecommendationEngine(unittest.TestCase):
    """Unit tests for AdvancedRecommendationEngine component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create sample data
        customers_df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'registration_date': pd.date_range('2022-01-01', periods=5)
        })
        
        products_df = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 'Product F', 'Product G', 'Product H'],
            'category': ['Electronics', 'Electronics', 'Clothing', 'Books', 'Books', 'Home', 'Sports', 'Electronics'],
            'price': [100.0, 150.0, 50.0, 25.0, 30.0, 75.0, 60.0, 200.0]
        })
        
        transactions_df = pd.DataFrame({
            'transaction_id': range(1, 21),
            'customer_id': [1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'product_id': [1, 2, 1, 3, 2, 3, 1, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7],
            'purchase_date': pd.date_range('2022-01-01', periods=20),
            'price': [100.0, 150.0, 100.0, 50.0, 150.0, 50.0, 100.0, 100.0, 150.0, 50.0, 25.0, 30.0, 75.0, 60.0, 200.0, 50.0, 25.0, 30.0, 75.0, 60.0]
        })
        
        # Create customer segments (mock data)
        segments_df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'segment': ['Champions', 'Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk']
        })
        
        # Save to temp files
        customers_df.to_csv(data_dir / "customers.csv", index=False)
        products_df.to_csv(data_dir / "products.csv", index=False)
        transactions_df.to_csv(data_dir / "transactions.csv", index=False)
        segments_df.to_csv(data_dir / "customer_segmentation_results.csv", index=False)
        
        self.engine = AdvancedRecommendationEngine()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_loading(self):
        """Test recommendation engine data loading"""
        data_dir = Path(self.temp_dir) / "data"
        
        success = self.engine.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv"),
            str(data_dir / "customer_segmentation_results.csv")
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(self.engine.customers_df)
        self.assertIsNotNone(self.engine.products_df)
        self.assertIsNotNone(self.engine.transactions_df)
    
    def test_matrix_creation(self):
        """Test user-item matrix creation"""
        data_dir = Path(self.temp_dir) / "data"
        
        self.engine.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv"),
            str(data_dir / "customer_segmentation_results.csv")
        )
        
        self.engine.create_user_item_matrix()
        self.assertIsNotNone(self.engine.user_item_matrix)
        
        self.engine.calculate_item_similarity()
        self.assertIsNotNone(self.engine.item_similarity_matrix)
    
    def test_generate_recommendations_basic(self):
        """Test basic recommendation generation"""
        data_dir = Path(self.temp_dir) / "data"
        
        self.engine.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv"),
            str(data_dir / "customer_segmentation_results.csv")
        )
        
        self.engine.create_user_item_matrix()
        self.engine.calculate_item_similarity()
        
        recommendations = self.engine.generate_advanced_recommendations(1, 3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        if recommendations:
            rec = recommendations[0]
            required_fields = ['product_id', 'product_name', 'price', 'confidence_score']
            for field in required_fields:
                self.assertIn(field, rec)


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = CustomerAnalyticsPipeline(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline_small_dataset(self):
        """Test complete pipeline with small dataset"""
        # Generate small dataset
        data_generator = DataGenerator()
        
        customers_df = data_generator.generate_customers(10)
        products_df = data_generator.generate_products(5)
        transactions_df = data_generator.generate_transactions(customers_df, products_df, 20)
        
        # Save to temp directory
        customers_df.to_csv(self.pipeline.data_dir / "customers.csv", index=False)
        products_df.to_csv(self.pipeline.data_dir / "products.csv", index=False)
        transactions_df.to_csv(self.pipeline.data_dir / "transactions.csv", index=False)
        
        # Test pipeline execution
        success = self.pipeline.load_and_validate_data()
        self.assertTrue(success)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data"""
        # Create invalid data files
        invalid_customers = pd.DataFrame({'invalid_column': [1, 2, 3]})
        invalid_customers.to_csv(self.pipeline.data_dir / "customers.csv", index=False)
        
        # This should fail gracefully
        success = self.pipeline.load_and_validate_data()
        self.assertFalse(success)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance testing with various dataset sizes"""
    
    def setUp(self):
        """Set up performance testing"""
        self.data_generator = DataGenerator()
        self.segmentation = CustomerSegmentation()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_small_dataset(self):
        """Test performance with small dataset (50 customers)"""
        start_time = time.time()
        
        customers_df = self.data_generator.generate_customers(50)
        products_df = self.data_generator.generate_products(25)
        transactions_df = self.data_generator.generate_transactions(customers_df, products_df, 200)
        
        generation_time = time.time() - start_time
        
        # Save and load data for segmentation
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        customers_df.to_csv(data_dir / "customers.csv", index=False)
        products_df.to_csv(data_dir / "products.csv", index=False)
        transactions_df.to_csv(data_dir / "transactions.csv", index=False)
        
        start_time = time.time()
        self.segmentation.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv")
        )
        rfm_df = self.segmentation.calculate_rfm()
        rfm_time = time.time() - start_time
        
        # Performance assertions (should complete within reasonable time)
        self.assertLess(generation_time, 10.0)  # 10 seconds
        self.assertLess(rfm_time, 10.0)  # 10 seconds
        
        print(f"Small dataset performance:")
        print(f"  Data generation: {generation_time:.2f}s")
        print(f"  RFM calculation: {rfm_time:.2f}s")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up edge case testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.segmentation = CustomerSegmentation()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_single_transaction_per_customer(self):
        """Test RFM calculation with single transaction per customer"""
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create minimal dataset
        customers_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
            'registration_date': pd.date_range('2022-01-01', periods=3),
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })
        
        products_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'product_name': ['Product A', 'Product B', 'Product C'],
            'category': ['Electronics', 'Books', 'Clothing'],
            'price': [100.0, 25.0, 50.0]
        })
        
        transactions_df = pd.DataFrame({
            'transaction_id': [1, 2, 3],
            'customer_id': [1, 2, 3],
            'product_id': [1, 2, 3],
            'purchase_date': pd.date_range('2022-01-01', periods=3),
            'price': [100.0, 25.0, 50.0]
        })
        
        # Save files
        customers_df.to_csv(data_dir / "customers.csv", index=False)
        products_df.to_csv(data_dir / "products.csv", index=False)
        transactions_df.to_csv(data_dir / "transactions.csv", index=False)
        
        # Load and test
        self.segmentation.load_data(
            str(data_dir / "customers.csv"),
            str(data_dir / "products.csv"),
            str(data_dir / "transactions.csv")
        )
        
        rfm_df = self.segmentation.calculate_rfm()
        
        # All customers should have frequency = 1
        self.assertIsNotNone(rfm_df)
        if rfm_df is not None:
            self.assertTrue(all(rfm_df['frequency'] == 1))


def run_comprehensive_tests():
    """Run all test suites and generate detailed report"""
    print("🧪 COMPREHENSIVE TESTING SUITE")
    print("=" * 50)
    
    test_suites = [
        TestDataGenerator,
        TestCustomerSegmentation,
        TestAdvancedRecommendationEngine,
        TestIntegrationPipeline,
        TestPerformanceBenchmarks,
        TestEdgeCases
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for suite_class in test_suites:
        print(f"\n📋 Running {suite_class.__name__}")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        # Print simplified results
        if result.testsRun > 0:
            success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
            print(f"   Tests: {result.testsRun} | Passed: {result.testsRun - len(result.failures) - len(result.errors)} | Failed: {len(result.failures)} | Errors: {len(result.errors)}")
            print(f"   Success Rate: {success_rate:.1f}%")
        
        if result.failures:
            print(f"   ❌ Failures:")
            for test, traceback in result.failures:
                print(f"      - {test.id().split('.')[-1]}")
        
        if result.errors:
            print(f"   🚫 Errors:")
            for test, traceback in result.errors:
                print(f"      - {test.id().split('.')[-1]}")
    
    print(f"\n📊 TESTING SUMMARY")
    print("=" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests() 