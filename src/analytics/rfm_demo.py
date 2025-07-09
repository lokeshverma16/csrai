#!/usr/bin/env python3
"""
Comprehensive RFM Analysis & Visualization Demonstration
=========================================================

This script demonstrates the complete RFM (Recency, Frequency, Monetary) analysis
pipeline including data generation, customer segmentation, and comprehensive visualizations.

Author: Data Science Portfolio Project
Date: July 2025
"""

import sys
import os
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)

def print_step(step_num, title):
    """Print formatted step"""
    print(f"\n{step_num}️⃣  {title}")
    print("-" * 60)

def main():
    """Run complete RFM analysis and visualization demonstration"""
    
    print_header("COMPREHENSIVE RFM ANALYSIS & VISUALIZATION DEMO")
    print("🎯 Complete Customer Analytics Pipeline Demonstration")
    print("📊 Including: Data Generation → RFM Analysis → Professional Visualizations")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Generation
        print_step("1", "DATA GENERATION")
        print("🔄 Generating realistic customer, product, and transaction data...")
        
        try:
            from data_generator import demonstrate_data_generation
            data_generator = demonstrate_data_generation()
            if data_generator is None:
                print("❌ Data generation failed")
                return False
            print("✅ Data generation completed successfully")
        except Exception as e:
            print(f"❌ Data generation error: {e}")
            return False
        
        # Step 2: RFM Customer Segmentation
        print_step("2", "RFM CUSTOMER SEGMENTATION")
        print("🎯 Performing comprehensive RFM analysis...")
        
        try:
            from customer_segmentation import demonstrate_rfm_analysis
            segmentation_system = demonstrate_rfm_analysis()
            if segmentation_system is None:
                print("❌ RFM analysis failed")
                return False
            print("✅ RFM customer segmentation completed successfully")
        except Exception as e:
            print(f"❌ RFM analysis error: {e}")
            return False
        
        # Step 3: Comprehensive Visualizations
        print_step("3", "COMPREHENSIVE VISUALIZATIONS")
        print("📊 Creating professional publication-ready visualizations...")
        
        try:
            from rfm_visualizations import demonstrate_rfm_visualizations
            viz_system = demonstrate_rfm_visualizations()
            if viz_system is None:
                print("❌ Visualization generation failed")
                return False
            print("✅ Comprehensive visualizations completed successfully")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return False
        
        # Step 4: Results Summary
        print_step("4", "RESULTS SUMMARY")
        print("📋 Generating final analysis summary...")
        
        # Check generated files
        data_files = ['data/customers.csv', 'data/products.csv', 'data/transactions.csv']
        analysis_files = ['data/customer_segmentation_results.csv']
        viz_files = [
            'visualizations/rfm_distributions.png',
            'visualizations/rfm_correlation_analysis.png', 
            'visualizations/customer_behavior_analysis.png',
            'visualizations/outlier_analysis.png',
            'visualizations/business_insights.png'
        ]
        
        print("\n📁 GENERATED FILES:")
        print("  📊 Data Files:")
        for file in data_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"     ✅ {file} ({size:,} bytes)")
            else:
                print(f"     ❌ {file} (missing)")
        
        print("  🎯 Analysis Files:")
        for file in analysis_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"     ✅ {file} ({size:,} bytes)")
            else:
                print(f"     ❌ {file} (missing)")
        
        print("  📈 Visualization Files:")
        for file in viz_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"     ✅ {file} ({size:,} bytes)")
            else:
                print(f"     ❌ {file} (missing)")
        
        # Summary statistics
        if segmentation_system and hasattr(segmentation_system, 'rfm_data'):
            rfm_data = segmentation_system.rfm_data
            print("\n📊 KEY METRICS:")
            print(f"     👥 Total Customers Analyzed: {len(rfm_data):,}")
            print(f"     💰 Total Revenue: ${rfm_data['monetary'].sum():,.2f}")
            print(f"     🎯 Customer Segments: {rfm_data['segment'].nunique()}")
            
            # Top segments
            top_segments = rfm_data['segment'].value_counts().head(3)
            print(f"     🏆 Top Segments:")
            for segment, count in top_segments.items():
                percentage = (count / len(rfm_data)) * 100
                print(f"        • {segment}: {count} customers ({percentage:.1f}%)")
        
        print_header("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
        print("🚀 Complete RFM Analysis Pipeline Executed")
        print("📊 Professional visualizations generated")
        print("💼 Ready for data science portfolio presentation")
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        print("🔍 Check error messages above for troubleshooting")
        return False

def quick_analysis():
    """Run quick analysis on existing data"""
    print_header("QUICK RFM ANALYSIS")
    print("🚀 Running analysis on existing data...")
    
    try:
        # Check if data exists
        required_files = ['data/customers.csv', 'data/products.csv', 'data/transactions.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("💡 Run full demonstration first: python3 rfm_demo.py")
            return False
        
        # Run RFM analysis only
        from customer_segmentation import demonstrate_rfm_analysis
        segmentation_system = demonstrate_rfm_analysis()
        
        if segmentation_system:
            print("✅ Quick RFM analysis completed!")
            return True
        else:
            print("❌ Quick analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick analysis error: {e}")
        return False

def quick_visualizations():
    """Generate visualizations on existing data"""
    print_header("QUICK VISUALIZATION GENERATION")
    print("📊 Generating visualizations on existing data...")
    
    try:
        # Check if RFM data exists
        if not os.path.exists('data/customer_segmentation_results.csv'):
            print("❌ RFM analysis results not found")
            print("💡 Run RFM analysis first: python3 rfm_demo.py --quick-analysis")
            return False
        
        # Generate visualizations only
        from rfm_visualizations import demonstrate_rfm_visualizations
        viz_system = demonstrate_rfm_visualizations()
        
        if viz_system:
            print("✅ Quick visualization generation completed!")
            return True
        else:
            print("❌ Visualization generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

if __name__ == "__main__":
    # Command line argument handling
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-analysis":
            success = quick_analysis()
        elif sys.argv[1] == "--quick-viz":
            success = quick_visualizations()
        elif sys.argv[1] == "--help":
            print("RFM Analysis Demonstration Script")
            print("Usage:")
            print("  python3 rfm_demo.py              # Full demonstration")
            print("  python3 rfm_demo.py --quick-analysis  # RFM analysis only")
            print("  python3 rfm_demo.py --quick-viz       # Visualizations only")
            print("  python3 rfm_demo.py --help           # Show this help")
            sys.exit(0)
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Run full demonstration
        success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 