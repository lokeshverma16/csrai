#!/usr/bin/env python3
"""
Advanced Hybrid Recommendation Engine with Transaction Analysis

This enhanced system includes:
- Purchase pattern analysis (timing, seasonality, basket composition)
- Segment-specific recommendation strategies
- Time-based recommendations with customer journey mapping
- Advanced features (cross-selling, upselling, bundles)
- Recommendation validation and A/B testing simulation
- Comprehensive performance dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import itertools
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedRecommendationEngine:
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.segmentation_df = None
        
        # Core recommendation components
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.customer_profiles = None
        
        # Advanced analysis components
        self.purchase_patterns = None
        self.seasonal_patterns = None
        self.basket_analysis = None
        self.price_sensitivity = None
        self.customer_journey = None
        
        # Recommendation strategies
        self.segment_strategies = {}
        self.cross_sell_matrix = None
        self.bundle_recommendations = None
        
        # Performance tracking
        self.recommendation_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.config = {
            'min_confidence': 0.2,
            'diversity_weight': 0.3,
            'seasonality_boost': 0.15,
            'recency_weight': 0.25,
            'cross_sell_threshold': 0.3,
            'bundle_min_support': 0.05
        }
    
    def load_data(self, customers_path='data/customers.csv', 
                  products_path='data/products.csv', 
                  transactions_path='data/transactions.csv',
                  segmentation_path='data/customer_segmentation_results.csv'):
        """Load all data and initialize advanced analytics"""
        try:
            print("📊 LOADING DATA FOR ADVANCED RECOMMENDATION ENGINE")
            print("-" * 60)
            
            self.customers_df = pd.read_csv(customers_path)
            self.products_df = pd.read_csv(products_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            # Convert date columns
            self.transactions_df['purchase_date'] = pd.to_datetime(self.transactions_df['purchase_date'])
            
            # Load segmentation data
            try:
                self.segmentation_df = pd.read_csv(segmentation_path)
                print(f"✅ Loaded customer segmentation data")
            except:
                print("⚠️  Customer segmentation data not found - will generate basic segments")
                self.segmentation_df = None
            
            print(f"📈 Data Summary:")
            print(f"   • Customers: {len(self.customers_df):,}")
            print(f"   • Products: {len(self.products_df):,}")
            print(f"   • Transactions: {len(self.transactions_df):,}")
            print(f"   • Date Range: {self.transactions_df['purchase_date'].min()} to {self.transactions_df['purchase_date'].max()}")
            
            # Initialize advanced analytics
            self._initialize_advanced_analytics()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _initialize_advanced_analytics(self):
        """Initialize all advanced analytics components"""
        print(f"\n🔧 INITIALIZING ADVANCED ANALYTICS")
        print("-" * 40)
        
        # Purchase pattern analysis
        print("Analyzing purchase patterns...")
        self._analyze_purchase_patterns()
        
        # Seasonal analysis
        print("Detecting seasonal patterns...")
        self._analyze_seasonal_patterns()
        
        # Basket analysis
        print("Performing market basket analysis...")
        self._analyze_basket_composition()
        
        # Price sensitivity analysis
        print("Calculating price sensitivity...")
        self._analyze_price_sensitivity()
        
        # Customer journey mapping
        print("Mapping customer journeys...")
        self._map_customer_journeys()
        
        # Initialize segment strategies
        print("Setting up segment strategies...")
        self._initialize_segment_strategies()
        
        print("✅ Advanced analytics initialized")
    
    def _analyze_purchase_patterns(self):
        """Analyze purchase patterns for each customer"""
        purchase_patterns = []
        
        for customer_id in self.customers_df['customer_id']:
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ].sort_values('purchase_date')
            
            if len(customer_transactions) < 2:
                pattern = {
                    'customer_id': customer_id,
                    'avg_days_between_purchases': None,
                    'purchase_frequency': 'new' if len(customer_transactions) == 0 else 'single',
                    'regularity_score': 0,
                    'last_purchase_days_ago': (datetime.now() - customer_transactions['purchase_date'].max()).days if len(customer_transactions) > 0 else None,
                    'purchase_count': len(customer_transactions),
                    'total_spent': customer_transactions['price'].sum() if len(customer_transactions) > 0 else 0
                }
            else:
                # Calculate days between purchases
                dates = customer_transactions['purchase_date']
                days_between = [(dates.iloc[i] - dates.iloc[i-1]).days for i in range(1, len(dates))]
                avg_days = np.mean(days_between)
                
                # Determine frequency pattern
                if avg_days <= 30:
                    frequency = 'frequent'  # Monthly or more
                elif avg_days <= 90:
                    frequency = 'regular'   # Quarterly
                elif avg_days <= 180:
                    frequency = 'occasional'  # Semi-annual
                else:
                    frequency = 'rare'      # Annual or less
                
                # Calculate regularity (inverse of coefficient of variation)
                regularity = 1 / (np.std(days_between) / avg_days + 1) if avg_days > 0 else 0
                
                pattern = {
                    'customer_id': customer_id,
                    'avg_days_between_purchases': avg_days,
                    'purchase_frequency': frequency,
                    'regularity_score': regularity,
                    'last_purchase_days_ago': (datetime.now() - dates.max()).days,
                    'purchase_count': len(customer_transactions),
                    'total_spent': customer_transactions['price'].sum()
                }
            
            purchase_patterns.append(pattern)
        
        self.purchase_patterns = pd.DataFrame(purchase_patterns)
        print(f"   Analyzed patterns for {len(self.purchase_patterns)} customers")
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal buying patterns"""
        # Add temporal features
        self.transactions_df['month'] = self.transactions_df['purchase_date'].dt.month
        self.transactions_df['quarter'] = self.transactions_df['purchase_date'].dt.quarter
        self.transactions_df['day_of_week'] = self.transactions_df['purchase_date'].dt.dayofweek
        
        # Seasonal analysis by category
        seasonal_by_category = self.transactions_df.merge(
            self.products_df[['product_id', 'category']], on='product_id'
        ).groupby(['category', 'month']).size().unstack(fill_value=0)
        
        # Calculate seasonal indices (monthly sales / average monthly sales)
        seasonal_indices = seasonal_by_category.div(seasonal_by_category.mean(axis=1), axis=0)
        
        # Identify peak seasons for each category
        peak_seasons = {}
        for category in seasonal_indices.index:
            category_data = seasonal_indices.loc[category]
            peak_month = category_data.idxmax()
            peak_strength = category_data.max()
            
            peak_seasons[category] = {
                'peak_month': peak_month,
                'peak_strength': peak_strength,
                'seasonal_pattern': 'high' if peak_strength > 1.3 else 'moderate' if peak_strength > 1.1 else 'low'
            }
        
        self.seasonal_patterns = {
            'by_category': seasonal_indices,
            'peak_seasons': peak_seasons,
            'current_month': datetime.now().month
        }
        
        print(f"   Identified seasonal patterns for {len(peak_seasons)} categories")
    
    def _analyze_basket_composition(self):
        """Perform market basket analysis to find complementary products"""
        # Group transactions by customer and date to identify baskets
        transaction_baskets = self.transactions_df.groupby(['customer_id', 'purchase_date'])['product_id'].apply(list).reset_index()
        transaction_baskets = transaction_baskets[transaction_baskets['product_id'].apply(len) > 1]  # Only multi-item baskets
        
        # Calculate product co-occurrence
        product_pairs = defaultdict(int)
        total_baskets = len(transaction_baskets)
        
        for basket in transaction_baskets['product_id']:
            for product1, product2 in itertools.combinations(sorted(basket), 2):
                product_pairs[(product1, product2)] += 1
        
        # Calculate support, confidence, and lift
        product_counts = self.transactions_df['product_id'].value_counts()
        basket_rules = []
        
        for (product1, product2), co_count in product_pairs.items():
            support = co_count / total_baskets
            confidence = co_count / product_counts[product1]
            lift = (co_count / total_baskets) / ((product_counts[product1] / len(self.transactions_df)) * (product_counts[product2] / len(self.transactions_df)))
            
            if support >= self.config['bundle_min_support'] and confidence >= 0.1:
                basket_rules.append({
                    'product1': product1,
                    'product2': product2,
                    'support': support,
                    'confidence': confidence,
                    'lift': lift
                })
        
        # Create cross-sell recommendations matrix
        cross_sell_dict = defaultdict(list)
        for rule in sorted(basket_rules, key=lambda x: x['lift'], reverse=True):
            if rule['lift'] > 1.0:  # Positive association
                cross_sell_dict[rule['product1']].append({
                    'product': rule['product2'],
                    'strength': rule['lift'],
                    'confidence': rule['confidence']
                })
        
        self.basket_analysis = {
            'rules': basket_rules,
            'cross_sell_matrix': dict(cross_sell_dict),
            'total_baskets': total_baskets
        }
        
        print(f"   Found {len(basket_rules)} product association rules")
    
    def _analyze_price_sensitivity(self):
        """Analyze price sensitivity for each customer"""
        price_sensitivity = []
        
        for customer_id in self.customers_df['customer_id']:
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ]
            
            if len(customer_transactions) == 0:
                sensitivity = {
                    'customer_id': customer_id,
                    'price_sensitivity': 'unknown',
                    'avg_price': 0,
                    'price_variance': 0,
                    'discount_affinity': 0
                }
            else:
                prices = customer_transactions['price']
                avg_price = prices.mean()
                price_variance = prices.var()
                
                # Calculate overall market price percentiles
                all_prices = self.transactions_df['price']
                price_percentile = (all_prices <= avg_price).mean()
                
                # Determine sensitivity based on price percentile and variance
                if price_percentile <= 0.33 and price_variance < avg_price * 0.5:
                    sensitivity_level = 'high'  # Consistently buys low-priced items
                elif price_percentile >= 0.67 and price_variance < avg_price * 0.3:
                    sensitivity_level = 'low'   # Consistently buys high-priced items
                else:
                    sensitivity_level = 'medium'  # Mixed or moderate pricing
                
                # Simulate discount affinity (would be based on actual discount data)
                discount_affinity = min(1.0, price_variance / avg_price) if avg_price > 0 else 0
                
                sensitivity = {
                    'customer_id': customer_id,
                    'price_sensitivity': sensitivity_level,
                    'avg_price': avg_price,
                    'price_variance': price_variance,
                    'discount_affinity': discount_affinity,
                    'price_percentile': price_percentile
                }
            
            price_sensitivity.append(sensitivity)
        
        self.price_sensitivity = pd.DataFrame(price_sensitivity)
        print(f"   Analyzed price sensitivity for {len(self.price_sensitivity)} customers")
    
    def _map_customer_journeys(self):
        """Map customer journeys and identify next best actions"""
        customer_journeys = []
        
        for customer_id in self.customers_df['customer_id']:
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ].sort_values('purchase_date')
            
            if len(customer_transactions) == 0:
                journey_stage = 'prospect'
                next_action = 'acquisition'
            else:
                # Determine journey stage based on purchase history
                days_since_last = (datetime.now() - customer_transactions['purchase_date'].max()).days
                purchase_count = len(customer_transactions)
                
                if purchase_count == 1:
                    if days_since_last <= 30:
                        journey_stage = 'new_customer'
                        next_action = 'second_purchase'
                    else:
                        journey_stage = 'one_time_buyer'
                        next_action = 'reactivation'
                elif purchase_count <= 5:
                    if days_since_last <= 60:
                        journey_stage = 'developing'
                        next_action = 'engagement'
                    else:
                        journey_stage = 'at_risk_new'
                        next_action = 'retention'
                else:
                    if days_since_last <= 90:
                        journey_stage = 'loyal'
                        next_action = 'expansion'
                    else:
                        journey_stage = 'at_risk_loyal'
                        next_action = 'win_back'
                
                # Analyze category progression
                categories_purchased = customer_transactions.merge(
                    self.products_df[['product_id', 'category']], on='product_id'
                )['category'].unique()
                
                journey = {
                    'customer_id': customer_id,
                    'journey_stage': journey_stage,
                    'next_best_action': next_action,
                    'purchase_count': purchase_count,
                    'days_since_last_purchase': days_since_last,
                    'categories_explored': len(categories_purchased),
                    'category_diversity': len(categories_purchased) / len(self.products_df['category'].unique()),
                    'total_value': customer_transactions['price'].sum()
                }
                
            customer_journeys.append(journey)
        
        self.customer_journey = pd.DataFrame(customer_journeys)
        print(f"   Mapped customer journeys for {len(self.customer_journey)} customers")
    
    def _initialize_segment_strategies(self):
        """Initialize advanced segment-specific recommendation strategies"""
        self.segment_strategies = {
            'Champions': {
                'focus': 'premium_exclusive',
                'product_filters': ['premium', 'new_arrivals', 'exclusive'],
                'price_boost': 1.3,  # Boost expensive items
                'category_expansion': True,
                'cross_sell_weight': 0.4,
                'explanation_style': 'premium'
            },
            'Loyal Customers': {
                'focus': 'brand_extensions',
                'product_filters': ['complementary', 'bulk_eligible'],
                'price_boost': 1.1,
                'category_expansion': True,
                'cross_sell_weight': 0.6,
                'explanation_style': 'loyalty'
            },
            'Potential Loyalists': {
                'focus': 'popular_value',
                'product_filters': ['popular', 'trending'],
                'price_boost': 0.9,  # Slight preference for value
                'category_expansion': True,
                'cross_sell_weight': 0.3,
                'explanation_style': 'social_proof'
            },
            'At-Risk': {
                'focus': 'win_back',
                'product_filters': ['previous_categories', 'discounted'],
                'price_boost': 0.7,  # Strong value emphasis
                'category_expansion': False,  # Stick to known preferences
                'cross_sell_weight': 0.2,
                'explanation_style': 'value'
            },
            'Cannot Lose Them': {
                'focus': 'personalized_high_value',
                'product_filters': ['personalized', 'high_value'],
                'price_boost': 1.5,  # Premium recommendations
                'category_expansion': True,
                'cross_sell_weight': 0.5,
                'explanation_style': 'exclusive'
            },
            'Hibernating': {
                'focus': 'reactivation',
                'product_filters': ['previous_categories', 'trending'],
                'price_boost': 0.8,
                'category_expansion': False,
                'cross_sell_weight': 0.1,
                'explanation_style': 'comeback'
            }
        }
        
        print(f"   Configured strategies for {len(self.segment_strategies)} segments")
    
    def create_user_item_matrix(self):
        """Create enhanced user-item matrix with temporal weighting"""
        print("🔢 Creating enhanced user-item matrix...")
        
        # Calculate recency weights (more recent purchases get higher weights)
        max_date = self.transactions_df['purchase_date'].max()
        self.transactions_df['days_ago'] = (max_date - self.transactions_df['purchase_date']).dt.days
        self.transactions_df['recency_weight'] = np.exp(-self.transactions_df['days_ago'] / 365)  # Exponential decay
        
        # Create weighted ratings
        user_item_data = self.transactions_df.groupby(['customer_id', 'product_id']).agg({
            'transaction_id': 'count',  # Purchase frequency
            'price': 'sum',             # Total spent
            'recency_weight': 'max'     # Most recent weight
        }).reset_index()
        
        # Calculate implicit ratings
        max_frequency = user_item_data['transaction_id'].max()
        max_spent = user_item_data['price'].max()
        
        user_item_data['rating'] = (
            (user_item_data['transaction_id'] / max_frequency) * 0.4 +
            (user_item_data['price'] / max_spent) * 0.4 +
            user_item_data['recency_weight'] * 0.2
        ) * 5
        
        # Create matrix
        self.user_item_matrix = user_item_data.pivot(
            index='customer_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        print(f"   Matrix shape: {self.user_item_matrix.shape}")
        sparsity = (self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100
        print(f"   Sparsity: {sparsity:.1f}%")
        
        return self.user_item_matrix
    
    def calculate_item_similarity(self):
        """Calculate enhanced item similarity matrix"""
        print("🔗 Calculating item similarity matrix...")
        
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Item-based similarity
        item_matrix = self.user_item_matrix.T
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_matrix.index,
            columns=item_matrix.index
        )
        
        print(f"   Similarity matrix shape: {self.item_similarity_matrix.shape}")
        return self.item_similarity_matrix 

    def generate_advanced_recommendations(self, customer_id, n_recommendations=3):
        """Generate advanced recommendations using all enhancement features"""
        print(f"🎯 Generating advanced recommendations for customer {customer_id}")
        
        try:
            # Initialize if needed
            if self.user_item_matrix is None:
                self.create_user_item_matrix()
            if self.item_similarity_matrix is None:
                self.calculate_item_similarity()
            
            # Get customer context
            customer_context = self._get_customer_context(customer_id)
            
            # Get candidate products
            candidate_products = self._get_candidate_products(customer_id, customer_context)
            
            if not candidate_products:
                return self._get_fallback_recommendations(customer_id, n_recommendations)
            
            # Calculate recommendation scores using multiple strategies
            recommendation_scores = self._calculate_advanced_scores(customer_id, candidate_products, customer_context)
            
            # Apply business rules and enhancements
            enhanced_scores = self._apply_advanced_business_rules(recommendation_scores, customer_context)
            
            # Select diverse recommendations
            selected_products = self._select_advanced_recommendations(enhanced_scores, n_recommendations, customer_context)
            
            # Format with advanced explanations
            formatted_recommendations = self._format_advanced_recommendations(
                selected_products, customer_id, customer_context
            )
            
            # Track recommendations for performance analysis
            self._track_recommendation(customer_id, formatted_recommendations)
            
            print(f"✅ Generated {len(formatted_recommendations)} advanced recommendations")
            return formatted_recommendations
            
        except Exception as e:
            print(f"❌ Error generating advanced recommendations: {e}")
            return self._get_fallback_recommendations(customer_id, n_recommendations)
    
    def _get_customer_context(self, customer_id):
        """Get comprehensive customer context for recommendations"""
        context = {'customer_id': customer_id}
        
        # Basic customer info
        customer_info = self.customers_df[self.customers_df['customer_id'] == customer_id]
        if len(customer_info) > 0:
            context['name'] = customer_info.iloc[0]['name']
        
        # Segmentation
        if self.segmentation_df is not None:
            segment_info = self.segmentation_df[self.segmentation_df['customer_id'] == customer_id]
            context['segment'] = segment_info.iloc[0]['segment'] if len(segment_info) > 0 else 'Unknown'
        else:
            context['segment'] = 'Unknown'
        
        # Purchase patterns
        if self.purchase_patterns is not None:
            pattern_info = self.purchase_patterns[self.purchase_patterns['customer_id'] == customer_id]
            if len(pattern_info) > 0:
                context['purchase_pattern'] = pattern_info.iloc[0].to_dict()
            else:
                context['purchase_pattern'] = {'purchase_frequency': 'new'}
        
        # Price sensitivity
        if self.price_sensitivity is not None:
            price_info = self.price_sensitivity[self.price_sensitivity['customer_id'] == customer_id]
            if len(price_info) > 0:
                context['price_sensitivity'] = price_info.iloc[0].to_dict()
            else:
                context['price_sensitivity'] = {'price_sensitivity': 'unknown'}
        
        # Customer journey
        if self.customer_journey is not None:
            journey_info = self.customer_journey[self.customer_journey['customer_id'] == customer_id]
            if len(journey_info) > 0:
                context['journey'] = journey_info.iloc[0].to_dict()
            else:
                context['journey'] = {'journey_stage': 'prospect'}
        
        # Purchase history
        customer_purchases = self.transactions_df[self.transactions_df['customer_id'] == customer_id]
        context['purchase_history'] = {
            'total_purchases': len(customer_purchases),
            'total_spent': customer_purchases['price'].sum() if len(customer_purchases) > 0 else 0,
            'categories': customer_purchases.merge(
                self.products_df[['product_id', 'category']], on='product_id'
            )['category'].tolist() if len(customer_purchases) > 0 else [],
            'last_purchase_date': customer_purchases['purchase_date'].max() if len(customer_purchases) > 0 else None
        }
        
        return context
    
    def _get_candidate_products(self, customer_id, customer_context):
        """Get candidate products based on customer context and segment strategy"""
        # Get purchased products to exclude
        if customer_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[customer_id]
            purchased_products = set(user_ratings[user_ratings > 0].index.tolist())
        else:
            purchased_products = set()
        
        # All available products
        all_products = set(self.products_df['product_id'].tolist())
        candidate_products = list(all_products - purchased_products)
        
        # Apply segment-specific filtering
        segment = customer_context['segment']
        if segment in self.segment_strategies:
            strategy = self.segment_strategies[segment]
            
            # Filter based on segment strategy
            if strategy['focus'] == 'premium_exclusive':
                # Premium products (top 20% by price)
                price_threshold = self.products_df['price'].quantile(0.8)
                premium_products = self.products_df[self.products_df['price'] >= price_threshold]['product_id'].tolist()
                candidate_products = [p for p in candidate_products if p in premium_products]
            
            elif strategy['focus'] == 'popular_value':
                # Popular products with good value
                popular_products = self.transactions_df['product_id'].value_counts().head(100).index.tolist()
                price_threshold = self.products_df['price'].quantile(0.6)
                value_products = self.products_df[self.products_df['price'] <= price_threshold]['product_id'].tolist()
                candidate_products = [p for p in candidate_products if p in popular_products or p in value_products]
            
            elif strategy['focus'] == 'win_back':
                # Products from previously purchased categories
                previous_categories = customer_context['purchase_history']['categories']
                if previous_categories:
                    category_products = self.products_df[
                        self.products_df['category'].isin(previous_categories)
                    ]['product_id'].tolist()
                    candidate_products = [p for p in candidate_products if p in category_products]
        
        return candidate_products[:200]  # Limit for performance
    
    def _calculate_advanced_scores(self, customer_id, candidate_products, customer_context):
        """Calculate advanced recommendation scores using multiple strategies"""
        scores = defaultdict(float)
        
        # 1. Collaborative filtering score
        if customer_id in self.user_item_matrix.index:
            cf_scores = self._calculate_collaborative_scores(customer_id, candidate_products)
            for product_id, score in cf_scores.items():
                scores[product_id] += score * 0.3
        
        # 2. Content-based score (category preferences)
        content_scores = self._calculate_content_scores(customer_id, candidate_products, customer_context)
        for product_id, score in content_scores.items():
            scores[product_id] += score * 0.25
        
        # 3. Cross-selling score
        cross_sell_scores = self._calculate_cross_sell_scores(customer_id, candidate_products)
        for product_id, score in cross_sell_scores.items():
            scores[product_id] += score * 0.2
        
        # 4. Seasonal/temporal score
        temporal_scores = self._calculate_temporal_scores(candidate_products, customer_context)
        for product_id, score in temporal_scores.items():
            scores[product_id] += score * 0.15
        
        # 5. Price affinity score
        price_scores = self._calculate_price_affinity_scores(candidate_products, customer_context)
        for product_id, score in price_scores.items():
            scores[product_id] += score * 0.1
        
        return dict(scores)
    
    def _calculate_collaborative_scores(self, customer_id, candidate_products):
        """Enhanced collaborative filtering with recency weighting"""
        scores = {}
        
        if self.item_similarity_matrix is None:
            return scores
        
        user_ratings = self.user_item_matrix.loc[customer_id]
        purchased_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not purchased_items:
            return scores
        
        for product_id in candidate_products:
            if product_id in self.item_similarity_matrix.index:
                score = 0
                similarity_sum = 0
                
                for purchased_item in purchased_items:
                    if purchased_item in self.item_similarity_matrix.columns:
                        similarity = self.item_similarity_matrix.loc[product_id, purchased_item]
                        user_rating = user_ratings[purchased_item]
                        
                        score += similarity * user_rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    scores[product_id] = score / similarity_sum
        
        return scores
    
    def _calculate_content_scores(self, customer_id, candidate_products, customer_context):
        """Calculate content-based scores from customer preferences"""
        scores = {}
        
        # Get category preferences
        purchased_categories = customer_context['purchase_history']['categories']
        if not purchased_categories:
            return scores
        
        category_counts = Counter(purchased_categories)
        total_purchases = sum(category_counts.values())
        category_preferences = {cat: count/total_purchases for cat, count in category_counts.items()}
        
        # Score products based on category preferences
        for product_id in candidate_products:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if len(product_info) > 0:
                product_category = product_info.iloc[0]['category']
                scores[product_id] = category_preferences.get(product_category, 0.1)  # Small base score for new categories
        
        return scores
    
    def _calculate_cross_sell_scores(self, customer_id, candidate_products):
        """Calculate cross-selling scores based on basket analysis"""
        scores = {}
        
        if not self.basket_analysis or 'cross_sell_matrix' not in self.basket_analysis:
            return scores
        
        # Get customer's purchase history
        customer_purchases = self.transactions_df[
            self.transactions_df['customer_id'] == customer_id
        ]['product_id'].tolist()
        
        cross_sell_matrix = self.basket_analysis['cross_sell_matrix']
        
        # Calculate cross-sell scores
        for product_id in candidate_products:
            total_score = 0
            count = 0
            
            for purchased_product in customer_purchases:
                if purchased_product in cross_sell_matrix:
                    for cross_sell_item in cross_sell_matrix[purchased_product]:
                        if cross_sell_item['product'] == product_id:
                            total_score += cross_sell_item['strength']
                            count += 1
            
            if count > 0:
                scores[product_id] = total_score / count
        
        return scores
    
    def _calculate_temporal_scores(self, candidate_products, customer_context):
        """Calculate temporal/seasonal scores"""
        scores = {}
        
        if not self.seasonal_patterns:
            return scores
        
        current_month = self.seasonal_patterns['current_month']
        
        for product_id in candidate_products:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if len(product_info) > 0:
                category = product_info.iloc[0]['category']
                
                # Check if category has seasonal boost this month
                if category in self.seasonal_patterns['peak_seasons']:
                    peak_info = self.seasonal_patterns['peak_seasons'][category]
                    if peak_info['peak_month'] == current_month:
                        scores[product_id] = peak_info['peak_strength']
                    else:
                        # Distance from peak month
                        month_distance = min(abs(current_month - peak_info['peak_month']), 
                                           12 - abs(current_month - peak_info['peak_month']))
                        scores[product_id] = max(0.5, 1 - (month_distance / 6))
                else:
                    scores[product_id] = 0.8  # Neutral score
        
        return scores
    
    def _calculate_price_affinity_scores(self, candidate_products, customer_context):
        """Calculate price affinity scores based on customer's price sensitivity"""
        scores = {}
        
        price_sensitivity = customer_context.get('price_sensitivity', {})
        customer_avg_price = price_sensitivity.get('avg_price', self.products_df['price'].median())
        sensitivity_level = price_sensitivity.get('price_sensitivity', 'medium')
        
        for product_id in candidate_products:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if len(product_info) > 0:
                product_price = product_info.iloc[0]['price']
                
                # Calculate price affinity based on customer's sensitivity
                if sensitivity_level == 'high':
                    # Price-sensitive customers prefer lower prices
                    if product_price <= customer_avg_price * 0.8:
                        scores[product_id] = 1.0
                    elif product_price <= customer_avg_price * 1.2:
                        scores[product_id] = 0.7
                    else:
                        scores[product_id] = 0.3
                elif sensitivity_level == 'low':
                    # Price-insensitive customers are open to premium
                    if product_price >= customer_avg_price * 1.2:
                        scores[product_id] = 1.0
                    elif product_price >= customer_avg_price * 0.8:
                        scores[product_id] = 0.8
                    else:
                        scores[product_id] = 0.6
                else:  # medium sensitivity
                    # Prefer products near their average price
                    price_ratio = product_price / customer_avg_price if customer_avg_price > 0 else 1
                    scores[product_id] = max(0.4, 1 - abs(1 - price_ratio))
        
        return scores
    
    def _apply_advanced_business_rules(self, scores, customer_context):
        """Apply advanced business rules based on segment strategy"""
        enhanced_scores = scores.copy()
        
        segment = customer_context['segment']
        if segment not in self.segment_strategies:
            return enhanced_scores
        
        strategy = self.segment_strategies[segment]
        
        # Apply segment-specific price boost
        price_boost = strategy.get('price_boost', 1.0)
        
        for product_id in enhanced_scores:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if len(product_info) > 0:
                product_price = product_info.iloc[0]['price']
                median_price = self.products_df['price'].median()
                
                # Apply price-based boost
                if price_boost > 1.0 and product_price > median_price:
                    enhanced_scores[product_id] *= price_boost
                elif price_boost < 1.0 and product_price < median_price:
                    enhanced_scores[product_id] *= (2 - price_boost)  # Boost value products
        
        # Apply journey-based boosts
        journey_stage = customer_context.get('journey', {}).get('journey_stage', 'unknown')
        
        if journey_stage == 'new_customer':
            # Boost popular/trending items for new customers
            popular_products = self.transactions_df['product_id'].value_counts().head(50).index.tolist()
            for product_id in enhanced_scores:
                if product_id in popular_products:
                    enhanced_scores[product_id] *= 1.2
        
        elif journey_stage in ['at_risk_new', 'at_risk_loyal']:
            # Boost previously purchased categories for at-risk customers
            previous_categories = customer_context['purchase_history']['categories']
            for product_id in enhanced_scores:
                product_info = self.products_df[self.products_df['product_id'] == product_id]
                if len(product_info) > 0:
                    product_category = product_info.iloc[0]['category']
                    if product_category in previous_categories:
                        enhanced_scores[product_id] *= 1.3
        
        return enhanced_scores
    
    def _select_advanced_recommendations(self, scores, n_recommendations, customer_context):
        """Select diverse recommendations with advanced logic"""
        if not scores:
            return []
        
        # Sort by score
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Category diversity enforcement
        selected = []
        used_categories = set()
        
        segment = customer_context['segment']
        strategy = self.segment_strategies.get(segment, {})
        enforce_diversity = strategy.get('category_expansion', True)
        
        # First pass: ensure category diversity if required
        if enforce_diversity:
            for product_id, score in sorted_products:
                if len(selected) >= n_recommendations:
                    break
                
                product_info = self.products_df[self.products_df['product_id'] == product_id]
                if len(product_info) > 0:
                    category = product_info.iloc[0]['category']
                    
                    if category not in used_categories or len(selected) == 0:
                        selected.append(product_id)
                        used_categories.add(category)
        
        # Second pass: fill remaining slots with highest scores
        for product_id, score in sorted_products:
            if len(selected) >= n_recommendations:
                break
            
            if product_id not in selected:
                selected.append(product_id)
        
        return selected[:n_recommendations]
    
    def _format_advanced_recommendations(self, product_ids, customer_id, customer_context):
        """Format recommendations with advanced explanations"""
        recommendations = []
        
        segment = customer_context['segment']
        strategy = self.segment_strategies.get(segment, {})
        explanation_style = strategy.get('explanation_style', 'general')
        
        for i, product_id in enumerate(product_ids, 1):
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            
            if len(product_info) == 0:
                continue
            
            product = product_info.iloc[0]
            
            # Calculate confidence score
            confidence = self._calculate_recommendation_confidence(product_id, customer_context)
            
            # Generate advanced explanation
            explanation = self._generate_advanced_explanation(
                product, customer_context, explanation_style
            )
            
            # Determine recommendation type
            rec_type = self._determine_recommendation_type(product_id, customer_context)
            
            recommendation = {
                'rank': i,
                'product_id': product_id,
                'product_name': product['product_name'],
                'category': product['category'],
                'price': product['price'],
                'confidence_score': round(confidence, 3),
                'explanation': explanation,
                'recommendation_type': rec_type,
                'customer_segment': segment,
                'journey_stage': customer_context.get('journey', {}).get('journey_stage', 'unknown'),
                'next_best_action': customer_context.get('journey', {}).get('next_best_action', 'engagement')
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_recommendation_confidence(self, product_id, customer_context):
        """Calculate confidence score for recommendation"""
        base_confidence = 0.5
        
        # Boost confidence based on available data
        if customer_context['purchase_history']['total_purchases'] > 5:
            base_confidence += 0.2  # More data = higher confidence
        
        if customer_context['segment'] in self.segment_strategies:
            base_confidence += 0.15  # Known segment = higher confidence
        
        # Check if product has strong associations
        if self.basket_analysis and 'cross_sell_matrix' in self.basket_analysis:
            customer_purchases = self.transactions_df[
                self.transactions_df['customer_id'] == customer_context['customer_id']
            ]['product_id'].tolist()
            
            cross_sell_strength = 0
            for purchased_product in customer_purchases:
                if purchased_product in self.basket_analysis['cross_sell_matrix']:
                    for item in self.basket_analysis['cross_sell_matrix'][purchased_product]:
                        if item['product'] == product_id:
                            cross_sell_strength += item['strength']
            
            if cross_sell_strength > 1.5:
                base_confidence += 0.15
        
        return min(1.0, base_confidence)
    
    def _generate_advanced_explanation(self, product, customer_context, style):
        """Generate advanced explanation based on recommendation style"""
        explanations = []
        
        if style == 'premium':
            explanations.append("Exclusive premium selection")
            if product['price'] > self.products_df['price'].quantile(0.8):
                explanations.append("Top-tier quality")
        
        elif style == 'loyalty':
            explanations.append("Recommended for loyal customers")
            if product['category'] in customer_context['purchase_history']['categories']:
                explanations.append(f"From your favorite {product['category']} category")
        
        elif style == 'social_proof':
            explanations.append("Popular choice among customers")
            explanations.append("Trending in your area")
        
        elif style == 'value':
            explanations.append("Great value option")
            if product['price'] < self.products_df['price'].median():
                explanations.append("Budget-friendly price")
        
        elif style == 'exclusive':
            explanations.append("Personally curated for you")
            explanations.append("High-value recommendation")
        
        elif style == 'comeback':
            explanations.append("Welcome back special")
            if product['category'] in customer_context['purchase_history']['categories']:
                explanations.append("From categories you enjoyed")
        
        else:
            explanations.append("Personalized for you")
        
        # Add seasonal context if relevant
        current_month = datetime.now().month
        if current_month in [11, 12] and product['category'] in ['Electronics', 'Home']:
            explanations.append("Perfect for the holiday season")
        
        return " • ".join(explanations)
    
    def _determine_recommendation_type(self, product_id, customer_context):
        """Determine the type of recommendation (cross-sell, upsell, etc.)"""
        customer_purchases = self.transactions_df[
            self.transactions_df['customer_id'] == customer_context['customer_id']
        ]
        
        if len(customer_purchases) == 0:
            return 'acquisition'
        
        # Check if it's a cross-sell (different category)
        purchased_categories = customer_purchases.merge(
            self.products_df[['product_id', 'category']], on='product_id'
        )['category'].unique()
        
        product_category = self.products_df[
            self.products_df['product_id'] == product_id
        ]['category'].iloc[0]
        
        if product_category not in purchased_categories:
            return 'cross_sell'
        
        # Check if it's an upsell (higher price in same category)
        category_purchases = customer_purchases.merge(
            self.products_df[['product_id', 'category', 'price']], on='product_id'
        )
        category_purchases = category_purchases[category_purchases['category'] == product_category]
        
        if len(category_purchases) > 0:
            avg_category_price = category_purchases['price'].mean()
            product_price = self.products_df[
                self.products_df['product_id'] == product_id
            ]['price'].iloc[0]
            
            if product_price > avg_category_price * 1.2:
                return 'upsell'
            elif product_price < avg_category_price * 0.8:
                return 'value_alternative'
        
        return 'repeat_category'
    
    def _get_fallback_recommendations(self, customer_id, n_recommendations):
        """Get fallback recommendations when advanced methods fail"""
        try:
            # Get popular products
            popular_products = self.transactions_df['product_id'].value_counts().head(n_recommendations * 2)
            
            recommendations = []
            for i, (product_id, count) in enumerate(popular_products.items(), 1):
                if i > n_recommendations:
                    break
                
                product_info = self.products_df[self.products_df['product_id'] == product_id]
                if len(product_info) > 0:
                    product = product_info.iloc[0]
                    
                    recommendation = {
                        'rank': i,
                        'product_id': product_id,
                        'product_name': product['product_name'],
                        'category': product['category'],
                        'price': product['price'],
                        'confidence_score': 0.6,
                        'explanation': 'Popular choice • Frequently purchased',
                        'recommendation_type': 'popular',
                        'customer_segment': 'unknown',
                        'journey_stage': 'unknown',
                        'next_best_action': 'engagement'
                    }
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️  Error in fallback recommendations: {e}")
            return []
    
    def _track_recommendation(self, customer_id, recommendations):
        """Track recommendations for performance analysis"""
        self.recommendation_history.append({
            'customer_id': customer_id,
            'timestamp': datetime.now(),
            'recommendations': recommendations,
            'num_recommendations': len(recommendations)
        }) 

    def simulate_ab_testing(self, customer_ids=None, test_variants=None):
        """Simulate A/B testing for recommendation validation"""
        print("🧪 SIMULATING A/B TESTING FRAMEWORK")
        print("-" * 50)
        
        if customer_ids is None:
            # Use random sample of customers
            customer_ids = self.customers_df['customer_id'].sample(min(100, len(self.customers_df))).tolist()
        
        if test_variants is None:
            test_variants = {
                'control': {'weight_collaborative': 0.3, 'weight_content': 0.25, 'diversity_weight': 0.3},
                'variant_a': {'weight_collaborative': 0.4, 'weight_content': 0.2, 'diversity_weight': 0.4},
                'variant_b': {'weight_collaborative': 0.2, 'weight_content': 0.35, 'diversity_weight': 0.2}
            }
        
        ab_results = {}
        
        for variant_name, config in test_variants.items():
            print(f"Testing variant: {variant_name}")
            
            # Temporarily update config
            original_config = self.config.copy()
            self.config.update(config)
            
            variant_metrics = {
                'recommendations_generated': 0,
                'total_confidence': 0,
                'category_diversity': 0,
                'cross_sell_rate': 0,
                'upsell_rate': 0,
                'predicted_revenue': 0
            }
            
            for customer_id in customer_ids:
                try:
                    recommendations = self.generate_advanced_recommendations(customer_id, 3)
                    
                    if recommendations:
                        variant_metrics['recommendations_generated'] += 1
                        variant_metrics['total_confidence'] += sum(rec['confidence_score'] for rec in recommendations)
                        
                        # Category diversity
                        categories = set(rec['category'] for rec in recommendations)
                        variant_metrics['category_diversity'] += len(categories)
                        
                        # Recommendation type analysis
                        for rec in recommendations:
                            if rec['recommendation_type'] == 'cross_sell':
                                variant_metrics['cross_sell_rate'] += 1
                            elif rec['recommendation_type'] == 'upsell':
                                variant_metrics['upsell_rate'] += 1
                            
                            # Predicted revenue (simplified)
                            variant_metrics['predicted_revenue'] += rec['price'] * rec['confidence_score'] * 0.1
                
                except Exception as e:
                    print(f"   Error testing customer {customer_id}: {e}")
            
            # Calculate averages
            if variant_metrics['recommendations_generated'] > 0:
                variant_metrics['avg_confidence'] = variant_metrics['total_confidence'] / (variant_metrics['recommendations_generated'] * 3)
                variant_metrics['avg_category_diversity'] = variant_metrics['category_diversity'] / variant_metrics['recommendations_generated']
                variant_metrics['cross_sell_rate'] = variant_metrics['cross_sell_rate'] / (variant_metrics['recommendations_generated'] * 3)
                variant_metrics['upsell_rate'] = variant_metrics['upsell_rate'] / (variant_metrics['recommendations_generated'] * 3)
            
            ab_results[variant_name] = variant_metrics
            
            # Restore original config
            self.config = original_config
        
        # Determine winner
        winner = max(ab_results.keys(), key=lambda v: ab_results[v]['predicted_revenue'])
        
        print(f"\n🏆 A/B Test Results:")
        for variant, metrics in ab_results.items():
            print(f"   {variant}: Revenue=${metrics['predicted_revenue']:.2f}, Confidence={metrics.get('avg_confidence', 0):.3f}")
        print(f"   Winner: {winner}")
        
        return ab_results
    
    def calculate_revenue_impact(self, customer_recommendations, baseline_revenue_per_customer=50):
        """Calculate potential revenue impact of recommendations"""
        print("💰 CALCULATING REVENUE IMPACT")
        print("-" * 35)
        
        total_potential_revenue = 0
        conversion_estimates = []
        
        for customer_id, recommendations in customer_recommendations.items():
            customer_potential = 0
            
            for rec in recommendations:
                # Simple conversion probability model
                base_conversion = 0.05  # 5% base conversion
                confidence_boost = rec['confidence_score'] * 0.1  # Up to 10% boost from confidence
                
                # Segment-specific conversion rates
                segment = rec.get('customer_segment', 'unknown')
                if segment == 'Champions':
                    segment_boost = 0.15
                elif segment == 'Loyal Customers':
                    segment_boost = 0.12
                elif segment == 'Potential Loyalists':
                    segment_boost = 0.08
                else:
                    segment_boost = 0.05
                
                conversion_probability = min(0.5, base_conversion + confidence_boost + segment_boost)
                expected_revenue = rec['price'] * conversion_probability
                
                customer_potential += expected_revenue
                conversion_estimates.append({
                    'customer_id': customer_id,
                    'product_id': rec['product_id'],
                    'price': rec['price'],
                    'conversion_probability': conversion_probability,
                    'expected_revenue': expected_revenue
                })
            
            total_potential_revenue += customer_potential
        
        # Calculate lift over baseline
        num_customers = len(customer_recommendations)
        baseline_total = num_customers * baseline_revenue_per_customer
        revenue_lift = ((total_potential_revenue - baseline_total) / baseline_total) * 100 if baseline_total > 0 else 0
        
        impact_metrics = {
            'total_potential_revenue': total_potential_revenue,
            'baseline_revenue': baseline_total,
            'revenue_lift_percent': revenue_lift,
            'avg_revenue_per_customer': total_potential_revenue / num_customers if num_customers > 0 else 0,
            'num_customers_analyzed': num_customers,
            'conversion_estimates': conversion_estimates
        }
        
        print(f"   Potential Revenue: ${total_potential_revenue:,.2f}")
        print(f"   Revenue Lift: {revenue_lift:.1f}%")
        print(f"   Avg per Customer: ${impact_metrics['avg_revenue_per_customer']:.2f}")
        
        return impact_metrics
    
    def generate_performance_dashboard(self, customer_recommendations):
        """Generate comprehensive performance dashboard"""
        print("📊 GENERATING PERFORMANCE DASHBOARD")
        print("-" * 40)
        
        dashboard = {
            'recommendation_metrics': {},
            'segment_analysis': {},
            'business_metrics': {},
            'recommendation_types': {},
            'confidence_analysis': {},
            'category_analysis': {}
        }
        
        # Recommendation metrics
        total_recommendations = sum(len(recs) for recs in customer_recommendations.values())
        avg_confidence = np.mean([
            rec['confidence_score'] 
            for recs in customer_recommendations.values() 
            for rec in recs
        ]) if total_recommendations > 0 else 0
        
        dashboard['recommendation_metrics'] = {
            'total_customers': len(customer_recommendations),
            'total_recommendations': total_recommendations,
            'avg_recommendations_per_customer': total_recommendations / len(customer_recommendations) if customer_recommendations else 0,
            'avg_confidence_score': avg_confidence,
            'success_rate': len([r for r in customer_recommendations.values() if r]) / len(customer_recommendations) if customer_recommendations else 0
        }
        
        # Segment analysis
        segment_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0, 'total_revenue_potential': 0})
        
        for recommendations in customer_recommendations.values():
            for rec in recommendations:
                segment = rec.get('customer_segment', 'unknown')
                segment_stats[segment]['count'] += 1
                segment_stats[segment]['total_confidence'] += rec['confidence_score']
                segment_stats[segment]['total_revenue_potential'] += rec['price']
        
        for segment in segment_stats:
            count = segment_stats[segment]['count']
            if count > 0:
                segment_stats[segment]['avg_confidence'] = segment_stats[segment]['total_confidence'] / count
                segment_stats[segment]['avg_price'] = segment_stats[segment]['total_revenue_potential'] / count
        
        dashboard['segment_analysis'] = dict(segment_stats)
        
        # Recommendation types analysis
        type_counts = defaultdict(int)
        for recommendations in customer_recommendations.values():
            for rec in recommendations:
                type_counts[rec['recommendation_type']] += 1
        
        dashboard['recommendation_types'] = dict(type_counts)
        
        # Confidence analysis
        all_confidences = [
            rec['confidence_score'] 
            for recs in customer_recommendations.values() 
            for rec in recs
        ]
        
        if all_confidences:
            dashboard['confidence_analysis'] = {
                'min_confidence': min(all_confidences),
                'max_confidence': max(all_confidences),
                'median_confidence': np.median(all_confidences),
                'high_confidence_rate': sum(1 for c in all_confidences if c >= 0.8) / len(all_confidences)
            }
        
        # Category analysis
        category_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0, 'total_price': 0})
        
        for recommendations in customer_recommendations.values():
            for rec in recommendations:
                category = rec['category']
                category_stats[category]['count'] += 1
                category_stats[category]['total_confidence'] += rec['confidence_score']
                category_stats[category]['total_price'] += rec['price']
        
        for category in category_stats:
            count = category_stats[category]['count']
            if count > 0:
                category_stats[category]['avg_confidence'] = category_stats[category]['total_confidence'] / count
                category_stats[category]['avg_price'] = category_stats[category]['total_price'] / count
        
        dashboard['category_analysis'] = dict(category_stats)
        
        # Calculate revenue impact
        revenue_impact = self.calculate_revenue_impact(customer_recommendations)
        dashboard['business_metrics'] = revenue_impact
        
        return dashboard
    
    def print_dashboard_summary(self, dashboard):
        """Print formatted dashboard summary"""
        print("\n" + "="*80)
        print("🎯 ADVANCED RECOMMENDATION ENGINE PERFORMANCE DASHBOARD")
        print("="*80)
        
        # Recommendation Overview
        metrics = dashboard['recommendation_metrics']
        print(f"\n📊 RECOMMENDATION OVERVIEW:")
        print(f"   • Total Customers Analyzed: {metrics['total_customers']:,}")
        print(f"   • Total Recommendations Generated: {metrics['total_recommendations']:,}")
        print(f"   • Average Recommendations per Customer: {metrics['avg_recommendations_per_customer']:.1f}")
        print(f"   • Success Rate: {metrics['success_rate']:.1%}")
        print(f"   • Average Confidence Score: {metrics['avg_confidence_score']:.3f}")
        
        # Business Impact
        business = dashboard['business_metrics']
        print(f"\n💰 BUSINESS IMPACT:")
        print(f"   • Potential Revenue: ${business['total_potential_revenue']:,.2f}")
        print(f"   • Revenue Lift: {business['revenue_lift_percent']:.1f}%")
        print(f"   • Average Revenue per Customer: ${business['avg_revenue_per_customer']:.2f}")
        
        # Segment Performance
        print(f"\n👥 SEGMENT PERFORMANCE:")
        for segment, stats in dashboard['segment_analysis'].items():
            if stats['count'] > 0:
                print(f"   • {segment}: {stats['count']} recs, "
                      f"Confidence: {stats['avg_confidence']:.3f}, "
                      f"Avg Price: ${stats['avg_price']:.2f}")
        
        # Recommendation Types
        print(f"\n🎲 RECOMMENDATION TYPES:")
        for rec_type, count in dashboard['recommendation_types'].items():
            percentage = (count / metrics['total_recommendations']) * 100 if metrics['total_recommendations'] > 0 else 0
            print(f"   • {rec_type}: {count} ({percentage:.1f}%)")
        
        # Top Categories
        print(f"\n📦 TOP RECOMMENDED CATEGORIES:")
        category_data = sorted(
            dashboard['category_analysis'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]
        
        for category, stats in category_data:
            print(f"   • {category}: {stats['count']} recs, "
                  f"Confidence: {stats['avg_confidence']:.3f}")
        
        # Quality Metrics
        confidence = dashboard['confidence_analysis']
        if confidence:
            print(f"\n⭐ QUALITY METRICS:")
            print(f"   • High Confidence Rate (≥0.8): {confidence['high_confidence_rate']:.1%}")
            print(f"   • Confidence Range: {confidence['min_confidence']:.3f} - {confidence['max_confidence']:.3f}")
            print(f"   • Median Confidence: {confidence['median_confidence']:.3f}")
    
    def generate_all_customer_recommendations(self, max_customers=None):
        """Generate recommendations for all customers"""
        print("🚀 GENERATING RECOMMENDATIONS FOR ALL CUSTOMERS")
        print("-" * 55)
        
        if max_customers:
            customer_ids = self.customers_df['customer_id'].head(max_customers).tolist()
        else:
            customer_ids = self.customers_df['customer_id'].tolist()
        
        all_recommendations = {}
        successful_generations = 0
        failed_generations = 0
        
        for i, customer_id in enumerate(customer_ids, 1):
            if i % 50 == 0:
                print(f"   Processed {i}/{len(customer_ids)} customers...")
            
            try:
                recommendations = self.generate_advanced_recommendations(customer_id, 3)
                if recommendations:
                    all_recommendations[customer_id] = recommendations
                    successful_generations += 1
                else:
                    failed_generations += 1
            except Exception as e:
                print(f"   Error generating recommendations for customer {customer_id}: {e}")
                failed_generations += 1
        
        print(f"\n✅ GENERATION COMPLETE:")
        print(f"   • Successful: {successful_generations:,}")
        print(f"   • Failed: {failed_generations:,}")
        print(f"   • Success Rate: {(successful_generations/(successful_generations+failed_generations)):.1%}")
        
        return all_recommendations

# Main execution function
def run_advanced_recommendation_demo():
    """Run comprehensive demonstration of advanced recommendation engine"""
    print("🚀 ADVANCED RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize engine
    engine = AdvancedRecommendationEngine()
    
    # Load data
    if not engine.load_data():
        print("❌ Failed to load data")
        return
    
    # Create matrices
    engine.create_user_item_matrix()
    engine.calculate_item_similarity()
    
    # Generate recommendations for sample customers
    print(f"\n🎯 GENERATING SAMPLE RECOMMENDATIONS")
    print("-" * 45)
    
    sample_customers = engine.customers_df['customer_id'].head(5).tolist()
    sample_recommendations = {}
    
    for customer_id in sample_customers:
        print(f"\n--- Customer {customer_id} ---")
        recommendations = engine.generate_advanced_recommendations(customer_id, 3)
        sample_recommendations[customer_id] = recommendations
        
        if recommendations:
            for rec in recommendations:
                print(f"   {rec['rank']}. {rec['product_name']} (${rec['price']:.2f})")
                print(f"      Confidence: {rec['confidence_score']:.3f} | Type: {rec['recommendation_type']}")
                print(f"      Segment: {rec['customer_segment']} | Stage: {rec['journey_stage']}")
                print(f"      Explanation: {rec['explanation']}")
    
    # Generate all customer recommendations
    print(f"\n🎯 GENERATING ALL CUSTOMER RECOMMENDATIONS")
    print("-" * 50)
    all_recommendations = engine.generate_all_customer_recommendations(max_customers=200)  # Limit for demo
    
    # Generate performance dashboard
    dashboard = engine.generate_performance_dashboard(all_recommendations)
    engine.print_dashboard_summary(dashboard)
    
    # A/B testing simulation
    print(f"\n🧪 A/B TESTING SIMULATION")
    print("-" * 30)
    ab_results = engine.simulate_ab_testing(list(all_recommendations.keys())[:50])
    
    print(f"\n🎉 ADVANCED RECOMMENDATION ENGINE DEMO COMPLETE!")
    print("="*60)
    
    return {
        'engine': engine,
        'sample_recommendations': sample_recommendations,
        'all_recommendations': all_recommendations,
        'dashboard': dashboard,
        'ab_results': ab_results
    }

if __name__ == "__main__":
    results = run_advanced_recommendation_demo() 