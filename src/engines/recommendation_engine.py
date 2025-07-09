import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationEngine:
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.segmentation_df = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.content_similarity_matrix = None
        self.customer_profiles = None
        self.product_features = None
        self.cluster_characteristics = None
        self.trending_products = None
        self.recommendation_cache = {}
        
        # Business rules configuration
        self.min_confidence_threshold = 0.1
        self.diversity_weight = 0.3
        self.popularity_boost = 0.2
        self.seasonal_adjustment = 0.1
        
    def load_data(self, customers_path='data/customers.csv', 
                  products_path='data/products.csv', 
                  transactions_path='data/transactions.csv',
                  segmentation_path='data/customer_segmentation_results.csv'):
        """Load all necessary data for recommendations"""
        try:
            self.customers_df = pd.read_csv(customers_path)
            self.products_df = pd.read_csv(products_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            # Load customer segmentation data if available
            try:
                self.segmentation_df = pd.read_csv(segmentation_path)
                print(f"✅ Loaded customer segmentation data")
            except:
                print("⚠️  Customer segmentation data not found - will generate basic segments")
                self.segmentation_df = None
            
            print(f"📊 Loaded {len(self.customers_df)} customers, {len(self.products_df)} products, {len(self.transactions_df)} transactions")
            
            # Initialize additional components - will be called later when needed
            # self._analyze_trending_products()
            # self._extract_cluster_characteristics()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        # Calculate user-item ratings based on purchase frequency and recency
        user_item_data = self.transactions_df.groupby(['customer_id', 'product_id']).agg({
            'transaction_id': 'count',  # Purchase frequency
            'purchase_date': 'max',     # Most recent purchase
            'price': 'sum'              # Total spent on product
        }).reset_index()
        
        # Rename columns
        user_item_data.columns = ['customer_id', 'product_id', 'frequency', 'last_purchase', 'total_spent']
        
        # Create implicit ratings (1-5 scale based on frequency and spending)
        max_frequency = user_item_data['frequency'].max()
        max_spent = user_item_data['total_spent'].max()
        
        # Normalize and combine frequency and spending for rating
        user_item_data['freq_norm'] = user_item_data['frequency'] / max_frequency
        user_item_data['spent_norm'] = user_item_data['total_spent'] / max_spent
        user_item_data['rating'] = (user_item_data['freq_norm'] * 0.6 + user_item_data['spent_norm'] * 0.4) * 5
        user_item_data['rating'] = user_item_data['rating'].clip(1, 5)
        
        # Create pivot table (user-item matrix)
        self.user_item_matrix = user_item_data.pivot(
            index='customer_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        print(f"Created user-item matrix: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.2f}%")
        
        return self.user_item_matrix
    
    def calculate_item_similarity(self):
        """Calculate item-item similarity matrix using cosine similarity"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Calculate cosine similarity between items (products)
        # Transpose to get item x user matrix
        item_matrix = self.user_item_matrix.T
        
        # Calculate similarity
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_matrix.index,
            columns=item_matrix.index
        )
        
        print(f"Calculated item similarity matrix: {self.item_similarity_matrix.shape}")
        return self.item_similarity_matrix
    
    def calculate_user_similarity(self):
        """Calculate user-user similarity matrix using cosine similarity"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Calculate cosine similarity between users
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"Calculated user similarity matrix: {self.user_similarity_matrix.shape}")
        return self.user_similarity_matrix
    
    def content_based_similarity(self):
        """Calculate content-based similarity using product features"""
        # Create feature strings for each product
        self.products_df['features'] = self.products_df['category'] + ' ' + self.products_df['product_name']
        
        # Use TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(self.products_df['features'])
        
        # Calculate cosine similarity
        content_similarity = cosine_similarity(tfidf_matrix)
        
        self.content_similarity_matrix = pd.DataFrame(
            content_similarity,
            index=self.products_df['product_id'],
            columns=self.products_df['product_id']
        )
        
        print(f"Calculated content similarity matrix: {self.content_similarity_matrix.shape}")
        return self.content_similarity_matrix
    
    def get_item_based_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using item-based collaborative filtering"""
        if self.item_similarity_matrix is None:
            self.calculate_item_similarity()
        
        if customer_id not in self.user_item_matrix.index:
            print(f"Customer {customer_id} not found in user-item matrix")
            return pd.DataFrame()
        
        # Get user's purchase history
        user_ratings = self.user_item_matrix.loc[customer_id]
        purchased_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not purchased_items:
            print(f"No purchase history found for customer {customer_id}")
            return self.get_popular_recommendations(n_recommendations)
        
        # Calculate recommendations
        recommendations = {}
        
        for item in self.user_item_matrix.columns:
            if item not in purchased_items:  # Don't recommend already purchased items
                score = 0
                similarity_sum = 0
                
                for purchased_item in purchased_items:
                    if item in self.item_similarity_matrix.index and purchased_item in self.item_similarity_matrix.columns:
                        similarity = self.item_similarity_matrix.loc[item, purchased_item]
                        user_rating = user_ratings[purchased_item]
                        
                        score += similarity * user_rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    recommendations[item] = score / similarity_sum
        
        # Sort recommendations
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = recommendations[:n_recommendations]
        
        # Create recommendation dataframe
        rec_df = pd.DataFrame(top_recommendations, columns=['product_id', 'score'])
        rec_df = rec_df.merge(self.products_df, on='product_id', how='left')
        
        print(f"Generated {len(rec_df)} item-based recommendations for customer {customer_id}")
        return rec_df
    
    def get_user_based_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using user-based collaborative filtering"""
        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()
        
        if customer_id not in self.user_item_matrix.index:
            print(f"Customer {customer_id} not found in user-item matrix")
            return pd.DataFrame()
        
        # Get similar users
        user_similarities = self.user_similarity_matrix.loc[customer_id]
        similar_users = user_similarities.sort_values(ascending=False)[1:11]  # Top 10 similar users (excluding self)
        
        # Get user's purchase history
        user_ratings = self.user_item_matrix.loc[customer_id]
        purchased_items = user_ratings[user_ratings > 0].index.tolist()
        
        # Calculate recommendations based on similar users
        recommendations = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        for similar_user, similarity in similar_users.items():
            if similarity > 0:  # Only consider positively similar users
                similar_user_ratings = self.user_item_matrix.loc[similar_user]
                
                for item, rating in similar_user_ratings.items():
                    if item not in purchased_items and rating > 0:
                        recommendations[item] += similarity * rating
                        similarity_sums[item] += similarity
        
        # Normalize scores
        final_recommendations = {}
        for item in recommendations:
            if similarity_sums[item] > 0:
                final_recommendations[item] = recommendations[item] / similarity_sums[item]
        
        # Sort recommendations
        recommendations_sorted = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = recommendations_sorted[:n_recommendations]
        
        # Create recommendation dataframe
        rec_df = pd.DataFrame(top_recommendations, columns=['product_id', 'score'])
        rec_df = rec_df.merge(self.products_df, on='product_id', how='left')
        
        print(f"Generated {len(rec_df)} user-based recommendations for customer {customer_id}")
        return rec_df
    
    def get_content_based_recommendations(self, customer_id, n_recommendations=10):
        """Get recommendations using content-based filtering"""
        if not hasattr(self, 'content_similarity_matrix'):
            self.content_based_similarity()
        
        if customer_id not in self.user_item_matrix.index:
            print(f"Customer {customer_id} not found")
            return pd.DataFrame()
        
        # Get user's purchase history
        user_ratings = self.user_item_matrix.loc[customer_id]
        purchased_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not purchased_items:
            return self.get_popular_recommendations(n_recommendations)
        
        # Calculate content-based scores
        recommendations = {}
        
        for item in self.products_df['product_id']:
            if item not in purchased_items:
                score = 0
                count = 0
                
                for purchased_item in purchased_items:
                    if item in self.content_similarity_matrix.index and purchased_item in self.content_similarity_matrix.columns:
                        similarity = self.content_similarity_matrix.loc[item, purchased_item]
                        user_rating = user_ratings[purchased_item]
                        
                        score += similarity * user_rating
                        count += 1
                
                if count > 0:
                    recommendations[item] = score / count
        
        # Sort recommendations
        recommendations_sorted = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = recommendations_sorted[:n_recommendations]
        
        # Create recommendation dataframe
        rec_df = pd.DataFrame(top_recommendations, columns=['product_id', 'score'])
        rec_df = rec_df.merge(self.products_df, on='product_id', how='left')
        
        print(f"Generated {len(rec_df)} content-based recommendations for customer {customer_id}")
        return rec_df
    
    def get_popular_recommendations(self, n_recommendations=10):
        """Get popular items as fallback recommendations"""
        # Calculate popularity based on frequency and revenue
        popularity = self.transactions_df.groupby('product_id').agg({
            'transaction_id': 'count',
            'price': 'sum'
        }).reset_index()
        
        popularity.columns = ['product_id', 'purchase_count', 'total_revenue']
        
        # Normalize and combine metrics
        popularity['popularity_score'] = (
            popularity['purchase_count'] / popularity['purchase_count'].max() * 0.7 +
            popularity['total_revenue'] / popularity['total_revenue'].max() * 0.3
        )
        
        # Sort by popularity
        popular_items = popularity.sort_values('popularity_score', ascending=False).head(n_recommendations)
        
        # Merge with product information
        rec_df = popular_items.merge(self.products_df, on='product_id', how='left')
        
        print(f"Generated {len(rec_df)} popular item recommendations")
        return rec_df
    
    def get_hybrid_recommendations(self, customer_id, n_recommendations=10, weights=None):
        """Get hybrid recommendations combining multiple approaches"""
        if weights is None:
            weights = {'item_based': 0.4, 'user_based': 0.3, 'content_based': 0.3}
        
        # Get recommendations from different approaches
        item_recs = self.get_item_based_recommendations(customer_id, n_recommendations * 2)
        user_recs = self.get_user_based_recommendations(customer_id, n_recommendations * 2)
        content_recs = self.get_content_based_recommendations(customer_id, n_recommendations * 2)
        
        # Combine scores
        all_recommendations = {}
        
        # Add item-based scores
        for _, row in item_recs.iterrows():
            product_id = row['product_id']
            all_recommendations[product_id] = all_recommendations.get(product_id, 0) + row['score'] * weights['item_based']
        
        # Add user-based scores
        for _, row in user_recs.iterrows():
            product_id = row['product_id']
            all_recommendations[product_id] = all_recommendations.get(product_id, 0) + row['score'] * weights['user_based']
        
        # Add content-based scores
        for _, row in content_recs.iterrows():
            product_id = row['product_id']
            all_recommendations[product_id] = all_recommendations.get(product_id, 0) + row['score'] * weights['content_based']
        
        # Sort final recommendations
        final_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = final_recommendations[:n_recommendations]
        
        # Create recommendation dataframe
        rec_df = pd.DataFrame(top_recommendations, columns=['product_id', 'hybrid_score'])
        rec_df = rec_df.merge(self.products_df, on='product_id', how='left')
        
        print(f"Generated {len(rec_df)} hybrid recommendations for customer {customer_id}")
        return rec_df
    
    def get_recommendations_for_segment(self, segment, n_recommendations=10):
        """Get recommendations for a specific customer segment"""
        try:
            # Load segmentation results
            segmentation_df = pd.read_csv('data/customer_segmentation_results.csv')
            segment_customers = segmentation_df[segmentation_df['segment'] == segment]['customer_id'].tolist()
            
            if not segment_customers:
                print(f"No customers found in segment: {segment}")
                return pd.DataFrame()
            
            # Get all recommendations for segment customers
            all_segment_recommendations = []
            
            for customer_id in segment_customers[:5]:  # Sample 5 customers for efficiency
                customer_recs = self.get_hybrid_recommendations(customer_id, n_recommendations)
                if not customer_recs.empty:
                    customer_recs['customer_id'] = customer_id
                    all_segment_recommendations.append(customer_recs)
            
            if not all_segment_recommendations:
                return pd.DataFrame()
            
            # Combine all recommendations
            combined_recs = pd.concat(all_segment_recommendations, ignore_index=True)
            
            # Calculate average scores for each product
            segment_recommendations = combined_recs.groupby('product_id').agg({
                'hybrid_score': 'mean',
                'product_name': 'first',
                'category': 'first',
                'price': 'first'
            }).reset_index()
            
            # Sort by average score
            segment_recommendations = segment_recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
            
            print(f"Generated {len(segment_recommendations)} recommendations for segment: {segment}")
            return segment_recommendations
            
        except Exception as e:
            print(f"Error generating segment recommendations: {e}")
            return pd.DataFrame()
    
    def evaluate_recommendations(self, test_ratio=0.2):
        """Evaluate recommendation quality using train-test split"""
        print("Evaluating recommendation system...")
        
        # Create train-test split
        test_transactions = self.transactions_df.sample(frac=test_ratio, random_state=42)
        train_transactions = self.transactions_df.drop(test_transactions.index)
        
        # Temporarily use train data
        original_transactions = self.transactions_df.copy()
        self.transactions_df = train_transactions
        self.user_item_matrix = None  # Reset to force recalculation
        
        # Calculate matrices with train data
        self.create_user_item_matrix()
        self.calculate_item_similarity()
        
        # Evaluate on test set
        test_customers = test_transactions['customer_id'].unique()
        hits = 0
        total_recommendations = 0
        
        for customer_id in test_customers[:50]:  # Evaluate on sample for efficiency
            # Get actual test purchases for this customer
            actual_purchases = test_transactions[test_transactions['customer_id'] == customer_id]['product_id'].tolist()
            
            if actual_purchases:
                # Get recommendations
                recommendations = self.get_item_based_recommendations(customer_id, 10)
                
                if not recommendations.empty:
                    recommended_items = recommendations['product_id'].tolist()
                    
                    # Check for hits
                    for item in actual_purchases:
                        if item in recommended_items:
                            hits += 1
                    
                    total_recommendations += len(recommended_items)
        
        # Calculate precision
        precision = hits / total_recommendations if total_recommendations > 0 else 0
        
        print(f"Recommendation System Evaluation:")
        print(f"Precision@10: {precision:.3f}")
        print(f"Hits: {hits}")
        print(f"Total Recommendations: {total_recommendations}")
        
        # Restore original data
        self.transactions_df = original_transactions
        self.user_item_matrix = None
        
        return precision

if __name__ == "__main__":
    # Example usage
    rec_engine = HybridRecommendationEngine()
    
    # Load data
    if rec_engine.load_data():
        
        # Create matrices
        rec_engine.create_user_item_matrix()
        rec_engine.calculate_item_similarity()
        rec_engine.calculate_user_similarity()
        rec_engine.content_based_similarity()
        
        # Test recommendations for a sample customer
        sample_customer = rec_engine.customers_df['customer_id'].iloc[0]
        print(f"\nTesting recommendations for customer: {sample_customer}")
        
        # Get different types of recommendations
        print("\n--- Item-Based Recommendations ---")
        item_recs = rec_engine.get_item_based_recommendations(sample_customer, 5)
        print(item_recs[['product_name', 'category', 'price', 'score']])
        
        print("\n--- User-Based Recommendations ---")
        user_recs = rec_engine.get_user_based_recommendations(sample_customer, 5)
        print(user_recs[['product_name', 'category', 'price', 'score']])
        
        print("\n--- Content-Based Recommendations ---")
        content_recs = rec_engine.get_content_based_recommendations(sample_customer, 5)
        print(content_recs[['product_name', 'category', 'price', 'score']])
        
        print("\n--- Hybrid Recommendations ---")
        hybrid_recs = rec_engine.get_hybrid_recommendations(sample_customer, 5)
        print(hybrid_recs[['product_name', 'category', 'price', 'hybrid_score']])
        
        # Evaluate system
        rec_engine.evaluate_recommendations()
    
    def _analyze_trending_products(self):
        """Analyze trending products based on recent sales"""
        if self.transactions_df is None:
            return
        
        try:
            # Convert purchase_date to datetime if it's not already
            self.transactions_df['purchase_date'] = pd.to_datetime(self.transactions_df['purchase_date'])
            
            # Get recent transactions (last 30 days from max date)
            max_date = self.transactions_df['purchase_date'].max()
            recent_date = max_date - timedelta(days=30)
            recent_transactions = self.transactions_df[self.transactions_df['purchase_date'] >= recent_date]
            
            # Calculate trending metrics
            self.trending_products = recent_transactions.groupby('product_id').agg({
                'transaction_id': 'count',
                'price': 'sum',
                'purchase_date': 'count'
            }).reset_index()
            
            self.trending_products.columns = ['product_id', 'recent_sales', 'recent_revenue', 'recent_transactions']
            
            # Calculate trend score
            max_sales = self.trending_products['recent_sales'].max() if len(self.trending_products) > 0 else 1
            max_revenue = self.trending_products['recent_revenue'].max() if len(self.trending_products) > 0 else 1
            
            self.trending_products['trend_score'] = (
                (self.trending_products['recent_sales'] / max_sales) * 0.6 +
                (self.trending_products['recent_revenue'] / max_revenue) * 0.4
            )
            
            print(f"✅ Analyzed {len(self.trending_products)} trending products")
            
        except Exception as e:
            print(f"⚠️  Error analyzing trending products: {e}")
            self.trending_products = pd.DataFrame()
    
    def _extract_cluster_characteristics(self):
        """Extract characteristics of customer clusters"""
        if self.segmentation_df is None or self.transactions_df is None:
            return
        
        try:
            # Merge segmentation with transaction data
            transaction_segments = self.transactions_df.merge(
                self.segmentation_df[['customer_id', 'segment']], 
                on='customer_id', 
                how='left'
            )
            
            # Analyze cluster characteristics
            self.cluster_characteristics = transaction_segments.groupby('segment').agg({
                'price': ['mean', 'std', 'min', 'max'],
                'customer_id': 'nunique',
                'transaction_id': 'count'
            }).round(2)
            
            # Flatten column names
            self.cluster_characteristics.columns = [
                'avg_price', 'price_std', 'min_price', 'max_price', 
                'unique_customers', 'total_transactions'
            ]
            
            # Calculate cluster preference scores
            cluster_categories = transaction_segments.groupby(['segment', 'category']).size().unstack(fill_value=0)
            cluster_categories_pct = cluster_categories.div(cluster_categories.sum(axis=1), axis=0)
            
            self.cluster_characteristics['top_category'] = cluster_categories_pct.idxmax(axis=1)
            self.cluster_characteristics['category_concentration'] = cluster_categories_pct.max(axis=1)
            
            print(f"✅ Extracted characteristics for {len(self.cluster_characteristics)} clusters")
            
        except Exception as e:
            print(f"⚠️  Error extracting cluster characteristics: {e}")
            self.cluster_characteristics = pd.DataFrame()
    
    def create_customer_profiles(self):
        """Create detailed customer profiles for content-based filtering"""
        print("🔧 Creating customer profiles...")
        
        if self.transactions_df is None or self.products_df is None:
            print("❌ Missing transaction or product data")
            return False
        
        try:
            # Merge transactions with product data
            enriched_transactions = self.transactions_df.merge(
                self.products_df[['product_id', 'category', 'price']], 
                on='product_id', 
                how='left'
            )
            
            # Calculate customer preferences
            customer_profiles = []
            
            for customer_id in self.customers_df['customer_id']:
                customer_transactions = enriched_transactions[
                    enriched_transactions['customer_id'] == customer_id
                ]
                
                if len(customer_transactions) == 0:
                    # New customer - use default profile
                    profile = {
                        'customer_id': customer_id,
                        'num_purchases': 0,
                        'avg_price': 0,
                        'preferred_categories': {},
                        'price_sensitivity': 'medium',
                        'category_diversity': 0
                    }
                else:
                    # Calculate category preferences
                    category_counts = customer_transactions['category'].value_counts()
                    total_purchases = len(customer_transactions)
                    category_prefs = (category_counts / total_purchases).to_dict()
                    
                    # Calculate price preferences
                    avg_price = customer_transactions['price'].mean()
                    price_std = customer_transactions['price'].std()
                    
                    # Determine price sensitivity
                    if avg_price < customer_transactions['price'].quantile(0.33):
                        price_sensitivity = 'high'  # Price sensitive
                    elif avg_price > customer_transactions['price'].quantile(0.66):
                        price_sensitivity = 'low'   # Premium buyer
                    else:
                        price_sensitivity = 'medium'
                    
                    # Calculate category diversity
                    category_diversity = len(category_counts) / total_purchases
                    
                    profile = {
                        'customer_id': customer_id,
                        'num_purchases': total_purchases,
                        'avg_price': avg_price,
                        'price_std': price_std if not pd.isna(price_std) else 0,
                        'preferred_categories': category_prefs,
                        'price_sensitivity': price_sensitivity,
                        'category_diversity': category_diversity
                    }
                
                customer_profiles.append(profile)
            
            self.customer_profiles = pd.DataFrame(customer_profiles)
            print(f"✅ Created profiles for {len(self.customer_profiles)} customers")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating customer profiles: {e}")
            return False
    
    def get_content_based_scores(self, customer_id, candidate_products):
        """Calculate content-based scores for candidate products"""
        if self.customer_profiles is None:
            self.create_customer_profiles()
        
        # Get customer profile
        customer_profile = self.customer_profiles[
            self.customer_profiles['customer_id'] == customer_id
        ]
        
        if len(customer_profile) == 0:
            return {}
        
        profile = customer_profile.iloc[0]
        scores = {}
        
        for product_id in candidate_products:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            
            if len(product_info) == 0:
                continue
            
            product = product_info.iloc[0]
            score = 0
            
            # Category preference score
            category = product['category']
            if category in profile['preferred_categories']:
                score += profile['preferred_categories'][category] * 0.6
            
            # Price preference score
            if profile['num_purchases'] > 0:
                price_diff = abs(product['price'] - profile['avg_price'])
                max_price_diff = profile['avg_price'] + 2 * profile['price_std'] if profile['price_std'] > 0 else profile['avg_price']
                
                if max_price_diff > 0:
                    price_score = max(0, 1 - (price_diff / max_price_diff))
                    score += price_score * 0.4
            else:
                # For new customers, neutral price score
                score += 0.5 * 0.4
            
            scores[product_id] = score
        
        return scores
    
    def get_collaborative_scores(self, customer_id, candidate_products, use_cluster=True):
        """Calculate collaborative filtering scores with cluster awareness"""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        scores = {}
        
        # Get customer's segment if available
        customer_segment = None
        if self.segmentation_df is not None and use_cluster:
            segment_info = self.segmentation_df[
                self.segmentation_df['customer_id'] == customer_id
            ]
            if len(segment_info) > 0:
                customer_segment = segment_info.iloc[0]['segment']
        
        if customer_id not in self.user_item_matrix.index:
            return self._get_segment_based_scores(customer_segment, candidate_products)
        
        # Get user's purchase history
        user_ratings = self.user_item_matrix.loc[customer_id]
        purchased_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not purchased_items:
            return self._get_segment_based_scores(customer_segment, candidate_products)
        
        # Item-based collaborative filtering
        if self.item_similarity_matrix is not None:
            for product_id in candidate_products:
                if product_id not in purchased_items:
                    score = 0
                    similarity_sum = 0
                    
                    for purchased_item in purchased_items:
                        if (product_id in self.item_similarity_matrix.index and 
                            purchased_item in self.item_similarity_matrix.columns):
                            
                            similarity = self.item_similarity_matrix.loc[product_id, purchased_item]
                            user_rating = user_ratings[purchased_item]
                            
                            score += similarity * user_rating
                            similarity_sum += abs(similarity)
                    
                    if similarity_sum > 0:
                        base_score = score / similarity_sum
                        
                        # Apply cluster boost if same segment customers like this product
                        if customer_segment is not None:
                            cluster_boost = self._get_cluster_boost(product_id, customer_segment)
                            base_score *= (1 + cluster_boost)
                        
                        scores[product_id] = base_score
        
        return scores
    
    def _get_segment_based_scores(self, segment, candidate_products):
        """Get scores based on segment popularity for new/cold customers"""
        if segment is None or self.segmentation_df is None:
            return self._get_popularity_scores(candidate_products)
        
        # Get customers in the same segment
        segment_customers = self.segmentation_df[
            self.segmentation_df['segment'] == segment
        ]['customer_id'].tolist()
        
        if not segment_customers:
            return self._get_popularity_scores(candidate_products)
        
        # Get popular products in this segment
        segment_transactions = self.transactions_df[
            self.transactions_df['customer_id'].isin(segment_customers)
        ]
        
        product_popularity = segment_transactions['product_id'].value_counts()
        max_popularity = product_popularity.max() if len(product_popularity) > 0 else 1
        
        scores = {}
        for product_id in candidate_products:
            popularity = product_popularity.get(product_id, 0)
            scores[product_id] = popularity / max_popularity
        
        return scores
    
    def _get_popularity_scores(self, candidate_products):
        """Get global popularity scores as fallback"""
        if self.trending_products is None or len(self.trending_products) == 0:
            return {product_id: 0.5 for product_id in candidate_products}
        
        scores = {}
        for product_id in candidate_products:
            trend_info = self.trending_products[
                self.trending_products['product_id'] == product_id
            ]
            
            if len(trend_info) > 0:
                scores[product_id] = trend_info.iloc[0]['trend_score']
            else:
                scores[product_id] = 0.3  # Default score for non-trending items
        
        return scores
    
    def _get_cluster_boost(self, product_id, segment):
        """Calculate cluster-specific boost for products"""
        if self.segmentation_df is None:
            return 0
        
        # Get segment customers
        segment_customers = self.segmentation_df[
            self.segmentation_df['segment'] == segment
        ]['customer_id'].tolist()
        
        if not segment_customers:
            return 0
        
        # Calculate product popularity in segment vs overall
        segment_transactions = self.transactions_df[
            self.transactions_df['customer_id'].isin(segment_customers)
        ]
        
        segment_product_count = len(segment_transactions[
            segment_transactions['product_id'] == product_id
        ])
        
        total_segment_transactions = len(segment_transactions)
        total_product_count = len(self.transactions_df[
            self.transactions_df['product_id'] == product_id
        ])
        total_transactions = len(self.transactions_df)
        
        if total_segment_transactions == 0 or total_transactions == 0:
            return 0
        
        # Calculate relative popularity
        segment_popularity = segment_product_count / total_segment_transactions
        overall_popularity = total_product_count / total_transactions
        
        if overall_popularity == 0:
            return 0
        
        # Boost if more popular in segment than overall
        boost = max(0, (segment_popularity / overall_popularity - 1) * 0.3)
        return min(boost, 0.5)  # Cap boost at 50%
    
    def generate_hybrid_recommendations(self, customer_id, n_recommendations=3):
        """Generate exactly 3 product recommendations using hybrid approach"""
        print(f"🎯 Generating recommendations for customer {customer_id}")
        
        try:
            # Initialize matrices if needed
            if self.user_item_matrix is None:
                self.create_user_item_matrix()
            if self.item_similarity_matrix is None:
                self.calculate_item_similarity()
            
            # Get customer's purchase history
            if customer_id in self.user_item_matrix.index:
                user_ratings = self.user_item_matrix.loc[customer_id]
                purchased_items = set(user_ratings[user_ratings > 0].index.tolist())
            else:
                purchased_items = set()
            
            # Get all available products excluding purchased ones
            all_products = set(self.products_df['product_id'].tolist())
            candidate_products = list(all_products - purchased_items)
            
            if len(candidate_products) == 0:
                print(f"⚠️  No new products to recommend for customer {customer_id}")
                return self._get_popular_recommendations_formatted(n_recommendations)
            
            # Get customer segment for targeted weighting
            customer_segment = self._get_customer_segment(customer_id)
            segment_weights = self._get_segment_weights(customer_segment)
            
            # Calculate scores from different approaches
            content_scores = self.get_content_based_scores(customer_id, candidate_products)
            collaborative_scores = self.get_collaborative_scores(customer_id, candidate_products)
            
            # Combine scores with segment-specific weights
            final_scores = {}
            
            for product_id in candidate_products:
                content_score = content_scores.get(product_id, 0)
                collaborative_score = collaborative_scores.get(product_id, 0)
                
                # Apply segment-specific weighting
                combined_score = (
                    content_score * segment_weights['content'] +
                    collaborative_score * segment_weights['collaborative']
                )
                
                # Apply business rules
                combined_score = self._apply_business_rules(
                    product_id, combined_score, customer_segment
                )
                
                if combined_score >= self.min_confidence_threshold:
                    final_scores[product_id] = combined_score
            
            if not final_scores:
                print(f"⚠️  No products meet confidence threshold for customer {customer_id}")
                return self._get_popular_recommendations_formatted(n_recommendations)
            
            # Ensure diversity in recommendations
            recommendations = self._select_diverse_recommendations(
                final_scores, n_recommendations
            )
            
            # Format recommendations with confidence scores
            formatted_recommendations = self._format_recommendations(
                recommendations, customer_id, customer_segment
            )
            
            print(f"✅ Generated {len(formatted_recommendations)} recommendations")
            return formatted_recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return self._get_popular_recommendations_formatted(n_recommendations)
    
    def _get_customer_segment(self, customer_id):
        """Get customer's segment"""
        if self.segmentation_df is not None:
            segment_info = self.segmentation_df[
                self.segmentation_df['customer_id'] == customer_id
            ]
            if len(segment_info) > 0:
                return segment_info.iloc[0]['segment']
        return 'unknown'
    
    def _get_segment_weights(self, segment):
        """Get recommendation weights based on customer segment"""
        segment_weights = {
            'Champions': {'content': 0.3, 'collaborative': 0.7},
            'Loyal Customers': {'content': 0.4, 'collaborative': 0.6},
            'At-Risk': {'content': 0.6, 'collaborative': 0.4},
            'New Customers': {'content': 0.7, 'collaborative': 0.3},
            'Need Attention': {'content': 0.5, 'collaborative': 0.5},
            'Lost': {'content': 0.8, 'collaborative': 0.2}
        }
        
        return segment_weights.get(segment, {'content': 0.5, 'collaborative': 0.5})
    
    def _apply_business_rules(self, product_id, base_score, customer_segment):
        """Apply business rules to recommendation scores"""
        # Get product info
        product_info = self.products_df[self.products_df['product_id'] == product_id]
        if len(product_info) == 0:
            return base_score
        
        product = product_info.iloc[0]
        adjusted_score = base_score
        
        # Apply segment-specific rules
        if customer_segment == 'Champions':
            # Champions get premium/new product boost
            if product['price'] > self.products_df['price'].quantile(0.8):
                adjusted_score *= 1.2
        
        elif customer_segment == 'At-Risk' or customer_segment == 'Need Attention':
            # Price-sensitive customers get value product boost
            if product['price'] < self.products_df['price'].quantile(0.4):
                adjusted_score *= 1.3
        
        elif customer_segment == 'Loyal Customers':
            # Loyal customers get brand extension boost (same category preference)
            # This would require brand information - using category as proxy
            adjusted_score *= 1.1
        
        # Apply trending product boost
        if self.trending_products is not None and len(self.trending_products) > 0:
            trend_info = self.trending_products[
                self.trending_products['product_id'] == product_id
            ]
            if len(trend_info) > 0:
                trend_score = trend_info.iloc[0]['trend_score']
                adjusted_score *= (1 + trend_score * self.popularity_boost)
        
        # Apply seasonal adjustment (simplified - would need actual seasonality data)
        current_month = datetime.now().month
        if current_month in [11, 12]:  # Holiday season
            if product['category'] in ['Electronics', 'Home']:
                adjusted_score *= (1 + self.seasonal_adjustment)
        
        # Simulate inventory consideration (random availability)
        import random
        availability = random.random()
        if availability < 0.1:  # 10% chance of low inventory
            adjusted_score *= 0.7  # Reduce score for low inventory items
        
        return adjusted_score
    
    def _select_diverse_recommendations(self, scores, n_recommendations):
        """Select diverse recommendations ensuring different categories when possible"""
        if len(scores) == 0:
            return []
        
        # Sort by score
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_products) <= n_recommendations:
            return [product_id for product_id, _ in sorted_products]
        
        # Ensure diversity by category
        selected = []
        used_categories = set()
        
        # First pass: select highest scoring items from different categories
        for product_id, score in sorted_products:
            if len(selected) >= n_recommendations:
                break
            
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if len(product_info) > 0:
                category = product_info.iloc[0]['category']
                
                if category not in used_categories or len(selected) == 0:
                    selected.append(product_id)
                    used_categories.add(category)
        
        # Second pass: fill remaining spots with highest scores
        if len(selected) < n_recommendations:
            for product_id, score in sorted_products:
                if len(selected) >= n_recommendations:
                    break
                
                if product_id not in selected:
                    selected.append(product_id)
        
        return selected[:n_recommendations]
    
    def _format_recommendations(self, product_ids, customer_id, customer_segment):
        """Format recommendations with detailed information and confidence scores"""
        recommendations = []
        
        for i, product_id in enumerate(product_ids, 1):
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            
            if len(product_info) == 0:
                continue
            
            product = product_info.iloc[0]
            
            # Calculate confidence score (simplified)
            content_scores = self.get_content_based_scores(customer_id, [product_id])
            collaborative_scores = self.get_collaborative_scores(customer_id, [product_id])
            
            content_score = content_scores.get(product_id, 0)
            collaborative_score = collaborative_scores.get(product_id, 0)
            
            # Weighted combination
            segment_weights = self._get_segment_weights(customer_segment)
            confidence = (
                content_score * segment_weights['content'] +
                collaborative_score * segment_weights['collaborative']
            )
            
            # Normalize confidence to 0-1 range
            confidence = min(1.0, max(0.0, confidence))
            
            # Generate explanation
            explanation = self._generate_explanation(
                product, content_score, collaborative_score, customer_segment
            )
            
            recommendation = {
                'rank': i,
                'product_id': product_id,
                'product_name': product['product_name'],
                'category': product['category'],
                'price': product['price'],
                'confidence_score': round(confidence, 3),
                'explanation': explanation,
                'customer_segment': customer_segment
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_explanation(self, product, content_score, collaborative_score, segment):
        """Generate explanation for why product was recommended"""
        explanations = []
        
        if content_score > collaborative_score:
            explanations.append(f"Matches your preferences in {product['category']}")
        else:
            explanations.append(f"Popular among similar customers")
        
        if segment == 'Champions':
            explanations.append("Premium quality product")
        elif segment in ['At-Risk', 'Need Attention']:
            explanations.append("Great value option")
        elif segment == 'Loyal Customers':
            explanations.append("From your preferred category")
        
        # Add trending explanation if applicable
        if self.trending_products is not None and len(self.trending_products) > 0:
            trend_info = self.trending_products[
                self.trending_products['product_id'] == product['product_id']
            ]
            if len(trend_info) > 0 and trend_info.iloc[0]['trend_score'] > 0.7:
                explanations.append("Currently trending")
        
        return " • ".join(explanations)
    
    def _get_popular_recommendations_formatted(self, n_recommendations):
        """Get popular recommendations as fallback, formatted consistently"""
        try:
            # Calculate global popularity
            popularity = self.transactions_df.groupby('product_id').agg({
                'transaction_id': 'count',
                'price': 'sum'
            }).reset_index()
            
            popularity.columns = ['product_id', 'purchase_count', 'total_revenue']
            
            # Calculate popularity score
            popularity['popularity_score'] = (
                popularity['purchase_count'] / popularity['purchase_count'].max() * 0.7 +
                popularity['total_revenue'] / popularity['total_revenue'].max() * 0.3
            )
            
            # Get top products
            top_products = popularity.nlargest(n_recommendations, 'popularity_score')
            
            # Format as recommendations
            recommendations = []
            for i, (_, row) in enumerate(top_products.iterrows(), 1):
                product_info = self.products_df[
                    self.products_df['product_id'] == row['product_id']
                ]
                
                if len(product_info) > 0:
                    product = product_info.iloc[0]
                    
                    recommendation = {
                        'rank': i,
                        'product_id': row['product_id'],
                        'product_name': product['product_name'],
                        'category': product['category'],
                        'price': product['price'],
                        'confidence_score': round(row['popularity_score'], 3),
                        'explanation': "Popular choice • Frequently purchased",
                        'customer_segment': 'unknown'
                    }
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating popular recommendations: {e}")
            return []
    
    def calculate_performance_metrics(self, sample_customers=None, n_recommendations=3):
        """Calculate comprehensive performance metrics for the recommendation system"""
        print("\n📊 CALCULATING RECOMMENDATION PERFORMANCE METRICS")
        print("=" * 60)
        
        try:
            if sample_customers is None:
                # Sample customers from different segments if possible
                if self.segmentation_df is not None:
                    sample_customers = []
                    for segment in self.segmentation_df['segment'].unique():
                        segment_customers = self.segmentation_df[
                            self.segmentation_df['segment'] == segment
                        ]['customer_id'].head(5).tolist()
                        sample_customers.extend(segment_customers)
                else:
                    sample_customers = self.customers_df['customer_id'].head(20).tolist()
            
            metrics = {
                'recommendation_coverage': 0,
                'diversity_score': 0,
                'cluster_patterns': {},
                'confidence_distribution': [],
                'category_distribution': {},
                'successful_recommendations': 0,
                'total_attempts': 0
            }
            
            all_recommended_products = set()
            all_recommendations = []
            cluster_recommendations = defaultdict(list)
            
            print(f"🔍 Analyzing {len(sample_customers)} sample customers...")
            
            for customer_id in sample_customers:
                try:
                    recommendations = self.generate_hybrid_recommendations(
                        customer_id, n_recommendations
                    )
                    
                    if recommendations:
                        metrics['successful_recommendations'] += 1
                        
                        for rec in recommendations:
                            all_recommended_products.add(rec['product_id'])
                            all_recommendations.append(rec)
                            
                            # Track confidence scores
                            metrics['confidence_distribution'].append(rec['confidence_score'])
                            
                            # Track category distribution
                            category = rec['category']
                            metrics['category_distribution'][category] = \
                                metrics['category_distribution'].get(category, 0) + 1
                            
                            # Track cluster patterns
                            segment = rec['customer_segment']
                            cluster_recommendations[segment].append(rec)
                    
                    metrics['total_attempts'] += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing customer {customer_id}: {e}")
                    continue
            
            # Calculate coverage
            total_products = len(self.products_df)
            metrics['recommendation_coverage'] = len(all_recommended_products) / total_products * 100
            
            # Calculate diversity
            metrics['diversity_score'] = self._calculate_diversity_score(all_recommendations)
            
            # Analyze cluster patterns
            for segment, recs in cluster_recommendations.items():
                if recs:
                    avg_confidence = np.mean([r['confidence_score'] for r in recs])
                    avg_price = np.mean([r['price'] for r in recs])
                    categories = [r['category'] for r in recs]
                    top_category = Counter(categories).most_common(1)[0] if categories else ('None', 0)
                    
                    metrics['cluster_patterns'][segment] = {
                        'avg_confidence': round(avg_confidence, 3),
                        'avg_price': round(avg_price, 2),
                        'top_category': top_category[0],
                        'category_focus': round(top_category[1] / len(categories) * 100, 1) if categories else 0,
                        'recommendation_count': len(recs)
                    }
            
            # Print detailed metrics
            self._print_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_diversity_score(self, recommendations):
        """Calculate diversity score based on category and price spread"""
        if not recommendations:
            return 0
        
        # Category diversity
        categories = [rec['category'] for rec in recommendations]
        unique_categories = len(set(categories))
        total_categories = len(self.products_df['category'].unique())
        category_diversity = unique_categories / total_categories
        
        # Price diversity (coefficient of variation)
        prices = [rec['price'] for rec in recommendations]
        if len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            price_diversity = price_std / price_mean if price_mean > 0 else 0
        else:
            price_diversity = 0
        
        # Combined diversity score (0-1 scale)
        diversity_score = (category_diversity * 0.7 + min(price_diversity, 1) * 0.3)
        return round(diversity_score, 3)
    
    def _print_performance_metrics(self, metrics):
        """Print detailed performance metrics"""
        print(f"📈 PERFORMANCE METRICS SUMMARY:")
        print(f"   Recommendation Coverage: {metrics['recommendation_coverage']:.1f}% of catalog")
        print(f"   Diversity Score: {metrics['diversity_score']:.3f}/1.0")
        print(f"   Success Rate: {metrics['successful_recommendations']}/{metrics['total_attempts']} customers")
        
        if metrics['confidence_distribution']:
            avg_confidence = np.mean(metrics['confidence_distribution'])
            min_confidence = min(metrics['confidence_distribution'])
            max_confidence = max(metrics['confidence_distribution'])
            print(f"   Confidence Scores: Avg={avg_confidence:.3f}, Min={min_confidence:.3f}, Max={max_confidence:.3f}")
        
        print(f"\n🎯 CATEGORY DISTRIBUTION:")
        total_recs = sum(metrics['category_distribution'].values())
        for category, count in sorted(metrics['category_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_recs * 100 if total_recs > 0 else 0
            print(f"   {category}: {count} recommendations ({percentage:.1f}%)")
        
        print(f"\n🏷️  CLUSTER PATTERNS:")
        for segment, pattern in metrics['cluster_patterns'].items():
            print(f"   {segment}:")
            print(f"      Avg Confidence: {pattern['avg_confidence']}")
            print(f"      Avg Price: ${pattern['avg_price']:,.2f}")
            print(f"      Top Category: {pattern['top_category']} ({pattern['category_focus']}%)")
            print(f"      Total Recommendations: {pattern['recommendation_count']}")
    
    def test_recommendation_quality(self, test_customers=None, n_test=10):
        """Test recommendation quality with sample customers"""
        print("\n🧪 TESTING RECOMMENDATION QUALITY")
        print("=" * 50)
        
        if test_customers is None:
            test_customers = self.customers_df['customer_id'].sample(n_test).tolist()
        
        test_results = []
        
        for customer_id in test_customers:
            print(f"\n👤 Testing Customer: {customer_id}")
            
            # Get customer info
            customer_info = self.customers_df[self.customers_df['customer_id'] == customer_id]
            if len(customer_info) > 0:
                customer_name = customer_info.iloc[0]['name']
                print(f"   Name: {customer_name}")
            
            # Get customer segment
            segment = self._get_customer_segment(customer_id)
            print(f"   Segment: {segment}")
            
            # Get purchase history
            customer_purchases = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ]
            if len(customer_purchases) > 0:
                purchase_summary = customer_purchases.merge(
                    self.products_df[['product_id', 'category']], 
                    on='product_id'
                )
                unique_categories = purchase_summary['category'].nunique()
                total_spent = customer_purchases['price'].sum()
                print(f"   Purchase History: {len(customer_purchases)} transactions, "
                      f"{unique_categories} categories, ${total_spent:,.2f} total")
            else:
                print(f"   Purchase History: New customer (no purchases)")
            
            # Generate recommendations
            recommendations = self.generate_hybrid_recommendations(customer_id, 3)
            
            print(f"   📦 Recommendations:")
            business_sense_score = 0
            
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec['product_name']} ({rec['category']}) - ${rec['price']:,.2f}")
                print(f"         Confidence: {rec['confidence_score']:.3f}")
                print(f"         Reason: {rec['explanation']}")
                
                # Simple business sense check
                if rec['confidence_score'] > 0.3:
                    business_sense_score += 1
                if segment == 'Champions' and rec['price'] > self.products_df['price'].median():
                    business_sense_score += 0.5
                elif segment in ['At-Risk', 'Need Attention'] and rec['price'] < self.products_df['price'].median():
                    business_sense_score += 0.5
            
            test_result = {
                'customer_id': customer_id,
                'segment': segment,
                'num_recommendations': len(recommendations),
                'avg_confidence': np.mean([r['confidence_score'] for r in recommendations]) if recommendations else 0,
                'business_sense_score': business_sense_score / 3,  # Normalize to 0-1
                'recommendations': recommendations
            }
            
            test_results.append(test_result)
            print(f"   ✅ Business Sense Score: {business_sense_score/3:.2f}/1.0")
        
        # Summary
        print(f"\n📊 TEST SUMMARY:")
        avg_confidence = np.mean([r['avg_confidence'] for r in test_results])
        avg_business_sense = np.mean([r['business_sense_score'] for r in test_results])
        success_rate = len([r for r in test_results if r['num_recommendations'] == 3]) / len(test_results)
        
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Business Sense Score: {avg_business_sense:.3f}")
        print(f"   Success Rate (3 recommendations): {success_rate:.1%}")
        
        return test_results 