import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class DataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime.now()
        
        # Product categories and sample names
        self.categories = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smart Watch', 
                           'Camera', 'Gaming Console', 'Smart TV', 'Wireless Speaker', 'Drone'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sneakers', 
                        'Sweater', 'Shorts', 'Blouse', 'Boots', 'Coat'],
            'Books': ['Fiction Novel', 'Science Book', 'Biography', 'Cookbook', 'Self-Help', 
                     'Mystery Novel', 'History Book', 'Art Book', 'Technical Manual', 'Poetry'],
            'Home': ['Coffee Maker', 'Vacuum Cleaner', 'Bed Sheets', 'Kitchen Knife Set', 'Lamp', 
                    'Pillow', 'Curtains', 'Storage Box', 'Plant Pot', 'Candle'],
            'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Football', 
                      'Water Bottle', 'Fitness Tracker', 'Bicycle', 'Swimming Goggles', 'Golf Club']
        }
        
        # Price ranges by category
        self.price_ranges = {
            'Electronics': (50, 2000),
            'Clothing': (20, 300),
            'Books': (10, 50),
            'Home': (15, 500),
            'Sports': (25, 800)
        }
    
    def generate_customers(self, num_customers=1000):
        """Generate realistic customer data"""
        customers = []
        
        for i in range(num_customers):
            # Generate registration date (weighted towards more recent dates)
            days_back = np.random.exponential(365)  # Exponential distribution for more recent customers
            days_back = min(days_back, (self.end_date - self.start_date).days)
            registration_date = self.end_date - timedelta(days=int(days_back))
            
            customer = {
                'customer_id': f'CUST_{i+1:05d}',
                'name': self.fake.name(),
                'email': self.fake.email(),
                'registration_date': registration_date.strftime('%Y-%m-%d'),
                'age': np.random.normal(38, 12),  # Normal distribution around 38 years
                'location': self.fake.city()
            }
            
            # Ensure age is reasonable
            customer['age'] = max(18, min(80, int(customer['age'])))
            
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_products(self, num_products=500):
        """Generate realistic product data"""
        products = []
        
        for i in range(num_products):
            category = random.choice(list(self.categories.keys()))
            base_name = random.choice(self.categories[category])
            
            # Add brand/model variations
            brands = ['Premium', 'Classic', 'Pro', 'Elite', 'Standard', 'Deluxe']
            product_name = f"{random.choice(brands)} {base_name}"
            
            # Generate price within category range
            min_price, max_price = self.price_ranges[category]
            price = round(np.random.uniform(min_price, max_price), 2)
            
            product = {
                'product_id': f'PROD_{i+1:05d}',
                'product_name': product_name,
                'category': category,
                'price': price
            }
            
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_transactions(self, customers_df, products_df, num_transactions=10000):
        """Generate realistic transaction data with customer behavior patterns"""
        transactions = []
        
        # Define customer behavior segments
        num_customers = len(customers_df)
        
        # Frequent buyers (20% of customers, 60% of transactions)
        frequent_customers = customers_df.sample(int(num_customers * 0.2))['customer_id'].tolist()
        frequent_transactions = int(num_transactions * 0.6)
        
        # Seasonal buyers (30% of customers, 25% of transactions)
        seasonal_customers = customers_df[~customers_df['customer_id'].isin(frequent_customers)].sample(int(num_customers * 0.3))['customer_id'].tolist()
        seasonal_transactions = int(num_transactions * 0.25)
        
        # One-time purchasers (50% of customers, 15% of transactions)
        remaining_customers = customers_df[~customers_df['customer_id'].isin(frequent_customers + seasonal_customers)]['customer_id'].tolist()
        onetime_transactions = num_transactions - frequent_transactions - seasonal_transactions
        
        transaction_id = 1
        
        # Generate frequent buyer transactions
        for _ in range(frequent_transactions):
            customer_id = random.choice(frequent_customers)
            customer_reg_date = customers_df[customers_df['customer_id'] == customer_id]['registration_date'].iloc[0]
            customer_reg_date = datetime.strptime(customer_reg_date, '%Y-%m-%d')
            
            # Purchase date after registration
            days_since_reg = (self.end_date - customer_reg_date).days
            if days_since_reg > 0:
                # Frequent buyers shop regularly
                purchase_date = customer_reg_date + timedelta(days=random.randint(0, days_since_reg))
            else:
                purchase_date = customer_reg_date
            
            # Frequent buyers tend to buy more items and higher-value products
            product = products_df.sample(1).iloc[0]
            quantity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            
            # Higher chance of premium products for frequent buyers
            if random.random() < 0.3:  # 30% chance to buy premium
                premium_products = products_df[products_df['price'] > products_df['price'].quantile(0.7)]
                if not premium_products.empty:
                    product = premium_products.sample(1).iloc[0]
            
            transaction = {
                'transaction_id': f'TXN_{transaction_id:07d}',
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'price': product['price']
            }
            
            transactions.append(transaction)
            transaction_id += 1
        
        # Generate seasonal buyer transactions
        for _ in range(seasonal_transactions):
            customer_id = random.choice(seasonal_customers)
            customer_reg_date = customers_df[customers_df['customer_id'] == customer_id]['registration_date'].iloc[0]
            customer_reg_date = datetime.strptime(customer_reg_date, '%Y-%m-%d')
            
            # Seasonal buyers shop during specific periods (holidays, sales)
            # Higher probability during Nov-Dec and summer months
            if random.random() < 0.4:  # 40% chance for holiday season
                purchase_month = random.choice([11, 12])  # Nov-Dec
                purchase_year = random.choice([2022, 2023, 2024])
                try:
                    purchase_date = datetime(purchase_year, purchase_month, random.randint(1, 28))
                except:
                    purchase_date = datetime(purchase_year, purchase_month, 15)
            else:
                days_since_reg = (self.end_date - customer_reg_date).days
                if days_since_reg > 0:
                    purchase_date = customer_reg_date + timedelta(days=random.randint(0, days_since_reg))
                else:
                    purchase_date = customer_reg_date
            
            product = products_df.sample(1).iloc[0]
            quantity = np.random.choice([1, 2], p=[0.8, 0.2])
            
            transaction = {
                'transaction_id': f'TXN_{transaction_id:07d}',
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'price': product['price']
            }
            
            transactions.append(transaction)
            transaction_id += 1
        
        # Generate one-time purchaser transactions
        for customer_id in remaining_customers[:onetime_transactions]:
            customer_reg_date = customers_df[customers_df['customer_id'] == customer_id]['registration_date'].iloc[0]
            customer_reg_date = datetime.strptime(customer_reg_date, '%Y-%m-%d')
            
            # One-time buyers purchase soon after registration
            days_since_reg = min(30, (self.end_date - customer_reg_date).days)
            if days_since_reg > 0:
                purchase_date = customer_reg_date + timedelta(days=random.randint(0, days_since_reg))
            else:
                purchase_date = customer_reg_date
            
            # One-time buyers tend to buy cheaper products
            budget_products = products_df[products_df['price'] < products_df['price'].quantile(0.5)]
            product = budget_products.sample(1).iloc[0]
            quantity = 1
            
            transaction = {
                'transaction_id': f'TXN_{transaction_id:07d}',
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'price': product['price']
            }
            
            transactions.append(transaction)
            transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_all_data(self, num_customers=1000, num_products=500, num_transactions=10000):
        """Generate all datasets and save as CSV files"""
        print("Generating customer data...")
        customers_df = self.generate_customers(num_customers)
        
        print("Generating product data...")
        products_df = self.generate_products(num_products)
        
        print("Generating transaction data...")
        transactions_df = self.generate_transactions(customers_df, products_df, num_transactions)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV files
        print("Saving data to CSV files...")
        customers_df.to_csv('data/customers.csv', index=False)
        products_df.to_csv('data/products.csv', index=False)
        transactions_df.to_csv('data/transactions.csv', index=False)
        
        print(f"Data generation complete!")
        print(f"Generated {len(customers_df)} customers, {len(products_df)} products, {len(transactions_df)} transactions")
        
        return customers_df, products_df, transactions_df
    
    def show_sample_data(self):
        """Display sample data from generated CSV files"""
        try:
            customers_df = pd.read_csv('data/customers.csv')
            products_df = pd.read_csv('data/products.csv')
            transactions_df = pd.read_csv('data/transactions.csv')
            
            print("\n" + "="*60)
            print("SAMPLE CUSTOMER DATA")
            print("="*60)
            print(customers_df.head(10))
            print(f"\nCustomers dataset shape: {customers_df.shape}")
            print(f"Age distribution: Mean={customers_df['age'].mean():.1f}, Std={customers_df['age'].std():.1f}")
            
            print("\n" + "="*60)
            print("SAMPLE PRODUCT DATA")
            print("="*60)
            print(products_df.head(10))
            print(f"\nProducts dataset shape: {products_df.shape}")
            print("\nProducts by category:")
            print(products_df['category'].value_counts())
            print(f"\nPrice distribution by category:")
            print(products_df.groupby('category')['price'].agg(['mean', 'min', 'max']).round(2))
            
            print("\n" + "="*60)
            print("SAMPLE TRANSACTION DATA")
            print("="*60)
            print(transactions_df.head(10))
            print(f"\nTransactions dataset shape: {transactions_df.shape}")
            
            # Data validation
            print("\n" + "="*60)
            print("DATA VALIDATION")
            print("="*60)
            
            # Check for orphaned records
            customer_ids_in_transactions = set(transactions_df['customer_id'].unique())
            customer_ids_in_customers = set(customers_df['customer_id'].unique())
            orphaned_customers = customer_ids_in_transactions - customer_ids_in_customers
            print(f"Orphaned customer IDs in transactions: {len(orphaned_customers)}")
            
            product_ids_in_transactions = set(transactions_df['product_id'].unique())
            product_ids_in_products = set(products_df['product_id'].unique())
            orphaned_products = product_ids_in_transactions - product_ids_in_products
            print(f"Orphaned product IDs in transactions: {len(orphaned_products)}")
            
            # Transaction statistics
            print(f"\nTransaction date range: {transactions_df['purchase_date'].min()} to {transactions_df['purchase_date'].max()}")
            print(f"Total revenue: ${transactions_df['price'].sum():,.2f}")
            print(f"Average transaction value: ${transactions_df['price'].mean():.2f}")
            
            # Customer behavior analysis
            customer_transaction_counts = transactions_df['customer_id'].value_counts()
            print(f"\nCustomer purchase behavior:")
            print(f"Customers with 1 purchase: {(customer_transaction_counts == 1).sum()}")
            print(f"Customers with 2-5 purchases: {((customer_transaction_counts >= 2) & (customer_transaction_counts <= 5)).sum()}")
            print(f"Customers with 6+ purchases: {(customer_transaction_counts >= 6).sum()}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run generate_all_data() first to create the CSV files.")

if __name__ == "__main__":
    # Note: You'll need to install faker: pip install faker
    try:
        from faker import Faker
    except ImportError:
        print("Warning: faker library not found. Using basic name generation.")
        print("Install faker for better data generation: pip install faker")
        
        # Simple fallback name generator
        class SimpleFaker:
            def name(self):
                first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Chris', 'Emma', 'Alex', 'Maria']
                last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
                return f"{random.choice(first_names)} {random.choice(last_names)}"
            
            def email(self):
                domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
                return f"user{random.randint(1000, 9999)}@{random.choice(domains)}"
            
            def city(self):
                cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
                return random.choice(cities)
        
        # Monkey patch the faker
        import sys
        sys.modules['faker'] = type('MockModule', (), {'Faker': SimpleFaker})()
    
    generator = DataGenerator()
    customers_df, products_df, transactions_df = generator.generate_all_data()
    generator.show_sample_data() 