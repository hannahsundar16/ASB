import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

class SyntheticStockDatasetGenerator:
    def __init__(self):
        self.stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'SPOT', 'ZOOM', 'SNOW', 'PLTR', 'RBLX']
        self.first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Jessica', 'William', 'Ashley', 'James', 'Amanda', 'Christopher', 'Jennifer', 'Daniel', 'Lisa', 'Matthew', 'Karen', 'Anthony', 'Nancy', 'Mark', 'Betty', 'Donald', 'Helen', 'Steven', 'Sandra', 'Paul', 'Donna', 'Andrew', 'Carol']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson']
        self.cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington']
        self.states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA', 'TX', 'FL', 'TX', 'OH', 'NC', 'CA', 'IN', 'WA', 'CO', 'DC']
        self.brokerages = ['Fidelity', 'Charles Schwab', 'E*TRADE', 'TD Ameritrade', 'Robinhood', 'Interactive Brokers', 'Merrill Edge', 'Vanguard', 'Ally Invest', 'Webull']
        self.email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
        self.street_names = ['Main St', 'Oak Ave', 'Pine Rd', 'Elm Dr', 'Maple Way', 'Cedar Ln', 'Park Ave', 'First St', 'Second St', 'Broadway']
        
        self.company_map = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'ORCL': 'Oracle Corporation',
            'INTC': 'Intel Corporation',
            'AMD': 'Advanced Micro Devices',
            'PYPL': 'PayPal Holdings Inc.',
            'UBER': 'Uber Technologies Inc.',
            'SPOT': 'Spotify Technology SA',
            'ZOOM': 'Zoom Video Communications',
            'SNOW': 'Snowflake Inc.',
            'PLTR': 'Palantir Technologies',
            'RBLX': 'Roblox Corporation'
        }

    def generate_dataset(self, num_records=100, date_range_days=30):
        """Generate synthetic stock dataset with PII"""
        dataset = []
        today = datetime.now()
        
        print(f"Generating {num_records} records...")
        
        for i in range(num_records):
            # Generate random date within range
            random_days = random.randint(0, date_range_days)
            trade_date = today - timedelta(days=random_days)
            
            # Generate random person
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            city_index = random.randint(0, len(self.cities) - 1)
            
            # Generate random stock data
            symbol = random.choice(self.stock_symbols)
            base_price = round(random.uniform(50, 550), 2)
            quantity = random.randint(1, 1000)
            transaction_type = random.choice(['BUY', 'SELL'])
            
            # Generate trade time during market hours
            trade_hour = random.randint(9, 15)
            trade_minute = random.randint(0, 59)
            trade_second = random.randint(0, 59)
            
            record = {
                'id': f"TXN_{str(i + 1).zfill(6)}",
                # PII Data
                'first_name': first_name,
                'last_name': last_name,
                'email': f"{first_name.lower()}.{last_name.lower()}@{random.choice(self.email_domains)}",
                'phone': f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                'address': f"{random.randint(1, 9999)} {random.choice(self.street_names)}",
                'city': self.cities[city_index],
                'state': self.states[city_index],
                'zip_code': str(random.randint(10000, 99999)),
                'ssn': f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
                'date_of_birth': (datetime(1950, 1, 1) + timedelta(days=random.randint(0, 365*50))).strftime('%Y-%m-%d'),
                'account_number': f"ACC_{random.randint(1000000, 9999999)}",
                'brokerage': random.choice(self.brokerages),
                
                # Stock Transaction Data
                'symbol': symbol,
                'company_name': self.company_map.get(symbol, 'Unknown Company'),
                'transaction_type': transaction_type,
                'quantity': quantity,
                'price': base_price,
                'total_value': round(base_price * quantity, 2),
                'commission': round(random.uniform(0.99, 10.99), 2),
                'trade_date': trade_date.strftime('%Y-%m-%d'),
                'trade_time': f"{trade_hour:02d}:{trade_minute:02d}:{trade_second:02d}",
                
                # Additional Financial Data
                'portfolio_value': round(random.uniform(10000, 500000), 2),
                'risk_tolerance': random.choice(['Conservative', 'Moderate', 'Aggressive']),
                'investment_goal': random.choice(['Retirement', 'Growth', 'Income', 'Speculation']),
                'annual_income': random.randint(30000, 200000),
                'net_worth': random.randint(50000, 2000000)
            }
            
            dataset.append(record)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} records...")
        
        return dataset

    def save_to_csv(self, dataset, filename=None):
        """Save dataset to CSV file"""
        if filename is None:
            filename = f"synthetic_stock_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(dataset)
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        return filename

    def save_to_json(self, dataset, filename=None):
        """Save dataset to JSON file"""
        if filename is None:
            filename = f"synthetic_stock_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
        return filename

    def print_statistics(self, dataset):
        """Print dataset statistics"""
        if not dataset:
            print("No data to analyze")
            return
        
        df = pd.DataFrame(dataset)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total Records: {len(dataset)}")
        print(f"Unique Stocks: {df['symbol'].nunique()}")
        print(f"Unique Users: {df['email'].nunique()}")
        print(f"Date Range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        print(f"Total Transaction Volume: ${df['total_value'].sum():,.2f}")
        print(f"Average Transaction Size: ${df['total_value'].mean():,.2f}")
        print(f"Buy vs Sell Ratio: {df[df['transaction_type'] == 'BUY'].shape[0]}:{df[df['transaction_type'] == 'SELL'].shape[0]}")
        
        print("\nTop 5 Most Traded Stocks:")
        print(df['symbol'].value_counts().head())
        
        print("\nSample Records:")
        print(df[['first_name', 'last_name', 'symbol', 'transaction_type', 'quantity', 'price', 'total_value', 'trade_date']].head())

def main():
    """Main execution function"""
    generator = SyntheticStockDatasetGenerator()
    
    # Get user input
    try:
        num_records = int(input("Enter number of records to generate (default 100): ") or "100")
        date_range = int(input("Enter date range in days (default 30): ") or "30")
    except ValueError:
        print("Invalid input. Using default values.")
        num_records = 100
        date_range = 30
    
    # Generate dataset
    dataset = generator.generate_dataset(num_records, date_range)
    
    # Print statistics
    generator.print_statistics(dataset)
    
    # Save options
    print("\nSave options:")
    print("1. CSV only")
    print("2. JSON only") 
    print("3. Both CSV and JSON")
    print("4. Don't save")
    
    choice = input("Enter your choice (1-4, default 3): ") or "3"
    
    if choice in ['1', '3']:
        generator.save_to_csv(dataset)
    if choice in ['2', '3']:
        generator.save_to_json(dataset)
    if choice == '4':
        print("Data generated but not saved.")
    
    print("This is synthetic data for testing purposes only!")
    print("All PII data is completely fictional and does not correspond to real individuals.")

if __name__ == "__main__":
    main()