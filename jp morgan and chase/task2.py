import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

class NaturalGasStorageContractPricer:
    """
    Natural Gas Storage Contract Pricing Model
    
    This class provides comprehensive pricing capabilities for natural gas storage contracts,
    incorporating all relevant cash flows, constraints, and operational considerations.
    """
    
    def __init__(self, price_data_file='Nat_Gas.csv'):
        """Initialize the pricer with historical price data"""
        self.load_price_data(price_data_file)
        self.setup_price_predictor()
        
    def load_price_data(self, filename):
        """Load and prepare historical natural gas price data"""
        self.df = pd.read_csv(filename)
        
        # Convert scientific notation to regular numbers
        self.df['Prices'] = self.df['Prices'].astype(float)
        
        # Convert dates to datetime
        self.df['Dates'] = pd.to_datetime(self.df['Dates'], format='%m/%d/%y')
        
        # Sort by date
        self.df = self.df.sort_values('Dates').reset_index(drop=True)
        
        print(f"Loaded price data: {len(self.df)} data points from {self.df['Dates'].min().date()} to {self.df['Dates'].max().date()}")
        
    def setup_price_predictor(self):
        """Setup simple price prediction based on seasonal averages"""
        # Calculate monthly seasonal averages
        self.df['Month'] = self.df['Dates'].dt.month
        self.monthly_averages = self.df.groupby('Month')['Prices'].mean()
        
        # Calculate overall statistics
        self.price_stats = {
            'mean': self.df['Prices'].mean(),
            'std': self.df['Prices'].std(),
            'min': self.df['Prices'].min(),
            'max': self.df['Prices'].max()
        }
        
    def predict_price(self, target_date):
        """
        Predict natural gas price for a given date
        Uses seasonal averages with trend adjustment
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        month = target_date.month
        
        # Get base seasonal price
        base_price = self.monthly_averages[month]
        
        # Add simple trend adjustment for future dates
        last_date = self.df['Dates'].max()
        if target_date > last_date:
            # Simple linear trend based on last 12 months
            recent_data = self.df.tail(12)
            trend = (recent_data['Prices'].iloc[-1] - recent_data['Prices'].iloc[0]) / 365
            days_ahead = (target_date - last_date).days
            trend_adjustment = trend * days_ahead
            predicted_price = base_price + trend_adjustment
        else:
            # For historical dates, use interpolation
            predicted_price = base_price
            
        return max(predicted_price, 0)  # Ensure non-negative prices
    
    def price_contract(self, 
                      injection_schedule,
                      withdrawal_schedule, 
                      max_storage_volume,
                      injection_rate_limit=None,
                      withdrawal_rate_limit=None,
                      storage_cost_per_mmbtu_per_month=0.05,
                      injection_cost_per_mmbtu=0.01,
                      withdrawal_cost_per_mmbtu=0.01,
                      transport_cost_per_trip=50000,
                      detailed_output=True):
        """
        Price a natural gas storage contract
        
        Parameters:
        -----------
        injection_schedule : list of tuples
            [(date, volume), ...] - dates and volumes for gas injection
        withdrawal_schedule : list of tuples  
            [(date, volume), ...] - dates and volumes for gas withdrawal
        max_storage_volume : float
            Maximum storage capacity in MMBtu
        injection_rate_limit : float, optional
            Maximum injection rate in MMBtu per day
        withdrawal_rate_limit : float, optional
            Maximum withdrawal rate in MMBtu per day  
        storage_cost_per_mmbtu_per_month : float
            Monthly storage cost per MMBtu stored
        injection_cost_per_mmbtu : float
            Cost per MMBtu for injection operations
        withdrawal_cost_per_mmbtu : float
            Cost per MMBtu for withdrawal operations
        transport_cost_per_trip : float
            Transportation cost per facility visit
        detailed_output : bool
            Whether to return detailed breakdown
            
        Returns:
        --------
        dict : Contract valuation results
        """
        
        # Initialize results dictionary
        results = {
            'contract_value': 0,
            'total_revenue': 0,
            'total_costs': 0,
            'purchase_costs': 0,
            'storage_costs': 0,
            'operational_costs': 0,
            'transport_costs': 0,
            'violations': [],
            'cash_flows': [],
            'storage_profile': []
        }
        
        # Convert schedules to sorted lists with datetime objects
        injection_events = [(pd.to_datetime(date), volume) for date, volume in injection_schedule]
        withdrawal_events = [(pd.to_datetime(date), volume) for date, volume in withdrawal_schedule]
        
        injection_events.sort(key=lambda x: x[0])
        withdrawal_events.sort(key=lambda x: x[0])
        
        # Combine and sort all events
        all_events = []
        for date, volume in injection_events:
            all_events.append((date, 'injection', volume))
        for date, volume in withdrawal_events:
            all_events.append((date, 'withdrawal', volume))
        all_events.sort(key=lambda x: x[0])
        
        if not all_events:
            return results
            
        # Track storage level over time
        current_storage = 0
        start_date = all_events[0][0]
        end_date = all_events[-1][0]
        
        # Process each event
        for event_date, event_type, volume in all_events:
            
            # Check rate limits
            if injection_rate_limit and event_type == 'injection':
                max_daily_injection = injection_rate_limit
                if volume > max_daily_injection:
                    results['violations'].append(f"Injection rate limit violated on {event_date.date()}: {volume} > {max_daily_injection}")
                    
            if withdrawal_rate_limit and event_type == 'withdrawal':
                max_daily_withdrawal = withdrawal_rate_limit
                if volume > max_daily_withdrawal:
                    results['violations'].append(f"Withdrawal rate limit violated on {event_date.date()}: {volume} > {max_daily_withdrawal}")
            
            # Get price for this date
            price = self.predict_price(event_date)
            
            if event_type == 'injection':
                # Check storage capacity
                if current_storage + volume > max_storage_volume:
                    excess = current_storage + volume - max_storage_volume
                    results['violations'].append(f"Storage capacity exceeded on {event_date.date()}: excess of {excess:.2f} MMBtu")
                    volume = max_storage_volume - current_storage  # Limit to available capacity
                
                if volume > 0:
                    # Purchase costs
                    purchase_cost = volume * price
                    results['purchase_costs'] += purchase_cost
                    
                    # Injection operational costs
                    injection_op_cost = volume * injection_cost_per_mmbtu
                    results['operational_costs'] += injection_op_cost
                    
                    # Transportation cost
                    results['transport_costs'] += transport_cost_per_trip
                    
                    # Update storage level
                    current_storage += volume
                    
                    # Record cash flow
                    total_injection_cost = purchase_cost + injection_op_cost + transport_cost_per_trip
                    results['cash_flows'].append({
                        'date': event_date,
                        'type': 'injection',
                        'volume': volume,
                        'price': price,
                        'cash_flow': -total_injection_cost,
                        'storage_level': current_storage
                    })
                    
            elif event_type == 'withdrawal':
                # Check available storage
                if volume > current_storage:
                    shortage = volume - current_storage
                    results['violations'].append(f"Insufficient storage for withdrawal on {event_date.date()}: shortage of {shortage:.2f} MMBtu")
                    volume = current_storage  # Limit to available storage
                
                if volume > 0:
                    # Sale revenue
                    sale_revenue = volume * price
                    results['total_revenue'] += sale_revenue
                    
                    # Withdrawal operational costs
                    withdrawal_op_cost = volume * withdrawal_cost_per_mmbtu
                    results['operational_costs'] += withdrawal_op_cost
                    
                    # Transportation cost
                    results['transport_costs'] += transport_cost_per_trip
                    
                    # Update storage level
                    current_storage -= volume
                    
                    # Record cash flow
                    net_withdrawal_revenue = sale_revenue - withdrawal_op_cost - transport_cost_per_trip
                    results['cash_flows'].append({
                        'date': event_date,
                        'type': 'withdrawal',
                        'volume': volume,
                        'price': price,
                        'cash_flow': net_withdrawal_revenue,
                        'storage_level': current_storage
                    })
        
        # Calculate storage costs
        # Create monthly storage profile
        current_date = start_date.replace(day=1)  # Start of first month
        end_month = end_date.replace(day=1) + relativedelta(months=1)  # End of last month
        
        monthly_storage = {}
        storage_level = 0
        
        # Initialize storage tracking
        event_idx = 0
        while current_date < end_month:
            month_end = current_date + relativedelta(months=1)
            
            # Process events in this month
            month_storage_levels = []
            
            # Find events in this month
            month_events = [e for e in all_events if current_date <= e[0] < month_end]
            
            if not month_events:
                # No events this month, use previous storage level
                monthly_storage[current_date] = storage_level
            else:
                # Calculate average storage level for the month
                daily_storage = []
                temp_storage = storage_level
                
                # Add storage level for each day of the month
                for day in pd.date_range(current_date, month_end - timedelta(days=1), freq='D'):
                    # Check if there are events on this day
                    day_events = [e for e in month_events if e[0].date() == day.date()]
                    
                    for event_date, event_type, volume in day_events:
                        if event_type == 'injection':
                            temp_storage += volume
                        elif event_type == 'withdrawal':
                            temp_storage -= volume
                    
                    daily_storage.append(max(temp_storage, 0))
                
                avg_storage = np.mean(daily_storage) if daily_storage else storage_level
                monthly_storage[current_date] = avg_storage
                storage_level = temp_storage
            
            current_date = month_end
        
        # Calculate total storage costs
        for month, avg_storage in monthly_storage.items():
            monthly_cost = avg_storage * storage_cost_per_mmbtu_per_month
            results['storage_costs'] += monthly_cost
            
            results['storage_profile'].append({
                'month': month,
                'average_storage': avg_storage,
                'storage_cost': monthly_cost
            })
        
        # Calculate total costs and contract value
        results['total_costs'] = (results['purchase_costs'] + 
                                results['storage_costs'] + 
                                results['operational_costs'] + 
                                results['transport_costs'])
        
        results['contract_value'] = results['total_revenue'] - results['total_costs']
        
        if detailed_output:
            self._print_detailed_results(results)
            
        return results
    
    def _print_detailed_results(self, results):
        """Print detailed contract valuation results"""
        print("="*60)
        print("NATURAL GAS STORAGE CONTRACT VALUATION")
        print("="*60)
        
        print(f"\n📊 FINANCIAL SUMMARY")
        print(f"Contract Value: ${results['contract_value']:,.2f}")
        print(f"Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"Total Costs: ${results['total_costs']:,.2f}")
        
        print(f"\n💰 COST BREAKDOWN")
        print(f"Purchase Costs: ${results['purchase_costs']:,.2f}")
        print(f"Storage Costs: ${results['storage_costs']:,.2f}")
        print(f"Operational Costs: ${results['operational_costs']:,.2f}")
        print(f"Transport Costs: ${results['transport_costs']:,.2f}")
        
        if results['violations']:
            print(f"\n⚠️  CONSTRAINT VIOLATIONS")
            for violation in results['violations']:
                print(f"  • {violation}")
        else:
            print(f"\n✅ NO CONSTRAINT VIOLATIONS")
        
        print(f"\n📅 CASH FLOW SUMMARY")
        for cf in results['cash_flows']:
            flow_type = "💰" if cf['cash_flow'] > 0 else "💸"
            print(f"{flow_type} {cf['date'].strftime('%Y-%m-%d')}: {cf['type'].title()} "
                  f"{cf['volume']:,.0f} MMBtu @ ${cf['price']:.2f} = ${cf['cash_flow']:,.2f}")
    
    def run_scenario_analysis(self, base_scenario, price_scenarios=None):
        """Run multiple pricing scenarios"""
        if price_scenarios is None:
            price_scenarios = ['base', 'high_prices', 'low_prices']
        
        results = {}
        
        for scenario in price_scenarios:
            print(f"\n--- Scenario: {scenario.upper()} ---")
            
            # Modify price prediction based on scenario
            if scenario == 'high_prices':
                # Increase all prices by 20%
                original_predict = self.predict_price
                self.predict_price = lambda date: original_predict(date) * 1.2
            elif scenario == 'low_prices':
                # Decrease all prices by 20%
                original_predict = self.predict_price
                self.predict_price = lambda date: original_predict(date) * 0.8
            
            # Price the contract
            scenario_result = self.price_contract(**base_scenario)
            results[scenario] = scenario_result
            
            # Reset price prediction if modified
            if scenario != 'base':
                self.predict_price = original_predict
        
        return results

# Test the pricing model
def test_pricing_model():
    """Test the pricing model with sample scenarios"""
    
    # Initialize the pricer
    pricer = NaturalGasStorageContractPricer('Nat_Gas.csv')
    
    print("🧪 TESTING NATURAL GAS STORAGE CONTRACT PRICING MODEL")
    print("="*60)
    
    # Test Scenario 1: Simple Summer-to-Winter Strategy
    print("\n📈 TEST SCENARIO 1: Summer-to-Winter Storage Strategy")
    
    scenario_1 = {
        'injection_schedule': [
            ('2024-07-31', 500000),  # Inject 500,000 MMBtu in July
            ('2024-08-31', 300000),  # Inject 300,000 MMBtu in August
        ],
        'withdrawal_schedule': [
            ('2024-12-31', 400000),  # Withdraw 400,000 MMBtu in December
            ('2025-01-31', 400000),  # Withdraw 400,000 MMBtu in January
        ],
        'max_storage_volume': 1000000,  # 1 million MMBtu capacity
        'injection_rate_limit': 50000,   # 50,000 MMBtu per day
        'withdrawal_rate_limit': 50000,  # 50,000 MMBtu per day
        'storage_cost_per_mmbtu_per_month': 0.05,
        'injection_cost_per_mmbtu': 0.01,
        'withdrawal_cost_per_mmbtu': 0.01,
        'transport_cost_per_trip': 50000
    }
    
    result_1 = pricer.price_contract(**scenario_1)
    
    # Test Scenario 2: Aggressive Storage Strategy
    print("\n\n📈 TEST SCENARIO 2: Aggressive Multi-Season Strategy")
    
    scenario_2 = {
        'injection_schedule': [
            ('2024-06-30', 200000),
            ('2024-07-31', 200000), 
            ('2024-08-31', 200000),
            ('2024-09-30', 200000),
        ],
        'withdrawal_schedule': [
            ('2024-11-30', 200000),
            ('2024-12-31', 200000),
            ('2025-01-31', 200000),
            ('2025-02-28', 200000),
        ],
        'max_storage_volume': 800000,
        'injection_rate_limit': 25000,
        'withdrawal_rate_limit': 25000,
        'storage_cost_per_mmbtu_per_month': 0.03,
        'injection_cost_per_mmbtu': 0.008,
        'withdrawal_cost_per_mmbtu': 0.008,
        'transport_cost_per_trip': 30000
    }
    
    result_2 = pricer.price_contract(**scenario_2)
    
    # Test Scenario 3: Capacity Constraint Test
    print("\n\n📈 TEST SCENARIO 3: Capacity Constraint Testing")
    
    scenario_3 = {
        'injection_schedule': [
            ('2024-07-31', 600000),  # Try to inject more than capacity
            ('2024-08-31', 500000),
        ],
        'withdrawal_schedule': [
            ('2024-12-31', 800000),  # Try to withdraw more than stored
        ],
        'max_storage_volume': 500000,  # Small capacity
        'storage_cost_per_mmbtu_per_month': 0.04,
        'injection_cost_per_mmbtu': 0.01,
        'withdrawal_cost_per_mmbtu': 0.01,
        'transport_cost_per_trip': 40000
    }
    
    result_3 = pricer.price_contract(**scenario_3)
    
    # Scenario Analysis
    print("\n\n📊 SCENARIO ANALYSIS: Price Sensitivity")
    scenario_results = pricer.run_scenario_analysis(scenario_1)
    
    print("\n📋 SCENARIO COMPARISON SUMMARY:")
    for scenario_name, result in scenario_results.items():
        print(f"{scenario_name.upper()}: Contract Value = ${result['contract_value']:,.2f}")
    
    return pricer, scenario_results

# Run the tests
if __name__ == "__main__":
    pricer, results = test_pricing_model()
