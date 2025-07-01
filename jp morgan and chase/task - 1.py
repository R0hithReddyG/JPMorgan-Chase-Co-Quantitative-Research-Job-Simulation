import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

class NaturalGasPricePredictor:
    """
    Natural Gas Price Prediction Model for Commodity Storage Contracts
    
    This class provides comprehensive analysis and prediction capabilities for natural gas prices,
    specifically designed for pricing commodity storage contracts with seasonal trend analysis.
    """
    
    def __init__(self, csv_file_path='Nat_Gas.csv'):
        """Initialize the predictor with natural gas price data"""
        self.df = None
        self.models = {}
        self.seasonal_stats = {}
        self.load_and_prepare_data(csv_file_path)
        
    def load_and_prepare_data(self, csv_file_path):
        """Load and prepare the natural gas price data"""
        # Load the CSV data
        self.df = pd.read_csv(csv_file_path)
        
        # Convert scientific notation to regular numbers
        self.df['Prices'] = self.df['Prices'].astype(float)
        
        # Convert dates to datetime
        self.df['Dates'] = pd.to_datetime(self.df['Dates'], format='%m/%d/%y')
        
        # Sort by date
        self.df = self.df.sort_values('Dates').reset_index(drop=True)
        
        # Create additional features for modeling
        self.create_features()
        
        print("Data loaded successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['Dates'].min()} to {self.df['Dates'].max()}")
        print(f"Price range: ${self.df['Prices'].min():.2f} to ${self.df['Prices'].max():.2f}")
        
    def create_features(self):
        """Create additional features for modeling"""
        # Extract date components
        self.df['Year'] = self.df['Dates'].dt.year
        self.df['Month'] = self.df['Dates'].dt.month
        self.df['Quarter'] = self.df['Dates'].dt.quarter
        
        # Create seasonal indicators
        self.df['Season'] = self.df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Create heating/cooling season indicators
        self.df['Heating_Season'] = self.df['Month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
        self.df['Peak_Winter'] = self.df['Month'].isin([12, 1, 2]).astype(int)
        self.df['Peak_Summer'] = self.df['Month'].isin([6, 7, 8]).astype(int)
        
        # Create time-based features for trend analysis
        self.df['Days_Since_Start'] = (self.df['Dates'] - self.df['Dates'].min()).dt.days
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in natural gas prices"""
        # Calculate monthly statistics
        monthly_stats = self.df.groupby('Month')['Prices'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        # Calculate seasonal statistics
        seasonal_stats = self.df.groupby('Season')['Prices'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        self.seasonal_stats = {
            'monthly': monthly_stats,
            'seasonal': seasonal_stats
        }
        
        print("\n=== SEASONAL ANALYSIS ===")
        print("\nMonthly Price Statistics:")
        print(monthly_stats)
        print("\nSeasonal Price Statistics:")
        print(seasonal_stats)
        
        # Identify peak and trough months
        peak_month = monthly_stats['mean'].idxmax()
        trough_month = monthly_stats['mean'].idxmin()
        
        print(f"\nPeak pricing month: {peak_month} (${monthly_stats.loc[peak_month, 'mean']:.2f})")
        print(f"Trough pricing month: {trough_month} (${monthly_stats.loc[trough_month, 'mean']:.2f})")
        
        return monthly_stats, seasonal_stats
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the data"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Natural Gas Price Analysis for Storage Contract Pricing', fontsize=16, fontweight='bold')
        
        # 1. Time Series Plot
        axes[0, 0].plot(self.df['Dates'], self.df['Prices'], marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Natural Gas Prices Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Seasonal Box Plot
        sns.boxplot(data=self.df, x='Month', y='Prices', ax=axes[0, 1])
        axes[0, 1].set_title('Monthly Price Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Yearly Trends
        sns.boxplot(data=self.df, x='Year', y='Prices', ax=axes[1, 0])
        axes[1, 0].set_title('Yearly Price Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Seasonal Patterns
        seasonal_avg = self.df.groupby('Season')['Prices'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        bars = axes[1, 1].bar(seasonal_avg.index, seasonal_avg.values, 
                             color=['lightblue', 'lightgreen', 'orange', 'brown'], alpha=0.7)
        axes[1, 1].set_title('Average Prices by Season', fontweight='bold')
        axes[1, 1].set_ylabel('Average Price ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Rolling Average
        self.df['Rolling_12M'] = self.df['Prices'].rolling(window=12, center=True).mean()
        axes[2, 0].plot(self.df['Dates'], self.df['Prices'], alpha=0.6, label='Monthly Prices')
        axes[2, 0].plot(self.df['Dates'], self.df['Rolling_12M'], color='red', linewidth=2, label='12-Month Rolling Average')
        axes[2, 0].set_title('Price Trends with 12-Month Rolling Average', fontweight='bold')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Price ($)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. Price Distribution
        axes[2, 1].hist(self.df['Prices'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2, 1].axvline(self.df['Prices'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${self.df["Prices"].mean():.2f}')
        axes[2, 1].axvline(self.df['Prices'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${self.df["Prices"].median():.2f}')
        axes[2, 1].set_title('Price Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('Price ($)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def train_models(self):
        """Train multiple models for price prediction"""
        # Prepare feature matrix
        feature_cols = ['Days_Since_Start', 'Month', 'Quarter', 'Heating_Season', 
                       'Peak_Winter', 'Peak_Summer', 'Month_Sin', 'Month_Cos']
        X = self.df[feature_cols]
        y = self.df['Prices']
        
        # Model 1: Linear Regression with Polynomial Features
        poly_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        poly_pipeline.fit(X, y)
        self.models['polynomial'] = poly_pipeline
        
        # Model 2: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X, y)
        self.models['random_forest'] = rf_model
        
        # Model 3: Seasonal Decomposition with Linear Trend
        # Create monthly averages for seasonal component
        monthly_seasonal = self.df.groupby('Month')['Prices'].mean()
        overall_mean = self.df['Prices'].mean()
        seasonal_component = self.df['Month'].map(monthly_seasonal) - overall_mean
        
        # Detrend the data
        detrended = self.df['Prices'] - seasonal_component
        trend_model = LinearRegression()
        trend_model.fit(self.df[['Days_Since_Start']], detrended)
        
        self.models['seasonal_trend'] = {
            'trend_model': trend_model,
            'monthly_seasonal': monthly_seasonal,
            'overall_mean': overall_mean
        }
        
        # Model 4: Simple Seasonal Model (for baseline)
        self.models['seasonal_naive'] = monthly_seasonal
        
        # Evaluate models
        self.evaluate_models(X, y)
        
    def evaluate_models(self, X, y):
        """Evaluate model performance"""
        print("\n=== MODEL EVALUATION ===")
        
        # Evaluate polynomial model
        poly_pred = self.models['polynomial'].predict(X)
        poly_rmse = np.sqrt(mean_squared_error(y, poly_pred))
        poly_r2 = r2_score(y, poly_pred)
        print(f"Polynomial Model - RMSE: {poly_rmse:.3f}, R²: {poly_r2:.3f}")
        
        # Evaluate random forest
        rf_pred = self.models['random_forest'].predict(X)
        rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
        rf_r2 = r2_score(y, rf_pred)
        print(f"Random Forest Model - RMSE: {rf_rmse:.3f}, R²: {rf_r2:.3f}")
        
        # Feature importance for Random Forest
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(feature_importance)
        
    def predict_price(self, target_date, method='ensemble'):
        """
        Predict natural gas price for a given date
        
        Parameters:
        target_date: string or datetime - target date for prediction
        method: string - prediction method ('ensemble', 'polynomial', 'random_forest', 'seasonal_trend', 'seasonal_naive')
        
        Returns:
        float: predicted price
        """
        # Convert string to datetime if necessary
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Extract features for the target date
        month = target_date.month
        quarter = (month - 1) // 3 + 1
        heating_season = 1 if month in [10, 11, 12, 1, 2, 3] else 0
        peak_winter = 1 if month in [12, 1, 2] else 0
        peak_summer = 1 if month in [6, 7, 8] else 0
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Calculate days since start
        days_since_start = (target_date - self.df['Dates'].min()).days
        
        # Create feature vector
        features = np.array([[days_since_start, month, quarter, heating_season, 
                             peak_winter, peak_summer, month_sin, month_cos]])
        
        if method == 'ensemble':
            # Ensemble of multiple models
            poly_pred = self.models['polynomial'].predict(features)[0]
            rf_pred = self.models['random_forest'].predict(features)[0]
            
            # Seasonal trend prediction
            st_model = self.models['seasonal_trend']
            trend_pred = st_model['trend_model'].predict([[days_since_start]])[0]
            seasonal_adj = st_model['monthly_seasonal'][month] - st_model['overall_mean']
            st_pred = trend_pred + seasonal_adj
            
            # Seasonal naive prediction
            sn_pred = self.models['seasonal_naive'][month]
            
            # Weighted ensemble (giving more weight to more complex models for future dates)
            if target_date > self.df['Dates'].max():
                # For future dates, rely more on trend-based models
                prediction = 0.4 * poly_pred + 0.3 * rf_pred + 0.2 * st_pred + 0.1 * sn_pred
            else:
                # For historical dates, rely more on data-driven models
                prediction = 0.3 * poly_pred + 0.4 * rf_pred + 0.2 * st_pred + 0.1 * sn_pred
                
        elif method == 'polynomial':
            prediction = self.models['polynomial'].predict(features)[0]
            
        elif method == 'random_forest':
            prediction = self.models['random_forest'].predict(features)[0]
            
        elif method == 'seasonal_trend':
            st_model = self.models['seasonal_trend']
            trend_pred = st_model['trend_model'].predict([[days_since_start]])[0]
            seasonal_adj = st_model['monthly_seasonal'][month] - st_model['overall_mean']
            prediction = trend_pred + seasonal_adj
            
        elif method == 'seasonal_naive':
            prediction = self.models['seasonal_naive'][month]
            
        else:
            raise ValueError("Invalid method. Choose from: 'ensemble', 'polynomial', 'random_forest', 'seasonal_trend', 'seasonal_naive'")
        
        return max(prediction, 0)  # Ensure non-negative prices
    
    def create_future_forecast(self, months_ahead=12):
        """Create forecast for future months"""
        last_date = self.df['Dates'].max()
        future_dates = []
        future_predictions = []
        
        for i in range(1, months_ahead + 1):
            future_date = last_date + relativedelta(months=i)
            # Set to end of month
            next_month = future_date.replace(day=1) + relativedelta(months=1)
            future_date = next_month - timedelta(days=1)
            
            future_dates.append(future_date)
            future_predictions.append(self.predict_price(future_date))
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        
        return forecast_df
    
    def plot_forecast(self, months_ahead=12):
        """Plot historical data with future forecast"""
        forecast_df = self.create_future_forecast(months_ahead)
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(self.df['Dates'], self.df['Prices'], 'o-', label='Historical Prices', 
                color='blue', linewidth=2, markersize=4)
        
        # Plot forecast
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'], 's--', 
                label='Forecasted Prices', color='red', linewidth=2, markersize=6)
        
        # Add vertical line at forecast start
        plt.axvline(x=self.df['Dates'].max(), color='gray', linestyle=':', alpha=0.7, 
                   label='Forecast Start')
        
        plt.title('Natural Gas Price Forecast for Storage Contract Pricing', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nFuture Price Forecast:")
        print(forecast_df.to_string(index=False))
        
        return forecast_df
    
    def get_seasonal_insights(self):
        """Provide insights for commodity storage contract strategy"""
        monthly_stats = self.seasonal_stats['monthly']
        
        print("\n=== COMMODITY STORAGE CONTRACT INSIGHTS ===")
        
        # Best injection months (low prices)
        best_injection = monthly_stats['mean'].nsmallest(3)
        print(f"\nBest months for injection (lowest prices):")
        for month, price in best_injection.items():
            month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
            print(f"  {month_name}: ${price:.2f}")
        
        # Best withdrawal months (high prices)
        best_withdrawal = monthly_stats['mean'].nlargest(3)
        print(f"\nBest months for withdrawal (highest prices):")
        for month, price in best_withdrawal.items():
            month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
            print(f"  {month_name}: ${price:.2f}")
        
        # Calculate potential storage spread
        max_spread = best_withdrawal.iloc[0] - best_injection.iloc[0]
        print(f"\nMaximum seasonal spread: ${max_spread:.2f}")
        print(f"This represents {(max_spread/best_injection.iloc[0]*100):.1f}% potential profit margin")
        
        # Volatility analysis
        print(f"\nPrice volatility by season:")
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = self.df[self.df['Season'] == season]['Prices']
            cv = season_data.std() / season_data.mean()
            print(f"  {season}: {cv:.3f} (CV)")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("NATURAL GAS PRICE ANALYSIS FOR STORAGE CONTRACTS")
        print("="*60)
        
        # Step 1: Analyze seasonal patterns
        self.analyze_seasonal_patterns()
        
        # Step 2: Create visualizations
        self.create_visualizations()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Create forecast
        forecast_df = self.plot_forecast(12)
        
        # Step 5: Provide insights
        self.get_seasonal_insights()
        
        return forecast_df

def main():
    """Main execution function"""
    # Initialize the predictor
    predictor = NaturalGasPricePredictor('Nat_Gas.csv')
    
    # Run complete analysis
    forecast = predictor.run_complete_analysis()
    
    # Example usage of the prediction function
    print("\n" + "="*60)
    print("PRICE PREDICTION EXAMPLES")
    print("="*60)
    
    test_dates = [
        '2024-12-31',  # Future date
        '2025-06-30',  # Future summer
        '2025-01-31',  # Future winter
        '2022-07-31',  # Historical summer
        '2023-01-31'   # Historical winter
    ]
    
    for date in test_dates:
        price = predictor.predict_price(date)
        date_obj = pd.to_datetime(date)
        season = 'Winter' if date_obj.month in [12,1,2] else 'Spring' if date_obj.month in [3,4,5] else 'Summer' if date_obj.month in [6,7,8] else 'Fall'
        print(f"Predicted price for {date} ({season}): ${price:.2f}")
    
    print("\n" + "="*60)
    print("STORAGE CONTRACT TRADING STRATEGY RECOMMENDATION")
    print("="*60)
    
    # Calculate optimal storage strategy
    summer_price = predictor.predict_price('2025-07-31')
    winter_price = predictor.predict_price('2025-01-31')
    
    print(f"Summer 2025 predicted price: ${summer_price:.2f}")
    print(f"Winter 2025 predicted price: ${winter_price:.2f}")
    print(f"Seasonal spread: ${winter_price - summer_price:.2f}")
    
    if winter_price > summer_price:
        print("\nRECOMMENDATION: BULLISH STORAGE STRATEGY")
        print("- Inject (buy) natural gas during summer months")
        print("- Store in underground facilities")
        print("- Withdraw (sell) during winter months")
        print(f"- Potential profit per unit: ${winter_price - summer_price:.2f}")
    else:
        print("\nRECOMMENDATION: BEARISH STORAGE STRATEGY")
        print("- Consider selling storage capacity or avoid long storage positions")
    
    return predictor

# Execute the analysis
if __name__ == "__main__":
    predictor = main()
