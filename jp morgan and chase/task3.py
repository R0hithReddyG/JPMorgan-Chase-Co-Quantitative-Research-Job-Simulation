import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Create sample dataset structure (replace with actual data loading)
def create_sample_data(n_samples=10000):
    """
    Creates sample loan data for demonstration purposes
    In practice, replace this with actual data loading
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'annual_income': np.random.lognormal(10.5, 0.5, n_samples),
        'loan_amount': np.random.lognormal(9.5, 0.8, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'employment_length': np.random.exponential(5, n_samples),
        'num_credit_lines': np.random.poisson(3, n_samples),
        'interest_rate': np.random.normal(12, 3, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 0.5
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic default probability based on features
    default_prob = (
        0.1 * (df['credit_score'] < 600).astype(int) +
        0.05 * (df['debt_to_income'] > 0.4).astype(int) +
        0.03 * (df['employment_length'] < 1).astype(int) +
        0.02 * (df['age'] < 25).astype(int) +
        np.random.normal(0, 0.02, n_samples)
    )
    
    df['default'] = (np.random.random(n_samples) < np.clip(default_prob, 0, 1)).astype(int)
    
    return df

class LoanDefaultPredictor:
    """
    Comprehensive loan default prediction system
    """
    
    def __init__(self, recovery_rate=0.10):
        self.recovery_rate = recovery_rate
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_performance = {}
        
    def preprocess_data(self, df, target_col='default', test_size=0.2):
        """
        Preprocesses the loan data including handling missing values,
        feature scaling, and train-test split
        """
        # Handle missing values
        df = df.fillna(df.median())
        
        # Feature engineering
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        df['credit_utilization'] = df['num_credit_lines'] / 10  # Normalized
        df['age_income_interaction'] = df['age'] * df['annual_income'] / 1000000
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_columns = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        Addresses class imbalance using various resampling techniques
        """
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X_train, y_train
            
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Trains multiple models for comparative analysis
        """
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Handle class imbalance for tree-based models
            if name in ['Random Forest', 'XGBoost', 'LightGBM']:
                X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                    X_train, y_train, method='smote'
                )
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Store model and performance
            self.models[name] = model
            self.model_performance[name] = {
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'accuracy': model.score(X_test, y_test),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - AUC: {self.model_performance[name]['auc_score']:.4f}")
    
    def get_feature_importance(self, model_name='Random Forest'):
        """
        Extracts feature importance from tree-based models
        """
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def predict_default_probability(self, loan_features, model_name='Random Forest'):
        """
        Predicts default probability for new loan applications
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Ensure features are in correct format
        if isinstance(loan_features, dict):
            loan_features = pd.DataFrame([loan_features])
        
        # Add engineered features
        loan_features['loan_to_income_ratio'] = (
            loan_features['loan_amount'] / loan_features['annual_income']
        )
        loan_features['credit_utilization'] = loan_features['num_credit_lines'] / 10
        loan_features['age_income_interaction'] = (
            loan_features['age'] * loan_features['annual_income'] / 1000000
        )
        
        # Scale features
        features_scaled = self.scaler.transform(loan_features[self.feature_columns])
        
        # Get probability
        default_probability = model.predict_proba(features_scaled)[:, 1]
        
        return default_probability[0] if len(default_probability) == 1 else default_probability
    
    def calculate_expected_loss(self, loan_amount, default_probability):
        """
        Calculates expected loss given loan amount and default probability
        Expected Loss = PD * LGD * EAD
        Where: PD = Probability of Default, LGD = Loss Given Default, EAD = Exposure at Default
        """
        loss_given_default = 1 - self.recovery_rate  # 90% loss if recovery rate is 10%
        exposure_at_default = loan_amount  # Assuming full exposure
        
        expected_loss = default_probability * loss_given_default * exposure_at_default
        return expected_loss
    
    def comprehensive_risk_assessment(self, loan_features):
        """
        Provides comprehensive risk assessment using multiple models
        """
        results = {}
        
        for model_name in self.models.keys():
            prob = self.predict_default_probability(loan_features, model_name)
            expected_loss = self.calculate_expected_loss(
                loan_features['loan_amount'], prob
            )
            
            results[model_name] = {
                'default_probability': prob,
                'expected_loss': expected_loss,
                'risk_rating': 'High' if prob > 0.3 else 'Medium' if prob > 0.1 else 'Low'
            }
        
        return results

# Implementation example
def main():
    # Create sample data (replace with actual data loading)
    print("Loading and preprocessing data...")
    df = create_sample_data(10000)
    
    # Initialize predictor
    predictor = LoanDefaultPredictor(recovery_rate=0.10)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.preprocess_data(df)
    
    # Train models
    print("\nTraining multiple models...")
    predictor.train_models(X_train, X_test, y_train, y_test)
    
    # Display model comparison
    print("\n=== Model Performance Comparison ===")
    performance_df = pd.DataFrame({
        model: {
            'AUC Score': metrics['auc_score'],
            'Accuracy': metrics['accuracy']
        }
        for model, metrics in predictor.model_performance.items()
    }).T
    
    print(performance_df.round(4))
    
    # Feature importance analysis
    print("\n=== Feature Importance (Random Forest) ===")
    importance = predictor.get_feature_importance('Random Forest')
    if importance is not None:
        print(importance.head(10))
    
    # Example prediction
    print("\n=== Example Loan Risk Assessment ===")
    sample_loan = {
        'age': 35,
        'annual_income': 75000,
        'loan_amount': 25000,
        'credit_score': 680,
        'employment_length': 3.5,
        'num_credit_lines': 4,
        'interest_rate': 11.5,
        'loan_term': 36,
        'debt_to_income': 0.25
    }
    
    risk_assessment = predictor.comprehensive_risk_assessment(sample_loan)
    
    for model, assessment in risk_assessment.items():
        print(f"\n{model}:")
        print(f"  Default Probability: {assessment['default_probability']:.4f}")
        print(f"  Expected Loss: ${assessment['expected_loss']:.2f}")
        print(f"  Risk Rating: {assessment['risk_rating']}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
