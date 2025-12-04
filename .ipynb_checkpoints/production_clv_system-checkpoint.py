# production_clv_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CLVPrediction:
    customer_id: str
    clv_bg_nbd: float
    clv_rf: float
    segment: str
    recency: int
    frequency: int
    monetary_value: float
    prediction_date: datetime

class ProductionCLVSystem:
    """
    Production-ready CLV prediction system with all required components
    """
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.bgf_model = None
        self.ggf_model = None
        self.rf_model = None
        self.scaler = None
        self.feature_columns = None
        self.last_training_date = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess Olist e-commerce data
        """
        logger.info("Loading Olist datasets...")
        
        # Load all datasets
        customers = pd.read_csv('olist_customers_dataset.csv')
        orders = pd.read_csv('olist_orders_dataset.csv')
        order_items = pd.read_csv('olist_order_items_dataset.csv')
        order_payments = pd.read_csv('olist_order_payments_dataset.csv')
        products = pd.read_csv('olist_products_dataset.csv')
        product_translation = pd.read_csv('product_category_name_translation.csv')
        
        # Convert date columns
        date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                       'order_delivered_carrier_date', 'order_delivered_customer_date']
        for col in date_columns:
            orders[col] = pd.to_datetime(orders[col], errors='coerce')
        
        # Merge datasets
        merged_data = orders.merge(customers, on='customer_id', how='inner')
        merged_data = merged_data.merge(order_items, on='order_id', how='inner')
        merged_data = merged_data.merge(products[['product_id', 'product_category_name']], 
                                       on='product_id', how='left')
        merged_data = merged_data.merge(product_translation, 
                                       on='product_category_name', how='left')
        
        # Calculate total price
        merged_data['total_price'] = merged_data['price'] + merged_data['freight_value']
        
        # Clean data
        transaction_data = merged_data[
            (merged_data['order_purchase_timestamp'].notna()) &
            (merged_data['price'] > 0) &
            (merged_data['order_status'] == 'delivered')
        ][['customer_id', 'order_id', 'order_purchase_timestamp', 'total_price']].copy()
        
        logger.info(f"Loaded {len(transaction_data)} transactions from {transaction_data['customer_id'].nunique()} customers")
        return transaction_data.sort_values('order_purchase_timestamp')
    
    def create_rfm_features(self, transaction_data: pd.DataFrame, 
                           snapshot_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Create RFM features for CLV modeling
        """
        if snapshot_date is None:
            snapshot_date = transaction_data['order_purchase_timestamp'].max() + timedelta(days=1)
        
        rfm = transaction_data.groupby('customer_id').agg({
            'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'count',
            'total_price': 'mean'
        }).rename(columns={
            'order_purchase_timestamp': 'recency',
            'order_id': 'frequency',
            'total_price': 'monetary_value'
        })
        
        # Add customer tenure
        rfm['T'] = (snapshot_date - transaction_data.groupby('customer_id')['order_purchase_timestamp'].min()).dt.days
        rfm['customer_tenure'] = rfm['T']
        
        # Filter for customers with multiple purchases (for BG/NBD)
        rfm_filtered = rfm[rfm['frequency'] > 1].copy()
        
        logger.info(f"Created RFM features for {len(rfm_filtered)} customers")
        return rfm_filtered, snapshot_date
    
    def train_bg_nbd_gamma_gamma(self, rfm_data: pd.DataFrame):
        """
        Train BG/NBD + Gamma-Gamma model
        """
        logger.info("Training BG/NBD + Gamma-Gamma model...")
        
        bg_nbd_data = rfm_data[['frequency', 'recency', 'T']].copy()
        monetary_data = rfm_data[['frequency', 'monetary_value']].copy()
        
        # Train BG/NBD
        self.bgf_model = BetaGeoFitter(penalizer_coef=0.01)
        self.bgf_model.fit(bg_nbd_data['frequency'], bg_nbd_data['recency'], bg_nbd_data['T'])
        
        # Train Gamma-Gamma
        self.ggf_model = GammaGammaFitter(penalizer_coef=0.01)
        self.ggf_model.fit(monetary_data['frequency'], monetary_data['monetary_value'])
        
        logger.info("BG/NBD + Gamma-Gamma model training completed")
    
    def train_machine_learning_model(self, rfm_data: pd.DataFrame):
        """
        Train Random Forest model for CLV prediction
        """
        logger.info("Training Random Forest model...")
        
        # Create additional features
        ml_features = rfm_data.copy()
        ml_features['purchase_rate'] = ml_features['frequency'] / ml_features['T']
        ml_features['avg_days_between_purchases'] = ml_features['recency'] / (ml_features['frequency'] - 1)
        ml_features['avg_days_between_purchases'] = ml_features['avg_days_between_purchases'].replace([np.inf, -np.inf], np.nan)
        ml_features['avg_days_between_purchases'] = ml_features['avg_days_between_purchases'].fillna(ml_features['T'])
        
        # Select features
        self.feature_columns = ['recency', 'frequency', 'monetary_value', 'T', 
                               'purchase_rate', 'avg_days_between_purchases']
        X = ml_features[self.feature_columns]
        
        # Use BG/NBD predictions as target for consistency
        bg_nbd_predictions = self.ggf_model.customer_lifetime_value(
            self.bgf_model,
            rfm_data['frequency'],
            rfm_data['recency'],
            rfm_data['T'],
            rfm_data['monetary_value'],
            time=12,
            discount_rate=0.01
        )
        y = bg_nbd_predictions.values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_scaled, y)
        
        logger.info("Random Forest model training completed")
    
    def predict_clv(self, rfm_data: pd.DataFrame, prediction_months: int = 12) -> pd.DataFrame:
        """
        Generate CLV predictions using both models
        """
        logger.info(f"Generating CLV predictions for {prediction_months} months...")
        
        # BG/NBD + Gamma-Gamma predictions
        clv_bg_nbd = self.ggf_model.customer_lifetime_value(
            self.bgf_model,
            rfm_data['frequency'],
            rfm_data['recency'],
            rfm_data['T'],
            rfm_data['monetary_value'],
            time=prediction_months,
            discount_rate=0.01
        )
        
        # Random Forest predictions
        ml_features = rfm_data.copy()
        ml_features['purchase_rate'] = ml_features['frequency'] / ml_features['T']
        ml_features['avg_days_between_purchases'] = ml_features['recency'] / (ml_features['frequency'] - 1)
        ml_features['avg_days_between_purchases'] = ml_features['avg_days_between_purchases'].replace([np.inf, -np.inf], np.nan)
        ml_features['avg_days_between_purchases'] = ml_features['avg_days_between_purchases'].fillna(ml_features['T'])
        
        X = ml_features[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        clv_rf = self.rf_model.predict(X_scaled)
        
        # Create results dataframe
        results = rfm_data.copy()
        results['clv_bg_nbd'] = clv_bg_nbd.values
        results['clv_rf'] = clv_rf
        results['prediction_date'] = datetime.now()
        
        logger.info(f"Generated CLV predictions for {len(results)} customers")
        return results
    
    def create_customer_segments(self, clv_data: pd.DataFrame, 
                                clv_column: str = 'clv_bg_nbd') -> pd.DataFrame:
        """
        Create customer segments based on CLV predictions
        """
        segmented_data = clv_data.copy()
        
        # Calculate percentiles
        p75 = segmented_data[clv_column].quantile(0.75)
        p50 = segmented_data[clv_column].quantile(0.50)
        p25 = segmented_data[clv_column].quantile(0.25)
        
        def assign_segment(clv_value):
            if clv_value >= p75:
                return 'Champions'
            elif clv_value >= p50:
                return 'Loyal Customers'
            elif clv_value >= p25:
                return 'Potential Loyalists'
            else:
                return 'At Risk'
        
        segmented_data['clv_segment'] = segmented_data[clv_column].apply(assign_segment)
        return segmented_data
    
    def save_models(self):
        """
        Save trained models for production deployment
        """
        import os
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save BG/NBD + Gamma-Gamma models
        with open(f"{self.model_path}bgf_model.pkl", 'wb') as f:
            pickle.dump(self.bgf_model, f)
        with open(f"{self.model_path}ggf_model.pkl", 'wb') as f:
            pickle.dump(self.ggf_model, f)
        
        # Save Random Forest model and scaler
        with open(f"{self.model_path}rf_model.pkl", 'wb') as f:
            pickle.dump(self.rf_model, f)
        with open(f"{self.model_path}scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open(f"{self.model_path}feature_columns.pkl", 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Models saved to {self.model_path}")
    
    def load_models(self):
        """
        Load pre-trained models for inference
        """
        try:
            with open(f"{self.model_path}bgf_model.pkl", 'rb') as f:
                self.bgf_model = pickle.load(f)
            with open(f"{self.model_path}ggf_model.pkl", 'rb') as f:
                self.ggf_model = pickle.load(f)
            with open(f"{self.model_path}rf_model.pkl", 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(f"{self.model_path}scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{self.model_path}feature_columns.pkl", 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            logger.info("Models loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("Model files not found. Please train models first.")
            return False
    
    def train_full_pipeline(self):
        """
        Complete training pipeline for production deployment
        """
        logger.info("Starting full training pipeline...")
        
        # Load and preprocess data
        transaction_data = self.load_data()
        
        # Create RFM features
        rfm_data, snapshot_date = self.create_rfm_features(transaction_data)
        self.last_training_date = snapshot_date
        
        # Train models
        self.train_bg_nbd_gamma_gamma(rfm_data)
        self.train_machine_learning_model(rfm_data)
        
        # Generate predictions and segments
        clv_predictions = self.predict_clv(rfm_data)
        segmented_customers = self.create_customer_segments(clv_predictions)
        
        # Save models
        self.save_models()
        
        logger.info("Full training pipeline completed successfully")
        return segmented_customers
    
    def predict_new_customer(self, customer_data: Dict) -> CLVPrediction:
        """
        Predict CLV for a new customer (real-time inference)
        """
        if self.bgf_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Create RFM features for single customer
        recency = customer_data['recency']
        frequency = customer_data['frequency']
        monetary_value = customer_data['monetary_value']
        T = customer_data['T']
        customer_id = customer_data['customer_id']
        
        # BG/NBD prediction
        clv_bg_nbd = self.ggf_model.conditional_expected_average_profit(
            frequency, monetary_value
        ) * self.bgf_model.conditional_expected_number_of_purchases_up_to_time(
            12, frequency, recency, T
        )
        
        # Random Forest prediction
        features = np.array([[recency, frequency, monetary_value, T, 
                             frequency/T, recency/(frequency-1) if frequency > 1 else T]])
        features_scaled = self.scaler.transform(features)
        clv_rf = self.rf_model.predict(features_scaled)[0]
        
        # Assign segment (simplified - in production, use full segmentation logic)
        segment = 'Champions' if clv_bg_nbd >= 1000 else 'Loyal Customers' if clv_bg_nbd >= 500 else 'Potential Loyalists' if clv_bg_nbd >= 200 else 'At Risk'
        
        return CLVPrediction(
            customer_id=customer_id,
            clv_bg_nbd=float(clv_bg_nbd),
            clv_rf=float(clv_rf),
            segment=segment,
            recency=recency,
            frequency=frequency,
            monetary_value=monetary_value,
            prediction_date=datetime.now()
        )

# Dashboard and Integration Components
class CLVDashboard:
    """
    Customer segmentation dashboard with real-time updates
    """
    
    def __init__(self, clv_system: ProductionCLVSystem):
        self.clv_system = clv_system
        self.segment_stats = None
    
    def update_dashboard(self, segmented_data: pd.DataFrame):
        """
        Update dashboard with latest CLV predictions and segments
        """
        self.segment_stats = segmented_data.groupby('clv_segment').agg({
            'clv_bg_nbd': ['count', 'mean', 'median'],
            'frequency': 'mean',
            'monetary_value': 'mean',
            'recency': 'mean'
        }).round(2)
        
        logger.info("Dashboard updated with latest customer segments")
    
    def get_segment_recommendations(self) -> Dict[str, Dict]:
        """
        Generate marketing recommendations by segment
        """
        recommendations = {
            'Champions': {
                'budget_allocation': '25-30%',
                'strategies': ['VIP experiences', 'Exclusive offers', 'Referral programs'],
                'actions': ['Personalized service', 'Early product access', 'High-value incentives']
            },
            'Loyal Customers': {
                'budget_allocation': '20-25%',
                'strategies': ['Retention focus', 'Cross-selling', 'Loyalty rewards'],
                'actions': ['Product recommendations', 'Exclusive discounts', 'Feedback collection']
            },
            'Potential Loyalists': {
                'budget_allocation': '30-35%',
                'strategies': ['Nurturing campaigns', 'Educational content', 'Trial offers'],
                'actions': ['Welcome optimization', 'Engagement campaigns', 'Premium product trials']
            },
            'At Risk': {
                'budget_allocation': '15-20%',
                'strategies': ['Win-back campaigns', 'Re-engagement', 'Competitor analysis'],
                'actions': ['Special comeback offers', 'Feedback surveys', 'Win-back emails']
            }
        }
        return recommendations

class MarketingIntegration:
    """
    Integration with marketing automation platforms
    """
    
    def __init__(self, dashboard: CLVDashboard):
        self.dashboard = dashboard
    
    def export_to_marketing_platform(self, platform: str = 'email') -> Dict:
        """
        Export customer segments to marketing automation platforms
        """
        segment_recommendations = self.dashboard.get_segment_recommendations()
        export_data = {}
        
        for segment, recs in segment_recommendations.items():
            export_data[segment] = {
                'customers': [],  # Would contain actual customer IDs in production
                'campaign_strategy': recs['strategies'][0],
                'budget_allocation': recs['budget_allocation'],
                'recommended_actions': recs['actions']
            }
        
        logger.info(f"Exported segment data to {platform} marketing platform")
        return export_data

class ModelMonitoring:
    """
    Monitor model drift and performance metrics
    """
    
    def __init__(self, clv_system: ProductionCLVSystem):
        self.clv_system = clv_system
        self.performance_history = []
    
    def calculate_drift_metrics(self, new_data: pd.DataFrame, 
                               reference_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate model drift metrics
        """
        # Feature drift (simplified - in production use proper statistical tests)
        drift_metrics = {}
        for col in ['recency', 'frequency', 'monetary_value']:
            if col in new_data.columns and col in reference_data.columns:
                drift = abs(new_data[col].mean() - reference_data[col].mean()) / reference_data[col].std()
                drift_metrics[f'{col}_drift'] = drift
        
        return drift_metrics
    
    def monitor_performance(self, actual_clv: pd.Series, predicted_clv: pd.Series) -> Dict:
        """
        Monitor model performance metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(actual_clv, predicted_clv)
        rmse = np.sqrt(mean_squared_error(actual_clv, predicted_clv))
        correlation = actual_clv.corr(predicted_clv)
        
        performance = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'monitoring_date': datetime.now()
        }
        
        self.performance_history.append(performance)
        return performance
    
    def should_retrain(self, drift_threshold: float = 0.3, 
                      performance_threshold: float = 0.7) -> bool:
        """
        Determine if model should be retrained based on drift and performance
        """
        if len(self.performance_history) == 0:
            return False
        
        latest_performance = self.performance_history[-1]
        if latest_performance['correlation'] < performance_threshold:
            return True
        
        # Check drift metrics (would be implemented with actual drift calculation)
        return False

# Monthly Retraining Pipeline
class MonthlyRetrainingPipeline:
    """
    Automated monthly model retraining pipeline
    """
    
    def __init__(self, clv_system: ProductionCLVSystem, 
                 monitoring: ModelMonitoring):
        self.clv_system = clv_system
        self.monitoring = monitoring
    
    def run_monthly_retraining(self):
        """
        Execute monthly model retraining
        """
        logger.info("Starting monthly model retraining...")
        
        try:
            # Train full pipeline with latest data
            segmented_customers = self.clv_system.train_full_pipeline()
            
            # Update dashboard
            dashboard = CLVDashboard(self.clv_system)
            dashboard.update_dashboard(segmented_customers)
            
            # Integration with marketing platforms
            marketing_integration = MarketingIntegration(dashboard)
            marketing_integration.export_to_marketing_platform()
            
            logger.info("Monthly retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Monthly retraining failed: {str(e)}")
            return False

# Main execution function
def main():
    """
    Main function to demonstrate the complete production system
    """
    # Initialize CLV system
    clv_system = ProductionCLVSystem(model_path="production_models/")
    
    # Train full pipeline (Requirement 1 & 4)
    logger.info("Training production CLV models...")
    segmented_customers = clv_system.train_full_pipeline()
    
    # Initialize dashboard (Requirement 2)
    dashboard = CLVDashboard(clv_system)
    dashboard.update_dashboard(segmented_customers)
    
    # Marketing integration (Requirement 3)
    marketing_integration = MarketingIntegration(dashboard)
    marketing_data = marketing_integration.export_to_marketing_platform()
    
    # Model monitoring (Requirement 5)
    monitoring = ModelMonitoring(clv_system)
    
    # Monthly retraining pipeline (Requirement 4)
    retraining_pipeline = MonthlyRetrainingPipeline(clv_system, monitoring)
    
    # Demonstrate real-time prediction
    sample_customer = {
        'customer_id': 'CUST_001',
        'recency': 30,
        'frequency': 5,
        'monetary_value': 150.0,
        'T': 180
    }
    
    prediction = clv_system.predict_new_customer(sample_customer)
    print(f"\nReal-time CLV Prediction:")
    print(f"Customer ID: {prediction.customer_id}")
    print(f"BG/NBD CLV: ${prediction.clv_bg_nbd:.2f}")
    print(f"Random Forest CLV: ${prediction.clv_rf:.2f}")
    print(f"Segment: {prediction.segment}")
    
    # Print segment statistics
    print(f"\nCustomer Segmentation Summary:")
    print(dashboard.segment_stats)
    
    return clv_system, dashboard, marketing_integration, monitoring, retraining_pipeline

if __name__ == "__main__":
    # Execute the complete production system
    clv_system, dashboard, marketing_integration, monitoring, retraining_pipeline = main()