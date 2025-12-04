# api_endpoints.py
from flask import Flask, request, jsonify
from production_clv_system import ProductionCLVSystem, CLVPrediction
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Initialize CLV system (load pre-trained models)
clv_system = ProductionCLVSystem(model_path="production_models/")
clv_system.load_models()

@app.route('/predict_clv', methods=['POST'])
def predict_clv():
    """
    API endpoint for real-time CLV prediction
    """
    try:
        customer_data = request.json
        prediction = clv_system.predict_new_customer(customer_data)
        
        return jsonify({
            'customer_id': prediction.customer_id,
            'clv_bg_nbd': prediction.clv_bg_nbd,
            'clv_rf': prediction.clv_rf,
            'segment': prediction.segment,
            'recency': prediction.recency,
            'frequency': prediction.frequency,
            'monetary_value': prediction.monetary_value,
            'prediction_date': prediction.prediction_date.isoformat()
        })
    except Exception as e:
        logger.error(f"CLV prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_segments', methods=['GET'])
def get_segments():
    """
    API endpoint to get current customer segments
    """
    # In production, this would query the latest segmented data
    return jsonify({
        'segments': ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk'],
        'last_updated': '2024-01-15T10:30:00Z'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'healthy', 'models_loaded': clv_system.bgf_model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)