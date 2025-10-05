"""
SageMaker Inference Script for Real-time Bankruptcy Predictions

Provides low-latency predictions for bankruptcy risk assessment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model artifacts
model = None
scaler = None
feature_names = None


def model_fn(model_dir):
    """
    Load model from model directory (called once when endpoint is created)
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Dictionary containing model, scaler, and feature names
    """
    global model, scaler, feature_names
    
    logger.info(f"Loading model from {model_dir}")
    
    # Load model
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    # Load feature names
    feature_path = os.path.join(model_dir, 'feature_names.json')
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    logger.info("Model loaded successfully")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare input data
    
    Args:
        request_body: Request payload
        content_type: Content type of request
        
    Returns:
        Prepared input data as DataFrame
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle single record or batch
        if isinstance(input_data, dict):
            # Single record
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Batch of records
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the features used in training
        df = df[feature_names]
        
        return df
    
    elif content_type == 'text/csv':
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
        
        # Select only the features used in training
        df = df[feature_names]
        
        return df
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_artifacts):
    """
    Make predictions on input data
    
    Args:
        input_data: Prepared input DataFrame
        model_artifacts: Dictionary with model, scaler, and feature names
        
    Returns:
        Predictions dictionary
    """
    logger.info(f"Making predictions for {len(input_data)} records")
    
    # Extract artifacts
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    
    # Scale features
    X_scaled = scaler.transform(input_data)
    
    # Get predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Prepare results
    results = []
    for i in range(len(input_data)):
        result = {
            'prediction': int(predictions[i]),
            'probability_not_bankrupt': float(probabilities[i][0]),
            'probability_bankrupt': float(probabilities[i][1]),
            'risk_level': get_risk_level(probabilities[i][1])
        }
        results.append(result)
    
    logger.info("Predictions completed successfully")
    
    return results


def output_fn(predictions, accept='application/json'):
    """
    Serialize predictions for response
    
    Args:
        predictions: Model predictions
        accept: Accepted response content type
        
    Returns:
        Serialized predictions
    """
    if accept == 'application/json':
        return json.dumps({
            'predictions': predictions,
            'model_version': '1.0',
            'timestamp': pd.Timestamp.now().isoformat()
        }), accept
    
    elif accept == 'text/csv':
        df = pd.DataFrame(predictions)
        return df.to_csv(index=False), accept
    
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def get_risk_level(bankruptcy_probability):
    """
    Categorize bankruptcy risk based on probability
    
    Args:
        bankruptcy_probability: Probability of bankruptcy (0-1)
        
    Returns:
        Risk level category
    """
    if bankruptcy_probability < 0.2:
        return 'LOW'
    elif bankruptcy_probability < 0.5:
        return 'MEDIUM'
    elif bankruptcy_probability < 0.8:
        return 'HIGH'
    else:
        return 'CRITICAL'


# Health check endpoint
def ping():
    """
    Health check for the model endpoint
    
    Returns:
        Status code
    """
    try:
        # Verify model is loaded
        if model is None or scaler is None or feature_names is None:
            return 503  # Service Unavailable
        
        return 200  # OK
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return 503


# Batch transform support
def transform_fn(model_artifacts, request_body, content_type, accept):
    """
    Complete transform function for batch predictions
    
    Args:
        model_artifacts: Model artifacts dictionary
        request_body: Input data
        content_type: Input content type
        accept: Output content type
        
    Returns:
        Serialized predictions
    """
    # Parse input
    input_data = input_fn(request_body, content_type)
    
    # Make predictions
    predictions = predict_fn(input_data, model_artifacts)
    
    # Format output
    response_body, response_content_type = output_fn(predictions, accept)
    
    return response_body, response_content_type


# Example usage for local testing
if __name__ == '__main__':
    # Example test data
    test_data = {
        'current_ratio': 1.5,
        'quick_ratio': 1.2,
        'net_profit_margin': 0.08,
        'roa': 0.05,
        'roe': 0.12,
        'debt_to_equity': 0.6,
        'debt_to_assets': 0.35,
        'asset_turnover': 1.1,
        'altman_z_score': 2.8,
        'working_capital_ratio': 0.15,
        'retained_earnings_ratio': 0.20,
        'ebit_to_assets': 0.08,
        'market_to_book_ratio': 1.5,
        'sales_to_assets': 1.2
    }
    
    # Simulate prediction
    print("Example bankruptcy prediction:")
    print(f"Input: {test_data}")
    
    # In production, this would use the actual model
    # For testing, we simulate a prediction
    simulated_prediction = {
        'prediction': 0,
        'probability_not_bankrupt': 0.95,
        'probability_bankrupt': 0.05,
        'risk_level': 'LOW'
    }
    
    print(f"Output: {simulated_prediction}")
    print(f"\nRisk Assessment: Company has {simulated_prediction['risk_level']} bankruptcy risk")
