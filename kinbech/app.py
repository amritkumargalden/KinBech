# app.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from config import config
import logging

# ==================== Initialize Flask ====================
app = Flask(__name__)
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Load Models & Data ====================
def load_models():
    """Load trained model, scaler, and encoders"""
    try:
        model = pickle.load(open(app.config['MODEL_PATH'], 'rb'))
        scaler = pickle.load(open(app.config['SCALER_PATH'], 'rb'))
        le_category = pickle.load(open(app.config['ENCODER_PATH'], 'rb'))
        feature_cols = pickle.load(open(app.config['FEATURES_PATH'], 'rb'))
        
        logger.info("✓ Models loaded successfully")
        return model, scaler, le_category, feature_cols
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {e}")
        raise

def load_dataset():
    """Load processed dataset for comparisons"""
    try:
        df = pd.read_csv(app.config['DATASET_PATH'])
        logger.info(f"✓ Dataset loaded: {len(df)} records")
        return df
    except FileNotFoundError:
        logger.warning("Dataset not found for comparisons")
        return pd.DataFrame()

# Load on startup
try:
    MODEL, SCALER, LE_CATEGORY, FEATURE_COLS = load_models()
    DATASET = load_dataset()
except Exception as e:
    logger.error(f"Fatal error loading models: {e}")
    MODEL = SCALER = LE_CATEGORY = FEATURE_COLS = None
    DATASET = pd.DataFrame()

# ==================== Helper Functions ====================
def get_comparable_listings(category, price, limit=5):
    """Find comparable listings from the dataset"""
    if DATASET.empty:
        return []
    
    try:
        # Filter by category
        category_data = DATASET[DATASET['Category'] == category].copy()
        
        if len(category_data) == 0:
            return []
        
        # Sort by price similarity
        category_data['price_diff'] = abs(category_data['Price_NPR'] - price)
        comparable = category_data.nsmallest(limit, 'price_diff')
        
        return comparable[['Title', 'Price_NPR', 'Condition_Score', 'Age_years', 'Brand']].to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching comparable listings: {e}")
        return []

def validate_prediction_input(data):
    """Validate user input"""
    errors = []
    
    try:
        original_price = float(data.get('original_price', 0))
        age_years = float(data.get('age_years', 0))
        condition = float(data.get('condition', 0))
        
        if original_price <= 0:
            errors.append("Original price must be greater than 0")
        if original_price > 10_000_000:
            errors.append("Price seems unrealistic (> 10M NPR)")
            
        if age_years < 0 or age_years > 50:
            errors.append("Age must be between 0 and 50 years")
            
        if condition < 1 or condition > 10:
            errors.append("Condition score must be between 1 and 10")
    
    except ValueError:
        errors.append("Invalid input format — please use numbers")
    
    return errors

# ==================== Routes ====================

@app.route('/')
def home():
    """Homepage"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Price prediction page"""
    if request.method == 'GET':
        # Get unique categories for dropdown
        categories = sorted(DATASET['Category'].unique().tolist()) if not DATASET.empty else []
        return render_template('predict.html', categories=categories)
    
    elif request.method == 'POST':
        try:
            # Validate input
            errors = validate_prediction_input(request.form)
            if errors:
                return render_template('predict.html', 
                                     error='; '.join(errors),
                                     categories=sorted(DATASET['Category'].unique().tolist()))
            
            # Get user inputs
            original_price = float(request.form.get('original_price'))
            age_years = float(request.form.get('age_years'))
            condition = float(request.form.get('condition'))
            storage_gb = float(request.form.get('storage_gb', 0))
            category = request.form.get('category', 'Electronics')
            
            # Feature engineering (match training pipeline)
            age_depreciation = age_years * 0.08
            condition_discount = (10 - condition) * 0.05
            total_depreciation = min(max(age_depreciation + condition_discount, 0.05), 0.85)
            
            estimated_original = original_price / (1 - total_depreciation)
            depreciation_rate = total_depreciation
            value_retention = 1 - total_depreciation
            condition_factor = condition / 10
            
            price_per_gb = original_price / storage_gb if storage_gb > 0 else 0
            
            # Encode category
            try:
                category_encoded = LE_CATEGORY.transform([category])[0]
            except:
                category_encoded = 0
            
            # Brand encoding (default to 0 for user input)
            brand_encoded = 0
            
            # Build feature array (must match training features)
            features = np.array([[
                estimated_original,
                age_years * 12,  # Convert to months
                condition,
                depreciation_rate,
                value_retention,
                storage_gb,
                category_encoded,
                brand_encoded
            ]])
            
            # Scale features
            features_scaled = SCALER.transform(features)
            
            # Predict
            predicted_price = MODEL.predict(features_scaled)[0]
            
            # Ensure positive prediction
            predicted_price = max(predicted_price, 100)
            
            # Get comparable listings
            comparables = get_comparable_listings(category, predicted_price, limit=5)
            
            # Calculate depreciation percentage
            depreciation_pct = ((original_price - predicted_price) / original_price) * 100
            
            # Store in session for logging (optional)
            session['last_prediction'] = {
                'original_price': original_price,
                'predicted_price': predicted_price,
                'category': category,
                'timestamp': datetime.now().isoformat()
            }
            
            return render_template('results.html',
                                 predicted_price=round(predicted_price, 2),
                                 original_price=round(original_price, 2),
                                 depreciation_pct=round(depreciation_pct, 2),
                                 category=category,
                                 age_years=age_years,
                                 condition=condition,
                                 comparable_listings=comparables)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template('predict.html',
                                 error=f"Prediction failed: {str(e)}",
                                 categories=sorted(DATASET['Category'].unique().tolist()))

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    try:
        if DATASET.empty:
            stats = {
                'total_listings': 0,
                'avg_price': 0,
                'min_price': 0,
                'max_price': 0,
                'avg_age': 0,
                'avg_condition': 0
            }
        else:
            stats = {
                'total_listings': len(DATASET),
                'avg_price': round(DATASET['Price_NPR'].mean(), 2),
                'min_price': round(DATASET['Price_NPR'].min(), 2),
                'max_price': round(DATASET['Price_NPR'].max(), 2),
                'avg_age': round(DATASET['Age_years'].mean(), 2),
                'avg_condition': round(DATASET['Condition_Score'].mean(), 2)
            }
        
        return render_template('dashboard.html', stats=stats)
    
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', stats={})

@app.route('/api/price-stats')
def price_stats():
    """API endpoint for price statistics"""
    try:
        if DATASET.empty:
            return jsonify({'error': 'No data available'}), 400
        
        stats = {
            'mean': float(DATASET['Price_NPR'].mean()),
            'median': float(DATASET['Price_NPR'].median()),
            'std': float(DATASET['Price_NPR'].std()),
            'min': float(DATASET['Price_NPR'].min()),
            'max': float(DATASET['Price_NPR'].max()),
            'count': int(len(DATASET))
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/category-stats')
def category_stats():
    """API endpoint for category statistics"""
    try:
        if DATASET.empty:
            return jsonify({'error': 'No data available'}), 400
        
        cat_stats = DATASET.groupby('Category').agg({
            'Price_NPR': ['mean', 'count'],
            'Condition_Score': 'mean',
            'Age_years': 'mean'
        }).round(2)
        
        result = cat_stats.to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'dataset_loaded': len(DATASET) > 0 if not DATASET.empty else False,
        'timestamp': datetime.now().isoformat()
    })

# ==================== Error Handlers ====================
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return render_template('500.html'), 500

# ==================== Run App ====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
