# app.py
import os
import pickle
from datetime import datetime
import logging

import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    from .config import config
except ImportError:
    from config import config

# ==================== Initialize Flask ====================
app = Flask(__name__)
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Load Models & Data ====================
def load_models():
    """Load the trained model and legacy metadata if available."""
    model_path = app.config['MODEL_PATH']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    logger.info("✓ Model loaded successfully")
    return model


def load_dataset():
    """Load processed dataset for comparisons."""
    try:
        df = pd.read_csv(app.config['DATASET_PATH'])
        logger.info(f"✓ Dataset loaded: {len(df)} records")
        return df
    except FileNotFoundError:
        logger.warning("Dataset not found for comparisons")
        return pd.DataFrame()


# Load on startup
try:
    MODEL = load_models()
    DATASET = load_dataset()
except Exception as e:
    logger.error(f"Fatal error loading models: {e}")
    MODEL = None
    DATASET = pd.DataFrame()

# ==================== Helper Functions ====================
def get_comparable_listings(category, price, limit=5):
    """Find comparable listings from the dataset."""
    if DATASET.empty:
        return []

    try:
        category_data = DATASET[DATASET['Category'] == category].copy()
        if len(category_data) == 0:
            return []

        category_data['price_diff'] = abs(category_data['Price_NPR'] - price)
        comparable = category_data.nsmallest(limit, 'price_diff')
        return comparable[['Title', 'Price_NPR', 'Condition_Score', 'Age_years', 'Brand']].to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching comparable listings: {e}")
        return []


def validate_prediction_input(data):
    """Validate user input."""
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


def _prediction_features(form_data):
    """Normalize form data for the trained sklearn Pipeline."""
    category = str(form_data.get('category', '')).strip()
    condition = str(form_data.get('condition', '')).strip()
    storage_gb = float(form_data.get('storage_gb', 0) or 0)
    age_years = float(form_data.get('age_years', 0) or 0)
    original_price = float(form_data.get('original_price', 0) or 0)

    if not category:
        raise ValueError('Category is required')
    if not condition:
        raise ValueError('Condition is required')

    return {
        'Category': category,
        'Condition': condition,
        'Storage_GB': storage_gb,
        'Age_years': age_years,
        'Price_NPR': original_price,
    }


# ==================== Routes ====================
@app.route('/')
def home():
    """Homepage."""
    total_listings = len(DATASET) if not DATASET.empty else 0
    total_predictions = 0
    return render_template('index.html', total_listings=total_listings, total_predictions=total_predictions)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Price prediction page."""
    categories = sorted(DATASET['Category'].unique().tolist()) if not DATASET.empty else []
    top_brands = DATASET['Brand'].value_counts().head(5).index.tolist() if not DATASET.empty else []
    top_categories = DATASET['Category'].value_counts().head(5).index.tolist() if not DATASET.empty else []
    popular_products = DATASET['Model'].value_counts().head(5).index.tolist() if not DATASET.empty else []

    if request.method == 'GET':
        return render_template(
            'predict.html',
            categories=categories,
            top_brands=top_brands,
            top_categories=top_categories,
            popular_products=popular_products,
        )

    if request.method == 'POST':
        if MODEL is None:
            return render_template(
                'predict.html',
                error='Model is unavailable. Please retrain the model first.',
                categories=categories,
                top_brands=top_brands,
                top_categories=top_categories,
                popular_products=popular_products,
            ), 503

        try:
            data = _prediction_features(request.form)
            validation_errors = validate_prediction_input(request.form)
            if validation_errors:
                return render_template(
                    'predict.html',
                    error='; '.join(validation_errors),
                    categories=categories,
                    top_brands=top_brands,
                    top_categories=top_categories,
                    popular_products=popular_products,
                ), 400

            feature_row = pd.DataFrame([data])
            predicted_price = float(MODEL.predict(feature_row[['Category', 'Condition', 'Storage_GB', 'Age_years']])[0])
            original_price = float(data['Price_NPR'])
            depreciation_pct = round(max(0.0, ((original_price - predicted_price) / original_price) * 100), 2) if original_price else 0.0
            comparables = get_comparable_listings(data['Category'], original_price, limit=5)

            return render_template(
                'results.html',
                predicted_price=round(predicted_price, 2),
                original_price=round(original_price, 2),
                depreciation_pct=depreciation_pct,
                category=data['Category'],
                age_years=data['Age_years'],
                condition=data['Condition'],
                comparable_listings=comparables,
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template(
                'predict.html',
                error=f"Error: {str(e)}",
                categories=categories,
                top_brands=top_brands,
                top_categories=top_categories,
                popular_products=popular_products,
            ), 500


@app.route('/dashboard')
def dashboard():
    """Analytics dashboard."""
    try:
        if DATASET.empty:
            stats = {'total_listings': 0, 'avg_price': 0, 'min_price': 0, 'max_price': 0, 'avg_age': 0, 'avg_condition': 0}
        else:
            stats = {
                'total_listings': len(DATASET),
                'avg_price': round(DATASET['Price_NPR'].mean(), 2),
                'min_price': round(DATASET['Price_NPR'].min(), 2),
                'max_price': round(DATASET['Price_NPR'].max(), 2),
                'avg_age': round(DATASET['Age_years'].mean(), 2),
                'avg_condition': round(DATASET['Condition_Score'].mean(), 2),
            }

        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', stats={})


@app.route('/api/price-stats')
def price_stats():
    """API endpoint for price statistics."""
    try:
        if DATASET.empty:
            return jsonify({'error': 'No data available'}), 400

        stats = {
            'mean': float(DATASET['Price_NPR'].mean()),
            'median': float(DATASET['Price_NPR'].median()),
            'std': float(DATASET['Price_NPR'].std()),
            'min': float(DATASET['Price_NPR'].min()),
            'max': float(DATASET['Price_NPR'].max()),
            'count': int(len(DATASET)),
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/category-stats')
def category_stats():
    """API endpoint for category statistics."""
    try:
        if DATASET.empty:
            return jsonify({'error': 'No data available'}), 400

        cat_stats = DATASET.groupby('Category').agg({
            'Price_NPR': ['mean', 'count'],
            'Condition_Score': 'mean',
            'Age_years': 'mean',
        }).round(2)

        result = cat_stats.to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'dataset_loaded': len(DATASET) > 0 if not DATASET.empty else False,
        'timestamp': datetime.now().isoformat(),
    })


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
