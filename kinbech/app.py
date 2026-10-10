# app.py
import os
import pickle
import math
import json
from datetime import datetime
import logging

import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    from .config import config
    from .prediction_log import PredictionLogError, log_prediction
except ImportError:
    from config import config
    from prediction_log import PredictionLogError, log_prediction

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


def load_model_version():
    """Read optional model metadata for prediction audit events."""
    metadata_path = app.config['MODEL_METADATA_PATH']
    try:
        with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
    except FileNotFoundError:
        logger.info("Model metadata is unavailable; prediction events will use no model version")
        return None
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read model metadata at %s: %s", metadata_path, error)
        return None

    version = metadata.get('model_version') if isinstance(metadata, dict) else None
    if not isinstance(version, str) or not version.strip():
        logger.warning("Model metadata at %s has no valid model_version", metadata_path)
        return None
    return version


# Load on startup
try:
    MODEL = load_models()
    DATASET = load_dataset()
except Exception as e:
    logger.error(f"Fatal error loading models: {e}")
    MODEL = None
    DATASET = pd.DataFrame()
MODEL_VERSION = load_model_version()
prediction_count = 0

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


def analyze_listing_price(category, brand, condition, listing_price, age_years=None):
    """Flag listing prices outside Tukey fences for sufficiently sized peer groups."""
    required_columns = {'Category', 'Brand', 'Condition', 'Price_NPR'}
    if DATASET.empty or not required_columns.issubset(DATASET.columns):
        return {
            'status': 'insufficient_data',
            'message': 'There is not enough market data to assess this listing price.',
            'sample_count': 0,
        }

    price_column = pd.to_numeric(DATASET['Price_NPR'], errors='coerce')
    peers = DATASET.assign(_price=price_column)
    peers = peers[peers['_price'].notna() & (peers['_price'] > 0)]
    condition_peers = peers[
        (peers['Category'] == category)
        & (peers['Brand'] == brand)
        & (peers['Condition'] == condition)
    ]
    groups = []
    if age_years is not None and 'Age_years' in condition_peers:
        peer_ages = pd.to_numeric(condition_peers['Age_years'], errors='coerce')
        age_matched = condition_peers[(peer_ages - age_years).abs() <= 1]
        groups.append(('category_brand_condition_age', age_matched))
    groups.extend((
        (
            'category_brand_condition',
            condition_peers,
        ),
        (
            'category_brand',
            peers[(peers['Category'] == category) & (peers['Brand'] == brand)],
        ),
        (
            'category',
            peers[peers['Category'] == category],
        ),
    ))
    comparison_level, comparable = next(
        (
            (level, group)
            for level, group in groups
            if len(group) >= 5
        ),
        (None, None),
    )
    if comparable is None:
        return {
            'status': 'insufficient_data',
            'message': 'There are fewer than 5 comparable listings, so this price could not be assessed.',
            'sample_count': len(peers[peers['Category'] == category]),
        }

    log_prices = comparable['_price'].map(math.log)
    first_quartile = float(log_prices.quantile(0.25))
    third_quartile = float(log_prices.quantile(0.75))
    interquartile_range = third_quartile - first_quartile
    lower_bound = math.exp(first_quartile - 1.5 * interquartile_range)
    upper_bound = math.exp(third_quartile + 1.5 * interquartile_range)
    if listing_price < lower_bound:
        status = 'unusually_low'
        message = 'This price is unusually low compared with similar listings. Review it before proceeding; an outlier alone is not proof of fraud.'
    elif listing_price > upper_bound:
        status = 'unusually_high'
        message = 'This price is unusually high compared with similar listings. Review it before proceeding; an outlier alone is not proof of fraud.'
    else:
        status = 'within_range'
        message = 'This price is within the expected range for similar listings.'

    return {
        'status': status,
        'message': message,
        'sample_count': len(comparable),
        'comparison_level': comparison_level,
        'comparison_basis': 'Category, brand, condition, and within 1 year of age'
        if comparison_level == 'category_brand_condition_age'
        else 'Category, brand, and condition'
        if comparison_level == 'category_brand_condition'
        else 'Category and brand'
        if comparison_level == 'category_brand'
        else 'Category',
        'reference_range': {
            'minimum': round(lower_bound, 2),
            'maximum': round(upper_bound, 2),
        },
    }


def validate_prediction_input(data):
    """Validate user input."""
    errors = []
    category = str(data.get('category', '')).strip()
    condition = str(data.get('condition', '')).strip()
    brand = str(data.get('brand', '')).strip()

    available_categories = set(DATASET['Category'].dropna().astype(str)) if not DATASET.empty else set()
    available_conditions = set(DATASET['Condition'].dropna().astype(str)) if not DATASET.empty else set()
    available_brands = set(DATASET['Brand'].dropna().astype(str)) if not DATASET.empty else set()
    if category not in available_categories:
        errors.append("Select a category from the available options")
    if condition not in available_conditions:
        errors.append("Select a condition from the available options")
    if brand not in available_brands:
        errors.append("Select a brand from the available options")

    for name, label, minimum, maximum, required in (
        ('storage_gb', 'Storage', 0, 100_000, False),
        ('age_years', 'Age', 0, 50, True),
        ('original_price', 'Original price', 0, 10_000_000, False),
        ('listing_price', 'Listing price', 0, 10_000_000, False),
    ):
        raw_value = data.get(name)
        if required and (raw_value is None or raw_value == ''):
            errors.append(f"{label} is required")
            continue
        try:
            value = float(raw_value or 0)
            if not math.isfinite(value):
                raise ValueError
            if value < minimum or value > maximum:
                errors.append(f"{label} must be between {minimum:,} and {maximum:,}")
        except (TypeError, ValueError):
            errors.append(f"{label} must be a valid number")

    return errors


def _prediction_features(form_data):
    """Normalize form data for the trained sklearn Pipeline."""
    category = str(form_data.get('category', '')).strip()
    condition = str(form_data.get('condition', '')).strip()
    brand = str(form_data.get('brand', '')).strip()
    storage_gb = float(form_data.get('storage_gb', 0) or 0)
    age_years = float(form_data.get('age_years', 0) or 0)

    if not category:
        raise ValueError('Category is required')
    if not condition:
        raise ValueError('Condition is required')
    if not brand:
        raise ValueError('Brand is required')

    return {
        'Category': category,
        'Condition': condition,
        'Brand': brand,
        'Storage_GB': storage_gb,
        'Age_years': age_years,
    }


def _predict_price(feature_values):
    feature_row = pd.DataFrame([feature_values])
    return float(MODEL.predict(
        feature_row[['Category', 'Condition', 'Brand', 'Storage_GB', 'Age_years']]
    )[0])


def _record_prediction(request_data, result, source):
    """Persist a successful estimate before reporting it to the caller."""
    return log_prediction(
        app.config['PREDICTION_LOG_PATH'],
        request_data=request_data,
        result=result,
        source=source,
        model_version=MODEL_VERSION,
    )


# ==================== Routes ====================
@app.route('/')
def home():
    """Homepage."""
    total_listings = len(DATASET) if not DATASET.empty else 0
    total_predictions = prediction_count
    return render_template('index.html', total_listings=total_listings, total_predictions=total_predictions)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Price prediction page."""
    global prediction_count
    categories = sorted(DATASET['Category'].unique().tolist()) if not DATASET.empty else []
    conditions = []
    if not DATASET.empty:
        condition_data = DATASET[['Condition', 'Condition_Score']].dropna()
        conditions = condition_data.sort_values(
            'Condition_Score', ascending=False
        )['Condition'].drop_duplicates().tolist()
    top_brands = DATASET['Brand'].value_counts().head(5).index.tolist() if not DATASET.empty else []
    brands = sorted(DATASET['Brand'].dropna().astype(str).unique().tolist()) if not DATASET.empty else []
    top_categories = DATASET['Category'].value_counts().head(5).index.tolist() if not DATASET.empty else []
    popular_products = DATASET['Model'].value_counts().head(5).index.tolist() if not DATASET.empty else []

    if request.method == 'GET':
        return render_template(
            'predict.html',
            categories=categories,
            conditions=conditions,
            brands=brands,
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
                conditions=conditions,
                brands=brands,
                top_brands=top_brands,
                top_categories=top_categories,
                popular_products=popular_products,
            ), 503
        try:
            validation_errors = validate_prediction_input(request.form)
            if validation_errors:
                return render_template(
                    'predict.html',
                    error='; '.join(validation_errors),
                    categories=categories,
                    conditions=conditions,
                    brands=brands,
                    top_brands=top_brands,
                    top_categories=top_categories,
                    popular_products=popular_products,
                ), 400

            data = _prediction_features(request.form)
            predicted_price = _predict_price(data)
            original_price = float(request.form.get('original_price', 0) or 0)
            listing_price = float(request.form.get('listing_price', 0) or 0)
            depreciation_pct = round(((original_price - predicted_price) / original_price) * 100, 2) if original_price else 0.0
            comparables = get_comparable_listings(data['Category'], predicted_price, limit=5)
            price_analysis = (
                analyze_listing_price(
                    data['Category'], data['Brand'], data['Condition'], listing_price
                )
                if listing_price
                else None
            )
            _record_prediction(
                {
                    **data,
                    'Original_Price_NPR': original_price or None,
                    'Listing_Price_NPR': listing_price or None,
                },
                {
                    'predicted_price': round(predicted_price, 2),
                    'original_price': original_price or None,
                    'depreciation_pct': depreciation_pct,
                    'listing_price_analysis': price_analysis,
                },
                'form',
            )
            prediction_count += 1

            return render_template(
                'results.html',
                predicted_price=round(predicted_price, 2),
                original_price=round(original_price, 2),
                depreciation_pct=depreciation_pct,
                listing_price=listing_price or None,
                price_analysis=price_analysis,
                category=data['Category'],
                age_years=data['Age_years'],
                condition=data['Condition'],
                brand=data['Brand'],
                comparable_listings=comparables,
            )
        except PredictionLogError:
            logger.exception("Could not log form prediction")
            return render_template(
                'predict.html',
                error='The prediction could not be recorded. Please try again.',
                categories=categories,
                conditions=conditions,
                brands=brands,
                top_brands=top_brands,
                top_categories=top_categories,
                popular_products=popular_products,
            ), 503
        except Exception as e:
            logger.exception("Prediction error: %s", e)
            return render_template(
                'predict.html',
                error=f"Error: {str(e)}",
                categories=categories,
                conditions=conditions,
                brands=brands,
                top_brands=top_brands,
                top_categories=top_categories,
                popular_products=popular_products,
            ), 500


@app.route('/api/options', methods=['GET', 'OPTIONS'])
def prediction_options():
    """Return model-supported categories, conditions, and brands."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if DATASET.empty:
        return jsonify({'error': 'No prediction data is available'}), 503

    conditions = DATASET[['Condition', 'Condition_Score']].dropna()
    conditions = conditions.sort_values('Condition_Score', ascending=False)
    return jsonify({
        'categories': sorted(DATASET['Category'].dropna().astype(str).unique().tolist()),
        'conditions': conditions['Condition'].astype(str).drop_duplicates().tolist(),
        'brands': sorted(DATASET['Brand'].dropna().astype(str).unique().tolist()),
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_api():
    """Return a price estimate for the EverSales storefront."""
    global prediction_count
    if request.method == 'OPTIONS':
        return ('', 204)
    if MODEL is None:
        return jsonify({'error': 'The prediction model is unavailable'}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Send prediction details as a JSON object'}), 400

    errors = validate_prediction_input(data)
    if errors:
        return jsonify({'error': '; '.join(errors), 'validation_errors': errors}), 400

    try:
        feature_values = _prediction_features(data)
        predicted_price = _predict_price(feature_values)
        original_price = float(data.get('original_price') or 0)
        listing_price = float(data.get('listing_price') or 0)
        depreciation_pct = (
            round(((original_price - predicted_price) / original_price) * 100, 2)
            if original_price else None
        )
        comparables = get_comparable_listings(
            feature_values['Category'], predicted_price, limit=5
        )
        original_price = original_price or None
        price_analysis = (
            analyze_listing_price(
                feature_values['Category'],
                feature_values['Brand'],
                feature_values['Condition'],
                listing_price,
            )
            if listing_price
            else None
        )
        result = {
            'predicted_price': round(predicted_price, 2),
            'original_price': original_price,
            'depreciation_pct': depreciation_pct,
            'listing_price': listing_price or None,
            'listing_price_analysis': price_analysis,
        }
        _record_prediction(
            {
                **feature_values,
                'Original_Price_NPR': original_price,
                'Listing_Price_NPR': listing_price or None,
            },
            result,
            'api',
        )
        prediction_count += 1

        return jsonify({
            **result,
            'category': feature_values['Category'],
            'condition': feature_values['Condition'],
            'brand': feature_values['Brand'],
            'age_years': feature_values['Age_years'],
            'comparables': [
                {
                    'title': str(item['Title']),
                    'price_npr': float(item['Price_NPR']),
                    'condition_score': float(item['Condition_Score']),
                    'age_years': float(item['Age_years']),
                    'brand': str(item['Brand']),
                }
                for item in comparables
            ],
        })
    except PredictionLogError:
        logger.exception("Could not log API prediction")
        return jsonify({'error': 'The prediction could not be recorded'}), 503
    except Exception as e:
        logger.exception("Prediction API error")
        return jsonify({'error': 'Could not calculate a price estimate'}), 500


@app.route('/api/analyze-listing-prices', methods=['POST', 'OPTIONS'])
def analyze_listing_prices_api():
    """Analyze marketplace listing prices in a bounded batch."""
    if request.method == 'OPTIONS':
        return ('', 204)
    if DATASET.empty:
        return jsonify({'error': 'The market dataset is unavailable'}), 503

    payload = request.get_json(silent=True)
    listings = payload.get('listings') if isinstance(payload, dict) else None
    if not isinstance(listings, list) or not listings or len(listings) > 50:
        return jsonify({
            'error': 'Send between 1 and 50 listings as a JSON array',
        }), 400

    categories = set(DATASET['Category'].dropna().astype(str))
    brands = set(DATASET['Brand'].dropna().astype(str))
    conditions = set(DATASET['Condition'].dropna().astype(str))
    results = []
    for index, listing in enumerate(listings):
        listing_id = listing.get('id', index) if isinstance(listing, dict) else index
        if (
            isinstance(listing_id, bool)
            or not isinstance(listing_id, (str, int, float))
            or (isinstance(listing_id, str) and len(listing_id) > 100)
            or (isinstance(listing_id, float) and not math.isfinite(listing_id))
        ):
            listing_id = index
        if not isinstance(listing, dict):
            results.append({
                'id': listing_id,
                'status': 'unassessed',
                'message': 'Listing details are invalid; price was not assessed.',
                'sample_count': 0,
            })
            continue

        category = str(listing.get('category', '')).strip()
        brand = str(listing.get('brand', '')).strip()
        condition = str(listing.get('condition', '')).strip()
        try:
            price_value = listing.get('price')
            age_value = listing.get('age_years')
            if isinstance(price_value, bool) or isinstance(age_value, bool):
                raise ValueError
            price = float(price_value)
            age_years = float(age_value)
        except (TypeError, ValueError):
            results.append({
                'id': listing_id,
                'status': 'unassessed',
                'message': 'A valid listing price and item age are required for a price check.',
                'sample_count': 0,
            })
            continue

        if not math.isfinite(price) or price <= 0 or price > 10_000_000:
            message = 'Listing price must be greater than 0 and no more than NPR 10,000,000.'
        elif not math.isfinite(age_years) or age_years < 0 or age_years > 50:
            message = 'Item age must be between 0 and 50 years.'
        elif category not in categories or brand not in brands or condition not in conditions:
            message = 'Category, brand, or condition is not supported by the KinBech market dataset.'
        else:
            results.append({
                'id': listing_id,
                **analyze_listing_price(
                    category, brand, condition, price, age_years=age_years
                ),
            })
            continue

        results.append({
            'id': listing_id,
            'status': 'unassessed',
            'message': message,
            'sample_count': 0,
        })

    return jsonify({'results': results})


@app.after_request
def allow_storefront_prediction_requests(response):
    """Expose the public, non-personalized prediction API to the EverSales UI."""
    if request.path.startswith('/api/'):
        allowed_origins = {
            origin.strip()
            for origin in os.getenv(
                'EVERSALES_ORIGINS',
                'http://127.0.0.1:8000,http://localhost:8000',
            ).split(',')
            if origin.strip()
        }
        origin = request.headers.get('Origin')
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.vary.add('Origin')
    return response


@app.route('/dashboard')
def dashboard():
    """Show market patterns in the available listing dataset."""
    trends = build_market_trends(DATASET)
    return render_template('dashboard.html', **trends)


def build_market_trends(dataset):
    """Aggregate the current listing snapshot for the market trends dashboard."""
    empty_stats = {
        'total_listings': 0,
        'avg_price': 0,
        'median_price': 0,
        'min_price': 0,
        'max_price': 0,
        'avg_age': 0,
        'avg_condition': 0,
    }
    empty_trends = {
        'stats': empty_stats,
        'category_trends': [],
        'brand_trends': [],
        'condition_trends': [],
        'age_trends': [],
        'location_trends': [],
        'insights': [],
    }
    if dataset.empty or 'Price_NPR' not in dataset:
        return empty_trends

    market = dataset.copy()
    market['_market_price'] = pd.to_numeric(market['Price_NPR'], errors='coerce')
    market = market[market['_market_price'].notna() & (market['_market_price'] > 0)]
    if market.empty:
        return empty_trends

    prices = market['_market_price']
    ages = (
        pd.to_numeric(market['Age_years'], errors='coerce')
        if 'Age_years' in market
        else pd.Series(dtype=float)
    )
    conditions = (
        pd.to_numeric(market['Condition_Score'], errors='coerce')
        if 'Condition_Score' in market
        else pd.Series(dtype=float)
    )
    stats = {
        'total_listings': len(market),
        'avg_price': round(float(prices.mean()), 2),
        'median_price': round(float(prices.median()), 2),
        'min_price': round(float(prices.min()), 2),
        'max_price': round(float(prices.max()), 2),
        'avg_age': round(float(ages.mean()), 2) if ages.notna().any() else 0,
        'avg_condition': round(float(conditions.mean()), 2) if conditions.notna().any() else 0,
    }

    def price_groups(column, limit=None, count_sorted=False):
        if column not in market:
            return []
        grouped = (
            market.dropna(subset=[column])
            .groupby(column, observed=True)['_market_price']
            .agg(median_price='median', listings='size')
            .reset_index()
        )
        if count_sorted:
            grouped = grouped.sort_values(
                ['listings', 'median_price'], ascending=[False, False]
            ).head(limit)
        else:
            grouped = grouped.sort_values('median_price', ascending=False)
            if limit is not None:
                grouped = grouped.head(limit)
        if grouped.empty:
            return []
        maximum = float(grouped['median_price'].max())
        return [
            {
                'label': str(row[column]),
                'median_price': round(float(row['median_price']), 2),
                'listings': int(row['listings']),
                'bar_pct': round(float(row['median_price']) / maximum * 100, 1)
                if maximum > 0
                else 0,
            }
            for _, row in grouped.iterrows()
        ]

    category_trends = price_groups('Category')
    brand_trends = price_groups('Brand', limit=8, count_sorted=True)
    location_trends = price_groups('Location', limit=8, count_sorted=True)

    condition_trends = []
    if 'Condition' in market:
        condition_data = market.dropna(subset=['Condition']).groupby(
            'Condition', observed=True
        ).agg(
            median_price=('_market_price', 'median'),
            listings=('_market_price', 'size'),
            condition_score=(
                'Condition_Score',
                'mean',
            ) if 'Condition_Score' in market else ('_market_price', 'size'),
        ).reset_index()
        if 'Condition_Score' not in market:
            condition_data = condition_data.drop(columns='condition_score')
        condition_data = condition_data.sort_values(
            'condition_score' if 'Condition_Score' in market else 'median_price'
        )
        maximum = float(condition_data['median_price'].max()) if not condition_data.empty else 0
        condition_trends = [
            {
                'label': str(row['Condition']),
                'median_price': round(float(row['median_price']), 2),
                'listings': int(row['listings']),
                'bar_pct': round(float(row['median_price']) / maximum * 100, 1)
                if maximum > 0
                else 0,
                'condition_score': round(float(row['condition_score']), 1)
                if 'Condition_Score' in market and pd.notna(row['condition_score'])
                else None,
            }
            for _, row in condition_data.iterrows()
        ]

    age_trends = []
    if 'Age_years' in market:
        age_data = market.assign(
            _age=pd.to_numeric(market['Age_years'], errors='coerce')
        ).dropna(subset=['_age'])
        age_data = age_data[age_data['_age'] >= 0].copy()
        age_data['_age_band'] = pd.cut(
            age_data['_age'],
            bins=[0, 1, 2, 3, 5, float('inf')],
            labels=['Under 1 year', '1–2 years', '2–3 years', '3–5 years', '5+ years'],
            right=False,
        )
        grouped_ages = (
            age_data.dropna(subset=['_age_band'])
            .groupby('_age_band', observed=False)['_market_price']
            .agg(median_price='median', listings='count')
            .reindex(['Under 1 year', '1–2 years', '2–3 years', '3–5 years', '5+ years'])
            .dropna(subset=['median_price'])
            .reset_index()
        )
        maximum = float(grouped_ages['median_price'].max()) if not grouped_ages.empty else 0
        age_trends = [
            {
                'label': str(row['_age_band']),
                'median_price': round(float(row['median_price']), 2),
                'listings': int(row['listings']),
                'bar_pct': round(float(row['median_price']) / maximum * 100, 1)
                if maximum > 0
                else 0,
            }
            for _, row in grouped_ages.iterrows()
        ]

    insights = []
    if category_trends:
        largest_category = market['Category'].value_counts().idxmax() if 'Category' in market else None
        if largest_category is not None:
            insights.append(
                f"{largest_category} is the most represented category "
                f"({int((market['Category'] == largest_category).sum())} listings)."
            )
        highest_category = category_trends[0]
        insights.append(
            f"{highest_category['label']} has the highest category median "
            f"at NPR {highest_category['median_price']:,.0f}."
        )
    if len(age_trends) >= 2:
        youngest, oldest = age_trends[0], age_trends[-1]
        insights.append(
            f"Median prices are NPR {youngest['median_price']:,.0f} for "
            f"{youngest['label'].lower()} items and NPR {oldest['median_price']:,.0f} "
            f"for {oldest['label'].lower()} items."
        )

    return {
        'stats': stats,
        'category_trends': category_trends,
        'brand_trends': brand_trends,
        'condition_trends': condition_trends,
        'age_trends': age_trends,
        'location_trends': location_trends,
        'insights': insights,
    }


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
    model_loaded = MODEL is not None
    dataset_loaded = not DATASET.empty
    is_healthy = model_loaded and dataset_loaded
    return jsonify({
        'status': 'healthy' if is_healthy else 'unhealthy',
        'model_loaded': model_loaded,
        'dataset_loaded': dataset_loaded,
        'timestamp': datetime.now().isoformat(),
    }), 200 if is_healthy else 503


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return render_template('500.html'), 500


# ==================== Run App ====================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=app.config['DEBUG'])
