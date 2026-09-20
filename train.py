import sys
import os
import pickle
import json
import pandas as pd
import numpy as np

# Ensure UTF-8 output for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from generate_dataset import generate_surprise_housing_data
from src.preprocessing import preprocess_housing_data
from src.models import (
    train_and_tune_lasso,
    train_and_tune_ridge,
    train_random_forest,
    train_xgboost,
    evaluate_all_models,
    extract_feature_importance
)

def format_inr(amount):
    """Format amount in Lakhs (L) or Crores (Cr)"""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    else:
        return f"₹{amount / 100000:.2f} L"

def main():
    print("=" * 75)
    print("SURPRISE HOUSING INDIA [INDIA] - REAL ESTATE INVESTMENT & PRICE PREDICTION PIPELINE")
    print("=" * 75)
    
    # 1. Load or Generate Dataset
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'train.csv')
    
    print("Generating Indian Real Estate Housing dataset...")
    df = generate_surprise_housing_data(n_samples=2000)
    df.to_csv(data_path, index=False)
        
    print(f"Dataset shape: {df.shape[0]} samples, {df.shape[1]} features.")
    
    # 2. Preprocessing & Feature Engineering
    print("\nExecuting Data Preprocessing, Ordinal Mapping & Indian Feature Engineering...")
    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw, scaler, feature_names = preprocess_housing_data(df)
    print(f"Features after One-Hot Encoding: {len(feature_names)}")
    print(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    
    # 3. Model Training & Hyperparameter Tuning
    print("\n1. Training & Tuning Lasso Regression (L1 Penalty)...")
    lasso_model, lasso_alpha, lasso_cv = train_and_tune_lasso(X_train, y_train_log)
    print(f"   -> Optimal Lasso Alpha: {lasso_alpha:.6f}")
    
    print("\n2. Training & Tuning Ridge Regression (L2 Penalty)...")
    ridge_model, ridge_alpha, ridge_cv = train_and_tune_ridge(X_train, y_train_log)
    print(f"   -> Optimal Ridge Alpha: {ridge_alpha:.6f}")
    
    print("\n3. Training Random Forest Regressor...")
    rf_model = train_random_forest(X_train, y_train_log)
    print("   -> Random Forest trained with 200 estimators.")
    
    print("\n4. Training XGBoost Regressor...")
    xgb_model = train_xgboost(X_train, y_train_log)
    print("   -> XGBoost Regressor trained with learning_rate=0.03.")
    
    models = {
        'Lasso Regression': lasso_model,
        'Ridge Regression': ridge_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    # 4. Evaluation
    print("\nEvaluating Models on Test Set (Converted to INR ₹ Scale)...")
    evaluation = evaluate_all_models(models, X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw)
    
    print("\n" + "=" * 85)
    print(f"{'Model':<20} | {'Train R2':<10} | {'Test R2':<10} | {'Test MAE':<16} | {'Test RMSE':<16}")
    print("-" * 85)
    for name, res in evaluation.items():
        mae_str = format_inr(res['Test_MAE'])
        rmse_str = format_inr(res['Test_RMSE'])
        print(f"{name:<20} | {res['Train_R2']:<10.4f} | {res['Test_R2']:<10.4f} | {mae_str:<16} | {rmse_str:<16}")
    print("=" * 85)
    
    # 5. Extract Top Feature Drivers
    lasso_top_pos = extract_feature_importance(lasso_model, feature_names).sort_values(ascending=False).head(10)
    lasso_top_neg = extract_feature_importance(lasso_model, feature_names).sort_values(ascending=True).head(5)
    
    print("\nTop Positive Indian House Price Drivers (Lasso Coefficients):")
    for feat, coef in lasso_top_pos.items():
        print(f"  + {feat:<40}: {coef:+.4f}")
        
    print("\nTop Negative Indian House Price Drivers (Lasso Coefficients):")
    for feat, coef in lasso_top_neg.items():
        print(f"  - {feat:<40}: {coef:+.4f}")

    # 6. Save Artifacts for Streamlit Dashboard
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    pickle.dump(lasso_model, open(os.path.join(models_dir, 'lasso.pkl'), 'wb'))
    pickle.dump(ridge_model, open(os.path.join(models_dir, 'ridge.pkl'), 'wb'))
    pickle.dump(rf_model, open(os.path.join(models_dir, 'rf.pkl'), 'wb'))
    pickle.dump(xgb_model, open(os.path.join(models_dir, 'xgb.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(models_dir, 'scaler.pkl'), 'wb'))
    
    # Clean up numpy array types in cv_results for json/pickle serialization
    lasso_cv_clean = {
        'alphas': lasso_cv['alphas'].tolist() if isinstance(lasso_cv['alphas'], np.ndarray) else lasso_cv['alphas'],
        'mean_test_scores': lasso_cv['mean_test_scores'].tolist() if isinstance(lasso_cv['mean_test_scores'], np.ndarray) else lasso_cv['mean_test_scores']
    }
    ridge_cv_clean = {
        'alphas': ridge_cv['alphas'].tolist() if isinstance(ridge_cv['alphas'], np.ndarray) else ridge_cv['alphas'],
        'mean_test_scores': ridge_cv['mean_test_scores'].tolist() if isinstance(ridge_cv['mean_test_scores'], np.ndarray) else ridge_cv['mean_test_scores']
    }
    
    metadata = {
        'feature_names': feature_names,
        'lasso_alpha': float(lasso_alpha),
        'ridge_alpha': float(ridge_alpha),
        'lasso_cv': lasso_cv_clean,
        'ridge_cv': ridge_cv_clean,
        'metrics': {k: {m: float(v[m]) for m in ['Train_R2', 'Test_R2', 'Train_MAE', 'Test_MAE', 'Train_RMSE', 'Test_RMSE', 'Test_MAPE']} for k, v in evaluation.items()}
    }
    
    pickle.dump(metadata, open(os.path.join(models_dir, 'metadata.pkl'), 'wb'))
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metadata['metrics'], f, indent=4)
        
    print(f"\nPipeline completed successfully! Model artifacts saved to '{models_dir}/'.")

if __name__ == "__main__":
    main()
