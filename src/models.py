import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    # R2 adjusted assumes p features (we compute basic R2 and MAE/RMSE)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def train_and_tune_lasso(X_train, y_train, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 2, 50)
        
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=Lasso(max_iter=10000, random_state=42),
        param_grid={'alpha': alphas},
        scoring='r2',
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_lasso = grid.best_estimator_
    best_alpha = grid.best_params_['alpha']
    
    cv_results = {
        'alphas': grid.param_grid['alpha'],
        'mean_test_scores': grid.cv_results_['mean_test_score'],
        'std_test_scores': grid.cv_results_['std_test_score']
    }
    
    return best_lasso, best_alpha, cv_results

def train_and_tune_ridge(X_train, y_train, alphas=None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
        
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=Ridge(max_iter=10000, random_state=42),
        param_grid={'alpha': alphas},
        scoring='r2',
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_ridge = grid.best_estimator_
    best_alpha = grid.best_params_['alpha']
    
    cv_results = {
        'alphas': grid.param_grid['alpha'],
        'mean_test_scores': grid.cv_results_['mean_test_score'],
        'std_test_scores': grid.cv_results_['std_test_score']
    }
    
    return best_ridge, best_alpha, cv_results

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_all_models(models, X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw):
    results = {}
    
    for name, model in models.items():
        # Log Scale Predictions
        y_train_pred_log = model.predict(X_train)
        y_test_pred_log = model.predict(X_test)
        
        # Original Dollar Scale Predictions
        y_train_pred_dollar = np.expm1(y_train_pred_log)
        y_test_pred_dollar = np.expm1(y_test_pred_log)
        
        # Metrics on Dollar scale
        train_metrics = calculate_metrics(y_train_raw, y_train_pred_dollar)
        test_metrics = calculate_metrics(y_test_raw, y_test_pred_dollar)
        
        results[name] = {
            'Train_R2': train_metrics['R2'],
            'Test_R2': test_metrics['R2'],
            'Train_MAE': train_metrics['MAE'],
            'Test_MAE': test_metrics['MAE'],
            'Train_RMSE': train_metrics['RMSE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Train_MAPE': train_metrics['MAPE'],
            'Test_MAPE': test_metrics['MAPE'],
            'y_test_pred_dollar': y_test_pred_dollar,
            'y_test_pred_log': y_test_pred_log
        }
        
    return results

def extract_feature_importance(model, feature_names):
    if hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_, index=feature_names)
        return coefs.sort_values(key=abs, ascending=False)
    elif hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)
    else:
        return None
