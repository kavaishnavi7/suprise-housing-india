import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Ordinal Mappings for Indian Property Attributes
TIER_MAP = {
    'Prime / Ultra-Luxury': 5,
    'Upscale': 4,
    'Mid-Tier': 3,
    'Suburban': 2,
    'Emerging': 1
}

VASTU_MAP = {
    'Full Vastu': 3,
    'Partial Vastu': 2,
    'Non-Vastu': 1
}

METRO_MAP = {
    'Walkable (<500m)': 3,
    'Near (1-2 km)': 2,
    'Distant (>3 km)': 1
}

FURNISH_MAP = {
    'Fully Furnished': 3,
    'Semi-Furnished': 2,
    'Unfurnished': 1
}

POSSESSION_MAP = {
    'Ready to Move': 3,
    'Under Construction': 2,
    'New Launch': 1
}

def preprocess_housing_data(df, test_size=0.2, random_state=42):
    """
    Preprocess Indian Real Estate Housing Dataset:
    - Encodes ordinal features (Locality Tier, Vastu, Metro Proximity, Furnishing, Possession)
    - Engineers domain features (Area Efficiency, Floor Ratio, Amenity Score)
    - One-hot encodes nominal categorical variables (City, Locality)
    - Enforces strict train-test split BEFORE fitting RobustScaler to eliminate data leakage
    - Log-transforms target variable SalePrice_INR
    """
    data = df.copy()
    
    # Ordinal Mapping
    if 'Locality_Tier' in data.columns:
        data['Locality_Tier_Score'] = data['Locality_Tier'].map(TIER_MAP).fillna(2)
        data = data.drop(columns=['Locality_Tier'])
        
    if 'Vastu_Compliance' in data.columns:
        data['Vastu_Score'] = data['Vastu_Compliance'].map(VASTU_MAP).fillna(2)
        data = data.drop(columns=['Vastu_Compliance'])
        
    if 'Metro_Proximity' in data.columns:
        data['Metro_Score'] = data['Metro_Proximity'].map(METRO_MAP).fillna(2)
        data = data.drop(columns=['Metro_Proximity'])
        
    if 'Furnishing_Status' in data.columns:
        data['Furnishing_Score'] = data['Furnishing_Status'].map(FURNISH_MAP).fillna(2)
        data = data.drop(columns=['Furnishing_Status'])
        
    if 'Possession_Status' in data.columns:
        data['Possession_Score'] = data['Possession_Status'].map(POSSESSION_MAP).fillna(2)
        data = data.drop(columns=['Possession_Status'])
        
    # Feature Engineering
    if 'Carpet_Area_SqFt' in data.columns and 'Builtup_Area_SqFt' in data.columns:
        data['Area_Efficiency_Ratio'] = data['Carpet_Area_SqFt'] / (data['Builtup_Area_SqFt'] + 1e-5)
        
    if 'Floor_Number' in data.columns and 'Total_Floors' in data.columns:
        data['Floor_Ratio'] = data['Floor_Number'] / (data['Total_Floors'] + 1e-5)
        
    # Amenity Score
    amenity_cols = ['Gated_Community', 'Power_Backup', 'Clubhouse', 'Reserved_Parking_Slots', 'RERA_Approved']
    present_amenities = [col for col in amenity_cols if col in data.columns]
    if present_amenities:
        data['Amenity_Score'] = data[present_amenities].sum(axis=1)
        
    # Drop raw ID column
    if 'Property_ID' in data.columns:
        data = data.drop(columns=['Property_ID'])
        
    # Target Variable separation
    target_col = 'SalePrice_INR' if 'SalePrice_INR' in data.columns else ('SalePrice' if 'SalePrice' in data.columns else None)
    
    if target_col and target_col in data.columns:
        X = data.drop(columns=[target_col])
        y_raw = data[target_col].values
        y_log = np.log1p(y_raw)
    else:
        X = data
        y_raw = None
        y_log = None
        
    # Impute missing continuous features
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
            
    # One-Hot Encoding for remaining nominal categories (City, Locality)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_encoded = X_encoded.fillna(0)
    
    if y_log is None:
        return X_encoded, None, None, None, None, None, None, X_encoded.columns.tolist()
        
    # Train-Test Split BEFORE fitting RobustScaler
    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
        X_encoded, y_log, y_raw, test_size=test_size, random_state=random_state
    )
    
    # Fit Scaler strictly on X_train to prevent leakage
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    feature_names = X_encoded.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train_raw, y_test_raw, scaler, feature_names
