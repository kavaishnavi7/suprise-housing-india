import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Surprise Housing India - Real Estate Valuation & Investment Platform",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark & Indian Accent Aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetricCard {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E6FF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8A8F98;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-title {
        background: linear-gradient(90deg, #FF9933 0%, #00E6FF 50%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .sub-title {
        color: #90A4AE;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card-box {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .badge-rera {
        background-color: rgba(19, 136, 8, 0.2);
        color: #2ECC71;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .badge-vastu {
        background-color: rgba(255, 153, 51, 0.2);
        color: #FF9933;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(255, 153, 51, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format INR amounts in Lakhs/Crores
def format_inr(amount):
    if amount >= 10000000:
        return f"₹ {amount / 10000000:.2f} Cr"
    else:
        return f"₹ {amount / 100000:.2f} Lakhs"

# Load Models & Artifacts
@st.cache_resource
def load_artifacts():
    models_dir = 'models'
    lasso = pickle.load(open(os.path.join(models_dir, 'lasso.pkl'), 'rb'))
    ridge = pickle.load(open(os.path.join(models_dir, 'ridge.pkl'), 'rb'))
    rf = pickle.load(open(os.path.join(models_dir, 'rf.pkl'), 'rb'))
    xgb = pickle.load(open(os.path.join(models_dir, 'xgb.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'), 'rb'))
    metadata = pickle.load(open(os.path.join(models_dir, 'metadata.pkl'), 'rb'))
    
    data_dir = 'data'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    return {
        'Lasso Regression': lasso,
        'Ridge Regression': ridge,
        'Random Forest': rf,
        'XGBoost': xgb
    }, scaler, metadata, train_df

try:
    models, scaler, metadata, train_df = load_artifacts()
    feature_names = metadata['feature_names']
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run 'python train.py' first.")
    st.stop()

# Header Section
st.markdown('<h1 class="header-title">Surprise Housing India 🇮🇳</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Real Estate Investment Decision Support & Automated Valuation System for the Indian Property Market</p>', unsafe_allow_html=True)

# Sidebar Inputs for Indian Real Estate Attributes
st.sidebar.header("📍 Property Location & Parameters")

city = st.sidebar.selectbox("Select Metro City", [
    'Mumbai', 'Bengaluru', 'Delhi NCR', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata'
], index=0)

locality_tier = st.sidebar.selectbox("Locality Tier", [
    'Prime / Ultra-Luxury', 'Upscale', 'Mid-Tier', 'Suburban', 'Emerging'
], index=1)

bhk = st.sidebar.slider("BHK Configuration", min_value=1, max_value=5, value=3, step=1)

carpet_area = st.sidebar.number_input("Carpet Area (Sq. Ft.)", min_value=300, max_value=7500, value=1450, step=50)

builtup_area = st.sidebar.number_input("Super Built-up Area (Sq. Ft.)", min_value=350, max_value=10000, value=int(carpet_area * 1.30), step=50)

st.sidebar.header("🏢 Building & Compliance Attributes")

possession = st.sidebar.selectbox("Possession Status", [
    'Ready to Move', 'Under Construction', 'New Launch'
], index=0)

rera_status = st.sidebar.radio("RERA Approval", ['Approved (Yes)', 'Pending / Unregistered (No)'], index=0)
rera_approved = 1 if rera_status == 'Approved (Yes)' else 0

vastu = st.sidebar.selectbox("Vastu Compliance", [
    'Full Vastu', 'Partial Vastu', 'Non-Vastu'
], index=0)

furnish = st.sidebar.selectbox("Furnishing Status", [
    'Fully Furnished', 'Semi-Furnished', 'Unfurnished'
], index=1)

metro_prox = st.sidebar.selectbox("Metro / Transport Connectivity", [
    'Walkable (<500m)', 'Near (1-2 km)', 'Distant (>3 km)'
], index=0)

st.sidebar.header("✨ Amenities & Building Details")
gated = st.sidebar.checkbox("Gated Society / Security", value=True)
power_backup = st.sidebar.checkbox("24/7 Power Backup", value=True)
clubhouse = st.sidebar.checkbox("Clubhouse & Swimming Pool", value=True)
parking_slots = st.sidebar.slider("Reserved Parking Slots", min_value=0, max_value=3, value=1)
floor_num = st.sidebar.slider("Floor Level", min_value=1, max_value=50, value=8)
total_floors = st.sidebar.slider("Total Building Floors", min_value=floor_num, max_value=50, value=20)
property_age = st.sidebar.slider("Property Age (Years)", min_value=0, max_value=30, value=3)

# Construct Input Vector matching preprocessed features
def construct_input_dataframe():
    input_dict = {
        'BHK': bhk,
        'Carpet_Area_SqFt': carpet_area,
        'Builtup_Area_SqFt': builtup_area,
        'RERA_Approved': rera_approved,
        'Floor_Number': floor_num,
        'Total_Floors': total_floors,
        'Gated_Community': 1 if gated else 0,
        'Power_Backup': 1 if power_backup else 0,
        'Clubhouse': 1 if clubhouse else 0,
        'Reserved_Parking_Slots': parking_slots,
        'Property_Age_Yrs': property_age,
        
        # Ordinal Features
        'Locality_Tier_Score': {'Prime / Ultra-Luxury': 5, 'Upscale': 4, 'Mid-Tier': 3, 'Suburban': 2, 'Emerging': 1}[locality_tier],
        'Vastu_Score': {'Full Vastu': 3, 'Partial Vastu': 2, 'Non-Vastu': 1}[vastu],
        'Metro_Score': {'Walkable (<500m)': 3, 'Near (1-2 km)': 2, 'Distant (>3 km)': 1}[metro_prox],
        'Furnishing_Score': {'Fully Furnished': 3, 'Semi-Furnished': 2, 'Unfurnished': 1}[furnish],
        'Possession_Score': {'Ready to Move': 3, 'Under Construction': 2, 'New Launch': 1}[possession],
        
        # Engineered Features
        'Area_Efficiency_Ratio': carpet_area / (builtup_area + 1e-5),
        'Floor_Ratio': floor_num / (total_floors + 1e-5),
        'Amenity_Score': (1 if gated else 0) + (1 if power_backup else 0) + (1 if clubhouse else 0) + parking_slots + rera_approved
    }
    
    # Create zeroed DataFrame with all model features
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    for k, v in input_dict.items():
        if k in input_df.columns:
            input_df.at[0, k] = float(v)
            
    # Set One-Hot Columns
    city_col = f"City_{city}"
    if city_col in input_df.columns:
        input_df.at[0, city_col] = 1.0
        
    return input_df

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Property Valuation", 
    "📊 Model Benchmarks", 
    "💡 Investment Drivers", 
    "📈 Price Sensitivity Simulator"
])

input_raw_df = construct_input_dataframe()
# Scale numeric columns using saved RobustScaler
input_scaled_df = pd.DataFrame(scaler.transform(input_raw_df), columns=feature_names)

# Make Predictions across all 4 models
predictions = {}
for name, model in models.items():
    pred_log = model.predict(input_scaled_df)[0]
    pred_inr = np.expm1(pred_log)
    predictions[name] = pred_inr

primary_model = "XGBoost"
predicted_price = predictions[primary_model]
price_per_sqft = predicted_price / carpet_area

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stMetricCard">
            <div class="metric-label">Estimated Market Value (XGBoost)</div>
            <div class="metric-value">{format_inr(predicted_price)}</div>
            <div style="font-size:0.9rem; color:#A0AEC0; margin-top:5px;">Range: {format_inr(predicted_price*0.95)} - {format_inr(predicted_price*1.05)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stMetricCard">
            <div class="metric-label">Effective Rate / Sq. Ft.</div>
            <div class="metric-value">₹ {price_per_sqft:,.0f} / sq.ft</div>
            <div style="font-size:0.9rem; color:#A0AEC0; margin-top:5px;">Carpet Area: {carpet_area:,} sq.ft</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        investment_grade = "STRONG BUY (Undervalued Potential)" if rera_approved and vastu == 'Full Vastu' else "FAIR MARKET VALUE"
        st.markdown(f"""
        <div class="stMetricCard">
            <div class="metric-label">Investment Rating</div>
            <div class="metric-value" style="color: #2ECC71;">{investment_grade}</div>
            <div style="font-size:0.9rem; color:#A0AEC0; margin-top:5px;">RERA: {"✅ Approved" if rera_approved else "⚠️ Pending"} | Vastu: {vastu}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Model Comparison Grid
    st.subheader("🤖 Multi-Model Price Predictions & Benchmark")
    comp_df = pd.DataFrame({
        'Model Architecture': list(predictions.keys()),
        'Valuation (INR ₹)': [format_inr(v) for v in predictions.values()],
        'Rate / Sq. Ft. (₹)': [f"₹ {v/carpet_area:,.0f}" for v in predictions.values()],
        'Test R² Score': [f"{metadata['metrics'][m]['Test_R2']:.4f}" for m in predictions.keys()],
        'Test MAE (Lakhs)': [f"₹ {metadata['metrics'][m]['Test_MAE']/100000:.2f} L" for m in predictions.keys()]
    })
    st.table(comp_df)
    
    # Property Summary Card
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("📌 Selected Property Configuration & Regulatory Badges")
    badge_cols = st.columns(4)
    with badge_cols[0]:
        st.write(f"**City:** {city}")
        st.write(f"**Locality Tier:** {locality_tier}")
    with badge_cols[1]:
        st.write(f"**Configuration:** {bhk} BHK")
        st.write(f"**Carpet Area:** {carpet_area} sq.ft")
    with badge_cols[2]:
        st.markdown(f'<span class="badge-rera">RERA {"Approved" if rera_approved else "Pending"}</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<span class="badge-vastu">{vastu}</span>', unsafe_allow_html=True)
    with badge_cols[3]:
        st.write(f"**Metro Proximity:** {metro_prox}")
        st.write(f"**Possession:** {possession}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📈 Regression Model Performance & Metrics (Indian Dataset)")
    
    metrics_data = []
    for model_name, m in metadata['metrics'].items():
        metrics_data.append({
            'Model': model_name,
            'Train R²': f"{m['Train_R2']:.4f}",
            'Test R²': f"{m['Test_R2']:.4f}",
            'Test MAE (INR)': format_inr(m['Test_MAE']),
            'Test RMSE (INR)': format_inr(m['Test_RMSE']),
            'Test MAPE (%)': f"{m['Test_MAPE']:.2f}%"
        })
    st.table(pd.DataFrame(metrics_data))
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### Lasso & Ridge Alpha Tuning CV Curves")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        lasso_cv = metadata['lasso_cv']
        ax.semilogx(lasso_cv['alphas'], lasso_cv['mean_test_scores'], label='Lasso CV R²', color='#00E6FF', lw=2)
        ax.axvline(metadata['lasso_alpha'], color='#FF9933', linestyle='--', label=f"Opt Alpha ({metadata['lasso_alpha']:.4f})")
        ax.set_xlabel('Alpha (Regularization Strength)')
        ax.set_ylabel('Cross-Validation R² Score')
        ax.legend()
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("##### Actual vs Predicted House Prices (Sample)")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        sns.scatterplot(x=train_df['Carpet_Area_SqFt'], y=train_df['SalePrice_INR']/100000, hue=train_df['City'], palette='tab10', alpha=0.7, ax=ax)
        ax.set_xlabel('Carpet Area (Sq. Ft.)')
        ax.set_ylabel('Sale Price (₹ Lakhs)')
        st.pyplot(fig)

with tab3:
    st.subheader("💡 Top House Price Drivers & Lasso Coefficients")
    
    lasso_model = models['Lasso Regression']
    coefs = pd.Series(lasso_model.coef_, index=feature_names)
    top_pos = coefs.sort_values(ascending=False).head(10)
    top_neg = coefs.sort_values(ascending=True).head(5)
    
    col_drivers1, col_drivers2 = st.columns(2)
    
    with col_drivers1:
        st.markdown("##### 🚀 Top Positive Price Multipliers")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        sns.barplot(x=top_pos.values, y=top_pos.index, palette='Greens_r', ax=ax)
        ax.set_xlabel('Lasso Coefficient Value')
        st.pyplot(fig)
        
    with col_drivers2:
        st.markdown("##### 🔻 Top Negative Price Penalty Drivers")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        sns.barplot(x=top_neg.values, y=top_neg.index, palette='Reds', ax=ax)
        ax.set_xlabel('Lasso Coefficient Value')
        st.pyplot(fig)

with tab4:
    st.subheader("📈 Interactive Market Value Sensitivity Simulator")
    st.write("Simulate how altering property attributes affects total valuation:")
    
    sim_area = st.slider("Simulated Carpet Area (Sq. Ft.)", 500, 4000, carpet_area, 100)
    sim_rera = st.radio("Simulated RERA Approval", [1, 0], index=0 if rera_approved else 1, format_func=lambda x: "RERA Approved" if x == 1 else "Pending")
    sim_metro = st.selectbox("Simulated Metro Connectivity", ['Walkable (<500m)', 'Near (1-2 km)', 'Distant (>3 km)'], index=0)
    
    # Calculate simulated valuation
    sim_df = input_raw_df.copy()
    sim_df['Carpet_Area_SqFt'] = sim_area
    sim_df['RERA_Approved'] = sim_rera
    sim_df['Metro_Score'] = {'Walkable (<500m)': 3, 'Near (1-2 km)': 2, 'Distant (>3 km)': 1}[sim_metro]
    
    sim_scaled = pd.DataFrame(scaler.transform(sim_df), columns=feature_names)
    sim_price_inr = np.expm1(models['XGBoost'].predict(sim_scaled)[0])
    
    delta = sim_price_inr - predicted_price
    
    st.markdown(f"""
    <div class="stMetricCard">
        <div class="metric-label">Simulated Property Valuation</div>
        <div class="metric-value">{format_inr(sim_price_inr)}</div>
        <div style="font-size:1.1rem; color:{'#2ECC71' if delta>=0 else '#E74C3C'}; margin-top:5px;">
            Valuation Delta: {'+' if delta>=0 else ''}{format_inr(delta)} ({delta/predicted_price*100:+.1f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
