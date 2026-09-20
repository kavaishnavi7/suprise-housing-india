import os
import numpy as np
import pandas as pd

def generate_surprise_housing_data(n_samples=2000, random_seed=42):
    """
    Generate synthetic Indian Real Estate Housing Dataset for Surprise Housing India.
    Includes cities, localities, BHK, carpet area, RERA status, Vastu compliance,
    metro proximity, possession status, and prices in INR (₹).
    """
    np.random.seed(random_seed)
    
    ids = np.arange(1001, 1001 + n_samples)
    
    # Major Indian Metro Cities & Distribution
    cities = ['Mumbai', 'Bengaluru', 'Delhi NCR', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata']
    city_weights = [0.25, 0.22, 0.20, 0.13, 0.10, 0.06, 0.04]
    chosen_city = np.random.choice(cities, size=n_samples, p=city_weights)
    
    # Locality Tiers
    locality_tiers = ['Prime / Ultra-Luxury', 'Upscale', 'Mid-Tier', 'Suburban', 'Emerging']
    tier_weights = [0.12, 0.28, 0.40, 0.15, 0.05]
    chosen_tier = np.random.choice(locality_tiers, size=n_samples, p=tier_weights)
    
    # Mapping Localities based on City & Tier
    locality_map = {
        ('Mumbai', 'Prime / Ultra-Luxury'): 'South Mumbai (Worli/Malabar Hill)',
        ('Mumbai', 'Upscale'): 'Bandra West / BKC',
        ('Mumbai', 'Mid-Tier'): 'Andheri West / Powai',
        ('Mumbai', 'Suburban'): 'Thane West',
        ('Mumbai', 'Emerging'): 'Navi Mumbai (Kharghar)',
        
        ('Bengaluru', 'Prime / Ultra-Luxury'): 'Indiranagar / Lavelle Road',
        ('Bengaluru', 'Upscale'): 'Koramangala / Sadashivnagar',
        ('Bengaluru', 'Mid-Tier'): 'HSR Layout / Whitefield',
        ('Bengaluru', 'Suburban'): 'Electronic City',
        ('Bengaluru', 'Emerging'): 'Yelahanka / Devanahalli',
        
        ('Delhi NCR', 'Prime / Ultra-Luxury'): 'Golf Course Road, Gurgaon',
        ('Delhi NCR', 'Upscale'): 'South Delhi (Vasant Vihar)',
        ('Delhi NCR', 'Mid-Tier'): 'Cyber City, Gurgaon',
        ('Delhi NCR', 'Suburban'): 'Noida Sector 62',
        ('Delhi NCR', 'Emerging'): 'Dwarka Expressway',
        
        ('Hyderabad', 'Prime / Ultra-Luxury'): 'Jubilee Hills',
        ('Hyderabad', 'Upscale'): 'Banjara Hills / Gachibowli',
        ('Hyderabad', 'Mid-Tier'): 'HITEC City / Kondapur',
        ('Hyderabad', 'Suburban'): 'Miyapur / Kukatpally',
        ('Hyderabad', 'Emerging'): 'Tellapur',
        
        ('Pune', 'Prime / Ultra-Luxury'): 'Koregaon Park',
        ('Pune', 'Upscale'): 'Kalyani Nagar / Prabhat Road',
        ('Pune', 'Mid-Tier'): 'Baner / Viman Nagar',
        ('Pune', 'Suburban'): 'Wakad / Kharadi',
        ('Pune', 'Emerging'): 'Hinjawadi Phase 3',
        
        ('Chennai', 'Prime / Ultra-Luxury'): 'Boat Club / Poes Garden',
        ('Chennai', 'Upscale'): 'Nungambakkam / Anna Nagar',
        ('Chennai', 'Mid-Tier'): 'Velachery / Adyar',
        ('Chennai', 'Suburban'): 'OMR (Perungudi)',
        ('Chennai', 'Emerging'): 'GST Road (Tambaram)',
        
        ('Kolkata', 'Prime / Ultra-Luxury'): 'Ballygunge / Alipore',
        ('Kolkata', 'Upscale'): 'Park Street / Salt Lake Sec 5',
        ('Kolkata', 'Mid-Tier'): 'New Town Action Area 1',
        ('Kolkata', 'Suburban'): 'EM Bypass',
        ('Kolkata', 'Emerging'): 'Rajarhat'
    }
    
    localities = [locality_map.get((c, t), f'{c} Locality') for c, t in zip(chosen_city, chosen_tier)]
    
    # BHK Configuration
    bhk_opts = [1, 2, 3, 4, 5]
    bhk_weights = [0.15, 0.45, 0.30, 0.08, 0.02]
    bhk = np.random.choice(bhk_opts, size=n_samples, p=bhk_weights)
    
    # Carpet Area in Sq. Ft. based on BHK
    area_means = {1: 520, 2: 950, 3: 1550, 4: 2400, 5: 3800}
    carpet_area = np.array([int(np.random.normal(area_means[b], area_means[b] * 0.15)) for b in bhk])
    carpet_area = np.clip(carpet_area, 300, 7500)
    
    # Built-up Area (typically 1.25x - 1.4x of Carpet Area in India)
    builtup_ratio = np.random.uniform(1.25, 1.38, size=n_samples)
    builtup_area = np.round(carpet_area * builtup_ratio).astype(int)
    
    # Possession Status
    possession_opts = ['Ready to Move', 'Under Construction', 'New Launch']
    possession = np.random.choice(possession_opts, size=n_samples, p=[0.60, 0.30, 0.10])
    
    # RERA Status (1 = Approved, 0 = Pending/Unregistered)
    rera_approved = np.random.choice([1, 0], size=n_samples, p=[0.88, 0.12])
    
    # Vastu Compliance
    vastu_opts = ['Full Vastu', 'Partial Vastu', 'Non-Vastu']
    vastu = np.random.choice(vastu_opts, size=n_samples, p=[0.55, 0.35, 0.10])
    
    # Furnishing Status
    furnish_opts = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
    furnish = np.random.choice(furnish_opts, size=n_samples, p=[0.35, 0.50, 0.15])
    
    # Metro / Transport Connectivity Proximity
    metro_opts = ['Walkable (<500m)', 'Near (1-2 km)', 'Distant (>3 km)']
    metro_prox = np.random.choice(metro_opts, size=n_samples, p=[0.42, 0.45, 0.13])
    
    # Floor level & Total Floors
    total_floors = np.random.choice([4, 7, 12, 22, 35, 50], size=n_samples, p=[0.15, 0.20, 0.35, 0.18, 0.08, 0.04])
    floor_num = np.array([np.random.randint(1, tf + 1) for tf in total_floors])
    
    # Amenities & Gated Community
    gated = np.random.choice([1, 0], size=n_samples, p=[0.85, 0.15])
    power_backup = np.random.choice([1, 0], size=n_samples, p=[0.90, 0.10])
    clubhouse = np.random.choice([1, 0], size=n_samples, p=[0.70, 0.30])
    reserved_parking = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.20, 0.60, 0.18, 0.02])
    
    # Property Age
    age_yrs = np.random.choice([0, 2, 5, 8, 12, 20, 30], size=n_samples, p=[0.25, 0.20, 0.22, 0.15, 0.10, 0.05, 0.03])
    
    # Base Price per Sq. Ft. (in INR ₹) by City and Locality Tier
    base_sqft_rates = {
        ('Mumbai', 'Prime / Ultra-Luxury'): 55000,
        ('Mumbai', 'Upscale'): 35000,
        ('Mumbai', 'Mid-Tier'): 22000,
        ('Mumbai', 'Suburban'): 14000,
        ('Mumbai', 'Emerging'): 9000,
        
        ('Delhi NCR', 'Prime / Ultra-Luxury'): 32000,
        ('Delhi NCR', 'Upscale'): 22000,
        ('Delhi NCR', 'Mid-Tier'): 14000,
        ('Delhi NCR', 'Suburban'): 8500,
        ('Delhi NCR', 'Emerging'): 6000,
        
        ('Bengaluru', 'Prime / Ultra-Luxury'): 24000,
        ('Bengaluru', 'Upscale'): 15000,
        ('Bengaluru', 'Mid-Tier'): 10000,
        ('Bengaluru', 'Suburban'): 6500,
        ('Bengaluru', 'Emerging'): 4800,
        
        ('Hyderabad', 'Prime / Ultra-Luxury'): 20000,
        ('Hyderabad', 'Upscale'): 13000,
        ('Hyderabad', 'Mid-Tier'): 8500,
        ('Hyderabad', 'Suburban'): 5800,
        ('Hyderabad', 'Emerging'): 4200,
        
        ('Pune', 'Prime / Ultra-Luxury'): 18000,
        ('Pune', 'Upscale'): 12000,
        ('Pune', 'Mid-Tier'): 8000,
        ('Pune', 'Suburban'): 5500,
        ('Pune', 'Emerging'): 4000,
        
        ('Chennai', 'Prime / Ultra-Luxury'): 19000,
        ('Chennai', 'Upscale'): 12500,
        ('Chennai', 'Mid-Tier'): 8200,
        ('Chennai', 'Suburban'): 5400,
        ('Chennai', 'Emerging'): 3900,
        
        ('Kolkata', 'Prime / Ultra-Luxury'): 16000,
        ('Kolkata', 'Upscale'): 10500,
        ('Kolkata', 'Mid-Tier'): 7000,
        ('Kolkata', 'Suburban'): 4800,
        ('Kolkata', 'Emerging'): 3500
    }
    
    base_rate_array = np.array([base_sqft_rates.get((c, t), 8000) for c, t in zip(chosen_city, chosen_tier)])
    
    # Feature Multipliers
    rera_mult = np.where(rera_approved == 1, 1.08, 0.92)
    
    vastu_mult_dict = {'Full Vastu': 1.06, 'Partial Vastu': 1.02, 'Non-Vastu': 0.96}
    vastu_mult = np.array([vastu_mult_dict[v] for v in vastu])
    
    metro_mult_dict = {'Walkable (<500m)': 1.10, 'Near (1-2 km)': 1.04, 'Distant (>3 km)': 0.95}
    metro_mult = np.array([metro_mult_dict[m] for m in metro_prox])
    
    possession_mult_dict = {'Ready to Move': 1.05, 'Under Construction': 0.95, 'New Launch': 0.90}
    possession_mult = np.array([possession_mult_dict[p] for p in possession])
    
    furnish_mult_dict = {'Fully Furnished': 1.08, 'Semi-Furnished': 1.03, 'Unfurnished': 1.00}
    furnish_mult = np.array([furnish_mult_dict[f] for f in furnish])
    
    amenity_mult = 1.0 + (gated * 0.05) + (power_backup * 0.03) + (clubhouse * 0.04) + (reserved_parking * 0.03)
    age_mult = 1.0 - (age_yrs * 0.008)
    age_mult = np.clip(age_mult, 0.70, 1.0)
    
    # Total Price calculation in ₹ (INR)
    effective_sqft_rate = base_rate_array * rera_mult * vastu_mult * metro_mult * possession_mult * furnish_mult * amenity_mult * age_mult
    raw_sale_price_inr = carpet_area * effective_sqft_rate
    
    # Multiplicative noise
    noise = np.random.lognormal(mean=0, sigma=0.08, size=n_samples)
    sale_price_inr = np.round(raw_sale_price_inr * noise, -4)  # Round to nearest ₹10,000
    
    df = pd.DataFrame({
        'Property_ID': ids,
        'City': chosen_city,
        'Locality': localities,
        'Locality_Tier': chosen_tier,
        'BHK': bhk,
        'Carpet_Area_SqFt': carpet_area,
        'Builtup_Area_SqFt': builtup_area,
        'Possession_Status': possession,
        'RERA_Approved': rera_approved,
        'Vastu_Compliance': vastu,
        'Furnishing_Status': furnish,
        'Metro_Proximity': metro_prox,
        'Floor_Number': floor_num,
        'Total_Floors': total_floors,
        'Gated_Community': gated,
        'Power_Backup': power_backup,
        'Clubhouse': clubhouse,
        'Reserved_Parking_Slots': reserved_parking,
        'Property_Age_Yrs': age_yrs,
        'SalePrice_INR': sale_price_inr
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_surprise_housing_data()
    out_path = os.path.join('data', 'train.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} Indian housing records saved to {out_path}.")
    print("Sample Data Preview:")
    print(df[['City', 'Locality', 'BHK', 'Carpet_Area_SqFt', 'RERA_Approved', 'Vastu_Compliance', 'SalePrice_INR']].head())
