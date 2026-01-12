import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from difflib import get_close_matches

class AadhaarBrain:
    def __init__(self, data_dir="/Users/rakeshmondal/Downloads/uidai data "):
        self.data_dir = data_dir
        self.enrol_df = None
        self.demo_df = None
        self.bio_df = None
        self.district_stats = None
        
    def load_data(self):
        print("Loading data...")
        self.enrol_df = self._load_folder("api_data_aadhar_enrolment")
        self.demo_df = self._load_folder("api_data_aadhar_demographic")
        self.bio_df = self._load_folder("api_data_aadhar_biometric")
        
        # Preprocess
        self.enrol_df = self._preprocess(self.enrol_df, 'enrolment')
        self.demo_df = self._preprocess(self.demo_df, 'update')
        self.bio_df = self._preprocess(self.bio_df, 'update')
        
    def _load_folder(self, folder_name):
        path = os.path.join(self.data_dir, folder_name)
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_list = []
        for f in all_files:
            try:
                df_list.append(pd.read_csv(f))
            except: pass
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame()

    def _preprocess(self, df, dtype):
        # Convert numeric
        for col in df.columns:
            if 'age' in col or 'total' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # Clean State Names
        if 'state' in df.columns:
            df = df.dropna(subset=['state'])
            df['state'] = df['state'].astype(str).str.strip().str.title()
            
            # Remove numeric states (Junk data)
            df = df[~df['state'].str.match(r'^\d+$')]
            
            # Typo Fixes
            replacements = {
                'West Bangal': 'West Bengal',
                'Westbengal': 'West Bengal',
                'West  Bengal': 'West Bengal',
                'Chhatisgarh': 'Chhattisgarh',
                'Uttaranchal': 'Uttarakhand',
                'Pondicherry': 'Puducherry',
                'Orissa': 'Odisha',
                'Andhra Pradesh': 'Andhra Pradesh',
                'West Bengli': 'West Bengal',
                'Jammu And Kashmir': 'Jammu & Kashmir',
                'Dadra And Nagar Haveli': 'Dadra & Nagar Haveli',
                'Daman And Diu': 'Daman & Diu',
                'Andaman And Nicobar Islands': 'Andaman & Nicobar Islands'
            }
            df['state'] = df['state'].replace(replacements)
            
            # Whitelist of valid Indian States and Union Territories (36 total)
            VALID_STATES = {
                # States (28)
                'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
                'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
                'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
                'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
                'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
                'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
                
                # Union Territories (8)
                'Andaman & Nicobar Islands', 'Chandigarh', 
                'Dadra & Nagar Haveli And Daman & Diu',
                'Delhi', 'Jammu & Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
            }
            
            # Filter out invalid states (cities, typos, etc.)
            df = df[df['state'].isin(VALID_STATES)]
            # Remove city names appearing as states (heuristic: if it's not in a known list, potentially drop? 
            # Or just rely on the user ignoring them. 'Jaipur' as state is bad.
            # Let's keep it simple for now and rely on the major fixes).

        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()
            
            # Remove trailing asterisks and other special chars
            df['district'] = df['district'].str.replace(r'\s*\*+\s*$', '', regex=True)
            
            # Remove parenthetical notes like (Bh), (Kar), (Mh), (R), (M), (Urban), etc.
            df['district'] = df['district'].str.replace(r'\s*\([^)]*\)', '', regex=True)
            
            # Clean up extra spaces
            df['district'] = df['district'].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # 1. Manual Mappings (Factual/Official/Phonetic Name Changes)
            manual_map = {
                # West Bengal (Phonetic fixes)
                'Haora': 'Howrah', 'Hugli': 'Hooghly', 'Midnapore': 'Paschim Medinipur', 
                'North 24 Paraganas': 'North 24 Parganas', 'South 24 Paraganas': 'South 24 Parganas',
                'North 24 Pgns': 'North 24 Parganas', 'South 24 Pgns': 'South 24 Parganas',
                'Parganas North': 'North 24 Parganas', 'Parganas South': 'South 24 Parganas',
                'Calcutta': 'Kolkata', 'South Dumdum': 'South Dum Dum',
                
                # Maharashtra (Official Changes & Typos)
                'Ahmednagar': 'Ahilyanagar', 'Aurangabad': 'Chhatrapati Sambhaji Nagar',
                'Osmanabad': 'Dharashiv', 'Beed': 'Bid', 'Bombay': 'Mumbai', 'Poona': 'Pune',
                'Mumbai Sub Urban': 'Mumbai Suburban', 'Raigarh': 'Raigad',
                
                # Kerala (Anglicized vs Local)
                'Trivandrum': 'Thiruvananthapuram', 'Quilon': 'Kollam', 'Alleppey': 'Alappuzha',
                'Trichur': 'Thrissur', 'Palghat': 'Palakkad', 'Calicut': 'Kozhikode', 
                'Cannanore': 'Kannur', 'Cochin': 'Kochi',
                
                # Karnataka (Previously added + refinements)
                'Bangalore': 'Bengaluru', 'Bangalore Rural': 'Bengaluru Rural', 'Bangalore South': 'Bengaluru South',
                'Belgaum': 'Belagavi', 'Bellary': 'Ballari', 'Mysore': 'Mysuru', 'Gulbarga': 'Kalaburagi',
                'Shimoga': 'Shivamogga', 'Tumkur': 'Tumakuru', 'Bijapur': 'Vijayapura', 'Chikmagalur': 'Chikkamagaluru',
                'Chickmagalur': 'Chikkamagaluru', 'Chamrajanagar': 'Chamarajanagar', 'Chamrajnagar': 'Chamarajanagar',
                'Hasan': 'Hassan', 'Davangere': 'Davanagere',
                
                # Uttar Pradesh (Refined)
                'Allahabad': 'Prayagraj', 'Bara Banki': 'Barabanki', 'Rae Bareli': 'Raebareli',
                'Budaun': 'Badaun', 'Sant Ravidas Nagar': 'Bhadohi', 'Kheri': 'Lakhimpur Kheri',
                'Firozpur': 'Ferozepur',
                
                # Andhra Pradesh / Telangana (Refined)
                'Vizag': 'Visakhapatnam', 'Vizianagaram': 'Vijayanagaram', 'Karim Nagar': 'Karimnagar',
                'K.V.Rangareddy': 'K.V. Rangareddy', 'Mahabub Nagar': 'Mahabubnagar',
                'Mahbubnagar': 'Mahabubnagar', 'Ananthapur': 'Anantapur', 'Ananthapuramu': 'Anantapur',
                'Cuddapah': 'Y. S. R', 'Kadapa': 'Y. S. R',
                
                # Others
                'Andamans': 'Andaman', 'Nicobars': 'Nicobar', 'Leh Ladakh': 'Leh',
                'Janjgir - Champa': 'Janjgir-Champa', 'Janjgir Champa': 'Janjgir-Champa',
                'Banas Kantha': 'Banaskantha', 'Panch Mahals': 'Panchmahals', 'Sabar Kantha': 'Sabarkantha',
                'Surendra Nagar': 'Surendranagar', 'Ahmadabad': 'Ahmedabad', 'Dohad': 'Dahod',
                'Yamuna Nagar': 'Yamunanagar', 'S.A.S Nagar': 'Sahibzada Ajit Singh Nagar',
                'Sas Nagar': 'Sahibzada Ajit Singh Nagar', 'Kaimur': 'Kaimur'
            }
            df['district'] = df['district'].replace(manual_map)

            # 2. String Cleaning (Fast)
            df['district'] = df['district'].str.replace(r'\s*\*+\s*$', '', regex=True)
            df['district'] = df['district'].str.replace(r'\s*\([^)]*\)', '', regex=True)
            df['district'] = df['district'].str.replace(r'\s+', ' ', regex=True).str.strip().str.title()

            # The fuzzy logic is too slow for millions of rows. 
            # We use the expanded manual_map for performance.
            # (Fuzzy logic could be run once to generate this map, but not every load)
            
            # Additional discovered variations from previous fuzzy runs
            extra_maps = {
                'Dohad': 'Dahod', 'Ahmadabad': 'Ahmedabad', 'Surendra Nagar': 'Surendranagar',
                'Banaskantha': 'Banas Kantha', 'Sabarkantha': 'Sabar Kantha',
                'Hooghly': 'Hooghly', 'Howrah': 'Howrah', # Ensure canonical
            }
            manual_map.update(extra_maps)
            df['district'] = df['district'].replace(manual_map)
        
        # Date Parsing (Robust)
        if 'date' in df.columns:
            # First try the standard format (fastest)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            # If everything is NaT, try inferring
            if df['date'].isna().all() and not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Final fallback: strip and try again if still some NaT
            df = df.dropna(subset=['date'])

        # Calculate Total Column
        if dtype == 'enrolment':
            cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        else: # update
            cols = [c for c in df.columns if 'age' in c]
        
        # Ensure numeric before sum
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        df['total'] = df[cols].sum(axis=1) if cols else 0
            
        return df

    def train_anomaly_model(self):
        print("Training Anomaly Model...")
        # Aggregating by District
        # Features: Total Volume, Variance (Std Dev over time)
        
        # We will use Updates (Demo + Bio) for anomalies as that's the stress point
        # Combine demo and bio for total update load
        d_grp = self.demo_df.groupby(['state', 'district'])['total'].agg(['sum', 'mean', 'std']).reset_index()
        d_grp.columns = ['state', 'district', 'demo_total', 'demo_mean', 'demo_std']
        
        b_grp = self.bio_df.groupby(['state', 'district'])['total'].agg(['sum', 'mean', 'std']).reset_index()
        b_grp.columns = ['state', 'district', 'bio_total', 'bio_mean', 'bio_std']
        
        # Merge
        features = pd.merge(d_grp, b_grp, on=['state', 'district'], how='outer').fillna(0)
        
        features['total_load'] = features['demo_total'] + features['bio_total']
        features['volatility'] = features['demo_std'] + features['bio_std']
        
        # Prepare for ML
        X = features[['total_load', 'volatility', 'bio_total', 'demo_total']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        features['anomaly'] = iso_forest.fit_predict(X_scaled)
        features['anomaly_score'] = iso_forest.decision_function(X_scaled)
        
        # -1 is anomaly, 1 is normal
        anomalies = features[features['anomaly'] == -1].copy()
        anomalies['risk_score'] = abs(anomalies['anomaly_score']) # Higher absolute val = more anomalous
        
        self.district_stats = features
        print(f"Detected {len(anomalies)} anomalies.")
        return anomalies.sort_values('risk_score', ascending=False)

    def get_district_stats(self, district_name):
        if self.district_stats is None:
            return None
        return self.district_stats[self.district_stats['district'] == district_name]

    def recommend_resources(self, district, current_load, projected_growth_pct=0.2):
        # Rule of Thumb (Hypothetical): 
        # 1 Kit handles 50 updates/day.
        # 1 Staff handles 40 enrolments/day.
        
        projected_load = current_load * (1 + projected_growth_pct)
        # Monthly load -> Daily (approx / 25 working days)
        daily_load = projected_load / 30 
        
        kits_needed = np.ceil(daily_load / 50)
        staff_needed = np.ceil(daily_load / 40)
        
        return {
            "projected_monthly_load": int(projected_load),
            "kits_required": int(kits_needed),
            "staff_required": int(staff_needed)
        }

if __name__ == "__main__":
    brain = AadhaarBrain()
    brain.load_data()
    anomalies = brain.train_anomaly_model()
    print(anomalies.head())
