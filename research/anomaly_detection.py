import pandas as pd
import glob
import os
import numpy as np

DATA_DIR = "/Users/rakeshmondal/Downloads/uidai data "

def load_all_data(folder_name):
    path = os.path.join(DATA_DIR, folder_name)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except:
            pass
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def detect_district_outliers(df, metric_col, name, threshold=3):
    print(f"\n--- Detecting District Outliers for {name} ---")
    # Aggr by District
    dist_stats = df.groupby(['state', 'district'])[metric_col].sum().reset_index()
    
    # Calculate Z-score per State (to account for state size diffs, though state average might be better)
    # Actually simpler: Z-score across specific 'similar' districts or just broadly.
    # Let's do Z-score within State to find districts that stand out from their neighbors.
    
    outliers = []
    for state, group in dist_stats.groupby('state'):
        if len(group) < 3: continue # Skip states with too few districts
        
        mean = group[metric_col].mean()
        std = group[metric_col].std()
        
        if std == 0: continue
        
        group['z_score'] = (group[metric_col] - mean) / std
        state_outliers = group[group['z_score'].abs() > threshold]
        
        if not state_outliers.empty:
            outliers.append(state_outliers)
            
    if outliers:
        all_outliers = pd.concat(outliers)
        print(f"Found {len(all_outliers)} outlier districts (Z-score > {threshold}).")
        print(all_outliers.sort_values('z_score', ascending=False).head(10))
    else:
        print("No significant outliers found within states.")

def analyze_seasonality(df, date_col='date', metric_col='count'):
    print(f"\n--- Seasonality Analysis for {metric_col} ---")
    df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
    df['month'] = df[date_col].dt.month
    
    monthly_avg = df.groupby('month')[metric_col].mean()
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    
    print(f"Peak Month: {peak_month} (Avg: {monthly_avg[peak_month]:.2f})")
    print(f"Lowest Month: {low_month} (Avg: {monthly_avg[low_month]:.2f})")
    
    # Simple check for variance
    variance = monthly_avg.std() / monthly_avg.mean()
    print(f"Seasonality Variation (CV): {variance:.2f}")

def main():
    # Enrolment
    enrol_df = load_all_data("api_data_aadhar_enrolment")
    for c in ['age_0_5', 'age_5_17', 'age_18_greater']:
        enrol_df[c] = pd.to_numeric(enrol_df[c], errors='coerce').fillna(0)
    enrol_df['total'] = enrol_df['age_0_5'] + enrol_df['age_5_17'] + enrol_df['age_18_greater']
    
    detect_district_outliers(enrol_df, 'total', "Total Enrolment")
    analyze_seasonality(enrol_df, metric_col='total')

    # Demographic Updates
    demo_df = load_all_data("api_data_aadhar_demographic")
    for c in ['demo_age_5_17', 'demo_age_17_']:
        demo_df[c] = pd.to_numeric(demo_df[c], errors='coerce').fillna(0)
    demo_df['total'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
    
    detect_district_outliers(demo_df, 'total', "Demographic Updates")
    analyze_seasonality(demo_df, metric_col='total')

if __name__ == "__main__":
    main()
