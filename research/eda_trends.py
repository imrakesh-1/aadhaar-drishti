import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/Users/rakeshmondal/Downloads/uidai data "
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_data(folder_name):
    path = os.path.join(DATA_DIR, folder_name)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []
    print(f"Loading {folder_name}...")
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error: {e}")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def process_date(df):
    # Try different formats if needed, assuming dd-mm-yyyy based on inspection
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    return df

def analyze_enrolment(df):
    print("\n--- Enrolment Analysis ---")
    df = process_date(df)
    
    # Ensure numeric
    cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['total_enrolment'] = df[cols].sum(axis=1)
    
    # 1. Trend over time (Monthly)
    monthly = df.groupby(pd.Grouper(key='date', freq='M'))['total_enrolment'].sum()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, marker='o')
    plt.title('Total Aadhaar Enrolments Over Time (Monthly)')
    plt.ylabel('Enrolments')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/enrolment_trend.png")
    print(f"Saved enrolment_trend.png")
    
    # 2. State-wise Total
    state_wise = df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 States by Enrolment:\n", state_wise)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=state_wise.values, y=state_wise.index, palette='viridis')
    plt.title('Top 10 States by Total Enrolments')
    plt.xlabel('Enrolments')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_states_enrolment.png")

    # 3. Age Group Distribution
    age_sums = df[cols].sum()
    print("\nEnrolment by Age Group:\n", age_sums)
    plt.figure(figsize=(8, 8))
    age_sums.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.ylabel('')
    plt.title('Enrolment Distribution by Age Group')
    plt.savefig(f"{OUTPUT_DIR}/enrolment_age_dist.png")

def analyze_updates(demo_df, bio_df):
    print("\n--- Update Analysis ---")
    demo_df = process_date(demo_df)
    bio_df = process_date(bio_df)
    
    # Numeric conversion
    for c in ['demo_age_5_17', 'demo_age_17_']:
        demo_df[c] = pd.to_numeric(demo_df[c], errors='coerce').fillna(0)
    for c in ['bio_age_5_17', 'bio_age_17_']:
        bio_df[c] = pd.to_numeric(bio_df[c], errors='coerce').fillna(0)
        
    demo_df['total_demo'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
    bio_df['total_bio'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']
    
    # Merge on month for comparison ? Or just separate plots.
    # Let's do separate monthly trends first.
    
    monthly_demo = demo_df.groupby(pd.Grouper(key='date', freq='M'))['total_demo'].sum()
    monthly_bio = bio_df.groupby(pd.Grouper(key='date', freq='M'))['total_bio'].sum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_demo.index, monthly_demo.values, label='Demographic Updates', marker='o')
    plt.plot(monthly_bio.index, monthly_bio.values, label='Biometric Updates', marker='x')
    plt.title('Demographic vs Biometric Updates Over Time')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/update_trends_comparison.png")
    print("Saved update_trends_comparison.png")
    
    # Biometric Age Split
    bio_age = bio_df[['bio_age_5_17', 'bio_age_17_']].sum()
    print("\nBiometric Updates by Age:\n", bio_age)
    
    # Demographic Age Split
    demo_age = demo_df[['demo_age_5_17', 'demo_age_17_']].sum()
    print("\nDemographic Updates by Age:\n", demo_age)

    # --- Phase 3: Yearly Enrolment Trend ---
    # Assuming 'date' is already processed in enrolment_df (enrol_df in main)
    # But analyze_updates doesn't have access to enrol_df directly here.
    # We'll add a separate call in main or modify analyze_enrolment.

def analyze_yearly_trends(df):
    print("\n--- Yearly Enrolment Trend ---")
    df = process_date(df)
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year')['total_enrolment'].sum()
    
    plt.figure(figsize=(10, 6))
    yearly.plot(kind='bar', color='skyblue')
    plt.title('Year-wise Aadhaar Enrolment Trend')
    plt.ylabel('Total Enrolments')
    plt.xlabel('Year')
    plt.savefig(f"{OUTPUT_DIR}/yearly_enrolment_trend.png")
    print("Saved yearly_enrolment_trend.png")

def analyze_biometric_age_split(df):
    print("\n--- Biometric Updates by Age Group ---")
    # Columns are bio_age_5_17 and bio_age_17_
    totals = df[['bio_age_5_17', 'bio_age_17_']].sum()
    totals.index = ['Age 5-17 (Mandatory)', 'Age 17+ (Adult)']
    
    plt.figure(figsize=(8, 6))
    totals.plot(kind='bar', color=['orange', 'teal'])
    plt.title('Biometric Updates: School-Age vs Adults')
    plt.ylabel('Total Updates')
    plt.xticks(rotation=0)
    plt.savefig(f"{OUTPUT_DIR}/biometric_age_split.png")
    print("Saved biometric_age_split.png")

def main():
    enrol_df = load_all_data("api_data_aadhar_enrolment")
    analyze_enrolment(enrol_df)
    analyze_yearly_trends(enrol_df)
    
    demo_df = load_all_data("api_data_aadhar_demographic")
    bio_df = load_all_data("api_data_aadhar_biometric")
    
    # Pre-process for numeric
    for c in ['bio_age_5_17', 'bio_age_17_']:
        bio_df[c] = pd.to_numeric(bio_df[c], errors='coerce').fillna(0)
        
    analyze_updates(demo_df, bio_df)
    analyze_biometric_age_split(bio_df)

if __name__ == "__main__":
    main()
