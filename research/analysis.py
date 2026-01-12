import pandas as pd
import os
import glob

DATA_DIR = "/Users/rakeshmondal/Downloads/uidai data "

def load_data(folder_name):
    # Depending on how the unzip worked, the folder might not have the trailing space
    # The folders created were just "api_data_aadhar_enrolment" etc. inside the cwd.
    # CWD is DATA_DIR.
    path = os.path.join(DATA_DIR, folder_name)
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print(f"No files found in {path}")
        return None
        
    print(f"Loading {len(all_files)} files from {folder_name}...")
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, dtype=str) 
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Successfully loaded {folder_name}. Shape: {combined_df.shape}")
        return combined_df
    return None

def inspect_df(name, df):
    if df is None:
        return
    print(f"\n--- Inspecting {name} ---")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\ncolumns:", df.columns.tolist())

def main():
    print("Starting Data Inspection...")
    
    # Enrolment
    enrol_df = load_data("api_data_aadhar_enrolment")
    inspect_df("Enrolment Data", enrol_df)

    # Demographic
    demo_df = load_data("api_data_aadhar_demographic")
    inspect_df("Demographic Update Data", demo_df)

    # Biometric
    bio_df = load_data("api_data_aadhar_biometric")
    inspect_df("Biometric Update Data", bio_df)

if __name__ == "__main__":
    main()
