from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

print("\n--- Inspecting District Values ---")
districts = brain.demo_df['district'].unique()
numeric_districts = [d for d in districts if str(d).isdigit()]
print(f"Found {len(numeric_districts)} numeric 'district' values: {numeric_districts[:10]}")

print("\n--- Typos/Inconsistencies Check (First 20 sorted) ---")
print(sorted([str(d) for d in districts])[:20])
