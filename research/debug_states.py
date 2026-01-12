from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

print("\n--- Unique States in Demographic Data ---")
states = brain.demo_df['state'].unique()
for s in states:
    print(f"'{s}' (Type: {type(s)})")

print("\n--- Sample Rows with '100000' as State (if exists) ---")
# Check if any state looks like a number
try:
    bad_rows = brain.demo_df[brain.demo_df['state'].astype(str).str.match(r'^\d+$')]
    if not bad_rows.empty:
        print(bad_rows.head())
except:
    pass
