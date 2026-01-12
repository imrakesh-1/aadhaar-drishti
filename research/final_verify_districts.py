from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

print("\n--- West Bengal District Verification ---")
wb_districts = sorted(brain.demo_df[brain.demo_df['state'] == 'West Bengal']['district'].unique())
for d in wb_districts:
    print(f"  - {d}")

# Check if 'Haora' or 'Hugli' still exist
for bad in ['Haora', 'Hugli', 'Midnapore']:
    if bad in wb_districts:
        print(f"❌ ERROR: '{bad}' still exists!")
    else:
        print(f"✅ SUCCESS: '{bad}' is consolidated.")

print("\n--- Top States Cleanliness Check ---")
for state in ['Maharashtra', 'Uttar Pradesh', 'Kerala', 'Tamil Nadu']:
    districts = sorted(brain.demo_df[brain.demo_df['state'] == state]['district'].unique())
    print(f"\n{state}: {len(districts)} districts")
    # Show first 10 for brevity
    for d in districts[:10]:
        print(f"  - {d}")
