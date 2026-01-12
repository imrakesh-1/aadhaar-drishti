from modeling import AadhaarBrain
import pandas as pd
from collections import defaultdict

brain = AadhaarBrain()
brain.load_data()

print("\n=== DISTRICT DATA QUALITY ANALYSIS ===\n")

# Group districts by state
state_districts = defaultdict(list)
for state in sorted(brain.demo_df['state'].unique()):
    districts = sorted(brain.demo_df[brain.demo_df['state'] == state]['district'].unique())
    state_districts[state] = districts

# Find potential duplicates (case-insensitive)
print("--- States with Potential Duplicate Districts ---\n")
for state, districts in state_districts.items():
    lower_map = defaultdict(list)
    for d in districts:
        lower_map[str(d).lower()].append(d)
    
    duplicates = {k: v for k, v in lower_map.items() if len(v) > 1}
    if duplicates:
        print(f"{state}:")
        for lower, variants in duplicates.items():
            print(f"  {variants}")
        print()

# Check for common issues
print("\n--- Districts with Special Characters ---")
special_char_districts = []
for state, districts in state_districts.items():
    for d in districts:
        if any(char in str(d) for char in ['*', '(', ')', '/', '\\', '#', '@']):
            special_char_districts.append((state, d))

for state, dist in special_char_districts[:20]:  # Show first 20
    print(f"  {state}: '{dist}'")

print(f"\nTotal: {len(special_char_districts)}")

# Sample a few states to show district counts
print("\n--- Sample District Counts ---")
for state in ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']:
    if state in state_districts:
        print(f"{state}: {len(state_districts[state])} districts")
