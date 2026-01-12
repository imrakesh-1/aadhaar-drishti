from modeling import AadhaarBrain
import pandas as pd
from collections import defaultdict

brain = AadhaarBrain()
brain.load_data()

print("\n=== FINDING ALL DUPLICATE DISTRICTS ===\n")

# Group districts by state
for state in sorted(brain.demo_df['state'].unique()):
    districts = list(brain.demo_df[brain.demo_df['state'] == state]['district'].unique())
    
    # Find duplicates by checking similar names
    lower_map = defaultdict(list)
    for d in districts:
        # Normalize for comparison: lowercase, remove spaces/punctuation
        normalized = str(d).lower().replace(' ', '').replace('-', '').replace('.', '')
        lower_map[normalized].append(d)
    
    # Show only states with duplicates
    duplicates = {k: v for k, v in lower_map.items() if len(v) > 1}
    if duplicates:
        print(f"\n{state}:")
        for normalized, variants in sorted(duplicates.items()):
            print(f"  Variants: {variants}")
            for v in variants:
                count = len(brain.demo_df[(brain.demo_df['state'] == state) & (brain.demo_df['district'] == v)])
                print(f"    '{v}': {count} records")
