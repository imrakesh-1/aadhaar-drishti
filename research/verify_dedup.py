from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

print("\n--- State Deduplication Check ---")
states = sorted(list(set(brain.demo_df['state'].dropna().unique())))
print(f"Total unique states: {len(states)}")
print("First 10 states:", states[:10])

# Check for near-duplicates (case variations)
state_lower = [s.lower() for s in states]
if len(state_lower) != len(set(state_lower)):
    print("⚠️ WARNING: Case-insensitive duplicates found!")
else:
    print("✓ No case-insensitive duplicates")

print("\n--- District Deduplication Check (Sample: Karnataka) ---")
if 'Karnataka' in states:
    districts = brain.demo_df[brain.demo_df['state'] == 'Karnataka']['district'].dropna().unique()
    districts_sorted = sorted(list(set(districts)))
    print(f"Total unique districts in Karnataka: {len(districts_sorted)}")
    print("First 10:", districts_sorted[:10])
