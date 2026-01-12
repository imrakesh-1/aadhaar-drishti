from modeling import AadhaarBrain
from collections import defaultdict

brain = AadhaarBrain()
brain.load_data()

print("\n=== COMPREHENSIVE DUPLICATE CHECK ===\n")

total_duplicates = 0

for state in sorted(brain.demo_df['state'].unique()):
    districts = list(brain.demo_df[brain.demo_df['state'] == state]['district'].unique())
    
    # Find duplicates by normalizing (lowercase, no spaces/hyphens)
    lower_map = defaultdict(list)
    for d in districts:
        normalized = str(d).lower().replace(' ', '').replace('-', '').replace('.', '')
        lower_map[normalized].append(d)
    
    duplicates = {k: v for k, v in lower_map.items() if len(v) > 1}
    if duplicates:
        total_duplicates += len(duplicates)
        print(f"{state}:")
        for normalized, variants in sorted(duplicates.items()):
            print(f"  {variants}")
        print()

if total_duplicates == 0:
    print("✓ NO DUPLICATES FOUND - All districts are clean!")
else:
    print(f"\n⚠️ Found {total_duplicates} duplicate sets")

# Check specific states the user mentioned
print("\n=== Specific State Checks ===")
for state in ['Andaman & Nicobar Islands', 'Karnataka', 'Gujarat', 'Andhra Pradesh']:
    if state in brain.demo_df['state'].unique():
        districts = sorted(brain.demo_df[brain.demo_df['state'] == state]['district'].unique())
        print(f"\n{state}: {len(districts)} districts")
        for d in districts:
            print(f"  - {d}")
