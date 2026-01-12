from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

# Official list of Indian States and Union Territories (as of 2026)
VALID_STATES_UTS = {
    # States (28)
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    
    # Union Territories (8)
    'Andaman & Nicobar Islands', 'Chandigarh', 
    'Dadra & Nagar Haveli And Daman & Diu',
    'Delhi', 'Jammu & Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
}

print(f"\n--- Official Count: {len(VALID_STATES_UTS)} States/UTs ---\n")

current_states = sorted(list(set(brain.demo_df['state'].dropna().unique())))
print(f"Current data has: {len(current_states)} unique values\n")

print("--- INVALID Entries (Not in official list) ---")
invalid = [s for s in current_states if s not in VALID_STATES_UTS]
for inv in invalid:
    count = len(brain.demo_df[brain.demo_df['state'] == inv])
    print(f"  '{inv}' ({count} records)")

print(f"\nTotal invalid entries: {len(invalid)}")
