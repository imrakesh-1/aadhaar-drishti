from modeling import AadhaarBrain
import pandas as pd

brain = AadhaarBrain()
brain.load_data()

print("\n--- Jammu Variations ---")
states = brain.demo_df['state'].unique()
jammu_states = [s for s in states if "Jammu" in str(s)]
for s in jammu_states:
    print(f"'{s}'")
