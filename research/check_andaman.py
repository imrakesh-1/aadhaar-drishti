from modeling import AadhaarBrain

brain = AadhaarBrain()
brain.load_data()

print("\n=== Andaman & Nicobar Islands Districts ===\n")

# Check in all three datasets
for name, df in [('Demographic', brain.demo_df), ('Biometric', brain.bio_df), ('Enrolment', brain.enrol_df)]:
    if df is not None and 'state' in df.columns:
        an_data = df[df['state'] == 'Andaman & Nicobar Islands']
        if not an_data.empty:
            districts = sorted(an_data['district'].unique())
            print(f"{name} Data:")
            for d in districts:
                count = len(an_data[an_data['district'] == d])
                print(f"  '{d}': {count} records")
            print()
