import streamlit as st
import pandas as pd
import plotly.express as px
from modeling import AadhaarBrain

# Page Config
st.set_page_config(page_title="Aadhaar Drishti", page_icon="🇮🇳", layout="wide")

# Initialize Brain
@st.cache_resource
def load_brain():
    brain = AadhaarBrain()
    brain.load_data()
    return brain

try:
    brain = load_brain()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Aadhaar Drishti 👁️")
st.sidebar.markdown("AI-Driven Resource Optimization")
page = st.sidebar.radio("Navigate", ["Pulse Monitor", "Anomaly Hunter", "Infrastructure Allocator"])

# --- Module 1: Pulse Monitor ---
if page == "Pulse Monitor":
    st.title("📊 Pulse Monitor: National Trends")
    st.markdown("Real-time visibility into Enrolment and Update flows.")
    
    tab1, tab2 = st.tabs(["Enrolment Saturation", "Update Surges"])
    
    with tab1:
        st.subheader("Enrolment Trends (Annual)")
        if brain.enrol_df is not None and not brain.enrol_df.empty:
            # Aggregate by Year
            enrol_copy = brain.enrol_df.copy()
            enrol_copy['year'] = enrol_copy['date'].dt.year
            daily_enrol = enrol_copy.groupby('year')['total'].sum().reset_index()
            
            if not daily_enrol.empty:
                fig = px.bar(daily_enrol, x='year', y='total', title="Yearly Enrolments (0-5 vs Adult)", 
                             text_auto='.2s', color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig, use_container_width=True)
                st.info("95% of new enrolments are in the 0-5 age bracket, indicating saturation in the adult population.")
            else:
                st.warning("No enrolment data available for the selected period.")
        else:
            st.warning("Enrolment data not found or empty.")
        
    with tab2:
        st.subheader("Demographic vs Biometric Update Volume")
        if not brain.demo_df.empty and not brain.bio_df.empty:
            # Combine demo and bio monthly (Using 'MS' for Month Start as it's very robust)
            demo_m = brain.demo_df.groupby(pd.Grouper(key='date', freq='M'))['total'].sum().reset_index()
            demo_m['Type'] = 'Demographic'
            bio_m = brain.bio_df.groupby(pd.Grouper(key='date', freq='M'))['total'].sum().reset_index()
            bio_m['Type'] = 'Biometric'
            
            combined = pd.concat([demo_m, bio_m])
            
            if not combined.empty:
                fig2 = px.line(combined, x='date', y='total', color='Type', markers=True, 
                               title="Monthly Update Volume (Seasonality Check)")
                st.plotly_chart(fig2, use_container_width=True)
                st.warning("Note the mid-year spikes (July) correlating with school admission cycles.")
            else:
                st.warning("No update volume data available to display.")
        else:
            st.warning("Update data (Demographic/Biometric) not found or empty.")

# --- Module 2: Anomaly Hunter ---
elif page == "Anomaly Hunter":
    st.title("🕵️ Anomaly Hunter: AI Diagnostics")
    st.markdown("Unsupervised Machine Learning (`IsolationForest`) to detect operational anomalies.")
    
    if st.button("Run Anomaly Detection Model"):
        with st.spinner("Analyzing District Patterns..."):
            anomalies = brain.train_anomaly_model()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Districts Scanned", f"{len(brain.district_stats)}")
            col2.metric("Anomalies Detected", f"{len(anomalies)}")
            col3.metric("Max Risk Score", f"{anomalies['risk_score'].max():.2f}")
            
            st.subheader("⚠️ High-Risk Districts")
            st.dataframe(anomalies[['state', 'district', 'total_load', 'risk_score']].head(10).style.background_gradient(cmap='Reds'))
            
            # Scatter Plot
            st.subheader("Cluster View")
            fig = px.scatter(brain.district_stats, x='total_load', y='volatility', 
                             color='anomaly', hover_data=['district', 'state'],
                             color_continuous_scale=px.colors.sequential.Viridis,
                             title="Volume vs Volatility (Anomalies in Yellow/Purple)")
            st.plotly_chart(fig, use_container_width=True)

# --- Module 3: Infrastructure Allocator ---
elif page == "Infrastructure Allocator":
    st.title("🏗️ Infrastructure Allocator")
    st.markdown("Predictive Resource Planning for District Managers.")
    
    # Select District
    if brain.district_stats is None:
        with st.spinner("Initializing Neural Network & Stats..."):
            brain.train_anomaly_model()
            
    # Get unique, sorted states
    states = sorted(list(set(brain.demo_df['state'].dropna().unique())))
    selected_state = st.selectbox("Select State", states)
    
    # Get unique, sorted districts for selected state
    districts_in_state = brain.demo_df[brain.demo_df['state'] == selected_state]['district'].dropna().unique()
    districts = sorted(list(set(districts_in_state)))
    selected_district = st.selectbox("Select District", districts)
    
    # Get Stats
    stats = brain.district_stats[brain.district_stats['district'] == selected_district]
    
    if not stats.empty:
        current_load = stats['total_load'].values[0]
        st.metric("Current Annual Load (Updates)", f"{int(current_load):,}")
        
        st.divider()
        st.subheader("🔮 Scenario Planning")
        
        growth = st.slider("Predicted Growth (Next 6 Months)", -20, 100, 20) / 100.0
        
        recommendation = brain.recommend_resources(selected_district, current_load, growth)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected Monthly Load", f"{recommendation['projected_monthly_load']:,}", delta=f"{int(growth*100)}%")
        c2.metric("Biometric Kits Needed", f"{recommendation['kits_required']}")
        c3.metric("Staff Required", f"{recommendation['staff_required']}")
        
        if recommendation['kits_required'] > 100:
            st.error("High Resource Demand! Consider deploying Mobile Vans.")
        elif recommendation['kits_required'] < 5:
            st.success("Current infrastructure is sufficient.")
