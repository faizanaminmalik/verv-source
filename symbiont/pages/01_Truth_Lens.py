#pip install streamlit --only-binary :all:
import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Symbiont: Truth Lens",
    page_icon="👁️",
    layout="centered"
)

# --- CSS for "Symbiont" Aesthetics ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("👁️ Symbiont")
st.caption("Decentralized Truth & Ecological Verification Node")

# --- Input Section ---
st.markdown("### 📡 Scan Incoming Info")
user_input = st.text_area(
    "Enter a news headline, product claim, or text snippet:",
    height=100,
    placeholder="Ex: 'Our new synthetic fabric is 100% good for the planet and lowers oil dependency.'"
)

# --- Logic & Visualization ---
if st.button("Activate Truth Lens", type="primary"):
    if not user_input:
        st.warning("⚠️ Please enter data to scan.")
    else:
        with st.spinner("Accessing Distributed Ledger... Querying Nodes..."):
            # Simulate network latency for realism
            time.sleep(1.2) 
            
            try:
                # 1. Call the Local API
                response = requests.post("http://127.0.0.1:8000/verify", json={"text": user_input})
                data = response.json()
                
                # 2. Parse Data
                analysis = data["analysis"]
                eco = data["ecological_impact"]
                warnings = data["warnings"]
                
                truth_score = analysis["truth_score"]
                trust_grade = analysis["trust_grade"]
                
                # --- Result Dashboard ---
                st.markdown("---")
                
                # Top Level Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Truth Score", f"{truth_score}/100", delta=trust_grade)
                with col2:
                    st.metric("Bias Level", f"{analysis['bias_level']}", help="0.0 = Fact, 1.0 = Opinion")
                with col3:
                    # Color code the eco impact
                    eco_label = eco['rating']
                    st.metric("Eco-Cost", eco_label)

                # Visual Logic for Truth Score
                if truth_score >= 75:
                    st.success(f"✅ VERIFIED: This statement appears reliable. ({analysis['network_consensus']})")
                elif truth_score >= 50:
                    st.warning(f"⚠️ CONTESTED: Mixed consensus in the network. ({analysis['network_consensus']})")
                else:
                    st.error(f"❌ UNVERIFIED: High likelihood of misinformation or subjectivity.")

                # Warnings & Eco-Details
                with st.expander("🔍 Deep Scan Details", expanded=True):
                    if warnings:
                        st.subheader("Flags Detected:")
                        for warn in warnings:
                            st.write(f"- 🚩 {warn}")
                    else:
                        st.write("No active flags detected. Content seems neutral.")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Is the backend server (main.py) running?")