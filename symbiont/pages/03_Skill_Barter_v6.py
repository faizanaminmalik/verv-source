import streamlit as st
import random
import uuid
import graphviz
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

# --- Neural Network Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

# --- Config ---
st.set_page_config(page_title="Symbiont: Evolutionary Economy", page_icon="📈", layout="wide")

# --- Extended "Fuzzy" Catalog for Semantic Matching ---
FUZZY_SKILL_CATALOG = [
    "Python Programming", "Web Development", "React JS", "Data Science",
    "Conversational Spanish", "French for Beginners", "Mandarin",
    "Sourdough Baking", "Gourmet Cooking", "Meal Prepping",
    "Guitar Lessons", "Piano Tutoring", "Music Theory",
    "Yoga Flow", "Pilates", "Personal Training",
    "Graphic Design", "Logo Creation", "Digital Art",
    "Financial Planning", "Tax Help", "Investment Advice"
]

class NeuralSkillMatcher:
    def __init__(self):
        self.users = {}
        if HAS_NEURAL:
            with st.spinner("Loading Neural Weights (all-MiniLM-L6-v2)..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_user(self, user):
        self.users[user['id']] = user

    def simulate_market_epochs(self, epochs=20, threshold=0.65):
        """
        Level 6: Runs a multi-day simulation where prices evolve based on 
        successful vs. failed trades (Adaptive Supply/Demand).
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame()

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame()

        # 1. Initialize Vectors
        offer_vecs = self.model.encode([u['offering'] for u in users_list], convert_to_tensor=True)
        need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)
        sim_matrix = util.cos_sim(need_vecs, offer_vecs)

        # 2. Initialize Prices (Everyone starts at 10.0)
        current_prices = {u['id']: 10.0 for u in users_list}
        
        # Track history for plotting
        price_history = []  # List of dicts
        transaction_log = []

        # 3. Run Simulation Loop (Days)
        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            day_txs = 0
            
            # Record prices at start of day
            snapshot = {"Day": day}
            # We average prices per skill type for cleaner plotting
            skill_prices = {} 
            
            # Reset daily availability (Everyone can sell once per day)
            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}

            # Attempt Trades
            for i, buyer in enumerate(users_list):
                if not buyers_available[buyer['id']]: continue

                # Find best affordable seller
                best_match = 0
                best_seller_idx = -1
                
                for j, seller in enumerate(users_list):
                    if i == j: continue
                    if not sellers_available[seller['id']]: continue
                    
                    score = sim_matrix[i][j].item()
                    if score > threshold and score > best_match:
                        price = current_prices[seller['id']]
                        # Check affordability
                        if buyer['credits'] >= price:
                            best_match = score
                            best_seller_idx = j

                # Execute Trade
                if best_seller_idx != -1:
                    seller = users_list[best_seller_idx]
                    price = current_prices[seller['id']]
                    
                    # Transfer Credits
                    buyer['credits'] -= price
                    seller['credits'] += price
                    
                    # Mark availability
                    buyers_available[buyer['id']] = False
                    sellers_available[seller['id']] = False
                    
                    # Market Feedback: Demand exists! Raise Price.
                    # "The Invisible Hand" algorithm
                    current_prices[seller['id']] *= 1.10  # +10% demand spike
                    
                    transaction_log.append({
                        "Day": day,
                        "Buyer": buyer['name'],
                        "Seller": seller['name'],
                        "Skill": seller['offering'],
                        "Price": price
                    })
                    day_txs += 1
                
            # End of Day Adjustments
            for u in users_list:
                # If seller didn't sell today, lower price to attract buyers tomorrow
                if sellers_available[u['id']]: # means they were available but no one bought
                    current_prices[u['id']] *= 0.95 # -5% discount strategy
                
                # Cap prices to prevent explosion
                current_prices[u['id']] = max(2.0, min(100.0, current_prices[u['id']]))
                
                # Aggregate for charts
                skill = u['offering']
                if skill not in skill_prices: skill_prices[skill] = []
                skill_prices[skill].append(current_prices[u['id']])

            # Store averages for plotting
            for skill, p_list in skill_prices.items():
                snapshot[skill] = sum(p_list) / len(p_list)
            
            price_history.append(snapshot)
            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(price_history)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

    for i in range(n):
        offer = random.choice(FUZZY_SKILL_CATALOG)
        need = random.choice([s for s in FUZZY_SKILL_CATALOG if s != offer])
        
        # Level 6: Higher starting wealth for liquidity
        starting_credits = random.randint(50, 150) 
        
        user = {
            "id": str(uuid.uuid4())[:8],
            "name": f"User_{random.randint(100,999)}",
            "offering": offer,
            "needing": need,
            "credits": starting_credits,
            "reputation": round(random.uniform(3.5, 5.0), 1)
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("📈 Symbiont: Evolutionary Economy")
st.markdown("### Level 6: Adaptive Market Simulation")
st.caption("The AI simulates 30 days of trading. Prices evolve based on successful trades (Demand) vs. unsold inventory (Supply).")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Simulation Settings")
    num_users = st.slider("Population Size", 10, 50, 25)
    epochs = st.slider("Simulation Duration (Days)", 10, 100, 30)
    match_threshold = st.slider("Match Threshold", 0.5, 0.9, 0.65)
    
    if st.button("Run Evolutionary Model"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating {epochs} days of trading..."):
            final_prices, tx_log, history_df = st.session_state.neural_matcher.simulate_market_epochs(epochs, match_threshold)
            
            st.session_state.final_prices = final_prices
            st.session_state.tx_log = tx_log
            st.session_state.history_df = history_df
            
        st.success("Equilibrium Reached!")
        st.metric("Total Trades", len(tx_log))

    # --- Ticker ---
    if 'final_prices' in st.session_state:
        st.markdown("---")
        st.subheader("🏷️ Final Clearing Prices")
        # Group by skill to find avg price
        skill_avgs = {}
        for uid, price in st.session_state.final_prices.items():
            u = st.session_state.neural_matcher.users[uid]
            sk = u['offering']
            if sk not in skill_avgs: skill_avgs[sk] = []
            skill_avgs[sk].append(price)
            
        # Sort by expensive
        sorted_skills = sorted(skill_avgs.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        
        st.write("**Most Expensive (High Demand)**")
        for sk, plist in sorted_skills[:3]:
            avg = sum(plist)/len(plist)
            st.write(f"• {sk}: **{avg:.2f} cr**")
            
        st.write("**Least Expensive (Low Demand)**")
        for sk, plist in sorted_skills[-3:]:
            avg = sum(plist)/len(plist)
            st.write(f"• {sk}: **{avg:.2f} cr**")

# --- Visualization ---
with col2:
    if 'history_df' in st.session_state and not st.session_state.history_df.empty:
        st.subheader("📊 Price Discovery Charts")
        st.caption("Watch how the AI adjusted prices over time to clear the market.")
        
        # Reshape for Streamlit Line Chart
        df = st.session_state.history_df.set_index("Day")
        
        # Filter to show only top 5 volatile skills to avoid clutter
        # Variance check
        variances = df.var().sort_values(ascending=False)
        top_skills = variances.head(7).index.tolist()
        
        # Use Matplotlib instead of Altair/Vega-Lite (which breaks on Python 3.14)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each skill line
        for skill in top_skills:
            ax.plot(df.index, df[skill], label=skill)
            
        ax.set_title("Price Volatility (Top 7 Skills)")
        ax.set_xlabel("Simulation Day")
        ax.set_ylabel("Price (Credits)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.info("""
        **Chart Interpretation:**
        - **Rising Line:** High Demand. The seller sold out frequently, so the AI raised prices.
        - **Falling Line:** Low Demand. The seller struggled to find buyers, so the AI discounted the service.
        - **Flat Line:** Market Equilibrium.
        """)
        
    if 'tx_log' in st.session_state and st.session_state.tx_log:
        st.subheader("📜 Last 5 Days Activity")
        recent = st.session_state.tx_log[-8:]
        for tx in reversed(recent):
            st.markdown(f"🗓️ **Day {tx['Day']}**: {tx['Buyer']} bought **{tx['Skill']}** from {tx['Seller']} for `{tx['Price']:.2f} cr`")
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize simulation to see the economy evolve.")