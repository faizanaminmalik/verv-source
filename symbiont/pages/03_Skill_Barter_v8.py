import streamlit as st
import random
import uuid
import graphviz
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Neural Network Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

# --- Config ---
st.set_page_config(page_title="Symbiont: Knowledge Economy", page_icon="🎓", layout="wide")

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
            with st.spinner("Loading Neural Brain & Pre-computing Skill Vectors..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                # Level 8 Optimization: Pre-encode the entire catalog to allow dynamic switching
                self.catalog_vecs = self.model.encode(FUZZY_SKILL_CATALOG, convert_to_tensor=True)
                self.skill_to_idx = {skill: i for i, skill in enumerate(FUZZY_SKILL_CATALOG)}

    def add_user(self, user):
        self.users[user['id']] = user

    def calculate_gini(self, wealths):
        """Calculates Gini Coefficient (0=Perfect Equality, 1=Perfect Inequality)"""
        if not wealths or sum(wealths) == 0: return 0.0
        sorted_wealths = sorted(wealths)
        n = len(wealths)
        numer = 2 * sum((i + 1) * w for i, w in enumerate(sorted_wealths))
        denom = n * sum(sorted_wealths)
        return (numer / denom) - (n + 1) / n

    def simulate_market_epochs(self, epochs=20, threshold=0.65, stimulus_amount=0, learning_rate=0.1):
        """
        Level 8: Simulation with Skill Acquisition and Career Switching.
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Apply Stimulus
        if stimulus_amount > 0:
            for u in users_list:
                u['credits'] += stimulus_amount

        # 1. Initialize Demand Vectors (Static needs for now)
        need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)

        # 2. Initialize Prices
        current_prices = {u['id']: 20.0 for u in users_list}
        
        # Track history
        price_history = [] 
        macro_history = []
        workforce_history = [] # Tracks # of sellers per skill
        transaction_log = []

        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            day_txs = 0
            day_volume = 0
            
            # --- Dynamic Supply Vectors (Level 8) ---
            # Re-map users to vectors daily because they might have switched professions
            current_offers = [u['offering'] for u in users_list]
            
            # Map current user offers to the pre-computed catalog vectors
            # This is much faster than re-encoding every day
            offer_indices = [self.skill_to_idx[off] for off in current_offers]
            current_offer_vecs = self.catalog_vecs[offer_indices]
            
            # Recalculate match matrix
            sim_matrix = util.cos_sim(need_vecs, current_offer_vecs)

            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}

            # Randomize buyer order
            buyer_indices = list(range(len(users_list)))
            random.shuffle(buyer_indices)

            # Attempt Trades
            for i in buyer_indices:
                buyer = users_list[i]
                if not buyers_available[buyer['id']]: continue

                best_match = 0
                best_seller_idx = -1
                
                # Buyer Strategy
                for j, seller in enumerate(users_list):
                    if i == j: continue
                    if not sellers_available[seller['id']]: continue
                    
                    score = sim_matrix[i][j].item()
                    if score > threshold:
                        price = current_prices[seller['id']]
                        if buyer['credits'] >= price:
                            # Value calculation
                            value_score = (score * 100) / (price + 1)
                            if value_score > best_match:
                                best_match = value_score
                                best_seller_idx = j

                # Execute Trade
                if best_seller_idx != -1:
                    seller = users_list[best_seller_idx]
                    price = current_prices[seller['id']]
                    skill_sold = seller['offering']
                    
                    # Transaction
                    buyer['credits'] -= price
                    seller['credits'] += price
                    buyers_available[buyer['id']] = False
                    sellers_available[seller['id']] = False
                    
                    # --- Level 8: Learning Mechanism ---
                    # Buyer gains XP in the skill they just bought
                    if skill_sold not in buyer['xp']: buyer['xp'][skill_sold] = 0
                    buyer['xp'][skill_sold] += learning_rate
                    
                    # Mastery Check: Did they learn enough to sell it?
                    if buyer['xp'][skill_sold] >= 1.0 and skill_sold not in buyer['known_skills']:
                        buyer['known_skills'].append(skill_sold)
                        # Optional: Notify log
                        # transaction_log.append({"Day": day, "Event": f"{buyer['name']} mastered {skill_sold}!"})
                    
                    # Price Adjustments (Supply/Demand)
                    current_prices[seller['id']] *= 1.10 # Seller raises price
                    
                    transaction_log.append({
                        "Day": day,
                        "Buyer": buyer['name'],
                        "Seller": seller['name'],
                        "Skill": skill_sold,
                        "Price": price
                    })
                    day_txs += 1
                    day_volume += price
                
            # --- End of Day Logic ---
            
            # 1. Price Decay for unsold inventory
            for u in users_list:
                if sellers_available[u['id']]:
                    current_prices[u['id']] *= 0.90 # Aggressive discount
                current_prices[u['id']] = max(2.0, min(200.0, current_prices[u['id']]))

            # 2. Career Switching (Adaptive Workforce)
            # Users look at their known skills and pick the one with highest market average
            # Calculate avg price per skill for today
            skill_market_avgs = {}
            counts = {}
            for u in users_list:
                sk = u['offering']
                p = current_prices[u['id']]
                skill_market_avgs[sk] = skill_market_avgs.get(sk, 0) + p
                counts[sk] = counts.get(sk, 0) + 1
            
            for sk in skill_market_avgs:
                skill_market_avgs[sk] /= counts[sk]
            
            # Agents decide: "Should I switch?"
            for u in users_list:
                best_skill = u['offering']
                best_potential_price = skill_market_avgs.get(best_skill, 0)
                
                # Check other skills they know
                for known in u['known_skills']:
                    market_rate = skill_market_avgs.get(known, 10.0) # Assume base if no market
                    # If market rate for known skill is > current offering + switching cost buffer
                    if market_rate > (best_potential_price * 1.2): 
                        best_skill = known
                        best_potential_price = market_rate
                
                # Switch profession
                if best_skill != u['offering']:
                    u['offering'] = best_skill
                    # Reset their personal price to market avg
                    current_prices[u['id']] = best_potential_price

            # 3. Data Collection
            current_wealths = [u['credits'] for u in users_list]
            gini = self.calculate_gini(current_wealths)
            
            macro_history.append({
                "Day": day,
                "Gini": gini,
                "Volume": day_volume,
                "Tx_Count": day_txs
            })

            snapshot_price = {"Day": day}
            snapshot_workforce = {"Day": day}
            
            # Track price averages
            for skill in FUZZY_SKILL_CATALOG:
                if skill in skill_market_avgs:
                    snapshot_price[skill] = skill_market_avgs[skill]
                else:
                    snapshot_price[skill] = None # No trade
                
                # Track workforce count
                count = sum([1 for u in users_list if u['offering'] == skill])
                snapshot_workforce[skill] = count

            price_history.append(snapshot_price)
            workforce_history.append(snapshot_workforce)
            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(price_history), pd.DataFrame(macro_history), pd.DataFrame(workforce_history)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

    strategies = ["Balanced", "Undercutter", "Premium"]
    weights = [0.6, 0.3, 0.1]

    for i in range(n):
        offer = random.choice(FUZZY_SKILL_CATALOG)
        need = random.choice([s for s in FUZZY_SKILL_CATALOG if s != offer])
        starting_credits = random.randint(80, 120) 
        strat = random.choices(strategies, weights)[0]
        
        user = {
            "id": str(uuid.uuid4())[:8],
            "name": f"User_{random.randint(100,999)}",
            "offering": offer,
            "needing": need,
            "credits": starting_credits,
            "strategy": strat,
            "known_skills": [offer], # Level 8: List of mastery
            "xp": {offer: 1.0}       # Level 8: Experience points (0.0 to 1.0+)
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🎓 Symbiont: The Knowledge Economy")
st.markdown("### Level 8: Civilization Engine")
st.caption("Simulation of **Skill Propagation** and **Labor Mobility**. Agents 'Learn' from trades and switch careers to chase higher wages.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas matplotlib`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Simulation Config")
    num_users = st.slider("Population Size", 10, 80, 40)
    epochs = st.slider("Duration (Days)", 20, 100, 40)
    learning_rate = st.slider("Learning Rate", 0.05, 0.5, 0.20, help="How fast buyers master new skills (0.2 = 5 trades to learn).")
    
    st.markdown("---")
    st.markdown("**Fiscal Policy**")
    stimulus = st.number_input("Stimulus Injection", 0, 500, 100)
    
    if st.button("Run Civilization Model"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating Evolution..."):
            final_prices, tx_log, history_df, macro_df, work_df = st.session_state.neural_matcher.simulate_market_epochs(epochs, 0.65, stimulus, learning_rate)
            
            st.session_state.final_prices = final_prices
            st.session_state.tx_log = tx_log
            st.session_state.history_df = history_df
            st.session_state.macro_df = macro_df
            st.session_state.work_df = work_df
            
        st.success("Evolution Complete")

    # --- Ticker ---
    if 'work_df' in st.session_state and not st.session_state.work_df.empty:
        st.markdown("---")
        st.subheader("👨‍💻 Workforce Stats")
        last_day = st.session_state.work_df.iloc[-1]
        
        # Sort by most popular jobs
        jobs = last_day.drop("Day").sort_values(ascending=False)
        st.write("**Most Common Jobs (Day " + str(int(last_day['Day'])) + ")**")
        for skill, count in jobs.head(5).items():
            if count > 0:
                st.write(f"• {skill}: **{int(count)} workers**")

# --- Visualization ---
with col2:
    if 'history_df' in st.session_state and not st.session_state.history_df.empty:
        # Tabbed View for Analysis
        tab1, tab2, tab3 = st.tabs(["Workforce Evolution", "Price Trends", "Inequality"])
        
        with tab1:
            st.subheader("🌊 The Great Resignation (Labor Mobility)")
            st.caption("Watch how the population switches jobs over time to chase profit.")
            
            w_df = st.session_state.work_df.set_index("Day")
            # Only show skills that actually had workers
            active_skills = w_df.columns[(w_df.sum() > 0)].tolist()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Stackplot to show share of workforce
            ax.stackplot(w_df.index, [w_df[s] for s in active_skills], labels=active_skills, alpha=0.8)
            ax.set_ylabel("Number of Workers")
            ax.set_xlabel("Day")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            st.pyplot(fig)
            
        with tab2:
            st.subheader("🏷️ Price Collapses")
            st.caption("As skills spread (Learning), supply increases, and prices should fall.")
            
            p_df = st.session_state.history_df.set_index("Day")
            variances = p_df.var().sort_values(ascending=False)
            top_skills = variances.head(5).index.tolist()
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for skill in top_skills:
                ax2.plot(p_df.index, p_df[skill], label=skill, linewidth=2)
            ax2.set_ylabel("Price (Credits)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        with tab3:
            st.subheader("⚖️ Wealth Inequality (Gini)")
            macro = st.session_state.macro_df.set_index("Day")
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(macro.index, macro['Gini'], color='red', linewidth=2)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Gini Coefficient")
            ax3.grid(True)
            st.pyplot(fig3)
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize simulation to see the economy evolve.")