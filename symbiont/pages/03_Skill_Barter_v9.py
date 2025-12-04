import streamlit as st
import random
import uuid
import graphviz
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

# --- Neural Network Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

# --- Config ---
st.set_page_config(page_title="Symbiont: Darwinian Market", page_icon="🧬", layout="wide")

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

    def simulate_market_epochs(self, epochs=20, threshold=0.65, stimulus_amount=0, learning_rate=0.1, shock_chance=0.05):
        """
        Level 9: Darwinian Simulation with Bankruptcy, Reproduction, and Market Shocks.
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Apply Stimulus
        if stimulus_amount > 0:
            for u in users_list:
                u['credits'] += stimulus_amount

        # 1. Initialize Demand Vectors (Dynamic now)
        # We store needs as indices to allow quick updates during shocks
        current_needs = [u['needing'] for u in users_list]

        # 2. Initialize Prices
        current_prices = {u['id']: 20.0 for u in users_list}
        
        # Track history
        price_history = [] 
        macro_history = []
        workforce_history = [] 
        strategy_history = [] # Tracks population of Undercutters vs Premium
        transaction_log = []
        
        # Stats
        deaths = 0
        births = 0

        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            # --- Level 9: Market Shock ---
            if random.random() < shock_chance:
                shock_skill = random.choice(FUZZY_SKILL_CATALOG)
                # 20% of population suddenly wants this skill
                for u in random.sample(users_list, k=int(len(users_list)*0.2)):
                    u['needing'] = shock_skill
                transaction_log.append({"Day": day, "Buyer": "MARKET", "Seller": "SHOCK", "Skill": shock_skill, "Price": 0})
            
            # Re-encode needs daily (in case of shocks)
            need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)

            day_txs = 0
            day_volume = 0
            
            # --- Dynamic Supply Vectors ---
            current_offers = [u['offering'] for u in users_list]
            offer_indices = [self.skill_to_idx[off] for off in current_offers]
            current_offer_vecs = self.catalog_vecs[offer_indices]
            
            sim_matrix = util.cos_sim(need_vecs, current_offer_vecs)

            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}

            # Randomize buyer order
            buyer_indices = list(range(len(users_list)))
            random.shuffle(buyer_indices)

            # Attempt Trades
            for i in buyer_indices:
                if i >= len(users_list): continue # Handle list shrinking
                buyer = users_list[i]
                if not buyers_available.get(buyer['id'], False): continue

                best_match = 0
                best_seller_idx = -1
                
                # Buyer Strategy
                for j, seller in enumerate(users_list):
                    if buyer['id'] == seller['id']: continue
                    if not sellers_available.get(seller['id'], False): continue
                    
                    score = sim_matrix[i][j].item()
                    if score > threshold:
                        price = current_prices.get(seller['id'], 20.0)
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
                    
                    # Learning
                    if skill_sold not in buyer['xp']: buyer['xp'][skill_sold] = 0
                    buyer['xp'][skill_sold] += learning_rate
                    if buyer['xp'][skill_sold] >= 1.0 and skill_sold not in buyer['known_skills']:
                        buyer['known_skills'].append(skill_sold)
                    
                    # Price Adjustments based on Strategy
                    if seller['strategy'] == "Premium":
                         current_prices[seller['id']] *= 1.15
                    elif seller['strategy'] == "Undercutter":
                         current_prices[seller['id']] *= 1.02 # Raise slowly to keep volume
                    else:
                         current_prices[seller['id']] *= 1.10
                    
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
            
            # 1. Survival & Reproduction (Darwinian Logic)
            survivors = []
            new_babies = []
            
            for u in users_list:
                # Inventory Decay
                if sellers_available.get(u['id']):
                    decay = 0.85 if u['strategy'] == "Undercutter" else 0.95
                    current_prices[u['id']] *= decay
                current_prices[u['id']] = max(2.0, min(200.0, current_prices[u['id']]))

                # Bankruptcy Check (Death)
                # Cost of living per day = 2 credits
                u['credits'] -= 2
                
                if u['credits'] > 0:
                    survivors.append(u)
                    
                    # Reproduction (Wealthy agents spawn apprentices)
                    if u['credits'] > 250:
                        u['credits'] -= 100 # Cost to spawn
                        baby = copy.deepcopy(u)
                        baby['id'] = str(uuid.uuid4())[:8]
                        baby['name'] = f"Gen2_{u['name']}"
                        baby['credits'] = 50
                        baby['reputation'] = 3.0
                        # Mutation: Small chance to change strategy
                        if random.random() < 0.1:
                            baby['strategy'] = random.choice(["Balanced", "Undercutter", "Premium"])
                        
                        new_babies.append(baby)
                        current_prices[baby['id']] = 15.0 # Starting price
                        births += 1
                else:
                    deaths += 1
            
            users_list = survivors + new_babies
            self.users = {u['id']: u for u in users_list} # Update main dict
            
            # 2. Career Switching
            skill_market_avgs = {}
            counts = {}
            for u in users_list:
                sk = u['offering']
                p = current_prices.get(u['id'], 10.0)
                skill_market_avgs[sk] = skill_market_avgs.get(sk, 0) + p
                counts[sk] = counts.get(sk, 0) + 1
            
            for sk in skill_market_avgs: skill_market_avgs[sk] /= counts[sk]
            
            for u in users_list:
                best_skill = u['offering']
                best_price = skill_market_avgs.get(best_skill, 0)
                for known in u['known_skills']:
                    rate = skill_market_avgs.get(known, 10.0)
                    if rate > (best_price * 1.3): 
                        best_skill = known
                        best_price = rate
                if best_skill != u['offering']:
                    u['offering'] = best_skill
                    current_prices[u['id']] = best_price

            # 3. Data Collection
            current_wealths = [u['credits'] for u in users_list]
            gini = self.calculate_gini(current_wealths)
            
            macro_history.append({
                "Day": day, "Gini": gini, "Volume": day_volume, 
                "Tx_Count": day_txs, "Population": len(users_list)
            })

            snapshot_price = {"Day": day}
            snapshot_workforce = {"Day": day}
            snapshot_strat = {"Day": day, "Balanced": 0, "Undercutter": 0, "Premium": 0}
            
            for u in users_list:
                snapshot_strat[u['strategy']] += 1
            
            for skill in FUZZY_SKILL_CATALOG:
                snapshot_price[skill] = skill_market_avgs.get(skill, None)
                snapshot_workforce[skill] = sum([1 for u in users_list if u['offering'] == skill])

            price_history.append(snapshot_price)
            workforce_history.append(snapshot_workforce)
            strategy_history.append(snapshot_strat)
            
            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(price_history), pd.DataFrame(macro_history), pd.DataFrame(workforce_history), pd.DataFrame(strategy_history)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

    strategies = ["Balanced", "Undercutter", "Premium"]
    weights = [0.5, 0.4, 0.1]

    for i in range(n):
        offer = random.choice(FUZZY_SKILL_CATALOG)
        need = random.choice([s for s in FUZZY_SKILL_CATALOG if s != offer])
        starting_credits = random.randint(60, 100) 
        strat = random.choices(strategies, weights)[0]
        
        user = {
            "id": str(uuid.uuid4())[:8],
            "name": f"User_{random.randint(100,999)}",
            "offering": offer,
            "needing": need,
            "credits": starting_credits,
            "strategy": strat,
            "known_skills": [offer],
            "xp": {offer: 1.0}
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🧬 Symbiont: The Darwinian Market")
st.markdown("### Level 9: Survival of the Fittest")
st.caption("A living economy. Wealthy agents spawn new apprentices (Reproduction). Poor agents go bankrupt (Death). Strategies evolve.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas matplotlib`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Ecosystem Config")
    num_users = st.slider("Initial Population", 20, 100, 50)
    epochs = st.slider("Simulation Days", 20, 100, 45)
    shock_chance = st.slider("Market Shock Probability", 0.0, 0.2, 0.05, help="Chance of a sudden demand shift.")
    
    if st.button("Run Evolution"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating Natural Selection..."):
            final_prices, tx_log, history_df, macro_df, work_df, strat_df = st.session_state.neural_matcher.simulate_market_epochs(epochs, 0.65, 0, 0.1, shock_chance)
            
            st.session_state.final_prices = final_prices
            st.session_state.tx_log = tx_log
            st.session_state.history_df = history_df
            st.session_state.macro_df = macro_df
            st.session_state.work_df = work_df
            st.session_state.strat_df = strat_df
            
        st.success("Evolution Complete")

    # --- Ticker ---
    if 'macro_df' in st.session_state and not st.session_state.macro_df.empty:
        st.markdown("---")
        st.subheader("💀 Vital Statistics")
        start_pop = num_users
        end_pop = st.session_state.macro_df.iloc[-1]['Population']
        
        col_a, col_b = st.columns(2)
        col_a.metric("Population", f"{int(end_pop)}", f"{int(end_pop - start_pop)}")
        col_b.metric("Avg Gini", f"{st.session_state.macro_df['Gini'].mean():.2f}")

# --- Visualization ---
with col2:
    if 'history_df' in st.session_state and not st.session_state.history_df.empty:
        tab1, tab2, tab3 = st.tabs(["Strategy Evolution", "Workforce & Prices", "Macro Health"])
        
        with tab1:
            st.subheader("🧬 Survival of Strategies")
            st.caption("Which pricing strategy conquered the market?")
            
            s_df = st.session_state.strat_df.set_index("Day")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stackplot(s_df.index, s_df["Balanced"], s_df["Undercutter"], s_df["Premium"], 
                         labels=["Balanced", "Undercutter", "Premium"], alpha=0.8,
                         colors=['#4caf50', '#ff9800', '#9c27b0'])
            ax.set_ylabel("Population Count")
            ax.set_xlabel("Day")
            ax.legend(loc='upper left')
            st.pyplot(fig)
            
            st.info("**Tip:** If 'Undercutters' dominate, it means the market favored high volume/low margin. If 'Premium' wins, quality/scarcity ruled.")

        with tab2:
            st.subheader("🌊 Workforce Trends")
            w_df = st.session_state.work_df.set_index("Day")
            active_skills = w_df.columns[(w_df.sum() > 0)].tolist()
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for skill in active_skills[:5]: # Top 5 active
                ax2.plot(w_df.index, w_df[skill], label=skill, linewidth=2)
            ax2.set_ylabel("Workers")
            ax2.legend()
            st.pyplot(fig2)

        with tab3:
            st.subheader("🏥 Population Health")
            macro = st.session_state.macro_df.set_index("Day")
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(macro.index, macro['Population'], color='blue', linewidth=2, label="Population")
            ax3.set_ylabel("Live Agents")
            ax3.legend()
            st.pyplot(fig3)
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize simulation to start the evolutionary clock.")