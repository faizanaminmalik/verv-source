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
st.set_page_config(page_title="Symbiont: Sovereign AI", page_icon="🏛️", layout="wide")

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

    def assign_guild(self, skill):
        """Level 10: Assigns agents to Guilds based on their skill type."""
        if skill in ["Python Programming", "Web Development", "React JS", "Data Science", "Financial Planning", "Tax Help", "Investment Advice"]:
            return "Technocrat Guild"
        elif skill in ["Graphic Design", "Logo Creation", "Digital Art", "Guitar Lessons", "Piano Tutoring", "Music Theory"]:
            return "Creative Union"
        elif skill in ["Sourdough Baking", "Gourmet Cooking", "Meal Prepping", "Conversational Spanish", "French for Beginners", "Mandarin"]:
            return "Artisan Collective"
        else:
            return "Service Alliance" # Wellness, etc.

    def simulate_market_epochs(self, epochs=20, threshold=0.65, initial_stimulus=0, learning_rate=0.1, shock_chance=0.05, enable_gov=True):
        """
        Level 10: Sovereign AI with Automated Governance and Guild Treasuries.
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        # Guild Setup
        guilds = {
            "Technocrat Guild": {"funds": 500, "bailouts": 0},
            "Creative Union": {"funds": 500, "bailouts": 0},
            "Artisan Collective": {"funds": 500, "bailouts": 0},
            "Service Alliance": {"funds": 500, "bailouts": 0}
        }

        # Apply Initial Stimulus
        if initial_stimulus > 0:
            for u in users_list: u['credits'] += initial_stimulus

        # Vectors
        current_needs = [u['needing'] for u in users_list]
        current_prices = {u['id']: 20.0 for u in users_list}
        
        # History
        price_history = [] 
        macro_history = []
        workforce_history = [] 
        strategy_history = []
        transaction_log = []
        gov_log = []
        
        # Governance State
        tax_rate = 0.05 # Start low
        ubi_amount = 2  # Daily basic income
        
        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            # --- GOVERNOR AI LOGIC ---
            if enable_gov and day > 1:
                # Analyze previous day
                prev_gini = macro_history[-1]["Gini"]
                prev_pop = macro_history[-1]["Population"]
                start_pop = len(self.users) # Approximation
                
                action = None
                
                # Rule 1: High Inequality -> Redistribute
                if prev_gini > 0.35:
                    tax_rate = min(0.50, tax_rate + 0.02)
                    ubi_amount += 3
                    action = f"Inequality Alert ({prev_gini:.2f}). Raising Taxes to {int(tax_rate*100)}% & UBI to {ubi_amount}."
                
                # Rule 2: Population Crash -> Stimulate
                elif prev_pop < (macro_history[0]["Population"] * 0.8):
                    tax_rate = max(0.01, tax_rate - 0.05)
                    ubi_amount += 10
                    action = f"Recession Alert. Slashing Taxes to {int(tax_rate*100)}% & Injecting Liquidity."
                
                # Rule 3: Stability -> Moderate
                elif prev_gini < 0.25:
                    tax_rate = max(0.05, tax_rate - 0.01)
                    if random.random() < 0.2:
                        action = "Economy Stable. Lowering tax burden."

                if action: gov_log.append({"Day": day, "Action": action})

            # Apply UBI (From Governor)
            if ubi_amount > 0:
                for u in users_list: u['credits'] += ubi_amount

            # Market Shocks
            if random.random() < shock_chance:
                shock_skill = random.choice(FUZZY_SKILL_CATALOG)
                for u in random.sample(users_list, k=int(len(users_list)*0.2)):
                    u['needing'] = shock_skill
                transaction_log.append({"Day": day, "Buyer": "MARKET", "Seller": "SHOCK", "Skill": shock_skill, "Price": 0})
            
            # Re-encode needs
            need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)

            day_txs = 0
            day_volume = 0
            
            # Supply Vectors
            current_offers = [u['offering'] for u in users_list]
            offer_indices = [self.skill_to_idx[off] for off in current_offers]
            current_offer_vecs = self.catalog_vecs[offer_indices]
            
            sim_matrix = util.cos_sim(need_vecs, current_offer_vecs)

            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}
            buyer_indices = list(range(len(users_list)))
            random.shuffle(buyer_indices)

            # --- TRADING LOOP ---
            for i in buyer_indices:
                if i >= len(users_list): continue
                buyer = users_list[i]
                if not buyers_available.get(buyer['id'], False): continue

                best_match = 0
                best_seller_idx = -1
                
                for j, seller in enumerate(users_list):
                    if buyer['id'] == seller['id']: continue
                    if not sellers_available.get(seller['id'], False): continue
                    
                    score = sim_matrix[i][j].item()
                    if score > threshold:
                        price = current_prices.get(seller['id'], 20.0)
                        if buyer['credits'] >= price:
                            value_score = (score * 100) / (price + 1)
                            if value_score > best_match:
                                best_match = value_score
                                best_seller_idx = j

                if best_seller_idx != -1:
                    seller = users_list[best_seller_idx]
                    price = current_prices[seller['id']]
                    skill_sold = seller['offering']
                    
                    # TAXATION
                    tax = price * tax_rate
                    net_income = price - tax
                    
                    # Guild Collection
                    guild_name = self.assign_guild(seller['offering'])
                    guilds[guild_name]['funds'] += tax
                    
                    # Transfer
                    buyer['credits'] -= price
                    seller['credits'] += net_income
                    
                    buyers_available[buyer['id']] = False
                    sellers_available[seller['id']] = False
                    
                    # Learning
                    if skill_sold not in buyer['xp']: buyer['xp'][skill_sold] = 0
                    buyer['xp'][skill_sold] += learning_rate
                    if buyer['xp'][skill_sold] >= 1.0 and skill_sold not in buyer['known_skills']:
                        buyer['known_skills'].append(skill_sold)
                    
                    # Strategy Adjustments
                    if seller['strategy'] == "Premium": current_prices[seller['id']] *= 1.15
                    elif seller['strategy'] == "Undercutter": current_prices[seller['id']] *= 1.02
                    else: current_prices[seller['id']] *= 1.10
                    
                    transaction_log.append({
                        "Day": day, "Buyer": buyer['name'], "Seller": seller['name'],
                        "Skill": skill_sold, "Price": price, "Tax": tax
                    })
                    day_txs += 1
                    day_volume += price
            
            # --- END OF DAY ---
            
            survivors = []
            new_babies = []
            
            for u in users_list:
                # Inventory Decay
                if sellers_available.get(u['id']):
                    decay = 0.85 if u['strategy'] == "Undercutter" else 0.95
                    current_prices[u['id']] *= decay
                current_prices[u['id']] = max(2.0, min(200.0, current_prices[u['id']]))

                # Living Cost
                u['credits'] -= 2
                
                # --- GUILD SAFETY NET (Bailout) ---
                if u['credits'] < 0:
                    g_name = self.assign_guild(u['offering'])
                    deficit = abs(u['credits']) + 5 # Give them a small buffer
                    if guilds[g_name]['funds'] >= deficit:
                        guilds[g_name]['funds'] -= deficit
                        guilds[g_name]['bailouts'] += 1
                        u['credits'] += deficit # Saved!

                # Death Check
                if u['credits'] > 0:
                    survivors.append(u)
                    
                    # Reproduction
                    if u['credits'] > 250:
                        u['credits'] -= 100
                        baby = copy.deepcopy(u)
                        baby['id'] = str(uuid.uuid4())[:8]
                        baby['name'] = f"Gen2_{u['name']}"
                        baby['credits'] = 50
                        baby['reputation'] = 3.0
                        if random.random() < 0.1:
                            baby['strategy'] = random.choice(["Balanced", "Undercutter", "Premium"])
                        new_babies.append(baby)
                        current_prices[baby['id']] = 15.0
                else:
                    pass # Dead
            
            users_list = survivors + new_babies
            self.users = {u['id']: u for u in users_list}
            
            # Career Switching
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

            # Data Collection
            current_wealths = [u['credits'] for u in users_list]
            gini = self.calculate_gini(current_wealths)
            
            # Store Guild Data as string representation for simplicity in DF
            guild_status = {k: v['funds'] for k,v in guilds.items()}

            macro_history.append({
                "Day": day, "Gini": gini, "Volume": day_volume, 
                "Population": len(users_list), "Tax_Rate": tax_rate, "UBI": ubi_amount,
                **guild_status
            })

            snapshot_strat = {"Day": day, "Balanced": 0, "Undercutter": 0, "Premium": 0}
            for u in users_list: snapshot_strat[u['strategy']] += 1
            strategy_history.append(snapshot_strat)
            
            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(price_history), pd.DataFrame(macro_history), pd.DataFrame(workforce_history), pd.DataFrame(strategy_history), gov_log

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
st.title("🏛️ Symbiont: The Sovereign AI")
st.markdown("### Level 10: Automated Governance & Guilds")
st.caption("A Cybernetic Society. The 'Governor AI' regulates Taxes/UBI to prevent collapse. Agents form Guilds to bail each other out.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas matplotlib`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Simulation Config")
    num_users = st.slider("Initial Population", 20, 100, 50)
    epochs = st.slider("Simulation Days", 20, 100, 45)
    enable_gov = st.checkbox("Enable AI Governor", value=True)
    
    if st.button("Initialize Sovereign State"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating Cybernetic Economy..."):
            final_prices, tx_log, history_df, macro_df, work_df, strat_df, gov_log = st.session_state.neural_matcher.simulate_market_epochs(epochs, 0.65, 0, 0.1, 0.05, enable_gov)
            
            st.session_state.final_prices = final_prices
            st.session_state.tx_log = tx_log
            st.session_state.history_df = history_df
            st.session_state.macro_df = macro_df
            st.session_state.work_df = work_df
            st.session_state.strat_df = strat_df
            st.session_state.gov_log = gov_log
            
        st.success("Simulation Complete")

    # --- Ticker ---
    if 'macro_df' in st.session_state and not st.session_state.macro_df.empty:
        st.markdown("---")
        st.subheader("💀 Civilization Score")
        last = st.session_state.macro_df.iloc[-1]
        
        # Calculate a "Civ Score"
        survival_rate = last['Population'] / num_users
        equality_score = 1 - last['Gini']
        civ_score = (survival_rate * 50) + (equality_score * 50)
        
        col_a, col_b = st.columns(2)
        col_a.metric("Civ Score", f"{int(civ_score)}/100")
        col_b.metric("Survival Rate", f"{int(survival_rate*100)}%")
        
        st.metric("Final Tax Rate", f"{int(last['Tax_Rate']*100)}%")

# --- Visualization ---
with col2:
    if 'history_df' in st.session_state and not st.session_state.history_df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Governor Log", "💰 Guild Treasuries", "🧬 Strategy War", "📈 Macro Health"])
        
        with tab1:
            st.subheader("AI Policy Decisions")
            if st.session_state.gov_log:
                for entry in st.session_state.gov_log:
                    st.write(f"**Day {entry['Day']}**: {entry['Action']}")
            else:
                st.write("Governor remained silent (Economy stable or Disabled).")

        with tab2:
            st.subheader("🛡️ Guild Safety Nets")
            st.caption("Funds accumulated by Guilds to bail out bankrupt members.")
            
            m_df = st.session_state.macro_df.set_index("Day")
            guild_cols = ["Technocrat Guild", "Creative Union", "Artisan Collective", "Service Alliance"]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for g in guild_cols:
                ax.plot(m_df.index, m_df[g], label=g, linewidth=2)
            ax.set_ylabel("Treasury (Credits)")
            ax.legend()
            st.pyplot(fig)

        with tab3:
            st.subheader("⚔️ Strategy Dominance")
            s_df = st.session_state.strat_df.set_index("Day")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stackplot(s_df.index, s_df["Balanced"], s_df["Undercutter"], s_df["Premium"], 
                         labels=["Balanced", "Undercutter", "Premium"], alpha=0.8,
                         colors=['#4caf50', '#ff9800', '#9c27b0'])
            ax2.set_ylabel("Population")
            ax2.legend(loc='upper left')
            st.pyplot(fig2)

        with tab4:
            st.subheader("🏥 Vital Signs")
            macro = st.session_state.macro_df.set_index("Day")
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(macro.index, macro['Gini'], color='red', label="Gini (Inequality)")
            ax3.set_ylim(0, 1)
            ax3.legend(loc='upper left')
            
            ax4 = ax3.twinx()
            ax4.plot(macro.index, macro['Population'], color='blue', label="Population", linestyle='--')
            ax4.legend(loc='upper right')
            
            st.pyplot(fig3)
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize the Sovereign State.")