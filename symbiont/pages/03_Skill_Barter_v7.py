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
st.set_page_config(page_title="Symbiont: Behavioral Economy", page_icon="🏛️", layout="wide")

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

    def calculate_gini(self, wealths):
        """Calculates Gini Coefficient (0=Perfect Equality, 1=Perfect Inequality)"""
        if not wealths or sum(wealths) == 0: return 0.0
        sorted_wealths = sorted(wealths)
        n = len(wealths)
        # Gini formula
        numer = 2 * sum((i + 1) * w for i, w in enumerate(sorted_wealths))
        denom = n * sum(sorted_wealths)
        return (numer / denom) - (n + 1) / n

    def simulate_market_epochs(self, epochs=20, threshold=0.65, stimulus_amount=0):
        """
        Level 7: Behavioral Simulation with heterogeneous agent strategies and macro-metrics.
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame(), pd.DataFrame()

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame(), pd.DataFrame()

        # Apply Stimulus (Central Bank Injection)
        if stimulus_amount > 0:
            for u in users_list:
                u['credits'] += stimulus_amount

        # 1. Initialize Vectors
        offer_vecs = self.model.encode([u['offering'] for u in users_list], convert_to_tensor=True)
        need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)
        sim_matrix = util.cos_sim(need_vecs, offer_vecs)

        # 2. Initialize Prices
        current_prices = {u['id']: 15.0 for u in users_list} # Slightly higher starting price
        
        # Track history
        price_history = [] 
        macro_history = []
        transaction_log = []

        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            day_txs = 0
            day_volume = 0
            
            snapshot = {"Day": day}
            skill_prices = {} 
            
            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}

            # Randomize buyer order to prevent "First Mover" bias
            buyer_indices = list(range(len(users_list)))
            random.shuffle(buyer_indices)

            # Attempt Trades
            for i in buyer_indices:
                buyer = users_list[i]
                if not buyers_available[buyer['id']]: continue

                best_match = 0
                best_seller_idx = -1
                
                # Buyer Strategy: Looks for best value (Match Quality / Price)
                for j, seller in enumerate(users_list):
                    if i == j: continue
                    if not sellers_available[seller['id']]: continue
                    
                    score = sim_matrix[i][j].item()
                    if score > threshold:
                        price = current_prices[seller['id']]
                        # Value Heuristic: High Score + Low Price = Good Deal
                        # Also check if seller reputation meets buyer standard
                        if buyer['credits'] >= price:
                            # Weighted score favoring match quality slightly over price
                            value_score = (score * 100) / (price + 1)
                            
                            if value_score > best_match:
                                best_match = value_score
                                best_seller_idx = j

                # Execute Trade
                if best_seller_idx != -1:
                    seller = users_list[best_seller_idx]
                    price = current_prices[seller['id']]
                    
                    buyer['credits'] -= price
                    seller['credits'] += price
                    
                    buyers_available[buyer['id']] = False
                    sellers_available[seller['id']] = False
                    
                    # --- Behavioral Pricing Logic (Supply Side) ---
                    strategy = seller['strategy']
                    
                    if strategy == "Undercutter": 
                        # Raises price slowly (volume focused)
                        current_prices[seller['id']] *= 1.05 
                    elif strategy == "Premium": 
                        # Raises price aggressively (luxury focused)
                        current_prices[seller['id']] *= 1.20 
                    else: # Balanced
                        current_prices[seller['id']] *= 1.10

                    transaction_log.append({
                        "Day": day,
                        "Buyer": buyer['name'],
                        "Seller": seller['name'],
                        "Skill": seller['offering'],
                        "Price": price
                    })
                    day_txs += 1
                    day_volume += price
                
            # End of Day Adjustments (Unsold Inventory)
            current_wealths = []
            for u in users_list:
                current_wealths.append(u['credits'])
                
                if sellers_available[u['id']]: # Failed to sell
                    strategy = u['strategy']
                    if strategy == "Undercutter":
                        # Slashes prices to clear inventory
                        current_prices[u['id']] *= 0.85 
                    elif strategy == "Premium":
                        # Refuses to discount heavily
                        current_prices[u['id']] *= 0.98 
                    else: # Balanced
                        current_prices[u['id']] *= 0.95

                # Clamp prices
                current_prices[u['id']] = max(2.0, min(150.0, current_prices[u['id']]))
                
                skill = u['offering']
                if skill not in skill_prices: skill_prices[skill] = []
                skill_prices[skill].append(current_prices[u['id']])

            # Macro Metrics
            gini = self.calculate_gini(current_wealths)
            macro_history.append({
                "Day": day,
                "Gini (Inequality)": gini,
                "Trade Volume": day_volume,
                "Transaction Count": day_txs
            })

            for skill, p_list in skill_prices.items():
                snapshot[skill] = sum(p_list) / len(p_list)
            
            price_history.append(snapshot)
            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(price_history), pd.DataFrame(macro_history)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

    # Strategy Distribution
    strategies = ["Balanced", "Undercutter", "Premium"]
    weights = [0.6, 0.3, 0.1] # Most people are balanced, some cheap, few premium

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
            "reputation": round(random.uniform(3.5, 5.0), 1),
            "strategy": strat
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🏛️ Symbiont: Behavioral Macro-Economy")
st.markdown("### Level 7: Intelligent Agents & Inequality Tracking")
st.caption("Agents now have personalities (Premium vs. Undercutter). The System tracks Gini Coefficient (Wealth Inequality) and Trade Velocity.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas matplotlib`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Central Bank Controls")
    num_users = st.slider("Population Size", 10, 80, 40)
    epochs = st.slider("Simulation Duration (Days)", 10, 60, 30)
    match_threshold = st.slider("Match Threshold", 0.5, 0.9, 0.65)
    
    st.markdown("---")
    st.markdown("**Fiscal Policy**")
    stimulus = st.number_input("Inject UBI Stimulus (Credits)", 0, 500, 0, step=50, help="Give everyone free money at start.")
    
    if st.button("Run Simulation"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating Economy ({epochs} Epochs)..."):
            final_prices, tx_log, history_df, macro_df = st.session_state.neural_matcher.simulate_market_epochs(epochs, match_threshold, stimulus)
            
            st.session_state.final_prices = final_prices
            st.session_state.tx_log = tx_log
            st.session_state.history_df = history_df
            st.session_state.macro_df = macro_df
            
        st.success("Simulation Complete")

    # --- Ticker ---
    if 'final_prices' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Market Report")
        
        # Calculate Economy Stats
        total_vol = st.session_state.macro_df['Trade Volume'].sum()
        avg_gini = st.session_state.macro_df['Gini (Inequality)'].mean()
        
        col_a, col_b = st.columns(2)
        col_a.metric("Total GDP", f"{int(total_vol)} cr")
        col_b.metric("Avg Inequality", f"{avg_gini:.2f}", help="0=Equal, 1=Unequal")

# --- Visualization ---
with col2:
    if 'history_df' in st.session_state and not st.session_state.history_df.empty:
        # Tabbed View for Analysis
        tab1, tab2, tab3 = st.tabs(["Price History", "Wealth Inequality", "Ledger"])
        
        with tab1:
            st.subheader("🏷️ Price Discovery")
            df = st.session_state.history_df.set_index("Day")
            variances = df.var().sort_values(ascending=False)
            top_skills = variances.head(7).index.tolist()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for skill in top_skills:
                ax.plot(df.index, df[skill], label=skill)
            ax.set_xlabel("Day")
            ax.set_ylabel("Price (Credits)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab2:
            st.subheader("⚖️ Macro-Economic Health")
            macro = st.session_state.macro_df.set_index("Day")
            
            # Dual Axis Plot
            fig2, ax1 = plt.subplots(figsize=(10, 5))
            
            color = 'tab:red'
            ax1.set_xlabel('Day')
            ax1.set_ylabel('Gini Coefficient (Inequality)', color=color)
            ax1.plot(macro.index, macro['Gini (Inequality)'], color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(0, 1) # Gini is always 0-1

            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('Daily Trade Volume', color=color)  
            ax2.bar(macro.index, macro['Trade Volume'], color=color, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title("Wealth Inequality vs. Market Activity")
            st.pyplot(fig2)
            
            st.info("""
            **Analysis:**
            - If **Gini (Red Line)** is rising, wealth is concentrating in a few "Super Sellers".
            - A **UBI Stimulus** often lowers Gini initially but may cause inflation (Price spikes in Tab 1).
            """)

        with tab3:
            st.subheader("📜 Transaction Log")
            recent = st.session_state.tx_log
            if recent:
                st.dataframe(pd.DataFrame(recent).sort_values(by="Day", ascending=False))
            else:
                st.write("No transactions occurred.")
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize simulation to see the behavioral economy in action.")