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
st.set_page_config(page_title="Symbiont: Civilization AI", page_icon="🚀", layout="wide")

# --- Level 11: The Tech Tree ---
TECH_TREE = {
    "Agrarian Age": [
        "Subsistence Farming", "Pottery", "Carpentry", "Blacksmithing", 
        "Herbalism", "Weaving", "Stone Masonry", "Animal Husbandry"
    ],
    "Industrial Age": [
        "Steam Mechanics", "Steel Working", "Railroad Logistics", "Telegraphy", 
        "Factory Management", "Chemical Engineering", "Textile Manufacturing"
    ],
    "Information Age": [
        "Python Programming", "React JS", "Data Science", "Digital Marketing", 
        "Cybersecurity", "UX Design", "Cloud Computing"
    ],
    "Cybernetic Age": [
        "AGI Alignment", "Neural Linking", "Mars Terraforming", "Gene Editing", 
        "Consciousness Uploading", "Quantum Cryptography", "Nanobot Repair"
    ]
}

ERA_ORDER = ["Agrarian Age", "Industrial Age", "Information Age", "Cybernetic Age"]

class NeuralSkillMatcher:
    def __init__(self):
        self.users = {}
        self.active_skills = [] # Dynamic list based on current Era
        self.current_era_idx = 0
        
        if HAS_NEURAL:
            with st.spinner("Loading Neural Brain..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.update_vectors()

    def update_vectors(self):
        """Re-computes vectors based on currently unlocked eras."""
        # Active skills include all previous eras up to current
        self.active_skills = []
        for i in range(self.current_era_idx + 1):
            self.active_skills.extend(TECH_TREE[ERA_ORDER[i]])
            
        if HAS_NEURAL:
            self.catalog_vecs = self.model.encode(self.active_skills, convert_to_tensor=True)
            self.skill_to_idx = {skill: i for i, skill in enumerate(self.active_skills)}

    def add_user(self, user):
        self.users[user['id']] = user

    def calculate_gini(self, wealths):
        if not wealths or sum(wealths) == 0: return 0.0
        sorted_wealths = sorted(wealths)
        n = len(wealths)
        numer = 2 * sum((i + 1) * w for i, w in enumerate(sorted_wealths))
        denom = n * sum(sorted_wealths)
        return (numer / denom) - (n + 1) / n

    def simulate_market_epochs(self, epochs=20, threshold=0.65, initial_stimulus=0, learning_rate=0.1, shock_chance=0.05, enable_gov=True):
        """
        Level 11: Civilization Simulation with Tech Tree and Era Progression.
        """
        if not HAS_NEURAL: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        users_list = list(self.users.values())
        if not users_list: return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        # Init Economy
        if initial_stimulus > 0:
            for u in users_list: u['credits'] += initial_stimulus

        current_prices = {u['id']: 20.0 for u in users_list}
        
        # History Logs
        price_history = [] 
        macro_history = []
        workforce_history = [] 
        strategy_history = []
        transaction_log = []
        gov_log = []
        
        # Civ State
        tax_rate = 0.05 
        ubi_amount = 2
        innovation_points = 0
        innovation_threshold = 50 # Points needed to unlock next Era (Scales up)
        
        progress_bar = st.progress(0)
        
        for day in range(1, epochs + 1):
            
            # --- ERA PROGRESSION LOGIC ---
            if innovation_points >= innovation_threshold and self.current_era_idx < len(ERA_ORDER) - 1:
                # Level Up!
                self.current_era_idx += 1
                new_era_name = ERA_ORDER[self.current_era_idx]
                
                # Update Brain
                self.update_vectors()
                
                # Reset Innovation for next level (harder)
                innovation_points = 0
                innovation_threshold *= 1.5 
                
                # Log Event
                gov_log.append({"Day": day, "Action": f"🎉 SCIENTIFIC BREAKTHROUGH! Welcome to the {new_era_name}."})
                
                # Early Adopters: Random 10% of pop learns new tech immediately
                new_techs = TECH_TREE[new_era_name]
                for u in random.sample(users_list, k=int(len(users_list)*0.1)):
                    u['offering'] = random.choice(new_techs)
                    current_prices[u['id']] = 50.0 # High value for new tech

            # --- GOVERNOR AI ---
            if enable_gov and day > 1:
                prev_gini = macro_history[-1]["Gini"] if macro_history else 0
                prev_pop = macro_history[-1]["Population"] if macro_history else len(users_list)
                
                action = None
                if prev_gini > 0.40:
                    tax_rate = min(0.60, tax_rate + 0.03)
                    ubi_amount += 5
                    action = f"Inequality Alert ({prev_gini:.2f}). Tax->{int(tax_rate*100)}% UBI->{ubi_amount}."
                elif prev_pop < (len(users_list) * 0.8): # Crisis
                    tax_rate = max(0.01, tax_rate - 0.1)
                    ubi_amount += 15
                    action = "Population Collapse Imminent. Emergency Stimulus."
                
                if action: gov_log.append({"Day": day, "Action": action})

            if ubi_amount > 0:
                for u in users_list: u['credits'] += ubi_amount

            # --- TRADING ---
            # Re-encode needs daily (as needs might shift to new eras)
            need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)
            
            # Supply Vectors
            current_offers = [u['offering'] for u in users_list]
            
            # Handle users offering old skills not in current vector space?
            # self.update_vectors() ensures all previous eras are included, so we are safe.
            offer_indices = [self.skill_to_idx[off] for off in current_offers]
            current_offer_vecs = self.catalog_vecs[offer_indices]
            
            sim_matrix = util.cos_sim(need_vecs, current_offer_vecs)

            sellers_available = {u['id']: True for u in users_list}
            buyers_available = {u['id']: True for u in users_list}
            buyer_indices = list(range(len(users_list)))
            random.shuffle(buyer_indices)

            day_volume = 0
            day_txs = 0

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
                            # Bias towards new era skills? Maybe implicit in demand.
                            if value_score > best_match:
                                best_match = value_score
                                best_seller_idx = j

                if best_seller_idx != -1:
                    seller = users_list[best_seller_idx]
                    price = current_prices[seller['id']]
                    
                    # Taxes
                    tax = price * tax_rate
                    net = price - tax
                    
                    buyer['credits'] -= price
                    seller['credits'] += net
                    buyers_available[buyer['id']] = False
                    sellers_available[seller['id']] = False
                    
                    # Strategy logic
                    if seller['strategy'] == "Premium": current_prices[seller['id']] *= 1.15
                    elif seller['strategy'] == "Undercutter": current_prices[seller['id']] *= 1.02
                    else: current_prices[seller['id']] *= 1.10
                    
                    # --- INNOVATION MECHANIC ---
                    # Trades generate research. New Era trades generate MORE research.
                    skill_era = self.get_skill_era(seller['offering'])
                    era_bonus = (ERA_ORDER.index(skill_era) + 1) 
                    innovation_points += (1 * era_bonus)

                    # --- LEARNING & OBSOLESCENCE ---
                    # Buyers update needs to match current Era
                    if random.random() < 0.1: # 10% chance to upgrade need
                        current_era_skills = TECH_TREE[ERA_ORDER[self.current_era_idx]]
                        buyer['needing'] = random.choice(current_era_skills)

                    transaction_log.append({
                        "Day": day, "Buyer": buyer['name'], "Seller": seller['name'],
                        "Skill": seller['offering'], "Price": price, "Era": skill_era
                    })
                    day_volume += price
                    day_txs += 1
            
            # --- END OF DAY ---
            
            survivors = []
            new_babies = []
            
            for u in users_list:
                # Inventory Decay (Obsolescence Check)
                skill_era = self.get_skill_era(u['offering'])
                
                # If skill is from an old era, decay price faster
                is_obsolete = ERA_ORDER.index(skill_era) < self.current_era_idx
                decay = 0.80 if is_obsolete else 0.95
                
                if sellers_available.get(u['id']):
                    current_prices[u['id']] *= decay
                
                u['credits'] -= 2 # Cost of Living
                
                if u['credits'] > 0:
                    survivors.append(u)
                    if u['credits'] > 250:
                        u['credits'] -= 100
                        baby = copy.deepcopy(u)
                        baby['id'] = str(uuid.uuid4())[:8]
                        baby['name'] = f"Gen_{day}_{u['name'][:3]}"
                        baby['credits'] = 50
                        # Baby starts with modern skills
                        current_era_skills = TECH_TREE[ERA_ORDER[self.current_era_idx]]
                        baby['offering'] = random.choice(current_era_skills)
                        current_prices[baby['id']] = 25.0
                        new_babies.append(baby)
            
            users_list = survivors + new_babies
            self.users = {u['id']: u for u in users_list}
            
            # Data Collection
            current_wealths = [u['credits'] for u in users_list]
            gini = self.calculate_gini(current_wealths)
            
            macro_history.append({
                "Day": day, "Gini": gini, "Volume": day_volume, 
                "Population": len(users_list), "Innovation": innovation_points,
                "Current_Era": ERA_ORDER[self.current_era_idx]
            })

            progress_bar.progress(day / epochs)

        return current_prices, transaction_log, pd.DataFrame(), pd.DataFrame(macro_history), pd.DataFrame(), pd.DataFrame(), gov_log

    def get_skill_era(self, skill):
        for era, skills in TECH_TREE.items():
            if skill in skills: return era
        return "Unknown"

# --- Helper to Generate Fake Users ---
def generate_community(n=20, start_era_idx=0):
    matcher = NeuralSkillMatcher()
    matcher.current_era_idx = start_era_idx
    matcher.update_vectors() # Ensure brain is ready
    
    if not HAS_NEURAL: return matcher

    strategies = ["Balanced", "Undercutter", "Premium"]
    
    # Start with Agrarian Skills
    start_skills = TECH_TREE[ERA_ORDER[start_era_idx]]

    for i in range(n):
        offer = random.choice(start_skills)
        need = random.choice(start_skills)
        
        user = {
            "id": str(uuid.uuid4())[:8],
            "name": f"Settler_{random.randint(100,999)}",
            "offering": offer,
            "needing": need,
            "credits": random.randint(60, 100),
            "strategy": random.choice(strategies),
            "known_skills": [offer],
            "xp": {offer: 1.0}
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🚀 Symbiont: Civilization AI")
st.markdown("### Level 11: The Tech Tree & Eras")
st.caption("Guide your society from the **Agrarian Age** to the **Cybernetic Age**. Innovation unlocks new Eras. Old skills become obsolete.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers pandas matplotlib`.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Civilization Config")
    num_users = st.slider("Starting Settlers", 20, 100, 50)
    epochs = st.slider("Simulation Years (Days)", 30, 150, 60)
    enable_gov = st.checkbox("Enable AI Governor", value=True)
    
    if st.button("Start Civilization"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        with st.spinner(f"Simulating History..."):
            final_prices, tx_log, _, macro_df, _, _, gov_log = st.session_state.neural_matcher.simulate_market_epochs(epochs, 0.65, 0, 0.1, 0.05, enable_gov)
            
            st.session_state.macro_df = macro_df
            st.session_state.gov_log = gov_log
            
        st.success("History Written!")

    # --- Civ State Ticker ---
    if 'macro_df' in st.session_state and not st.session_state.macro_df.empty:
        st.markdown("---")
        st.subheader("🏛️ State of the Union")
        last = st.session_state.macro_df.iloc[-1]
        
        current_era = last['Current_Era']
        st.info(f"**Current Era: {current_era}**")
        
        # Progress Bar for Era
        era_idx = ERA_ORDER.index(current_era)
        progress = (era_idx + 1) / len(ERA_ORDER)
        st.progress(progress)
        
        col_a, col_b = st.columns(2)
        col_a.metric("Population", f"{int(last['Population'])}")
        col_b.metric("Innovation Bank", f"{int(last['Innovation'])} pts")

# --- Visualization ---
with col2:
    if 'macro_df' in st.session_state and not st.session_state.macro_df.empty:
        tab1, tab2, tab3 = st.tabs(["📜 Historical Events", "📈 Economy & Tech", "🏥 Vital Signs"])
        
        with tab1:
            st.subheader("Annals of History")
            # Combine Gov Log and Tech Breakthroughs
            if st.session_state.gov_log:
                for entry in st.session_state.gov_log:
                    icon = "🎉" if "BREAKTHROUGH" in entry['Action'] else "🏛️"
                    st.write(f"**Year {entry['Day']}**: {icon} {entry['Action']}")

        with tab2:
            st.subheader("Growth & Innovation")
            macro = st.session_state.macro_df.set_index("Day")
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            color = 'tab:green'
            ax1.set_xlabel('Year')
            ax1.set_ylabel('GDP (Volume)', color=color)
            ax1.plot(macro.index, macro['Volume'], color=color, alpha=0.6)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  
            color = 'tab:purple'
            ax2.set_ylabel('Innovation Points', color=color)  
            ax2.plot(macro.index, macro['Innovation'], color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Mark Era changes
            # We detect where 'Current_Era' changes
            changes = macro[macro['Current_Era'].shift() != macro['Current_Era']]
            for day in changes.index:
                if day > 1:
                    plt.axvline(x=day, color='orange', linestyle=':', alpha=0.8)
                    plt.text(day, 0, f" {changes.loc[day]['Current_Era']}", rotation=90, fontsize=8)

            st.pyplot(fig)

        with tab3:
            st.subheader("Civilization Health")
            macro = st.session_state.macro_df.set_index("Day")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(macro.index, macro['Population'], label="Population", color="blue")
            ax3.set_ylabel("Citizens")
            st.pyplot(fig3)
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize the timeline.")