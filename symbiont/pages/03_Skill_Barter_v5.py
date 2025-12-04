import streamlit as st
import random
import uuid
import graphviz
import time
import math

# --- Neural Network Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

# --- Config ---
st.set_page_config(page_title="Symbiont: AI Time Bank", page_icon="🏦", layout="wide")

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

    def calculate_market_prices(self):
        """
        Level 5: AI determines the 'Fair Market Value' (Credits) of every skill
        based on real-time Supply vs. Demand scarcity in the community.
        """
        if not HAS_NEURAL: return {}

        users_list = list(self.users.values())
        if not users_list: return {}

        all_offers = [u['offering'] for u in users_list]
        all_needs = [u['needing'] for u in users_list]
        
        # Encode all descriptions to vectors
        offer_vecs = self.model.encode(all_offers, convert_to_tensor=True)
        need_vecs = self.model.encode(all_needs, convert_to_tensor=True)
        
        market_analysis = {}
        
        # Analyze each user's offering
        for i, user in enumerate(users_list):
            my_vec = offer_vecs[i]
            
            # 1. Calculate Supply (How many others offer similar things?)
            # We treat cosine similarity > 0.7 as a "Competitor"
            supply_scores = util.cos_sim(my_vec, offer_vecs)[0]
            supply_count = sum([1 for s in supply_scores if s > 0.7])
            
            # 2. Calculate Demand (How many people need this?)
            demand_scores = util.cos_sim(my_vec, need_vecs)[0]
            demand_count = sum([1 for s in demand_scores if s > 0.7])
            
            # 3. Pricing Algorithm (Logarithmic Scarcity)
            # Base Rate: 10 Credits/Hour
            # Formula: Price increases if Demand > Supply
            ratio = (demand_count + 1) / (supply_count + 1)
            fair_price = 10 * (1 + math.log(ratio + 0.5))
            fair_price = max(5.0, min(50.0, fair_price)) # Clamp price between 5 and 50
            
            market_analysis[user['id']] = {
                "skill": user['offering'],
                "price": round(fair_price, 2),
                "supply": supply_count,
                "demand": demand_count
            }
            
        return market_analysis

    def find_paid_transactions(self, prices, threshold=0.65):
        """
        Finds transactions where a user has enough credits to BUY a service
        without needing a barter match.
        """
        if not HAS_NEURAL: return []

        transactions = []
        users_list = list(self.users.values())
        
        # Precompute vectors
        offer_vecs = self.model.encode([u['offering'] for u in users_list], convert_to_tensor=True)
        need_vecs = self.model.encode([u['needing'] for u in users_list], convert_to_tensor=True)
        
        # Matrix: Rows=Needs, Cols=Offers
        sim_matrix = util.cos_sim(need_vecs, offer_vecs)
        
        processed_buyers = set()

        for i, buyer in enumerate(users_list):
            # Find the best seller for this buyer's need
            best_match_score = 0
            best_seller_idx = -1
            
            for j, seller in enumerate(users_list):
                if i == j: continue
                
                score = sim_matrix[i][j].item()
                if score > threshold and score > best_match_score:
                    best_match_score = score
                    best_seller_idx = j
            
            # If a valid seller exists
            if best_seller_idx != -1:
                seller = users_list[best_seller_idx]
                price = prices[seller['id']]['price']
                
                # Can Buyer afford it?
                if buyer['credits'] >= price:
                    transactions.append({
                        "buyer": buyer,
                        "seller": seller,
                        "skill_sold": seller['offering'],
                        "buyer_need": buyer['needing'],
                        "price": price,
                        "match_score": best_match_score
                    })
        
        # Sort by highest match score
        return sorted(transactions, key=lambda x: x['match_score'], reverse=True)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

    for i in range(n):
        offer = random.choice(FUZZY_SKILL_CATALOG)
        need = random.choice([s for s in FUZZY_SKILL_CATALOG if s != offer])
        
        # Level 5: Users now have a Wallet (Credits)
        # Simulating Universal Basic Income + Savings
        starting_credits = random.randint(10, 60) 
        
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
st.title("🏦 Symbiont: AI Time Bank")
st.markdown("### Level 5: The Semantic Economy")
st.caption("AI dynamically prices skills based on community Supply & Demand, enabling credit-based transactions when direct barter fails.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers` to enable the Neural Brain.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Economic Parameters")
    num_users = st.slider("Population Size", 10, 60, 30)
    match_threshold = st.slider("AI Similarity Threshold", 0.4, 0.9, 0.65)
    
    if st.button("Simulate Economy"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        # 1. AI Pricing Step
        with st.spinner("AI Calculating Market Rates..."):
            prices = st.session_state.neural_matcher.calculate_market_prices()
            st.session_state.prices = prices
        
        # 2. Transaction Step
        with st.spinner("Matching Buyers & Sellers..."):
            txs = st.session_state.neural_matcher.find_paid_transactions(prices, threshold=match_threshold)
            st.session_state.transactions = txs
            
        st.success(f"Market Open! {len(txs)} Deals Executed.")
        
    # --- Market Ticker ---
    if 'prices' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Live Market Rates")
        st.caption("AI-Calculated Prices (Credits/Session)")
        
        # Display top 5 most expensive and cheapest skills
        price_list = list(st.session_state.prices.values())
        price_list.sort(key=lambda x: x['price'], reverse=True)
        
        st.markdown("**🔥 High Demand (Expensive)**")
        for p in price_list[:3]:
            st.write(f"• **{p['skill']}**: {p['price']} cr (S:{p['supply']} D:{p['demand']})")
            
        st.markdown("**📉 High Supply (Cheap)**")
        for p in price_list[-3:]:
            st.write(f"• **{p['skill']}**: {p['price']} cr (S:{p['supply']} D:{p['demand']})")

# --- Visualization ---
with col2:
    if 'transactions' in st.session_state and st.session_state.transactions:
        st.subheader("💳 Transaction Ledger")
        
        # Graph Visualization
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', bgcolor='transparent')
        
        # Display limited transactions to avoid clutter
        display_txs = st.session_state.transactions[:12]
        
        for tx in display_txs:
            buyer = tx['buyer']
            seller = tx['seller']
            price = tx['price']
            
            # Nodes
            # Buyer (Yellow = Spending) -> Seller (Green = Earning)
            label_b = f"{buyer['name']}\nHas: {buyer['credits']} cr\nNeeds: {tx['buyer_need']}"
            label_s = f"{seller['name']}\nHas: {seller['credits']} cr\nSells: {tx['skill_sold']}"
            
            graph.node(buyer['id'], label=label_b, shape="box", style="filled", fillcolor="#fff9c4")
            graph.node(seller['id'], label=label_s, shape="box", style="filled", fillcolor="#c8e6c9")
            
            # Edge (Payment Flow)
            graph.edge(buyer['id'], seller['id'], label=f"Pays {price} cr", color="#2e7d32", fontcolor="#2e7d32")

        st.graphviz_chart(graph)
        
        # List View
        st.markdown("### 📝 Recent Activity")
        for tx in display_txs:
            st.markdown(f"""
            💸 **{tx['buyer']['name']}** paid **{tx['price']} credits** to **{tx['seller']['name']}** *For: {tx['skill_sold']} (Match: {int(tx['match_score']*100)}%)*
            """)
            
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Initialize the population to open the market.")
    else:
        st.warning("No transactions possible. Users may be too poor or needs don't match.")