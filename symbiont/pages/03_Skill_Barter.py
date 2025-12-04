import streamlit as st
import random
import uuid
import graphviz
import time

# --- Neural Network Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

# --- Config ---
st.set_page_config(page_title="Symbiont: Neural Skill Barter", page_icon="🧠", layout="wide")

# --- Extended "Fuzzy" Catalog for Semantic Matching ---
# The AI will be able to match "Coding" with "Python", or "Baking" with "Cooking"
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
        # Load a lightweight, efficient transformer model
        # This downloads ~20MB on first run
        if HAS_NEURAL:
            with st.spinner("Loading Neural Weights (all-MiniLM-L6-v2)..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_user(self, user):
        self.users[user['id']] = user

    def find_semantic_swaps(self, threshold=0.65):
        """
        Uses Vector Embeddings to find 'Good Enough' matches.
        Threshold 0.65 means 'Semantically Similar'.
        """
        if not HAS_NEURAL:
            return []

        matches = []
        users_list = list(self.users.values())
        processed_pairs = set()

        # 1. Pre-compute embeddings for all skills to speed up search
        # In production, these vectors would be stored in a Vector DB (Pinecone/Milvus)
        all_offers = [u['offering'] for u in users_list]
        all_needs = [u['needing'] for u in users_list]
        
        offer_vectors = self.model.encode(all_offers, convert_to_tensor=True)
        need_vectors = self.model.encode(all_needs, convert_to_tensor=True)

        # 2. Find Double Coincidence of Wants using Cosine Similarity
        for i in range(len(users_list)):
            user_a = users_list[i]
            
            for j in range(i + 1, len(users_list)):
                user_b = users_list[j]
                
                # Check A's Offer vs B's Need
                sim_a_to_b = util.cos_sim(offer_vectors[i], need_vectors[j]).item()
                
                # Check B's Offer vs A's Need
                sim_b_to_a = util.cos_sim(offer_vectors[j], need_vectors[i]).item()

                # If both are high enough matches
                if sim_a_to_b > threshold and sim_b_to_a > threshold:
                    
                    avg_similarity = (sim_a_to_b + sim_b_to_a) / 2.0
                    
                    matches.append({
                        "user_a": user_a,
                        "user_b": user_b,
                        "skill_a_to_b": user_a['offering'],
                        "skill_b_to_a": user_b['offering'],
                        "similarity_score": avg_similarity,
                        "match_reason": f"AI detected: '{user_a['offering']}' ≈ '{user_b['needing']}'"
                    })

        # Sort by best semantic match
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    
    # If we don't have the library, we can't run
    if not HAS_NEURAL:
        return matcher

    for i in range(n):
        offer = random.choice(FUZZY_SKILL_CATALOG)
        need = random.choice([s for s in FUZZY_SKILL_CATALOG if s != offer])
        
        user = {
            "id": str(uuid.uuid4())[:8],
            "name": f"User_{random.randint(100,999)}",
            "offering": offer,
            "needing": need,
            "reputation": round(random.uniform(3.5, 5.0), 1)
        }
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🧠 Neural Skill Barter")
st.markdown("### Level 2: Semantic Matching Engine")
st.caption("Using Transformer Embeddings to find 'conceptually similar' skills (e.g., Baking ≈ Cooking).")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers` to enable the Neural Brain.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Community Pulse")
    num_users = st.slider("Active Users", 10, 50, 20)
    match_threshold = st.slider("AI Similarity Threshold", 0.4, 0.9, 0.60, help="Lower = looser matches, Higher = stricter.")
    
    if st.button("Scan with Neural Net"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        # Run Semantic Search
        start_time = time.time()
        st.session_state.neural_matches = st.session_state.neural_matcher.find_semantic_swaps(threshold=match_threshold)
        duration = time.time() - start_time
        
        st.success(f"Scanned {num_users} profiles in {duration:.2f}s")
        st.info(f"Found {len(st.session_state.neural_matches)} Semantic Swaps!")

    if 'neural_matches' in st.session_state and st.session_state.neural_matches:
        st.subheader("2. Swap Feed")
        for m in st.session_state.neural_matches[:5]:
            score = int(m['similarity_score'] * 100)
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px;">
                <b>🤖 Match Confidence: {score}%</b><br>
                👤 {m['user_a']['name']} offers <b>{m['skill_a_to_b']}</b><br>
                ⬇️ <i>(Matches need for "{m['user_b']['needing']}")</i><br>
                👤 {m['user_b']['name']} offers <b>{m['skill_b_to_a']}</b><br>
                ⬇️ <i>(Matches need for "{m['user_a']['needing']}")</i>
            </div>
            """, unsafe_allow_html=True)

# --- Visualization ---
with col2:
    if 'neural_matches' in st.session_state and st.session_state.neural_matches:
        st.subheader("🔗 Semantic Graph")
        
        # Create Graphviz object
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', bgcolor='transparent')
        
        # Add Nodes and Edges for Matches
        for m in st.session_state.neural_matches:
            # Nodes
            color_a = "#e1f5fe"
            color_b = "#fff9c4"
            
            graph.node(m['user_a']['id'], label=f"{m['user_a']['name']}\nHas: {m['user_a']['offering']}\nWants: {m['user_a']['needing']}", shape="box", style="filled", fillcolor=color_a)
            graph.node(m['user_b']['id'], label=f"{m['user_b']['name']}\nHas: {m['user_b']['offering']}\nWants: {m['user_b']['needing']}", shape="box", style="filled", fillcolor=color_b)
            
            # Edges (The Swap) - Thickness based on confidence
            penwidth = str(max(1, m['similarity_score'] * 3))
            graph.edge(m['user_a']['id'], m['user_b']['id'], label=f"{int(m['similarity_score']*100)}% Match", color="purple", penwidth=penwidth)
            
        st.graphviz_chart(graph)
        
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Generate a community to spin up the Neural Network.")
    else:
        st.warning("No semantic swaps found. Try lowering the threshold or increasing users.")