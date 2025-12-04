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

    def _compute_similarity_matrix(self, users_list):
        """
        Creates a matrix of how well User i's offer satisfies User j's need.
        """
        all_offers = [u['offering'] for u in users_list]
        all_needs = [u['needing'] for u in users_list]
        
        offer_vectors = self.model.encode(all_offers, convert_to_tensor=True)
        need_vectors = self.model.encode(all_needs, convert_to_tensor=True)
        
        # Calculate cosine similarity between every offer and every need
        # matrix[i][j] = similarity(User i Offer, User j Need)
        return util.cos_sim(offer_vectors, need_vectors)

    def find_matches(self, threshold=0.65, enable_circular=True):
        """
        Finds both Direct Swaps (A <-> B) and Circular Swaps (A -> B -> C -> A).
        """
        if not HAS_NEURAL:
            return [], []

        users_list = list(self.users.values())
        n = len(users_list)
        sim_matrix = self._compute_similarity_matrix(users_list)
        
        direct_matches = []
        circular_matches = []
        matched_users = set()

        # 1. Direct Swaps (A <-> B)
        for i in range(n):
            for j in range(i + 1, n):
                if i in matched_users or j in matched_users: continue

                # Score A->B and B->A
                score_ab = sim_matrix[i][j].item()
                score_ba = sim_matrix[j][i].item()

                if score_ab > threshold and score_ba > threshold:
                    direct_matches.append({
                        "type": "Direct",
                        "users": [users_list[i], users_list[j]],
                        "skills": [users_list[i]['offering'], users_list[j]['offering']],
                        "score": (score_ab + score_ba) / 2
                    })
                    matched_users.add(i)
                    matched_users.add(j)

        if not enable_circular:
            return direct_matches, []

        # 2. Circular Swaps (A -> B -> C -> A)
        # Only look at unmatched users to find hidden value
        remaining_indices = [x for x in range(n) if x not in matched_users]
        
        for i in remaining_indices:
            for j in remaining_indices:
                if i == j: continue
                # Does A satisfy B?
                if sim_matrix[i][j].item() > threshold:
                    
                    for k in remaining_indices:
                        if k == i or k == j: continue
                        
                        # Check the triangle: A->B, B->C, C->A
                        score_ab = sim_matrix[i][j].item()
                        score_bc = sim_matrix[j][k].item()
                        score_ca = sim_matrix[k][i].item()
                        
                        if score_bc > threshold and score_ca > threshold:
                            # Verify we haven't already used these in a previous cycle
                            if i in matched_users or j in matched_users or k in matched_users:
                                continue

                            avg_score = (score_ab + score_bc + score_ca) / 3
                            circular_matches.append({
                                "type": "Circular",
                                "users": [users_list[i], users_list[j], users_list[k]],
                                "skills": [users_list[i]['offering'], users_list[j]['offering'], users_list[k]['offering']],
                                "score": avg_score
                            })
                            # Mark as matched so we don't reuse them
                            matched_users.update([i, j, k])

        return sorted(direct_matches, key=lambda x: x['score'], reverse=True), \
               sorted(circular_matches, key=lambda x: x['score'], reverse=True)

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = NeuralSkillMatcher()
    if not HAS_NEURAL: return matcher

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
st.title("🧠 Symbiont: Circular Skill Economy")
st.markdown("### Level 3: Multi-Hop Semantic Trading")
st.caption("AI detects Direct Swaps (A↔B) and Triangular Chains (A→B→C→A) to maximize community liquidity.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers` to enable the Neural Brain.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Community Pulse")
    num_users = st.slider("Active Users", 10, 80, 40)
    match_threshold = st.slider("AI Similarity Threshold", 0.4, 0.9, 0.60, help="Lower = looser matches.")
    enable_circular = st.checkbox("Enable Circular Trading (Level 3)", value=True)
    
    if st.button("Run AI Engine"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        # Run Semantic Search
        start_time = time.time()
        direct, circular = st.session_state.neural_matcher.find_matches(threshold=match_threshold, enable_circular=enable_circular)
        duration = time.time() - start_time
        
        st.session_state.direct_matches = direct
        st.session_state.circular_matches = circular
        
        total_value = len(direct)*2 + len(circular)*3
        st.success(f"Optimized in {duration:.2f}s")
        st.metric("Total Users Helped", total_value)

    # Display Feeds
    if 'direct_matches' in st.session_state:
        st.subheader("Direct Swaps (1-to-1)")
        if not st.session_state.direct_matches: st.write("No direct swaps found.")
        for m in st.session_state.direct_matches[:3]:
            st.markdown(f"✅ **{m['users'][0]['name']}** ({m['skills'][0]}) ↔ **{m['users'][1]['name']}** ({m['skills'][1]})")
            
    if 'circular_matches' in st.session_state and enable_circular:
        st.subheader("Circular Chains (3-way)")
        if not st.session_state.circular_matches: st.write("No circular chains found.")
        for m in st.session_state.circular_matches[:3]:
             st.markdown(f"🔄 **{m['users'][0]['name']}** ➔ **{m['users'][1]['name']}** ➔ **{m['users'][2]['name']}** ➔ 🏁")

# --- Visualization ---
with col2:
    if 'direct_matches' in st.session_state:
        st.subheader("🔗 Economy Graph")
        
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', bgcolor='transparent')
        
        # Plot Direct Matches
        for m in st.session_state.direct_matches:
            u1, u2 = m['users']
            graph.node(u1['id'], label=f"{u1['name']}\n{u1['offering']}", shape="box", style="filled", fillcolor="#e1f5fe")
            graph.node(u2['id'], label=f"{u2['name']}\n{u2['offering']}", shape="box", style="filled", fillcolor="#e1f5fe")
            graph.edge(u1['id'], u2['id'], color="green", penwidth="2")
            graph.edge(u2['id'], u1['id'], color="green", penwidth="2")

        # Plot Circular Matches
        if enable_circular:
            for m in st.session_state.circular_matches:
                u1, u2, u3 = m['users']
                # Highlight Circular nodes in Orange
                for u in [u1, u2, u3]:
                    graph.node(u['id'], label=f"{u['name']}\n{u['offering']}", shape="ellipse", style="filled", fillcolor="#ffe0b2")
                
                # Edges A->B->C->A
                graph.edge(u1['id'], u2['id'], color="orange", style="dashed", penwidth="2")
                graph.edge(u2['id'], u3['id'], color="orange", style="dashed", penwidth="2")
                graph.edge(u3['id'], u1['id'], color="orange", style="dashed", penwidth="2")

        st.graphviz_chart(graph)
        
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Generate a community to spin up the Level 3 Neural Network.")