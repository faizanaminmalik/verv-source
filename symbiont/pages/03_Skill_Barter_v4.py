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
st.set_page_config(page_title="Symbiont: Deep Pathfinder", page_icon="🧬", layout="wide")

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
        
        return util.cos_sim(offer_vectors, need_vectors)

    def find_deep_chains(self, threshold=0.65, max_depth=4):
        """
        Level 4 Algorithm: Depth-First Search (DFS) for Cycle Detection.
        Finds value chains of length 2 (Direct) up to max_depth.
        """
        if not HAS_NEURAL:
            return []

        users_list = list(self.users.values())
        n = len(users_list)
        sim_matrix = self._compute_similarity_matrix(users_list)
        
        # 1. Build Adjacency Graph (Who satisfies whom?)
        # graph[u] = [list of users u can help]
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if sim_matrix[i][j].item() > threshold:
                    adj[i].append(j)

        cycles = []
        
        # 2. Recursive DFS to find cycles
        def dfs(start_node, current_node, path, visited_indices):
            # Stop if path gets too long
            if len(path) > max_depth:
                return

            neighbors = adj[current_node]
            for neighbor in neighbors:
                if neighbor == start_node:
                    # Cycle found! 
                    # Only accept if length >= 2 (Direct Swap or longer)
                    if len(path) >= 2:
                        # Normalize: Check if this is a unique cycle we haven't seen.
                        # We enforce that we only record the cycle if the start_node is the 
                        # smallest index in the loop. This prevents duplicates like [A,B,C] vs [B,C,A].
                        if start_node == min(path):
                            # Calculate average score for the chain
                            chain_indices = path + [start_node]
                            total_score = 0
                            chain_users = []
                            chain_skills = []
                            
                            for k in range(len(path)):
                                u_idx = chain_indices[k]
                                v_idx = chain_indices[k+1]
                                total_score += sim_matrix[u_idx][v_idx].item()
                                chain_users.append(users_list[u_idx])
                                chain_skills.append(users_list[u_idx]['offering'])
                            
                            cycles.append({
                                "length": len(path),
                                "users": chain_users,
                                "skills": chain_skills,
                                "score": total_score / len(path)
                            })
                
                elif neighbor not in visited_indices:
                    # Continue searching deeper
                    visited_indices.add(neighbor)
                    path.append(neighbor)
                    dfs(start_node, neighbor, path, visited_indices)
                    # Backtrack
                    path.pop()
                    visited_indices.remove(neighbor)

        # Run DFS from every node
        for i in range(n):
            dfs(i, i, [i], {i})

        # Sort by length (Direct first), then by score
        return sorted(cycles, key=lambda x: (x['length'], -x['score']))

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
st.title("🧬 Symbiont: Deep Pathfinder")
st.markdown("### Level 4: Recursive Value Chains")
st.caption("AI uses Depth-First Search (DFS) to discover complex, multi-person trading loops (A→B→C→D→A) that connect the entire community.")

if not HAS_NEURAL:
    st.error("🚨 Missing Dependency: Please run `pip install sentence-transformers` to enable the Neural Brain.")
    st.stop()

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Community Pulse")
    num_users = st.slider("Active Users", 20, 100, 50)
    match_threshold = st.slider("AI Similarity Threshold", 0.4, 0.9, 0.60, help="Lower = looser matches.")
    max_depth = st.slider("Max Chain Length (Hops)", 2, 6, 4, help="Level 4 Power: How deep should the AI search?")
    
    if st.button("Run Pathfinder Engine"):
        st.session_state.neural_matcher = generate_community(num_users)
        
        # Run Level 4 Search
        start_time = time.time()
        chains = st.session_state.neural_matcher.find_deep_chains(threshold=match_threshold, max_depth=max_depth)
        duration = time.time() - start_time
        
        st.session_state.chains = chains
        
        # Calculate impact
        total_people = sum([c['length'] for c in chains])
        st.success(f"Pathfinding complete in {duration:.2f}s")
        st.metric("Liquidity Unlocked", f"{len(chains)} Loops", f"{total_people} People Served")

    # Display Feeds
    if 'chains' in st.session_state:
        st.markdown("---")
        st.subheader("⛓️ Value Chain Feed")
        
        if not st.session_state.chains:
            st.warning("No loops found. Try lowering threshold or increasing users.")
            
        # Group by length for cleaner display
        direct = [c for c in st.session_state.chains if c['length'] == 2]
        deep = [c for c in st.session_state.chains if c['length'] > 2]
        
        if deep:
            st.markdown(f"**Found {len(deep)} Deep Chains (Level 4+):**")
            for c in deep[:5]:
                # Format: A -> B -> C -> A
                names = [u['name'] for u in c['users']]
                skills = c['skills']
                chain_str = ""
                for i in range(len(names)):
                    chain_str += f"**{names[i]}** ({skills[i]}) ➡️ "
                chain_str += "🏁"
                st.info(chain_str)
        
        if direct:
            st.markdown(f"**Found {len(direct)} Direct Swaps:**")
            for c in direct[:3]:
                st.write(f"✅ {c['users'][0]['name']} ↔ {c['users'][1]['name']}")

# --- Visualization ---
with col2:
    if 'chains' in st.session_state and st.session_state.chains:
        st.subheader("🌐 Network Topology")
        
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', bgcolor='transparent')
        
        # We limit visualization to top 15 chains to prevent browser crash on large graphs
        top_chains = st.session_state.chains[:15]
        
        drawn_nodes = set()
        
        for chain in top_chains:
            users = chain['users']
            length = chain['length']
            
            # Color logic: Green for Direct (2), Orange for Triangles (3), Purple for Deep (4+)
            edge_color = "green" if length == 2 else ("orange" if length == 3 else "purple")
            style = "solid" if length == 2 else "dashed"
            penwidth = str(max(1, chain['score'] * 3))

            for i in range(len(users)):
                u_curr = users[i]
                u_next = users[(i + 1) % len(users)] # Wrap around to start
                
                # Draw Nodes
                if u_curr['id'] not in drawn_nodes:
                    graph.node(u_curr['id'], label=f"{u_curr['name']}\n{u_curr['offering']}", shape="box", style="filled", fillcolor="#e1f5fe")
                    drawn_nodes.add(u_curr['id'])
                
                # Draw Edges
                graph.edge(u_curr['id'], u_next['id'], color=edge_color, style=style, penwidth=penwidth)

        st.graphviz_chart(graph)
        
    elif 'neural_matcher' not in st.session_state:
        st.info("👈 Generate a community to spin up the Level 4 Engine.")