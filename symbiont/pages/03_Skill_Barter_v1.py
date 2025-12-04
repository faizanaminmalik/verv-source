import streamlit as st
import random
import uuid
import graphviz
from skill_barter import SkillMatcher, SkillUser, SKILL_CATALOG

# --- Config ---
st.set_page_config(page_title="Symbiont: Skill Barter", page_icon="🤝", layout="wide")

# --- Helper to Generate Fake Users ---
def generate_community(n=20):
    matcher = SkillMatcher()
    for i in range(n):
        offer = random.choice(SKILL_CATALOG)
        # Ensure need is different from offer
        need = random.choice([s for s in SKILL_CATALOG if s != offer])
        
        user = SkillUser(
            id=str(uuid.uuid4())[:8],
            name=f"User_{random.randint(100,999)}",
            offering=offer,
            needing=need,
            reputation=round(random.uniform(3.5, 5.0), 1)
        )
        matcher.add_user(user)
    return matcher

# --- UI Layout ---
st.title("🤝 Skill Barter Protocol")
st.markdown("### The Moneyless Talent Exchange")
st.caption("Matches users who have complementary needs (e.g., 'I teach Python' ↔ 'You teach Yoga')")

col1, col2 = st.columns([1, 2])

# --- Sidebar Controls ---
with col1:
    st.subheader("1. Community Pulse")
    num_users = st.slider("Active Users in Neighborhood", 10, 100, 30)
    
    if st.button("Scan for Talent"):
        st.session_state.skill_matcher = generate_community(num_users)
        st.session_state.skill_matches = st.session_state.skill_matcher.find_direct_swaps()
        st.success(f"Scanned {num_users} profiles.")
        st.info(f"Found {len(st.session_state.skill_matches)} Perfect Swaps!")

    if 'skill_matches' in st.session_state and st.session_state.skill_matches:
        st.subheader("2. Swap Feed")
        for m in st.session_state.skill_matches[:5]:
            st.markdown(f"""
            **Match Found!** (Quality: {m.match_quality*100}%)  
            👤 {m.user_a.name} gives **{m.skill_a_to_b}** 👤 {m.user_b.name} gives **{m.skill_b_to_a}** ---
            """)

# --- Visualization ---
with col2:
    if 'skill_matches' in st.session_state and st.session_state.skill_matches:
        st.subheader("🔗 Connection Graph")
        
        # Create Graphviz object
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', bgcolor='transparent')
        
        # Add Nodes and Edges for Matches
        for m in st.session_state.skill_matches:
            # Nodes
            graph.node(m.user_a.id, label=f"{m.user_a.name}\n(Has: {m.user_a.offering})", shape="box", style="filled", fillcolor="#e1f5fe")
            graph.node(m.user_b.id, label=f"{m.user_b.name}\n(Has: {m.user_b.offering})", shape="box", style="filled", fillcolor="#fff9c4")
            
            # Edges (The Swap)
            graph.edge(m.user_a.id, m.user_b.id, label=f"Teaches {m.skill_a_to_b}", color="green")
            graph.edge(m.user_b.id, m.user_a.id, label=f"Teaches {m.skill_b_to_a}", color="green")
            
        st.graphviz_chart(graph)
        
    elif 'skill_matcher' not in st.session_state:
        st.info("👈 Generate a community to see the connections form.")
    else:
        st.warning("No perfect swaps found in this iteration. Try increasing the user count!")