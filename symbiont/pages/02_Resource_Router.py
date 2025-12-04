import streamlit as st
import pydeck as pdk
import pandas as pd
import random
import uuid
from resource_router import ResourceRouter, UserNode, Location, Resource, ResourceType

# --- Config ---
st.set_page_config(page_title="Symbiont: Resource Grid", page_icon="📦", layout="wide")

# --- Helper Functions ---
def generate_random_location(center_lat=40.7128, center_lon=-74.0060, radius_km=5.0):
    """Generates a random Lat/Lon within a radius of a center point."""
    r = radius_km / 111.0 
    u = random.random()
    v = random.random()
    w = r * (u ** 0.5)
    t = 2 * 3.14159 * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    return Location(center_lat + x, center_lon + y)

import math # Ensure math is available

# --- Session State (To keep data alive between button clicks) ---
if 'router' not in st.session_state:
    st.session_state.router = ResourceRouter()
    st.session_state.agents = []
    st.session_state.matches = []
    st.session_state.initialized = False

# --- UI Layout ---
st.title("📦 Symbiont Resource Router")
st.markdown("### Decentralized Last-Mile Logistics Engine")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("1. Initialize Grid")
    num_agents = st.slider("Number of Agents", 10, 100, 40)
    
    if st.button("Generate Virtual City"):
        # Reset
        st.session_state.router = ResourceRouter()
        st.session_state.agents = []
        st.session_state.matches = []
        
        # Create Dummy Data
        item_pool = ["Apple", "Bread", "Laptop Charger", "Drill", "Winter Coat", "Milk", "Chairs"]
        
        for i in range(num_agents):
            uid = str(uuid.uuid4())[:8]
            loc = generate_random_location()
            agent = UserNode(id=uid, name=f"Agent_{uid}", location=loc)
            
            # 50/50 Chance of being Supplier or Receiver
            if random.random() > 0.5:
                # Supplier
                item_name = random.choice(item_pool)
                res = Resource(
                    id=str(uuid.uuid4()), name=item_name,
                    type=ResourceType.FOOD, expiry_hours=random.randint(2, 72),
                    owner_id=uid
                )
                agent.inventory.append(res)
                role = "Supplier"
                color = [200, 30, 0, 160] # Red
            else:
                # Receiver
                agent.wishlist.append(random.choice(item_pool))
                role = "Receiver"
                color = [0, 100, 240, 160] # Blue
                
            st.session_state.router.add_node(agent)
            st.session_state.agents.append({
                "id": uid,
                "role": role,
                "lat": loc.lat,
                "lon": loc.lon,
                "color": color,
                "info": f"{role}: {agent.inventory[0].name if role == 'Supplier' else agent.wishlist[0]}"
            })
        
        st.session_state.initialized = True
        st.success(f"Generated {num_agents} Nodes!")

    st.markdown("---")
    st.subheader("2. Run AI Matcher")
    
    if st.button("Run Proximal Decay Algorithm", disabled=not st.session_state.initialized):
        matches = st.session_state.router.find_matches()
        st.session_state.matches = matches
        if matches:
            st.success(f"Found {len(matches)} Optimal Routes!")
        else:
            st.warning("No matches found. Try generating again.")

# --- Visualization Logic (PyDeck) ---
with col2:
    if st.session_state.initialized:
        # Prepare Data for Map
        agents_df = pd.DataFrame(st.session_state.agents)
        
        layers = []
        
        # Layer 1: The Nodes (Dots)
        nodes_layer = pdk.Layer(
            "ScatterplotLayer",
            data=agents_df,
            get_position='[lon, lat]',
            get_color='color',
            get_radius=100,
            pickable=True,
            auto_highlight=True
        )
        layers.append(nodes_layer)
        
        # Layer 2: The Matches (Arcs/Lines)
        if st.session_state.matches:
            match_data = []
            for m in st.session_state.matches:
                match_data.append({
                    "source": [m.supplier.location.lon, m.supplier.location.lat],
                    "target": [m.receiver.location.lon, m.receiver.location.lat],
                    "item": m.resource.name,
                    "score": m.match_score
                })
            
            match_df = pd.DataFrame(match_data)
            
            arcs_layer = pdk.Layer(
                "ArcLayer",
                data=match_df,
                get_source_position="source",
                get_target_position="target",
                get_source_color=[200, 30, 0],   # Red (Supplier)
                get_target_color=[0, 100, 240],  # Blue (Receiver)
                get_width=3,
                pickable=True
            )
            layers.append(arcs_layer)
        
        # Render Map
        view_state = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=11, pitch=40)
        
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "{info}"}
        ))
        
        # Stats Table
        if st.session_state.matches:
            st.markdown("### 📊 Trade Manifest")
            
            # Format data for display
            display_data = []
            for m in st.session_state.matches:
                display_data.append({
                    "Item": m.resource.name,
                    "From": m.supplier.name,
                    "To": m.receiver.name,
                    "Distance (km)": f"{m.distance_km:.2f}",
                    "Carbon Saved (kg)": f"{m.saved_carbon_kg:.2f}",
                    "Match Score": f"{m.match_score:.2f}"
                })
            st.dataframe(pd.DataFrame(display_data))
            
    else:
        st.info("👈 Generate a virtual city to begin simulation.")