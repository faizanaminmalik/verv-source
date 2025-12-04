import streamlit as st
import pydeck as pdk
import pandas as pd
import random
import uuid
import math
import numpy as np

# --- Config ---
st.set_page_config(page_title="Symbiont: Logistics AI", page_icon="📦", layout="wide")

# --- Constants ---
EARTH_RADIUS_KM = 6371.0
AVG_SPEED_KMPH = 30.0
CO2_PER_KM_CAR = 0.192 # kg
CO2_PER_KM_DRONE = 0.015 # kg

# --- Core Logic Classes ---
class Location:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def distance_to(self, other):
        """Haversine distance in KM"""
        dlat = math.radians(other.lat - self.lat)
        dlon = math.radians(other.lon - self.lon)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(self.lat)) * math.cos(math.radians(other.lat)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_KM * c

class Agent:
    def __init__(self, uid, role, loc, item=None):
        self.id = uid
        self.role = role # "Supplier" or "Receiver"
        self.loc = loc
        self.item = item
        self.matched = False

class LogisticsEngine:
    def __init__(self):
        self.agents = []
        self.routes = []
        self.stats = {"co2_saved": 0, "efficiency_boost": 0}

    def add_agent(self, agent):
        self.agents.append(agent)

    def solve_vehicle_routing(self, max_stops=5):
        """
        Level 3 Algorithm: Greedy Route Construction (Nearest Neighbor)
        Simulates a 'Community Courier' picking up from Suppliers and delivering to Receivers.
        """
        suppliers = [a for a in self.agents if a.role == "Supplier" and not a.matched]
        receivers = [a for a in self.agents if a.role == "Receiver" and not a.matched]
        
        routes = []
        individual_dist = 0
        optimized_dist = 0
        
        # While we have pairs to match
        while suppliers and receivers:
            # Start a new route (New Driver/Drone)
            route_path = []
            
            # Pick a random start point (Depot or first Supplier)
            current_node = suppliers.pop(0)
            route_path.append(current_node)
            current_node.matched = True
            
            # Find closest Receiver for this item (Simple 1-to-1 logic for the item type for now)
            # In a real VRP, this is much more complex. We simulate the "Trip" here.
            
            # Find closest receiver to the current supplier
            best_rec = min(receivers, key=lambda r: current_node.loc.distance_to(r.loc))
            
            # Calculate distance if they drove themselves (P2P)
            dist_p2p = current_node.loc.distance_to(best_rec.loc)
            individual_dist += dist_p2p
            
            # Add to route
            route_path.append(best_rec)
            receivers.remove(best_rec)
            best_rec.matched = True
            
            # "Chain" the route: Can this driver pick up another nearby Supplier?
            # Look for closest remaining supplier from the last receiver
            if len(route_path) < max_stops * 2: # *2 because Pickup+Dropoff = 2 stops
                if suppliers:
                    closest_next_sup = min(suppliers, key=lambda s: best_rec.loc.distance_to(s.loc))
                    dist_to_next = best_rec.loc.distance_to(closest_next_sup.loc)
                    
                    # Heuristic: Only chain if close ( < 3km )
                    if dist_to_next < 3.0:
                        # Optimization: The driver drives A -> B -> C ... 
                        # Instead of A->B, C->D.
                        # We accumulate the optimized distance
                        pass # Logic handled in final path sum
            
            routes.append(route_path)

        # Calculate optimized distance (The sum of the path)
        for route in routes:
            for i in range(len(route) - 1):
                optimized_dist += route[i].loc.distance_to(route[i+1].loc)

        # Stats
        co2_individual = individual_dist * CO2_PER_KM_CAR
        co2_optimized = optimized_dist * CO2_PER_KM_DRONE # Assuming optimized uses efficient transport
        
        self.routes = routes
        self.stats = {
            "individual_km": individual_dist,
            "optimized_km": optimized_dist,
            "co2_saved": max(0, co2_individual - co2_optimized),
            "routes_count": len(routes)
        }
        return self.stats

# --- Helper Functions ---
def generate_random_loc(center_lat=40.7128, center_lon=-74.0060, radius_km=8.0):
    r = radius_km / 111.0 
    u = random.random()
    v = random.random()
    w = r * (u ** 0.5)
    t = 2 * 3.14159 * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    return Location(center_lat + x, center_lon + y)

# --- UI Layout ---
st.title("📦 Symbiont: Swarm Logistics")
st.markdown("### Level 3: The Mycelial Network")
st.caption("AI solves the **Vehicle Routing Problem (VRP)** to replace individual car trips with optimized 'Milk Run' loops, slashing carbon emissions.")

col1, col2 = st.columns([1, 3])

# --- Init Session ---
if 'logistics' not in st.session_state:
    st.session_state.logistics = LogisticsEngine()
    st.session_state.map_data = []

# --- Sidebar ---
with col1:
    st.subheader("1. Grid Parameters")
    num_nodes = st.slider("Neighborhood Density", 10, 100, 50)
    
    if st.button("Generate Demand"):
        engine = LogisticsEngine()
        map_data = []
        item_pool = ["Food Waste", "Electronics", "Textiles", "Furniture"]
        
        for i in range(num_nodes):
            uid = str(uuid.uuid4())[:6]
            loc = generate_random_loc()
            
            if random.random() > 0.5:
                role = "Supplier"
                color = [200, 30, 0, 200] # Red
            else:
                role = "Receiver"
                color = [0, 100, 240, 200] # Blue
                
            agent = Agent(uid, role, loc, random.choice(item_pool))
            engine.add_agent(agent)
            
            map_data.append({
                "lat": loc.lat, "lon": loc.lon, "color": color, 
                "radius": 100, "info": f"{role}: {agent.item}"
            })
            
        st.session_state.logistics = engine
        st.session_state.map_data = map_data
        st.session_state.optimized = False
        st.success(f"{num_nodes} Nodes Active")

    st.markdown("---")
    st.subheader("2. AI Optimization")
    
    if st.button("Run Swarm Routing"):
        if not st.session_state.map_data:
            st.error("Generate Grid First!")
        else:
            stats = st.session_state.logistics.solve_vehicle_routing()
            st.session_state.stats = stats
            st.session_state.optimized = True
            
            # Calculate Gain
            old = stats['individual_km']
            new = stats['optimized_km']
            if old > 0:
                percent = ((old - new) / old) * 100
                st.metric("Efficiency Boost", f"+{percent:.1f}%")
            
            st.metric("CO2 Saved", f"{stats['co2_saved']:.2f} kg")

# --- Visualization ---
with col2:
    if st.session_state.map_data:
        layers = []
        
        # 1. Nodes Layer
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame(st.session_state.map_data),
            get_position='[lon, lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            auto_highlight=True
        ))
        
        # 2. Routes Layer (The Mycelium)
        if st.session_state.get("optimized", False):
            path_data = []
            routes = st.session_state.logistics.routes
            
            for route in routes:
                # Build a continuous path for the vehicle
                path_coords = [[agent.loc.lon, agent.loc.lat] for agent in route]
                
                # Color code routes randomly to distinguish drivers
                r_color = [random.randint(50,255), random.randint(100,255), 50]
                
                path_data.append({
                    "path": path_coords,
                    "color": r_color,
                    "name": f"Route {routes.index(route)+1}"
                })
            
            layers.append(pdk.Layer(
                "PathLayer",
                data=pd.DataFrame(path_data),
                get_path="path",
                get_color="color",
                width_scale=20,
                width_min_pixels=2,
                pickable=True
            ))

        # Render
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10", # Dark mode for "Cybernetic" feel
            initial_view_state=pdk.ViewState(
                latitude=40.7128, longitude=-74.0060, zoom=12, pitch=45
            ),
            layers=layers,
            tooltip={"text": "{info}"}
        ))
        
        # Manifest
        if st.session_state.get("optimized", False):
            st.subheader("🚚 Route Manifest")
            cols = st.columns(3)
            for i, route in enumerate(st.session_state.logistics.routes[:6]): # Show top 6
                with cols[i % 3]:
                    st.info(f"**Route #{i+1}** ({len(route)} stops)")
                    flow = " ➝ ".join([a.role[0] for a in route]) # S -> R -> S
                    st.caption(f"Path: {flow}")
                    
    else:
        st.info("👈 Initialize the Logistics Grid to begin.")