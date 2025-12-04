import streamlit as st
import pydeck as pdk
import pandas as pd
import random
import uuid
import math
import numpy as np
import copy
import matplotlib.pyplot as plt

# --- Config ---
st.set_page_config(page_title="Symbiont: Swarm Logistics", page_icon="📦", layout="wide")

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
        self.role = role # "Supplier", "Receiver", or "Depot"
        self.loc = loc
        self.item = item
        self.matched = False

class LogisticsEngine:
    def __init__(self):
        self.agents = []
        self.routes = []
        self.depot = None
        self.stats = {"co2_saved": 0, "efficiency_boost": 0}

    def add_agent(self, agent):
        if agent.role == "Depot":
            self.depot = agent
        else:
            self.agents.append(agent)

    def _calculate_route_cost(self, route):
        """Total distance of a route (Depot -> Nodes -> Depot)"""
        if not route: return 0
        dist = 0
        # Start at Depot
        curr = self.depot.loc
        for node in route:
            dist += curr.distance_to(node.loc)
            curr = node.loc
        # Return to Depot (Hub-and-Spoke model)
        dist += curr.distance_to(self.depot.loc)
        return dist

    def solve_simulated_annealing(self, initial_temp=1000, cooling_rate=0.995, max_iterations=2000):
        """
        Level 4 Algorithm: Simulated Annealing with 2-Opt Swaps.
        Finds the global optimum by probabilistically accepting worse solutions 
        early on to escape local minima.
        """
        active_nodes = [a for a in self.agents] # We route everyone
        if not active_nodes or not self.depot: return {}

        # 1. Initial Solution: Random Shuffle
        current_route = list(active_nodes)
        random.shuffle(current_route)
        
        current_cost = self._calculate_route_cost(current_route)
        
        best_route = list(current_route)
        best_cost = current_cost
        
        temp = initial_temp
        
        # Tracking for chart
        cost_history = []

        # 2. Annealing Loop
        progress_bar = st.progress(0)
        for i in range(max_iterations):
            # Create neighbor by swapping two random nodes (2-Opt)
            new_route = list(current_route)
            idx1, idx2 = random.sample(range(len(new_route)), 2)
            new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
            
            new_cost = self._calculate_route_cost(new_route)
            
            # Acceptance Probability
            if new_cost < current_cost:
                accept = True
            else:
                # Boltzmann distribution: higher temp = higher chance to accept bad move
                delta = new_cost - current_cost
                probability = math.exp(-delta / temp)
                accept = random.random() < probability
            
            if accept:
                current_route = new_route
                current_cost = new_cost
                
                # Keep track of absolute best
                if current_cost < best_cost:
                    best_route = list(current_route)
                    best_cost = current_cost
            
            # Cool down
            temp *= cooling_rate
            cost_history.append(current_cost)
            
            if i % 100 == 0:
                progress_bar.progress(i / max_iterations)
        
        progress_bar.progress(1.0)

        # Split into sub-routes for display (e.g. max 5 stops per drone)
        # This is a simple post-processing split
        final_routes = []
        chunk_size = 5
        for i in range(0, len(best_route), chunk_size):
            chunk = best_route[i:i + chunk_size]
            final_routes.append(chunk)

        # Baseline Calculation (Individual trips Depot -> Node -> Depot)
        baseline_cost = sum([self.depot.loc.distance_to(a.loc) * 2 for a in active_nodes])
        
        co2_baseline = baseline_cost * CO2_PER_KM_CAR
        co2_optimized = best_cost * CO2_PER_KM_DRONE
        
        self.routes = final_routes
        self.stats = {
            "individual_km": baseline_cost,
            "optimized_km": best_cost,
            "co2_saved": max(0, co2_baseline - co2_optimized),
            "iterations": max_iterations,
            "history": cost_history
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
st.markdown("### Level 4: Swarm Intelligence")
st.caption("AI uses **Simulated Annealing** (Probabilistic Optimization) to untangle complex supply chains into efficient 'Milk Run' loops rooted at a central Hub.")

col1, col2 = st.columns([1, 3])

# --- Init Session ---
if 'logistics' not in st.session_state:
    st.session_state.logistics = LogisticsEngine()
    st.session_state.map_data = []

# --- Sidebar ---
with col1:
    st.subheader("1. Grid Parameters")
    num_nodes = st.slider("Nodes Density", 10, 80, 40)
    
    if st.button("Generate Demand"):
        engine = LogisticsEngine()
        map_data = []
        item_pool = ["Food Waste", "Electronics", "Textiles", "Furniture"]
        
        # Center Depot
        center_loc = Location(40.7128, -74.0060)
        depot = Agent("DEPOT", "Depot", center_loc, "Hub")
        engine.add_agent(depot)
        map_data.append({
            "lat": 40.7128, "lon": -74.0060, "color": [0, 255, 0, 255], 
            "radius": 300, "info": "♻️ CENTRAL HUB"
        })
        
        for i in range(num_nodes):
            uid = str(uuid.uuid4())[:6]
            loc = generate_random_loc()
            
            if random.random() > 0.5:
                role = "Supplier"
                color = [200, 30, 0, 200] 
            else:
                role = "Receiver"
                color = [0, 100, 240, 200] 
                
            agent = Agent(uid, role, loc, random.choice(item_pool))
            engine.add_agent(agent)
            
            map_data.append({
                "lat": loc.lat, "lon": loc.lon, "color": color, 
                "radius": 100, "info": f"{role}: {agent.item}"
            })
            
        st.session_state.logistics = engine
        st.session_state.map_data = map_data
        st.session_state.optimized = False
        st.success(f"{num_nodes} Nodes + 1 Hub Active")

    st.markdown("---")
    st.subheader("2. AI Settings")
    temp = st.slider("Initial Temperature", 100, 5000, 1000, help="High temp = more exploration (chaos).")
    cooling = st.slider("Cooling Rate", 0.8, 0.999, 0.99, help="How fast the AI 'freezes' the solution.")
    
    if st.button("Run Simulated Annealing"):
        if not st.session_state.map_data:
            st.error("Generate Grid First!")
        else:
            with st.spinner("Annealing route structures..."):
                stats = st.session_state.logistics.solve_simulated_annealing(temp, cooling)
            
            st.session_state.stats = stats
            st.session_state.optimized = True
            
            # Calculate Gain
            old = stats['individual_km']
            new = stats['optimized_km']
            if old > 0:
                percent = ((old - new) / old) * 100
                st.metric("Efficiency Boost", f"+{percent:.1f}%")
            
            st.metric("CO2 Saved", f"{stats['co2_saved']:.2f} kg")

    if st.session_state.get("optimized", False) and "history" in st.session_state.stats:
        st.markdown("---")
        st.caption("Optimization Curve")
        # Use Matplotlib instead of Altair due to Python 3.14 compatibility
        fig, ax = plt.subplots()
        ax.plot(st.session_state.stats["history"])
        ax.set_ylabel("Total Route Distance (km)")
        ax.set_xlabel("Iterations")
        ax.set_title("Simulated Annealing Progress")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

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
        
        # 2. Routes Layer
        if st.session_state.get("optimized", False):
            path_data = []
            routes = st.session_state.logistics.routes
            depot_loc = st.session_state.logistics.depot.loc
            
            for route in routes:
                # Path: Depot -> A -> B -> ... -> Depot
                path_coords = [[depot_loc.lon, depot_loc.lat]] # Start Depot
                for agent in route:
                    path_coords.append([agent.loc.lon, agent.loc.lat])
                path_coords.append([depot_loc.lon, depot_loc.lat]) # End Depot
                
                # Neon colors
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
                width_min_pixels=3,
                pickable=True
            ))

        # Render
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10", 
            initial_view_state=pdk.ViewState(
                latitude=40.7128, longitude=-74.0060, zoom=12, pitch=45
            ),
            layers=layers,
            tooltip={"text": "{info}"}
        ))
        
        # Manifest
        if st.session_state.get("optimized", False):
            st.subheader("🚚 Fleet Manifest")
            cols = st.columns(3)
            for i, route in enumerate(st.session_state.logistics.routes[:6]): 
                with cols[i % 3]:
                    st.info(f"**Drone #{i+1}** ({len(route)} stops)")
                    # Preview first 3 stops
                    flow = "Depot ➝ " + " ➝ ".join([a.role[0] for a in route[:3]]) 
                    if len(route) > 3: flow += "..."
                    st.caption(f"{flow} ➝ Depot")
                    
    else:
        st.info("👈 Initialize the Logistics Grid to begin.")