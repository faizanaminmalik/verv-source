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
st.set_page_config(page_title="Symbiont: Genetic Logistics", page_icon="🧬", layout="wide")

# --- Constants ---
EARTH_RADIUS_KM = 6371.0
CO2_PER_KM_CAR = 0.192 # kg
CO2_PER_KM_DRONE = 0.015 # kg
DRONE_CAPACITY_KG = 50 # Max weight a drone can carry before returning to hub

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
    def __init__(self, uid, role, loc, item=None, weight=0):
        self.id = uid
        self.role = role # "Supplier", "Receiver", or "Depot"
        self.loc = loc
        self.item = item
        self.weight = weight # Weight of the package in kg

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

    def _evaluate_fitness(self, individual):
        """
        Decodes a chromosome (list of agents) into actual routes based on Capacity.
        Returns: (Total Distance, List of Routes)
        """
        routes = []
        current_route = []
        current_load = 0
        total_dist = 0
        
        # Start at Depot
        curr_loc = self.depot.loc
        
        for agent in individual:
            # Check Capacity
            if current_load + agent.weight > DRONE_CAPACITY_KG:
                # Return to Depot
                total_dist += curr_loc.distance_to(self.depot.loc)
                routes.append(current_route)
                
                # Start new vehicle
                current_route = []
                current_load = 0
                curr_loc = self.depot.loc
            
            # Travel to Agent
            dist = curr_loc.distance_to(agent.loc)
            total_dist += dist
            current_route.append(agent)
            current_load += agent.weight
            curr_loc = agent.loc
            
        # Return last vehicle to Depot
        if current_route:
            total_dist += curr_loc.distance_to(self.depot.loc)
            routes.append(current_route)
            
        return total_dist, routes

    def solve_genetic_algorithm(self, pop_size=100, generations=50, mutation_rate=0.05):
        """
        Level 5 Algorithm: Genetic Algorithm for CVRP (Capacitated Vehicle Routing).
        Evolves a population of route orders to minimize distance while obeying weight limits.
        """
        if not self.agents or not self.depot: return {}

        # 1. Initialize Population (Random Permutations)
        population = []
        for _ in range(pop_size):
            individual = list(self.agents)
            random.shuffle(individual)
            population.append(individual)
            
        best_fitness = float('inf')
        best_routes = []
        fitness_history = []
        
        progress_bar = st.progress(0)
        
        # 2. Evolution Loop
        for gen in range(generations):
            # Evaluate all
            ranked_pop = []
            for ind in population:
                fit, routes = self._evaluate_fitness(ind)
                ranked_pop.append((fit, ind))
                
                if fit < best_fitness:
                    best_fitness = fit
                    best_routes = routes
            
            # Sort by fitness (lowest distance is best)
            ranked_pop.sort(key=lambda x: x[0])
            fitness_history.append(best_fitness)
            
            # Selection (Elitism + Tournament)
            next_gen = []
            
            # Keep top 10% (Elites)
            elite_count = int(pop_size * 0.1)
            next_gen.extend([ind for fit, ind in ranked_pop[:elite_count]])
            
            # Breed the rest
            while len(next_gen) < pop_size:
                # Tournament Selection
                parent1 = self._tournament_select(ranked_pop)
                parent2 = self._tournament_select(ranked_pop)
                
                # Crossover (Ordered Crossover - OX1)
                child = self._ordered_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    self._mutate(child)
                    
                next_gen.append(child)
            
            population = next_gen
            
            if gen % 5 == 0:
                progress_bar.progress(gen / generations)
        
        progress_bar.progress(1.0)

        # Baseline Cost (Individual Trips)
        baseline_cost = sum([self.depot.loc.distance_to(a.loc) * 2 for a in self.agents])
        
        co2_baseline = baseline_cost * CO2_PER_KM_CAR
        co2_optimized = best_fitness * CO2_PER_KM_DRONE
        
        self.routes = best_routes
        self.stats = {
            "individual_km": baseline_cost,
            "optimized_km": best_fitness,
            "co2_saved": max(0, co2_baseline - co2_optimized),
            "generations": generations,
            "history": fitness_history,
            "drone_count": len(best_routes)
        }
        return self.stats

    def _tournament_select(self, ranked_pop, k=5):
        """Pick k random individuals, return the best."""
        sample = random.sample(ranked_pop, k)
        return min(sample, key=lambda x: x[0])[1]

    def _ordered_crossover(self, p1, p2):
        """
        OX1 Crossover: Preserves order of genes to avoid duplicates in permutation.
        """
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        # Copy slice from Parent 1
        child[start:end] = p1[start:end]
        
        # Fill remaining spots with Parent 2 (skipping duplicates)
        p2_genes = [item for item in p2 if item not in child[start:end]]
        
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_genes[idx]
                idx += 1
        return child

    def _mutate(self, individual):
        """Swap Mutation: Swap two random nodes."""
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

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
st.title("📦 Symbiont: Genetic Logistics")
st.markdown("### Level 5: The Genetic Hive Mind")
st.caption("AI uses **Evolutionary Algorithms** to solve the Capacitated Vehicle Routing Problem (CVRP). Drones have weight limits and must evolve efficient paths to serve the grid.")

col1, col2 = st.columns([1, 3])

# --- Init Session ---
if 'logistics' not in st.session_state:
    st.session_state.logistics = LogisticsEngine()
    st.session_state.map_data = []

# --- Sidebar ---
with col1:
    st.subheader("1. Grid Parameters")
    num_nodes = st.slider("Neighborhood Density", 10, 80, 30)
    
    if st.button("Generate Demand"):
        engine = LogisticsEngine()
        map_data = []
        item_pool = [
            ("Food Waste", 5), ("Old TV", 15), ("Textiles", 8), 
            ("Furniture", 40), ("Glass", 10), ("Batteries", 2)
        ]
        
        # Center Depot
        center_loc = Location(40.7128, -74.0060)
        depot = Agent("DEPOT", "Depot", center_loc, "Hub")
        engine.add_agent(depot)
        map_data.append({
            "lat": 40.7128, "lon": -74.0060, "color": [0, 255, 0, 255], 
            "radius": 300, "info": "♻️ RECYCLING HUB"
        })
        
        for i in range(num_nodes):
            uid = str(uuid.uuid4())[:6]
            loc = generate_random_loc()
            
            # Role (Supplier has items, Receiver takes items - simple model: all go via depot for now)
            # For CVRP visuals, we assume everyone is a pickup point going to Depot
            role = "Pickup"
            item_name, weight = random.choice(item_pool)
            color = [200, 30, 0, 200] 
                
            agent = Agent(uid, role, loc, item_name, weight)
            engine.add_agent(agent)
            
            map_data.append({
                "lat": loc.lat, "lon": loc.lon, "color": color, 
                "radius": 50 + (weight * 2), # Size based on weight
                "info": f"📦 {item_name} ({weight}kg)"
            })
            
        st.session_state.logistics = engine
        st.session_state.map_data = map_data
        st.session_state.optimized = False
        st.success(f"{num_nodes} Pickup Points Active")

    st.markdown("---")
    st.subheader("2. Evolution Settings")
    pop_size = st.slider("Population Size", 50, 500, 100, help="How many route plans exist per generation.")
    generations = st.slider("Generations", 10, 200, 50, help="How many times the population reproduces.")
    
    if st.button("Evolve Solution"):
        if not st.session_state.map_data:
            st.error("Generate Grid First!")
        else:
            with st.spinner("Breeding optimal routes..."):
                stats = st.session_state.logistics.solve_genetic_algorithm(pop_size, generations)
            
            st.session_state.stats = stats
            st.session_state.optimized = True
            
            st.metric("Total Distance", f"{stats['optimized_km']:.1f} km")
            st.metric("CO2 Saved", f"{stats['co2_saved']:.2f} kg")
            st.metric("Active Drones", f"{stats['drone_count']}")

    if st.session_state.get("optimized", False) and "history" in st.session_state.stats:
        st.markdown("---")
        st.caption("Genetic Fitness (Distance vs Generation)")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(st.session_state.stats["history"], color='purple')
        ax.set_ylabel("Distance (km)")
        ax.set_xlabel("Generation")
        ax.grid(True, linestyle='--', alpha=0.3)
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
            
            for i, route in enumerate(routes):
                # Path: Depot -> A -> B -> ... -> Depot
                path_coords = [[depot_loc.lon, depot_loc.lat]] # Start Depot
                
                route_load = 0
                path_text = []
                
                for agent in route:
                    path_coords.append([agent.loc.lon, agent.loc.lat])
                    route_load += agent.weight
                    path_text.append(f"{agent.weight}kg")
                    
                path_coords.append([depot_loc.lon, depot_loc.lat]) # End Depot
                
                # Unique color per drone
                r_color = [random.randint(50,255), random.randint(100,255), 50]
                
                path_data.append({
                    "path": path_coords,
                    "color": r_color,
                    "name": f"Drone {i+1} (Load: {route_load}/{DRONE_CAPACITY_KG}kg)"
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
            tooltip={"text": "{info}\n{name}"}
        ))
        
        # Manifest
        if st.session_state.get("optimized", False):
            st.subheader("🚚 Hive Fleet Manifest")
            
            # Show top routes
            cols = st.columns(3)
            for i, route in enumerate(st.session_state.logistics.routes): 
                with cols[i % 3]:
                    load = sum(a.weight for a in route)
                    load_percent = int((load / DRONE_CAPACITY_KG) * 100)
                    
                    st.info(f"**Drone #{i+1}** | Stops: {len(route)}")
                    st.progress(load_percent / 100)
                    st.caption(f"Payload: {load}/{DRONE_CAPACITY_KG} kg")
                    
    else:
        st.info("👈 Initialize the Logistics Grid to begin.")