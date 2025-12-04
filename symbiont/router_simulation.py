import random
import uuid
import matplotlib.pyplot as plt
from resource_router import ResourceRouter, UserNode, Location, Resource, ResourceType

def generate_random_location(center_lat=40.7128, center_lon=-74.0060, radius_km=5.0):
    """Generates a random Lat/Lon within a radius of a center point."""
    # Convert radius from km to degrees (approx)
    r = radius_km / 111.0 
    u = random.random()
    v = random.random()
    w = r * (u ** 0.5)
    t = 2 * 3.14159 * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    return Location(center_lat + x, center_lon + y) # Use simple euclidean approx for small radius

import math # Re-importing math to ensure it is available in this scope if pasted separately

# --- Setup Simulation ---
print("--- Initializing Symbiont Resource Grid ---")
router = ResourceRouter()

# 1. Create Agents (The Nodes)
# We simulate a neighborhood in New York City
item_pool = ["Apple", "Bread", "Laptop Charger", "Drill", "Winter Coat", "Milk", "Chairs"]

agents = []
for i in range(30):
    uid = str(uuid.uuid4())[:8]
    loc = generate_random_location()
    agent = UserNode(id=uid, name=f"Agent_{uid}", location=loc)
    
    # Randomly assign Supply (Inventory) OR Demand (Wishlist)
    if random.random() > 0.5:
        # Supplier
        item_name = random.choice(item_pool)
        res = Resource(
            id=str(uuid.uuid4()),
            name=item_name,
            type=ResourceType.FOOD if item_name in ["Apple", "Bread", "Milk"] else ResourceType.MATERIAL,
            expiry_hours=random.randint(2, 72),
            owner_id=uid
        )
        agent.inventory.append(res)
        role = "Supplier"
    else:
        # Receiver
        agent.wishlist.append(random.choice(item_pool))
        role = "Receiver"
        
    router.add_node(agent)
    agents.append((agent, role))

print(f"Generated {len(agents)} agents scattered across 5km radius.")

# 2. Run the Algorithm
print("Running Proximal Decay Matching Algorithm...")
matches = router.find_matches()

print(f"\nFound {len(matches)} Optimal Matches:")
print("-" * 60)
print(f"{'SUPPLIER':<15} | {'RECEIVER':<15} | {'ITEM':<15} | {'DIST (km)':<10} | {'SCORE':<5}")
print("-" * 60)

for m in matches[:10]: # Show top 10
    print(f"{m.supplier.name:<15} | {m.receiver.name:<15} | {m.resource.name:<15} | {m.distance_km:.2f}       | {m.match_score:.2f}")

if not matches:
    print("No matches found. (Try running again for different random seed)")

# 3. Visualization (Scatter Plot)
# We will plot Suppliers (Red), Receivers (Blue), and Match Links (Green lines)
print("\nGenerating Grid Map...")

lats_sup, lons_sup = [], []
lats_rec, lons_rec = [], []

for agent, role in agents:
    if role == "Supplier":
        lats_sup.append(agent.location.lat)
        lons_sup.append(agent.location.lon)
    else:
        lats_rec.append(agent.location.lat)
        lons_rec.append(agent.location.lon)

plt.figure(figsize=(10, 8))
plt.style.use('ggplot')

# Plot Nodes
plt.scatter(lons_sup, lats_sup, c='red', s=100, label='Supplier (Excess)', alpha=0.6)
plt.scatter(lons_rec, lats_rec, c='blue', s=100, label='Receiver (Need)', alpha=0.6)

# Plot Connections (The Matches)
match_count = 0
for m in matches:
    # Only plot high quality matches to avoid clutter
    if m.match_score > 0.5:
        plt.plot([m.supplier.location.lon, m.receiver.location.lon],
                 [m.supplier.location.lat, m.receiver.location.lat],
                 'g--', alpha=0.5, linewidth=1)
        match_count += 1

plt.title(f"Symbiont Network: {match_count} Optimized Resource Flows")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

print(f"Visualization complete. {match_count} active trade routes plotted.")
plt.show()