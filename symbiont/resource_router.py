import math
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# --- Constants ---
EARTH_RADIUS_KM = 6371.0

class ResourceType(Enum):
    FOOD = "Food"
    MATERIAL = "Material" 
    SKILL = "Skill"
    ENERGY = "Energy"

@dataclass
class Location:
    lat: float
    lon: float

    def distance_to(self, other: 'Location') -> float:
        """
        Haversine formula to calculate distance in Kilometers between two points.
        """
        dlat = math.radians(other.lat - self.lat)
        dlon = math.radians(other.lon - self.lon)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(self.lat)) * math.cos(math.radians(other.lat)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_KM * c

@dataclass
class Resource:
    id: str
    name: str
    type: ResourceType
    expiry_hours: float  # Hours until worthless
    owner_id: str
    quantity: int = 1
    
    def current_value(self) -> float:
        """Simple linear decay function."""
        # Value drops as expiry approaches (simulation)
        return max(0.1, self.expiry_hours / 24.0)

@dataclass
class UserNode:
    id: str
    name: str
    location: Location
    inventory: List[Resource] = field(default_factory=list)
    wishlist: List[str] = field(default_factory=list) # List of item names they need
    reputation_score: float = 100.0

class MatchResult:
    def __init__(self, supplier: UserNode, receiver: UserNode, resource: Resource, distance: float):
        self.supplier = supplier
        self.receiver = receiver
        self.resource = resource
        self.distance_km = distance
        self.saved_carbon_kg = self._calculate_carbon_savings()
        self.match_score = self._calculate_score()

    def _calculate_carbon_savings(self) -> float:
        # Heuristic: 1kg of wasted food = ~2.5kg CO2e
        # Subtract transport emissions (approx 0.2kg CO2e per km driven)
        base_saving = 2.5 * self.resource.quantity
        transport_cost = 0.2 * self.distance_km
        return max(0, base_saving - transport_cost)

    def _calculate_score(self) -> float:
        """
        The Secret Sauce: Proximal Decay Algorithm.
        High Score = Close distance + High Urgency (low expiry)
        """
        distance_factor = 1 / (1 + self.distance_km) # Higher if closer
        urgency_factor = 1 / (1 + self.resource.expiry_hours) # Higher if expiring soon
        return (distance_factor * 0.7) + (urgency_factor * 0.3)

class ResourceRouter:
    def __init__(self):
        self.nodes: Dict[str, UserNode] = {}
        self.max_travel_km = 15.0

    def add_node(self, node: UserNode):
        self.nodes[node.id] = node

    def find_matches(self) -> List[MatchResult]:
        """
        Scans the entire grid to match Inventory (Supply) with Wishlists (Demand).
        Returns a sorted list of optimal matches.
        """
        potential_matches = []

        # O(N^2) naive approach - satisfactory for city-block scale simulations
        # In production, use a QuadTree or GeoHash index.
        for supplier_id, supplier in self.nodes.items():
            for resource in supplier.inventory:
                
                # Look for a receiver
                for receiver_id, receiver in self.nodes.items():
                    if supplier_id == receiver_id:
                        continue # Can't trade with self

                    # Check if receiver wants this type of item
                    # (Simple keyword matching for MVP)
                    if any(need.lower() in resource.name.lower() for need in receiver.wishlist):
                        
                        dist = supplier.location.distance_to(receiver.location)
                        
                        if dist <= self.max_travel_km:
                            match = MatchResult(supplier, receiver, resource, dist)
                            potential_matches.append(match)

        # Sort by best match score (Highest first)
        return sorted(potential_matches, key=lambda x: x.match_score, reverse=True)