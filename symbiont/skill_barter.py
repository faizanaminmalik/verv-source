import uuid
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# --- Data Structures ---
SKILL_CATALOG = [
    "Python Coding", "Spanish", "Gardening", "Plumbing", 
    "Guitar", "Cooking", "Financial Planning", "Yoga", 
    "Graphic Design", "Public Speaking"
]

@dataclass
class SkillUser:
    id: str
    name: str
    offering: str  # The skill they have
    needing: str   # The skill they want
    reputation: float = 5.0

@dataclass
class SwapMatch:
    user_a: SkillUser
    user_b: SkillUser
    skill_a_to_b: str
    skill_b_to_a: str
    match_quality: float # 0.0 to 1.0

class SkillMatcher:
    def __init__(self):
        self.users: Dict[str, SkillUser] = {}

    def add_user(self, user: SkillUser):
        self.users[user.id] = user

    def find_direct_swaps(self) -> List[SwapMatch]:
        """
        Finds perfect P2P matches:
        User A offers X, needs Y
        User B offers Y, needs X
        """
        matches = []
        processed_ids = set()

        users_list = list(self.users.values())

        # O(N^2) matching for MVP
        for i in range(len(users_list)):
            user_a = users_list[i]
            if user_a.id in processed_ids:
                continue

            for j in range(i + 1, len(users_list)):
                user_b = users_list[j]
                if user_b.id in processed_ids:
                    continue

                # Check for perfect swap
                # A offers what B needs AND B offers what A needs
                if (user_a.offering == user_b.needing) and \
                   (user_b.offering == user_a.needing):
                    
                    match = SwapMatch(
                        user_a=user_a,
                        user_b=user_b,
                        skill_a_to_b=user_a.offering,
                        skill_b_to_a=user_b.offering,
                        match_quality= (user_a.reputation + user_b.reputation) / 10.0
                    )
                    matches.append(match)
                    processed_ids.add(user_a.id)
                    processed_ids.add(user_b.id)
                    break # Stop looking for matches for A once one is found

        return matches