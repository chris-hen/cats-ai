from dataclasses import dataclass
from cats.game_state import CLAIM_TYPES


@dataclass(frozen=True)
class Action:
    action_type: str


@dataclass(frozen=True, init=False)
class MakeClaimAction(Action):
    amount: int
    cat_type: str
    proof_indices: tuple[int, ...]
    swap_indices: tuple[int, ...]

    def __init__(self, amount: int, cat_type: str, proof_indices=None, swap_indices=None):
        if cat_type not in CLAIM_TYPES:
            raise ValueError(f"Invalid claim type: {cat_type}")

        if proof_indices is None:
            proof_indices = ()

        if swap_indices is None:
            swap_indices = ()

        object.__setattr__(self, "action_type", "make_claim")
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "cat_type", cat_type)
        object.__setattr__(self, "proof_indices", tuple(proof_indices))
        object.__setattr__(self, "swap_indices", tuple(swap_indices))

    def __repr__(self):
        symbols = {
            "alive": "A",
            "dead": "D",
            "empty": "E",
        }

        parts = [f"{self.amount}{symbols[self.cat_type]}"]

        if self.proof_indices:
            parts.append(f"proof={self.proof_indices}")

        if self.swap_indices:
            parts.append(f"swap={self.swap_indices}")

        return f"Claim({', '.join(parts)})"


@dataclass(frozen=True, init=False)
class DoubtAction(Action):
    def __init__(self):
        object.__setattr__(self, "action_type", "doubt")

    def __repr__(self):
        return "Doubt()"


@dataclass(frozen=True, init=False)
class UsePhysicatAction(Action):
    target_index: int | None

    def __init__(self, target_index=None):
        object.__setattr__(self, "action_type", "use_physicat")
        object.__setattr__(self, "target_index", target_index)

    def __repr__(self):
        if self.target_index is None:
            return "UsePhysicat()"
        return f"UsePhysicat(target_index={self.target_index})"