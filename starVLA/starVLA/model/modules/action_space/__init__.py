from starVLA.model.modules.action_space.motion_token_adapter import MotionTokenActionAdapter
from starVLA.model.modules.action_space.wbc_adapter import WBCActionAdapter
from starVLA.model.modules.action_space.humanoid_motion_hands_adapter import HumanoidMotionHandsActionAdapter
from starVLA.model.modules.action_space.identity_adapter import IdentityActionAdapter


ACTION_ADAPTERS = {
    "identity": IdentityActionAdapter,
    "motion_token": MotionTokenActionAdapter,
    "wbc": WBCActionAdapter,
    "humanoid_motion_hands": HumanoidMotionHandsActionAdapter,
}


def build_action_adapter(action_type: str, action_dim: int, control_dim: int, hidden_dim: int):
    if action_type not in ACTION_ADAPTERS:
        raise ValueError(f"Unsupported action_type `{action_type}`. Expected one of {list(ACTION_ADAPTERS)}.")
    return ACTION_ADAPTERS[action_type](
        action_dim=action_dim,
        control_dim=control_dim,
        hidden_dim=hidden_dim,
    )
