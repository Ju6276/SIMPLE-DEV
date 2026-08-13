from starVLA.model.modules.action_space.action_adapter_base import ActionAdapterBase


class HumanoidMotionHandsActionAdapter(ActionAdapterBase):
    """
    Raw action layout for merged_dataset_001:
      motion_token (64) + left_hand_joints (7) + right_hand_joints (7) = 78
    """

    pass
