import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict, Union

import ai2thor
import numpy as np
from ai2thor.controller import Controller
from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    ADDITIONAL_ARM_ARGS,
    DONE,
    DROP,
    LOOK_DOWN,
    LOOK_UP,
    MOVE_AHEAD,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_X_M,
    MOVE_ARM_X_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Z_M,
    MOVE_ARM_Z_P,
    MOVE_BACK,
    PICKUP,
    ROTATE_ELBOW_M,
    ROTATE_ELBOW_P,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    ROTATE_WRIST_PITCH_M,
    ROTATE_WRIST_PITCH_P,
    ROTATE_WRIST_ROLL_M,
    ROTATE_WRIST_ROLL_P,
    ROTATE_WRIST_YAW_M,
    ROTATE_WRIST_YAW_P,
)
from typing_extensions import NotRequired


def arm_agent_step(
    controller: Controller,
    action: str,
    move_agent_dist: Optional[float] = None,
    rotate_degrees: Optional[float] = None,
    move_base_dist: float = 0.05,
    move_arm_dist: float = 0.05,
    rotate_wrist_degrees: float = 15,
    rotate_elbow_degrees: float = 15,
    simplify_physics: bool = False,
    render_image: bool = True,
) -> ai2thor.server.Event:
    """Take a step in the ai2thor environment."""
    last_frame: Optional[np.ndarray] = None
    if not render_image:
        last_frame = controller.last_event.current_frame

    action_dict: Dict[str, Any] = {}
    if simplify_physics:
        action_dict["simplifyPhysics"] = True

    if action == PICKUP:
        action_dict["action"] = "PickupObject"

    elif action == DROP:
        action_dict["action"] = "ReleaseObject"

    elif action in [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT]:
        for key, value in ADDITIONAL_ARM_ARGS.items():
            action_dict[key] = value
        if action == MOVE_AHEAD:
            action_dict["action"] = "MoveAhead"
            action_dict["moveMagnitude"] = move_agent_dist

        elif action == MOVE_BACK:
            action_dict["action"] = "MoveBack"
            action_dict["moveMagnitude"] = move_agent_dist

        elif action == ROTATE_RIGHT:
            action_dict["action"] = "RotateRight"
            action_dict["degrees"] = rotate_degrees

        elif action == ROTATE_LEFT:
            action_dict["action"] = "RotateLeft"
            action_dict["degrees"] = rotate_degrees

    elif "MoveArm" in action:
        for key, value in ADDITIONAL_ARM_ARGS.items():
            action_dict[key] = value
        if "MoveArmHeight" in action:
            action_dict["action"] = "MoveArmBaseUp"

            if action == "MoveArmHeightP":
                action_dict["distance"] = move_base_dist
            if action == "MoveArmHeightM":
                action_dict["distance"] = -move_base_dist
        else:
            action_dict["action"] = "MoveArmRelative"
            offset = {"x": 0.0, "y": 0.0, "z": 0.0}

            axis, plus_or_minus = list(action[-2:].lower())
            assert axis in ["x", "y", "z"] and plus_or_minus in ["p", "m"]

            offset[axis] = (1 if plus_or_minus == "p" else -1) * move_arm_dist

            action_dict["offset"] = offset

    elif action.startswith("RotateArmWrist"):
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        for key, value in copy_additions.items():
            action_dict[key] = value

        action_dict["action"] = "RotateWristRelative"

        tmp = action.replace("RotateArmWrist", "").lower()
        axis, plus_or_minus = tmp[:-1], tmp[-1]

        assert axis in ["pitch", "yaw", "roll"], plus_or_minus in ["p", "m"]

        action_dict[axis] = (
            1 if plus_or_minus == "p" else -1
        ) * rotate_wrist_degrees

    elif action.startswith("RotateArmElbow"):
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        for key, value in copy_additions.items():
            action_dict[key] = value

        action_dict["action"] = "RotateElbowRelative"

        plus_or_minus = action[-1].lower()

        assert plus_or_minus in ["p", "m"]

        action_dict["degrees"] = (
            1 if plus_or_minus == "p" else -1
        ) * rotate_elbow_degrees

    elif action in [LOOK_UP, LOOK_DOWN]:
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        for key, value in copy_additions.items():
            action_dict[key] = value
        if action == LOOK_UP:
            action_dict["action"] = LOOK_UP
        elif action == LOOK_DOWN:
            action_dict["action"] = LOOK_DOWN
    else:
        raise NotImplementedError

    sr = controller.step(action_dict)

    if not render_image:
        assert last_frame is not None
        controller.last_event.frame = last_frame

    return sr
