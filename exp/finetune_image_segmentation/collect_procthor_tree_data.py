from termios import TIOCM_DSR
import warnings
import random
import json
import sys
import cv2
import tqdm
import os

from ai2thor.controller import Controller
import ai2thor
from allenact_plugins.ithor_plugin.ithor_util import (
    horizontal_to_vertical_fov,
)
from data_collection_utils import (
    pos_to_id,
    teleport_to,
    take_step,
    find_shortes_terminal_path
)
from boolset.tasks_and_samplers import (
    AgentPose,
    HouseAugmenter,
    ProcTHORDataset,
    Vector3,
)

import prior
from allenact.embodiedai.sensors.vision_sensors import DepthSensor

# configurations
THOR_COMMIT_ID = "627521a56508f212749a779d358ab17df10e0d8e"
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
HORIZONTAL_FIELD_OF_VIEW = 100
STEP_SIZE = 0.2
ROTATION_DEGREES = 45.0
VISIBILITY_DISTANCE = 1.5

dataset = prior.load_dataset("procthor-10k")
train_scenes = dataset["train"]
val_scenes = dataset["val"]
SCENES = ProcTHORDataset(
    train_scenes + val_scenes
)

# define dataset collection parameters
TRAIN = False if sys.argv[1] == "test" else True
NUM_ANCHORS = 1000 if TRAIN else 100
NUM_STEPS = 4
ROT_ANGLE = 30
ACTIONS = ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]
IMG_ROOT = '../data/interactron/train' if TRAIN else '../data/interactron/test'
ANN_PATH = '../data/interactron/annotations/interactron_v1_train.json' if TRAIN \
    else '../data/interactron/annotations/interactron_v1_test.json'
CTRL = Controller(
    commit_id=THOR_COMMIT_ID,
    server_class=ai2thor.fifo_server.FifoServer,
    include_private_scenes=False,
    fastActionEmit=True,
    snapToGrid=False,
    autoSimulation=False,
    autoSyncTransforms=True,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    fieldOfView=horizontal_to_vertical_fov(
        horizontal_fov_in_degrees=HORIZONTAL_FIELD_OF_VIEW,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    ),
    makeAgentsVisible=True,
    visibilityScheme="Distance",
    agentMode="arm",
    rotateStepDegrees=ROTATION_DEGREES,
    visibilityDistance=VISIBILITY_DISTANCE,
    gridSize=STEP_SIZE,
    useMassThreshold=True,
    massThreshold=10,
    platform="CloudRendering"
)


def rollout_rec(root_state, state_table, d=0):
    # if we reached the end of the rollout return empty dict of next steps
    if d >= NUM_STEPS:
        return {}
    # otherwise generate the data for the steps we can take from this state
    if pos_to_id(root_state) in state_table and len(state_table[pos_to_id(root_state)]['actions']) > 0:
        steps = state_table[pos_to_id(root_state)]['actions']
    else:
        steps = {}
        for action in ACTIONS:
            new_state = take_step(CTRL, root_state, action)
            steps[action] = pos_to_id(new_state)
            if pos_to_id(new_state) not in state_table:
                state_table[pos_to_id(new_state)] = new_state
                state_table[pos_to_id(new_state)]["actions"] = {}
    for state_name in steps.values():
        state = state_table[state_name]
        next_steps = rollout_rec(state, state_table, d=d+1)
        if len(state_table[pos_to_id(state)]["actions"]) == 0:
            state_table[pos_to_id(state)]["actions"] = next_steps
    return steps


def collect_dataset(procthor_dataset: ProcTHORDataset):
    if NUM_ANCHORS % len(SCENES) != 0:
        warnings.warn("The number of anchors specified (%d) is not integer divisible by the number"
                      "of scenes (%d). To maintain dataset balance the number of anchors will"
                      "be reduced to (%d)" % (NUM_ANCHORS, len(SCENES), NUM_ANCHORS // len(SCENES)))
    samples_per_scene = NUM_ANCHORS // len(SCENES)
    annotations = {
        "data": [],
        "metadata": {
            "actions": ACTIONS,
            "max_steps": NUM_STEPS,
            "rotation_angle": ROT_ANGLE,
            "scenes": SCENES
        }
    }
    for scene_id in range(len(procthor_dataset)):
        currect_house = procthor_dataset.initialize_house(
            controller=CTRL,
            index=scene_id
        )
        rotations = [{"x": 0.0, "y": float(theta), "z": 0.0} for theta in range(0, 360, ROT_ANGLE)]
        horizons = [0]
        standing = [True]
        for i in range(samples_per_scene):
            # try generating data until you have a complete validated tree
            validated_root = False
            while not validated_root:
                # randomize scene
                CTRL.reset(scene=scene)
                # find a valid root
                num_valid_objects = 0
                while num_valid_objects < 3:
                    # select random starting rotation, horizon and standing state
                    p = random.choice(CTRL.step(action="GetReachablePositions").metadata["actionReturn"])
                    r = random.choice(rotations)
                    h = random.choice(horizons)
                    s = random.choice(standing)
                    root = teleport_to(CTRL, {"pos": p, "rot": r, "hor": h, "stand": s})
                    num_valid_objects = len(root["detections"])
                # generate data from this root
                root_id = pos_to_id(root)
                state_table = {root_id: root}
                state_table[root_id]["actions"] = {}
                state_table[root_id]["actions"] = rollout_rec(root, state_table)
                # check to see that all paths in this tree are at least as long as out max depth
                validated_root = find_shortes_terminal_path(root_id, state_table, max_depth=NUM_STEPS) >= NUM_STEPS
            # save data
            scene_name = "{}_{:05d}".format(scene, i)
            os.makedirs("{}/{}".format(IMG_ROOT, scene_name), exist_ok=True)
            for state, values in state_table.items():
                cv2.imwrite("{}/{}/{}.jpg".format(IMG_ROOT, scene_name, pos_to_id(values)), values["img"])
            # reformat state table into dataset format
            light_state_table = {}
            for name, fields in state_table.items():
                light_state_table[name] = {
                    "pos": fields["pos"],
                    "rot": fields["rot"],
                    "hor": fields["hor"],
                    "stand": fields["stand"],
                    "detections": fields["detections"],
                    "actions": fields["actions"]
                }
            annotations["data"].append({
                "scene_name": scene_name,
                "state_table": light_state_table,
                "root": root_id
            })
    # save annotations
    with open(ANN_PATH, 'w') as f:
        json.dump(annotations, f)

    # close env
    CTRL.stop()


if __name__ == '__main__':
    collect_dataset()
