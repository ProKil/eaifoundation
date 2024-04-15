import json
import os
import random
import sys
import warnings
from termios import TIOCM_DSR

import ai2thor
import cv2
import tqdm
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_util import (
    horizontal_to_vertical_fov,
)
from graph_to_scene import graph_to_scene
from relationship_graph.graphs import build_relationship_graph

from boolset.tasks_and_samplers import (
    AgentPose,
    HouseAugmenter,
    ProcTHORDataset,
    Vector3,
)

THOR_COMMIT_ID = "345a5fc046f25305c66367a484c9ae297107c877"
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
HORIZONTAL_FIELD_OF_VIEW = 100
STEP_SIZE = 0.2
ROTATION_DEGREES = 45.0
VISIBILITY_DISTANCE = 1.5

from allenact.embodiedai.sensors.vision_sensors import DepthSensor

import prior

if __name__ == "__main__":
    scenes = [
        f"ArchitecTHOR-{split}-0{idx}"
        for split in ["Val", "Test"]
        for idx in range(5)
    ]
    scenes_to_store = {"info": dict(split="train"), "scenes": list()}
    for scene in tqdm.tqdm(scenes):
        controller = Controller(
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
            platform="CloudRendering",
            branch="nanna-grasp-force",
            scene=scene,
            renderSemanticSegmentation=True,
        )

        metadata = controller.step("AdvancePhysicsStep", simSeconds=2).metadata

        relationship_graph = build_relationship_graph(
            controller=controller, rooms=[]
        )

        scenes_to_store["scenes"].append(graph_to_scene(relationship_graph))

    json.dump(scenes_to_store, open(f"architecthor.json", "w"), indent=4)
