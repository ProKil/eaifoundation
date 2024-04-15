import pickle
import unittest

import datasets
from ai2thor.controller import Controller

from question_answering.questions import CountingQuestion
from question_answering.questions.questions import (
    AdjacentRelationshipQuestion,
    ExistenceQuestion,
    OnRelationshipQuestion,
)
from question_answering.solvers import solve
from relationship_graph.graphs import build_relationship_graph


class TestQuestionSolvers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_built = False

    def get_graph(self):
        if not self.graph_built:
            if "houses_dict" not in locals():
                houses_dict = datasets.load_dataset(
                    "allenai/houses", use_auth_token=True
                )

            if "c" not in locals():
                c = Controller(
                    branch="bbox_distance",
                    scene="Procedural",
                    agentMode="arm",
                    fastActionEmit=True,
                )

            c.reset()

            train_houses = houses_dict["train"]

            # house = pickle.loads(train_houses[10]["house"])
            house = pickle.loads(train_houses[1]["house"])
            c.step("CreateHouse", house=house, raise_for_failure=True)

            c.step("TeleportFull", **house["metadata"]["agent"], raise_for_failure=True)

            for _ in range(6):
                c.step("MoveAhead")

            metadata = c.step("AdvancePhysicsStep", simSeconds=2).metadata

            self.graph = build_relationship_graph(
                controller=c,
                object_ids_subset=set(
                    o["objectId"]
                    for o in metadata["objects"]
                    if o["objectType"] not in ["Floor", "Wall", "wall"]
                    and o["objectId"].split("|")[0] not in ["wall", "room"]
                ),
                rooms=house["rooms"],
            )

            self.graph_built = True
            return self.graph

        else:
            return self.graph

    def test_solve_counting_question(self):
        graph = self.get_graph()

        question1 = CountingQuestion()
        question1.obj_ids = ("Kitchen", "Drawer")
        self.assertEqual(solve(question1, graph), 0)  # 0 drawers in kitchen

        question2 = CountingQuestion()
        question2.obj_ids = ("Kitchen", "Painting")
        self.assertEqual(solve(question2, graph), 2)  # 2 paintings in kitchen

        question3 = CountingQuestion()
        question3.obj_ids = ("LivingRoom", "DiningTable")
        self.assertEqual(
            solve(question3, graph), 5
        )  # 5 dining tables in living rooms

    def test_solve_existence_question(self):
        graph = self.get_graph()

        question1 = ExistenceQuestion()
        question1.obj_ids = ("Bathroom", "Window")
        self.assertEqual(solve(question1, graph), False)

        question2 = ExistenceQuestion()
        question2.obj_ids = ("Kitchen", "GarbageBag")
        self.assertEqual(solve(question2, graph), True)

        question3 = ExistenceQuestion()
        question3.obj_ids = ("LivingRoom", "DiningTable")
        self.assertEqual(solve(question3, graph), True)

    def test_solve_relationship_question(self):
        graph = self.get_graph()

        question1 = OnRelationshipQuestion()
        question1.obj_ids = ("TeddyBear", "Bed")
        self.assertEqual(solve(question1, graph), True)

        question2 = AdjacentRelationshipQuestion()
        question2.obj_ids = ("Dresser", "Toilet")
        self.assertEqual(solve(question2, graph), True)

        question1 = OnRelationshipQuestion()
        question1.obj_ids = ("Watch", "DeskLamp")
        self.assertEqual(solve(question1, graph), False)

        question2 = AdjacentRelationshipQuestion()
        question2.obj_ids = ("Window", "Sink")
        self.assertEqual(solve(question2, graph), False)
