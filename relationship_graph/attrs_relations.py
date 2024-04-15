import enum
from typing import Any, Dict, Optional

from allenact_plugins.ithor_plugin.ithor_environment import (
    IThorEnvironment,
)
from relationship_graph.constants import (
    FULLY_OPEN_THRES,
    MOVEMENT_THRES,
    PARTIALLY_OPEN_THRES,
    ROTATION_THRES,
)


class RelationshipEnum(enum.Enum):
    # Agent-Object
    SEES = enum.auto()
    HOLDS = enum.auto()
    TOUCHES = enum.auto()

    # Object-Object
    ON = enum.auto()
    NEAR = enum.auto()
    ADJACENT = enum.auto()

    # Agent conditioned Object-Object
    RIGHT = enum.auto()
    LEFT = enum.auto()
    ABOVE = enum.auto()
    BELOW = enum.auto()
    FRONT = enum.auto()
    BEHIND = enum.auto()

    # Object-Object, Room-Object, Room-Agent
    CONTAINS = enum.auto()
    HEAVIERTHAN = enum.auto()
    LIGHTERTHAN = enum.auto()
    LARGERTHAN = enum.auto()
    SMALLERTHAN = enum.auto()
    LONGERTHAN = enum.auto()
    SHORTERTHAN = enum.auto()


class EntityPairEnum(enum.Enum):
    OBJ_OBJ = enum.auto()
    AGENT_OBJ = enum.auto()
    ROOM_OBJ = enum.auto()
    ROOM_AGENT = enum.auto()


_ = RelationshipEnum
ENTITY_PAIR_TO_RELATIONSHIPS = {
    EntityPairEnum.OBJ_OBJ: [
        _.ON,
        _.NEAR,
        _.ADJACENT,
        _.CONTAINS,
        _.RIGHT,
        _.LEFT,
        _.ABOVE,
        _.BELOW,
        _.FRONT,
        _.BEHIND,
    ],
    EntityPairEnum.AGENT_OBJ: [
        _.SEES,
        _.HOLDS,
        _.TOUCHES,
    ],
    EntityPairEnum.ROOM_OBJ: [_.CONTAINS],
    EntityPairEnum.ROOM_AGENT: [_.CONTAINS],
}


class AttributeEnum(enum.Enum):
    pass


class NodeTypeEnum(enum.Enum):
    AGENT = enum.auto()
    ROOM = enum.auto()
    OBJECT = enum.auto()


class OpenableEnum(enum.Enum):
    CLOSED = enum.auto()
    PARTIALLY_OPEN = enum.auto()
    OPEN = enum.auto()

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> Optional["OpenableEnum"]:
        if not obj["openable"]:
            return None

        if obj["openness"] <= PARTIALLY_OPEN_THRES:
            return cls.CLOSED
        elif obj["openness"] <= FULLY_OPEN_THRES:
            return cls.PARTIALLY_OPEN
        else:
            return cls.CLOSED


class BrokenEnum(enum.Enum):
    BROKEN = enum.auto()
    NOT_BROKEN = enum.auto()

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> Optional["BrokenEnum"]:
        if not obj["breakable"]:
            return None

        return cls.BROKEN if obj["isBroken"] else cls.NOT_BROKEN


class MovementEnum(enum.Enum):
    MOVED = enum.auto()
    NOT_MOVED = enum.auto()

    @classmethod
    def from_dict(
        cls, obj: Dict[str, Any], init_obj: Optional[Dict[str, Any]]
    ) -> Optional["MovementEnum"]:
        if not obj["moveable"] or init_obj is None:
            return None

        if (
            IThorEnvironment.position_dist(
                obj["position"], init_obj["position"]
            )
            >= MOVEMENT_THRES
            or IThorEnvironment.rotation_dist(
                obj["rotation"], init_obj["rotation"]
            )
            >= ROTATION_THRES
        ):
            return cls.MOVED
        else:
            return cls.NOT_MOVED
