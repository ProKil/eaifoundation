import json
from typing import Any, Dict, List, Set, Tuple


def load_categories() -> Tuple[Set[str], Set[str], Set[str]]:
    d = {
        "cube": [
            "TeddyBear",
            "Boots",
            "Lettuce",
            "Tomato",
            "Safe",
            "GarbageCan",
            "Kettle",
            "Ottoman",
            "LaundryHamper",
            "CoffeeMachine",
            "Apple",
            "Potato",
            "BasketBall",
            "Egg",
            "Microwave",
            "Watch",
            "Mug",
            "TissueBox",
            "Toaster",
        ],
        "flat": [
            "Desktop",
            "Plate",
            "KeyChain",
            "Book",
            "DishSponge",
            "Pan",
            "SoapBar",
            "CreditCard",
            "Painting",
            "CellPhone",
            "RemoteControl",
            "CD",
        ],
        "long": [
            "Dumbbell",
            "PaperTowelRoll",
            "Plunger",
            "Candle",
            "Ladle",
            "FloorLamp",
            "WineBottle",
            "Fork",
            "VacuumCleaner",
            "Bottle",
        ],
    }
    return set(d["cube"]), set(d["flat"]), set(d["long"])


def get_size(obj: Dict[str, Any]) -> List[float]:
    x = obj["axisAlignedBoundingBox"]["size"]["x"]
    y = obj["axisAlignedBoundingBox"]["size"]["y"]
    z = obj["axisAlignedBoundingBox"]["size"]["z"]
    return sorted([x, y, z])


if __name__ == "__main__":
    print(load_categories())
