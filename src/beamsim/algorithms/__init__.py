from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.exhaustive import Exhaustive
from beamsim.algorithms.nns import NNS
from beamsim.algorithms.tabu import Tabu
from beamsim.algorithms.angular_prediction import AngularPrediction
from beamsim.algorithms.ci import ContextInformation
from beamsim.algorithms.mcmd import MCMD

ALL_ALGORITHMS = {
    "exhaustive": Exhaustive,
    "nns": NNS,
    "tabu": Tabu,
    "angular_prediction": AngularPrediction,
    "ci": ContextInformation,
    "mcmd": MCMD,
}

__all__ = [
    "Algorithm",
    "Exhaustive",
    "NNS",
    "Tabu",
    "AngularPrediction",
    "ContextInformation",
    "MCMD",
    "ALL_ALGORITHMS",
]
