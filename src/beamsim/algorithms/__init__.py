from beamsim.algorithms.angular_prediction import AngularPrediction
from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.ci import ContextInformation
from beamsim.algorithms.exhaustive import Exhaustive
from beamsim.algorithms.mcmd import MCMD
from beamsim.algorithms.nns import NNS
from beamsim.algorithms.nns_bs_sequential import NNSBSSequential
from beamsim.algorithms.perfect import Perfect
from beamsim.algorithms.tabu import Tabu

ALL_ALGORITHMS = {
    "exhaustive": Exhaustive,
    "nns": NNS,
    "nns_bs_sequential": NNSBSSequential,
    "tabu": Tabu,
    "angular_prediction": AngularPrediction,
    "ci": ContextInformation,
    "mcmd": MCMD,
    "perfect": Perfect,
}

__all__ = [
    "ALL_ALGORITHMS",
    "MCMD",
    "NNS",
    "Algorithm",
    "AngularPrediction",
    "ContextInformation",
    "Exhaustive",
    "NNSBSSequential",
    "Perfect",
    "Tabu",
]
