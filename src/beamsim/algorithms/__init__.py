from beamsim.algorithms.angular_prediction import AngularPrediction
from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.ci import ContextInformation
from beamsim.algorithms.dl_predictor import DLPredictor
from beamsim.algorithms.exhaustive import Exhaustive
from beamsim.algorithms.hbm import HBM
from beamsim.algorithms.mcmd import MCMD
from beamsim.algorithms.nns import NNS
from beamsim.algorithms.nns_bs_sequential import NNSBSSequential
from beamsim.algorithms.omp_compressive import OMPCompressive
from beamsim.algorithms.perfect import Perfect
from beamsim.algorithms.tabu import Tabu
from beamsim.algorithms.thompson import ThompsonGaussian
from beamsim.algorithms.ucb1 import UCB1

ALL_ALGORITHMS = {
    "exhaustive": Exhaustive,
    "nns": NNS,
    "nns_bs_sequential": NNSBSSequential,
    "tabu": Tabu,
    "angular_prediction": AngularPrediction,
    "ci": ContextInformation,
    "mcmd": MCMD,
    "perfect": Perfect,
    "ucb1": UCB1,
    "thompson": ThompsonGaussian,
    "hbm": HBM,
    "omp_compressive": OMPCompressive,
    "dl_predictor": DLPredictor,
}

__all__ = [
    "ALL_ALGORITHMS",
    "HBM",
    "MCMD",
    "NNS",
    "UCB1",
    "Algorithm",
    "AngularPrediction",
    "ContextInformation",
    "DLPredictor",
    "Exhaustive",
    "NNSBSSequential",
    "OMPCompressive",
    "Perfect",
    "Tabu",
    "ThompsonGaussian",
]
