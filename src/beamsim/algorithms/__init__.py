from beamsim.algorithms.agemx import AgeMx
from beamsim.algorithms.angular_prediction import AngularPrediction
from beamsim.algorithms.bai import BAIPureExploration
from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.ci import ContextInformation
from beamsim.algorithms.ci_mbs import ContextInformationMBS
from beamsim.algorithms.dl_lstm_predictor import DLLSTMPredictor
from beamsim.algorithms.dl_predictor import DLPredictor
from beamsim.algorithms.ekf_tracker import EKFTracker
from beamsim.algorithms.exhaustive import Exhaustive
from beamsim.algorithms.hbm import HBM
from beamsim.algorithms.mamba import MAMBA
from beamsim.algorithms.mcmd import MCMD
from beamsim.algorithms.nns import NNS
from beamsim.algorithms.nns_bs_sequential import NNSBSSequential
from beamsim.algorithms.nns_tabu import NNSTabu
from beamsim.algorithms.omp_compressive import OMPCompressive
from beamsim.algorithms.perfect import Perfect
from beamsim.algorithms.position_mab import PositionMAB
from beamsim.algorithms.random_search import RandomSearch
from beamsim.algorithms.tabu import Tabu
from beamsim.algorithms.thompson import ThompsonGaussian
from beamsim.algorithms.ucb1 import UCB1

ALL_ALGORITHMS = {
    "exhaustive": Exhaustive,
    "nns": NNS,
    "nns_bs_sequential": NNSBSSequential,
    "nns_tabu": NNSTabu,
    "tabu": Tabu,
    "angular_prediction": AngularPrediction,
    "ci": ContextInformation,
    "ci_mbs": ContextInformationMBS,
    "agemx": AgeMx,
    "random": RandomSearch,
    "mcmd": MCMD,
    "perfect": Perfect,
    "ucb1": UCB1,
    "thompson": ThompsonGaussian,
    "hbm": HBM,
    "omp_compressive": OMPCompressive,
    "dl_predictor": DLPredictor,
    "dl_lstm_predictor": DLLSTMPredictor,
    "mamba": MAMBA,
    "ekf_tracker": EKFTracker,
    "position_mab": PositionMAB,
    "bai_pure_explore": BAIPureExploration,
}

__all__ = [
    "ALL_ALGORITHMS",
    "HBM",
    "MAMBA",
    "MCMD",
    "NNS",
    "UCB1",
    "AgeMx",
    "Algorithm",
    "AngularPrediction",
    "BAIPureExploration",
    "ContextInformation",
    "ContextInformationMBS",
    "DLLSTMPredictor",
    "DLPredictor",
    "EKFTracker",
    "Exhaustive",
    "NNSBSSequential",
    "NNSTabu",
    "OMPCompressive",
    "Perfect",
    "PositionMAB",
    "RandomSearch",
    "Tabu",
    "ThompsonGaussian",
]
