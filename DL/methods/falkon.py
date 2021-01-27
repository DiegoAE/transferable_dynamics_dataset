"""
Learning using the kernel method Falkon.
"""

import argparse
import numpy as np

import torch
from falkon import Falkon, kernels
from falkon.options import FalkonOptions

from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData


class FalkonDynLearner(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon, difference_learning,
            averaging, streaming, settings=None):
        super().__init__(history_length, prediction_horizon,
                difference_learning, averaging=averaging, streaming=streaming)
        self.models_ = []
        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=False)

        # TODO: The penalty, sigma and M are hardcoded for now.
        for i in range(self.observation_dimension):
            kernel = kernels.GaussianKernel(sigma=5)
            flk = Falkon(kernel=kernel, penalty=1e-4, M=1000, options=opt)
            self.models_.append(flk)

    def _learn(self, training_inputs, training_targets):
        inputs_torch = torch.from_numpy(training_inputs)
        targets_torch = torch.from_numpy(training_targets)
        for i in range(self.observation_dimension):
            self.models_[i].fit(inputs_torch, targets_torch[:,i])

    def _learn_from_stream(self, training_generator, generator_size):
        raise NotImplementedError

    def _predict(self, inputs):
        assert self.models_, "a trained model must be available"
        inputs_torch = torch.from_numpy(inputs)
        prediction = np.zeros((inputs.shape[0], self.observation_dimension))
        for i, model in enumerate(self.models_):
            prediction[:, i] = model.predict(inputs_torch)
        return prediction

    def name(self):
        return "falkon-kernel-machine"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    parser.add_argument("--horizon", type=int, default=1,
            help="Prediction horizon")
    parser.add_argument("--history", type=int, default=1,
            help="History length")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)

    # Learning in batch mode.
    dynamics_model = FalkonDynLearner(args.history, args.horizon, True, False,
            False)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())

