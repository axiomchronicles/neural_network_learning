import torch
import enum

class LearningRate(enum.Enum):
    BEST_FIT = 1e-3 # 0.001 (1 x 10-3)
    FAST = 1e-2 # 0.001 (1 x 10-2)
    PATTERN = 5e-3 # 0.005 (5 x 10-3)
    SMART = 1e-4 # 0.0001 (1 x 10-4)

class WeightDecay(enum.Enum):
    BEST_FIT = 1e-4 # 0.001 (1 x 10-3)
    FAST = 1e-2 # 0.001 (1 x 10-2)
    PATTERN = 5e-3 # 0.005 (5 x 10-3)
    SMART = 1e-5 # 0.0001 (1 x 10-4)

class CriterionHyperParameters(enum.Enum):
    ...


criterion: torch.nn.Module = torch.nn.MSELoss()
# optimizer = torch.optim.Adam()