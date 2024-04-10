"""Probability distributions."""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

class SCARL_Categorical_Distribution(CategoricalDistribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),  # First linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(in_features=latent_dim, out_features=self.action_dim)  # Second linear layer
        )
        return action_logits





def make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False,use_two_linear_layers=False,dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return SCARL_Categorical_Distribution(int(action_space.n), **dist_kwargs) if use_two_linear_layers else CategoricalDistribution(int(action_space.n), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )