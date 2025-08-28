# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from numpy.typing import ArrayLike

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import Tokenizer

class ContinuousTokenizer(Tokenizer):
    def __init__(
        self,
        data_mean: ArrayLike,
        data_std: ArrayLike,
        stats: DataStatistics,
        normalize: bool = True,
    ):
        super().__init__()
        self._data_mean = torch.nn.Parameter(
            torch.tensor(data_mean, dtype=torch.float32), requires_grad=False
        )
        self._data_std = torch.nn.Parameter(
            torch.tensor(data_std, dtype=torch.float32), requires_grad=False
        )
        self.stats = stats
        self.normalize = normalize

    @classmethod
    def create(
        cls, key: str, train_dataset: DatasetProtocol, normalize: bool = True
    ) -> "ContinuousTokenizer":
        data = []
        stats = train_dataset.trajectory_statistics()[key]
        data_mean = stats.mean
        data_std = stats.std
        data_std[data_std < 0.1] = 1  # do not normalize if std is too small
        return cls(data_mean, data_std, stats, normalize=normalize)

    @property
    def discrete(self) -> bool:
        return False

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3

        # Perform normalization where attention mask is 1
        #att_mask = self.
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)
            # normalize trajectory

            trajectory[:] = (trajectory[:] - mean) / std
            #trajectory = (trajectory - mean) / std
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        
        # CHANGE THIS ACCORDING TO WHAT SHOULD BE THE OUTPUT DIMENSION OF THE TRAJECTORY
        # IF PREDICTING THE NEXT ACTION, THEN THE OUTPUT DIMENSION SHOULD BE 1
        # FOR THE STATE IT CAN BE 4 IF LOOKING FOR EXACT RECONTROUCTION, OR COULD CHANGE
        # ACCORDING TO THE REQUIREMENT EXAMPLE('POI_NAME', 'X') INSTEAD OF ('AGENT_ID', 'X', 'Y', 'POI_NAME')
        
        assert trajectory.dim() == 4 # [batch, traj_length, channel, output_dim]
        assert trajectory.size(2) == 1 #POSSIBLY. num of channels 
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)

            # denormalize trajectory
            return trajectory.squeeze(2) * std + mean
        else:
            return trajectory.squeeze(2)