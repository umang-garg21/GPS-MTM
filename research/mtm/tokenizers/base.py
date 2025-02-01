# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar("T", bound="Tokenizer")


class Tokenizer(torch.nn.Module, ABC):
    @abstractmethod
    def create(cls: Type[T], key: str, train_dataset: Dataset, **kwargs) -> T:
        """Create a new instance of the model."""
        pass

    @property
    @abstractmethod
    def discrete(self) -> bool:
        """Whether the tokenizer is discrete or continuous."""
        pass

    @abstractmethod
    def encode(
        self, trajectory: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode a trajectory with attention masks.

        Args:
            trajectory (torch.Tensor): shape=(batch_size, L, ...)
            attention_mask (torch.Tensor): shape=(batch_size, L), 1 for valid tokens, 0 for padding.

        Returns:
            tokenized_trajectories (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)
        """
        pass

    @abstractmethod
    def decode(
        self, tokenized_trajectory: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode a trajectory with attention masks.

        Args:
            tokenized_trajectory (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)
            attention_mask (torch.Tensor): shape=(batch_size, L), 1 for valid tokens, 0 for padding.

        Returns:
            decoded_trajectory (torch.Tensor): shape=(batch_size, L, ...)
        """
        pass


class TokenizerManager(torch.nn.Module):
    def __init__(self, tokenizers: Dict[str, Tokenizer]):
        super().__init__()
        self.tokenizers = torch.nn.ModuleDict(tokenizers)

    def encode(
        self,
        trajectories: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode all trajectories with optional attention masks.

        Args:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...).
            attention_masks (Dict[str, torch.Tensor]): Each mask has shape=(batch_size, L).

        Returns:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size).
        """
        out_trajectories = {}
        for key, value in trajectories.items():
            if key in self.tokenizers.keys():
                # Encode different keys ('states', 'actions') using appropriate tokenizer.
                out_trajectories[key] = self.tokenizers[key].encode(value)
                # out_trajectories[key] is 0 where attention_masks is 0, keep with 1.
                if attention_masks is not None:
                    out_trajectories[key][attention_masks==0] = 0
                assert len(out_trajectories[key].shape) == 4

        return out_trajectories

    def decode(
        self,
        tokenized_trajectories: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode all trajectories with optional attention masks.

        Args:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size).
            attention_masks (Dict[str, torch.Tensor]): Each mask has shape=(batch_size, L).

        Returns:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...).
        """
        out_trajectories = {}
        for key, value in tokenized_trajectories.items():
            out_trajectories[key] = self.tokenizers[key].decode(value)
            # out_trajectories[key] is 0 where attention_masks is 0, keep with 1.
            if attention_masks is not None:
                out_trajectories[key][attention_masks==0] = 0
                
        return out_trajectories