# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from research.mtm.tokenizers.base import TokenizerManager
from research.utils.plot_utils import PlotHandler as ph


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
    Returns:
        out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = np.arange(pos, dtype=np.float32)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@torch.no_grad()
def make_plots_with_masks(
    predict_fn,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    masks_list: List[Dict[str, torch.Tensor]],
    prefixs: List[str],
    batch_idxs: Tuple[int, ...] = (0,),
    max_n_plots: int = 3,
):
    eval_logs = {}
    for masks, prefix in zip(masks_list, prefixs):
        eval_name = f"{prefix}_eval"

        encoded_trajectories = tokenizer_manager.encode(trajectories)
        decoded_gt_trajectories = tokenizer_manager.decode(encoded_trajectories)
        predictions = predict_fn(encoded_trajectories, masks)
        decoded_trajs = tokenizer_manager.decode(predictions)

        mse_loss = 0
        for k, v in decoded_trajs.items():
            _mse = F.mse_loss(
                v.to(torch.float32), trajectories[k].to(torch.float32)
            ).item()
            eval_logs[f"{eval_name}/mse_{k}"] = _mse
            mse_loss += _mse
        eval_logs[f"{eval_name}/mse_sum"] = mse_loss

        mse_loss = 0
        for k, v in decoded_gt_trajectories.items():
            _mse = F.mse_loss(
                v.to(torch.float32), trajectories[k].to(torch.float32)
            ).item()
            eval_logs[f"{eval_name}/lower_bound_mse_{k}"] = _mse
            mse_loss += _mse
        eval_logs[f"{eval_name}/lower_bound_mse_sum"] = mse_loss

        for batch_idx in batch_idxs:
            for k, _ in decoded_trajs.items():
                traj = trajectories[k][batch_idx].cpu().numpy()
                if len(traj.shape) == 1:
                    traj = traj[:, None]
                pred_traj = decoded_trajs[k][batch_idx].cpu().numpy()
                if len(pred_traj.shape) == 1:
                    pred_traj = pred_traj[:, None]
                dec_gt_traj = decoded_gt_trajectories[k][batch_idx].cpu().numpy()
                if len(dec_gt_traj.shape) == 1:
                    dec_gt_traj = dec_gt_traj[:, None]
                logit_traj = predictions[k][batch_idx].cpu().numpy()

                if k == "images" and batch_idx == batch_idxs[0]:
                    traj = traj
                    pred_traj = pred_traj
                    dec_gt_traj = dec_gt_traj
                    # log images to wandb
                    sub_r = 2  # subsample ratio
                    eval_logs[f"{eval_name}/i_traj"] = [wandb.Image(t) for t in traj][
                        ::sub_r
                    ]
                    eval_logs[f"{eval_name}/i_pred_traj"] = [
                        wandb.Image(t) for t in pred_traj
                    ][::sub_r]
                    eval_logs[f"{eval_name}/i_dec_gt_traj"] = [
                        wandb.Image(t) for t in dec_gt_traj
                    ][::sub_r]
                    continue

                for i in range(min(max_n_plots, traj.shape[-1])):
                    gt_i = traj[:, i]
                    re_i = pred_traj[:, i]
                    dec_gt_i = dec_gt_traj[:, i]
                    mask = masks[k]
                    if len(mask.shape) == 1:
                        # only along time dimension: repeat across the given dimension
                        mask = mask[:, None].repeat(1, traj.shape[1])
                    select_mask = mask[:, i].cpu().numpy()
                    unmasked_gt_i = gt_i[select_mask == 1]
                    unmasked_gt_i_index = np.arange(len(gt_i))[select_mask == 1]
                    vmax = max(np.max(gt_i), np.max(re_i))
                    vmin = min(np.min(gt_i), np.min(re_i))
                    y_range = vmax - vmin
                    with ph.plot_context() as (fig, ax):
                        ax.plot(gt_i, "-o", label="ground truth")
                        ax.plot(
                            re_i, "-o", label="reconstructed", markerfacecolor="none"
                        )
                        # blue color
                        ax.plot(
                            dec_gt_i,
                            "--o",
                            label="gt_reconstructed",
                            markerfacecolor="none",
                            color="blue",
                        )
                        ax.plot(
                            unmasked_gt_i_index,
                            unmasked_gt_i,
                            "o",
                            label="unmasked ground truth",
                        )
                        ax.set_ylim(
                            vmin - y_range / 5,
                            vmax + y_range / 5,
                        )
                        ax.legend()
                        eval_logs[
                            f"{eval_name}/batch={batch_idx}|{i}_{k}"
                        ] = wandb.Image(ph.plot_as_image(fig))

                    if i < logit_traj.shape[1]:
                        logits = torch.tensor(logit_traj[:, i, :])
                        probs = torch.softmax(
                            logits / 2, dim=1
                        )  # divide by 2 to make the plot more readable
                        x = probs.detach().cpu().numpy()
                        with ph.plot_context() as (fig, ax):
                            ax.imshow(np.flipud(x.T), aspect=0.3)
                            eval_logs[
                                f"{eval_name}/batch={batch_idx}|{i}_{k}_logits"
                            ] = wandb.Image(ph.plot_as_image(fig))
    return eval_logs


@dataclasses.dataclass
class MTMConfig:
    n_embd: int = 128
    n_head: int = 2
    n_enc_layer: int = 1
    n_dec_layer: int = 1
    dropout: float = 0
    embd_pdrop: float = 0
    resid_pdrop: float = 0
    attn_pdrop: float = 0
    norm: str = "l2"
    loss: str = "total"
    reduce_use_sum: bool = False
    loss_keys: Optional[List[str]] = None
    latent_dim: Optional[int] = None
    use_masked_loss: bool = False

    def create(self, data_shape, traj_length):
        return MTM(data_shape, traj_length, self)

class MTM(nn.Module):
    def __init__(
        self,
        data_shapes: Dict[str, Tuple[int, ...]],
        traj_length: int,
        config: MTMConfig,
    ):
        """Initialize a masked model.

        Args:
            data_shapes (Dict[str, Tuple[int, int]]): data_shapes
            traj_length (int): traj_length
            attention_masks (Dict[str, torch.Tensor]): attention_masks
            config (MTMConfig): config
        """
        super().__init__()
        self.data_shapes = data_shapes
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.config = config
        # self.max_len = config.traj_length * 2
        self.max_len = traj_length
        # self.mask_ratio = config.mask_ratio
        self.norm = config.norm
        # print("norm", self.norm)
        self.encoder_embed_dict = nn.ModuleDict()
        self.decoder_embed_dict = nn.ModuleDict()
        self.mask_token_dict = nn.ParameterDict()

        self.encoder_per_dim_encoding = nn.ParameterDict()
        self.decoder_per_dim_encoding = nn.ParameterDict()

        for key, shape in data_shapes.items():
            self.encoder_embed_dict[key] = nn.Linear(shape[1], self.n_embd)
            if self.config.latent_dim is None:
                self.decoder_embed_dict[key] = nn.Linear(self.n_embd, self.n_embd)
                self.mask_token_dict[key] = nn.Parameter(torch.zeros(1, 1, self.n_embd))
            else:
                self.decoder_embed_dict[key] = nn.Linear(
                    self.config.latent_dim, self.n_embd
                )
                self.mask_token_dict[key] = nn.Parameter(
                    torch.zeros(1, 1, self.config.latent_dim)
                )
            self.encoder_per_dim_encoding[key] = nn.Parameter(
                torch.zeros(1, 1, shape[0], self.n_embd)
            )
            self.decoder_per_dim_encoding[key] = nn.Parameter(
                torch.zeros(1, 1, shape[0], self.n_embd)
            )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_enc_layer,
            norm=nn.LayerNorm(self.n_embd),
        )
        if self.config.latent_dim is not None:
            self.encoder_projection = nn.Sequential(
                *[nn.GELU(), nn.Linear(self.n_embd, self.config.latent_dim)]
            )

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_dec_layer,
            norm=nn.LayerNorm(self.n_embd),
        )

        self.output_head_dict = nn.ModuleDict()
        for key, shape in data_shapes.items():
            self.output_head_dict[key] = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.Linear(self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, shape[-1]),
            )
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float()[None, :, None, :] / 2.0
        self.register_buffer("pos_embed", pe)

    @staticmethod
    def forward_loss(
        targets: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        discrete_map: Dict[str, bool],
        norm="l2",
        reduce_use_sum=False,
        attention_masks: Dict[str, torch.Tensor] = None,
        loss_keys: Optional[List[str]] = None,
    ):
        """
        Compute the loss for the given targets and predictions, incorporating attention masks.

        Args:
            targets: Dictionary of target tensors (e.g., states, actions).
            preds: Dictionary of predicted tensors (e.g., states, actions).
            masks: Dictionary of masks indicating valid tokens.
            attention_masks: Dictionary of attention masks to filter padded positions.
            discrete_map: Mapping of keys to indicate if the values are discrete or continuous.
            norm: Normalization type ('l2' or 'mae').
            reduce_use_sum: Whether to use sum reduction for loss computation.
            loss_keys: Keys to include in the final loss calculation.

        Returns:
            Tuple of total loss, per-key losses, masked losses, and masked complete losses.
        """
        losses = {}
        masked_losses = {}
        masked_c_losses = {}

        for key in targets.keys():
            target = targets[key]
            pred = preds[key]
            mask = masks[key]

            if attention_masks is not None:
                #mask = mask * attention_masks
                mask_expanded = mask.unsqueeze(0).expand(attention_masks.shape[0], -1, -1)  # Shape: (5, 221, 1)
                # Expand attention_masks to align dimensions
                attention_masks_expanded = attention_masks.unsqueeze(2)  # Shape: (5, 221, 1)
                mask = mask_expanded * attention_masks_expanded

            if len(mask.shape) == 1:
                mask = mask[:, None].repeat(1, target.shape[2])

            if discrete_map[key]:
                raw_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
            else:
                if norm == "l2":
                    target = target / torch.norm(target, dim=-1, keepdim=True)
                raw_loss = nn.MSELoss(reduction="none")(pred, target)
                
            #Actual loss is only upto attention_mask length. Focus the model to learn relevant tokens not 0s.
            actual_loss= raw_loss * attention_masks.unsqueeze(-1).unsqueeze(-1)

            # Track masked loss to know how mask tokens are performing.
            if mask.sum() == 0:
                masked_c_loss = torch.tensor(0.0).to(raw_loss.device)
                # solve  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                #RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
            else:
                masked_c_loss = (
                    (raw_loss * mask[:, :, :, None]).sum(dim=(1, 2, 3)) / mask.sum()
                ).mean()
            
            #replaced raw_loss with actual loss.
            losses[key] = actual_loss.mean(dim=(2, 3)).mean()
            masked_c_losses[key] = masked_c_loss

        if loss_keys is None:
            total_loss = torch.sum(torch.stack(list(losses.values())))
            total_masked_c_loss = torch.sum(torch.stack(list(masked_c_losses.values())))
        else:
            total_loss = torch.sum(torch.stack([losses[key] for key in loss_keys]))
            total_masked_c_loss = torch.sum(torch.stack([masked_c_losses[key] for key in loss_keys]))

        return total_loss, losses, total_masked_c_loss, masked_c_losses
    
    def _index(self, x, use_mask):
        assert len(use_mask.shape) == 1
        ids = (use_mask == 1).nonzero(as_tuple=True)[0]
        zero_ids = (use_mask == 0).nonzero(as_tuple=True)[0]

        idx_array = torch.hstack((ids, zero_ids))
        ids_restore = torch.argsort(idx_array)

        x = x[:, ids]
        keep_len = len(ids)
        return x, ids_restore, keep_len
    
    def _index_and_concat(self, x, use_masks):
        assert len(use_masks.shape) == 2  # Ensure batch-wise masks
        batch_size,_,__ = x.shape

        concatenated_x = []
        ids_restore_list = []
        keep_lens = []

        for i in range(batch_size):
            # Mask for the current batch element
            use_mask = use_masks[i]

            # Extract indices based on the mask
            ids = (use_mask == 1).nonzero(as_tuple=True)[0]
            zero_ids = (use_mask == 0).nonzero(as_tuple=True)[0]

            # Combine and calculate restore indices
            idx_array = torch.hstack((ids, zero_ids))
            ids_restore = torch.argsort(idx_array)

            # Apply mask to the current batch element
            concatenated_x.append(x[i, ids])  # Keep only selected features
            ids_restore_list.append(ids_restore)
            keep_lens.append(len(ids))
            #print("i, len(ids), len(zero_ids), len(ids_restore), len(concat_x[i].shape)", i, len(ids), len(zero_ids), len(ids_restore), concatenated_x[i].shape)

        # Concatenate the results from all batch elements
        #concatenated_x = torch.stack(concatenated_x) # (batch_size, seq_len, feature_dim)
        #ids_restore_list = torch.stack(ids_restore_list)
        #keep_lens = torch.tensor(keep_lens)

        return concatenated_x, ids_restore_list, keep_lens

    def attention_mask_expand(self, x, use_mask, attention_masks=None):
        assert len(use_mask.shape) == 1
        # repeat the mask across the batch dimension
        use_mask = use_mask.repeat(x.shape[0], 1)
        if attention_masks is not None:
            # Update attention mask replacing 1 with 0 where the mask index is 0 in the use_mask
            # Do not touch the original zeros in the attention mask

            # The attention mask is a 2D tensor with shape (batch_size, seq_len)
            # The use_mask is a 2D tensor with shape (batch_size, seq_len)
            # The output will be a 2D tensor with shape (batch_size, seq_len)
            updated_attention_masks = attention_masks * use_mask

        return updated_attention_masks
    
    def trajectory_encoding(
        self, trajectories, attention_masks=None
    ) -> Dict[str, torch.Tensor]:
        encoded_trajectories = {}
        for key, traj in trajectories.items():
            if attention_masks is not None:
                traj = traj * attention_masks[:, :, None, None]
            encoded_traj = (
                self.encoder_embed_dict[key](traj.to(torch.float32))
                + self.encoder_per_dim_encoding[key]
                + self.pos_embed[:, : traj.shape[1], :, :]
            )
            # Original code
            #encoded_trajectories[key] = encoded_traj.reshape(
            #    traj.shape[0], -1, traj.shape[-1]
            #)

            # Changed code. RECHECK.
            encoded_trajectories[key] = encoded_traj.reshape(
                encoded_traj.shape[0], -1, encoded_traj.shape[-1]
            )
        return encoded_trajectories

    def process_masks(
        self, trajectories, masks, flatten_shape=True
    ) -> Dict[str, torch.Tensor]:
        batch_size = None
        batched_masks = {}
        for k, v in trajectories.items():
            assert (
                v.shape[2] == self.data_shapes[k][0]
            ), f"{v.shape}, {self.data_shapes}"
            assert (
                v.shape[3] == self.data_shapes[k][1]
            ), f"{v.shape}, {self.data_shapes}"

            mask = masks[k]
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, v.shape[2])
            elif len(mask.shape) == 2:
                pass
            else:
                raise NotImplementedError(f"mask shape = {mask.shape}")

            if batch_size is None:
                batch_size = v.shape[0]
            else:
                assert batch_size == v.shape[0]
                batch_size = v.shape[0]

            if flatten_shape:
                mask = mask.reshape(-1)

            batched_masks[k] = mask
        return batched_masks

    def forward(self, trajectories, masks, attention_masks=None):
        batched_masks = self.process_masks(trajectories, masks)
        embedded_trajectories = self.trajectory_encoding(trajectories, attention_masks)

        encoded_trajectories, ids_restore, keep_length = self.forward_encoder(
            embedded_trajectories, batched_masks, attention_masks=attention_masks
        )
        # extract the trajectories
        return self.forward_decoder(encoded_trajectories, ids_restore, keep_length, attention_masks=attention_masks)

    def encode(self, trajectories, masks) -> Dict[str, torch.Tensor]:
        batched_masks = self.process_masks(trajectories, masks)
        embedded_trajectories = self.trajectory_encoding(trajectories)

        encoded_trajectories, ids_restore, keep_length = self.forward_encoder(
            embedded_trajectories, batched_masks)

        return encoded_trajectories

    def forward_encoder(self, trajectories, masks, attention_masks=None):
        features = []
        ids_restore = {}
        keep_len = {}
        updated_att_masks= {}
        max_len={}
        # process obs
        batch_size =trajectories[list(trajectories.keys())[0]].shape[0]
        keys = list(trajectories.keys())  # get the keys in a list to maintain order
        for k in keys:
            traj = trajectories[k]
            mask = masks[k]

            # Updated attention masks include 
            updated_att_masks[k] = self.attention_mask_expand(traj, mask, attention_masks)
            x, ids_restore[k], keep_len[k] = self._index_and_concat(traj, updated_att_masks[k])
            #x, ids_restore[k], keep_len[k] = self._index(traj, mask)
            #stacked_attention_masks= attention_masks
            max_len[k]=max(keep_len[k])
            features.append(x)

        feats=[]
        src_key_padding_mask = []
        # Expand x to max_len
        for i, k in enumerate(keys):
            padding_mask = torch.zeros(batch_size, max_len[k], dtype=torch.bool)
            for b in range(batch_size):
                if features[i][b].shape[0] < max_len[k]:
                    # padding mask is one where the feature is padded
                    padding_mask[b, features[i][b].shape[0]:] = 1
                    features[i][b] = F.pad(features[i][b], (0, 0, 0, max_len[k] - features[i][b].shape[0]))
                # STACK and create tensor along batch dimension from features[i][b]

            # torch stack a list of tensors along a new dimension
            feats.append(torch.stack(features[i], dim=0))
            src_key_padding_mask.append(padding_mask)
        

        # 5x55 for states, 5x95 for actions [55+95= 150] 
        # 55 is max_len for states and 95 is max_len for actions
        feats=torch.cat(feats, dim=1)
        src_key_padding_mask=torch.cat(src_key_padding_mask, dim=1).to(feats.device)

        # ONLY SELECTED SORTED FEATURES ARE PASSED TO THE ENCODER
        x = self.encoder(feats, src_key_padding_mask=src_key_padding_mask)  # src key padding mask to ignore padded values
        
        if self.config.latent_dim is not None:
            x = self.encoder_projection(x)  # project down
        """
        for k in keys:
            v = keep_len[k]
            encoded_trajectories[k] = x[:, idx : idx + v]
            idx += v
        """
        encoded_trajectories = {}
        idx=0
        for i, k in enumerate(keys):
            v = max_len[k]
            encoded_trajectories[k] = x[:, idx : idx + v, :]
            idx += v
            #print("b, v, encoded_trajectories[k].shape",b, v, encoded_trajectories[k].shape)

        return encoded_trajectories, ids_restore, keep_len

    def _decoder_trajectory_encoding(self, trajectories) -> Dict[str, torch.Tensor]:
        encoded_trajectories = {}
        for key, traj in trajectories.items():
            data_shape = self.data_shapes[key]
            b, _, f = traj.shape
            re_traj = traj.reshape(b, -1, data_shape[0], f)
            t = re_traj.shape[1]
            encoded_traj = (
                self.decoder_embed_dict[key](re_traj)
                + self.decoder_per_dim_encoding[key]
                + self.pos_embed[:, :t, :, :]
            )
            b, t, p, c = encoded_traj.shape
            x = encoded_traj.reshape(b, t*p, c)
            encoded_trajectories[key] = x
        return encoded_trajectories

    def forward_decoder(
        self,
        trajectories: Dict[str, torch.Tensor],
        ids_restore: Dict[str, torch.Tensor],
        keep_lengths: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            trajectories (Dict[str, torch.Tensor]): trajectories. Each trajectory is of shape (batch_size, T*tokens_per_time, feature_dim)
            ids_restore (Dict[str, torch.Tensor]): ids_restore
            keep_lengths (Dict[str, torch.Tensor]): keep_lengths
        """
        encoded_trajectories_with_mask = {}
        keys = list(trajectories.keys())    # get the keys in a list to maintain order
       
        for k in keys:
            x_ = []
            traj = trajectories[k]
            batch_size = traj.shape[0]
            #assert len(ids_restore[k].shape) == 1

            # due to variable length trajectory this will be differnt for each element in the batch. 
            # outer looped by keys.
            for b in range(batch_size):
                num_mask_tokens = ids_restore[k][b].shape[0] - keep_lengths[k][b]

                #mask_tokens = self.mask_token_dict[k].repeat(batch_size, num_mask_tokens, 1)
                # create mask tokens for each trajectory
                mask_tokens_per_sample = self.mask_token_dict[k].repeat(1, num_mask_tokens, 1)
                
                # concatenate the mask tokens to the end of the trajectory, and reorder them in the original order.
                x_sample = torch.cat([traj[b][:keep_lengths[k][b]].unsqueeze(0), mask_tokens_per_sample], dim=1)
                assert (
                    ids_restore[k][b].shape[0] == x_sample.shape[1]
                ), f"{ids_restore[k]}, {x_sample.shape}"
                x_sample = torch.gather(x_sample, 1, ids_restore[k][b][None, :, None].repeat(1, 1, x_sample.shape[-1]))
                
                #concatenate all x samples into x_
                x_.append(x_sample)
            x_ = torch.cat(x_, dim=0)
            # re organize the indicies to be in their original positions
            # x_ = torch.gather(
            #     x_,
            #     1,
            #     ids_restore[k][None, :, None].repeat(batch_size, 1, traj.shape[-1]),
            # )

            # concatenate x_samples to x_ for each batch element
            #encoded_trajectories_with_mask[k] = torch.cat([encoded_trajectories_with_mask[k], x_], dim=0)


            encoded_trajectories_with_mask[k] = x_

        decoder_embedded_trajectories = self._decoder_trajectory_encoding(
            encoded_trajectories_with_mask 
        )
        concat_trajectories = torch.cat(
            [decoder_embedded_trajectories[k] for k in keys], dim=1
        )
        # concat attention masks number of keys times, dim=1
        att_mask_src_key = []
        for k in keys:
            att_mask_src_key.append(attention_masks)
        att_mask_src_key = torch.cat(att_mask_src_key, dim=1)
        
        x = self.decoder(concat_trajectories, src_key_padding_mask=att_mask_src_key) # Apply attention mask here
        extracted_trajectories = {}
        pos = 0
        for k in keys:
            b, t_p, f = decoder_embedded_trajectories[k].shape
            output_head = self.output_head_dict[k]
            traj_segment = x[:, pos : pos + t_p, :]
            p = self.data_shapes[k][0]
            extracted_trajectories[k] = output_head(traj_segment.reshape(b, -1, p, f))
            pos += t_p
        return extracted_trajectories

    def mask_git_forward(self, trajectories, masks, temperature=1.0, ratio=1.0):
        """Use MaskGIT style decoding

        Assumes that the last dimension is logits (only works for discrete model case)
        """
        p_masks = self.process_masks(trajectories, masks, flatten_shape=False)
        masks_copy = {k: torch.clone(v) for k, v in p_masks.items()}
        trajectories_copy = {k: torch.clone(v) for k, v in trajectories.items()}

        if ratio == 1.0:
            return self(trajectories_copy, masks_copy)

        num_choose = int(
            ratio
            * trajectories_copy["states"].shape[1]
            * trajectories_copy["states"].shape[2]
        )

        def masks_filled(_masks):
            for k, v in _masks.items():
                # print(v.sum(), v.shape[0] * v.shape[1])
                if v.sum() != v.shape[0] * v.shape[1]:
                    return False
            return True

        assert trajectories_copy["states"].shape[0] == 1

        while not masks_filled(masks_copy):
            traj_predictions = self(trajectories_copy, masks_copy)
            # sample from the logits
            for k, traj_logits in traj_predictions.items():
                B, L, I, _ = traj_logits.shape

                flattened_logits = traj_logits.reshape(B * L * I, -1) / temperature
                samples = torch.multinomial(F.softmax(flattened_logits, dim=-1), 1)
                p_for_sample = F.softmax(flattened_logits, dim=-1).gather(-1, samples)

                # pick the ratio number of highest likelihood sample that is not already masked
                indices_ = torch.argsort(p_for_sample, dim=0, descending=True)

                flattened_mask = masks_copy[k].reshape(L * I, -1)
                flattened_trajectory = trajectories_copy[k].reshape(B * L * I, -1)

                indices = indices_[~flattened_mask.bool()[indices_[:, 0]]]
                indices = indices[:num_choose]
                flattened_mask[indices, :] = 1
                flattened_trajectory[indices] = F.one_hot(
                    samples[indices][:, 0], num_classes=flattened_trajectory.shape[1]
                ).float()

                # fill in the trajectories
                trajectories_copy[k] = flattened_trajectory.reshape(B, L, I, -1)
                masks_copy[k] = flattened_mask.reshape(L, I)

        return trajectories_copy

    @staticmethod
    def configure_optimizers(
        model, learning_rate: float, weight_decay: float, betas: Tuple[float, float]
    ):
        """Create optimizers.

        This long function is unfortunately doing something very simple and is being very defensive:

        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        allowlist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blocklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, allowlist_weight_modules):
                    # weights of allowed modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blocklist_weight_modules):
                    # weights of blocked modules will NOT be weight decayed
                    no_decay.add(fpn)

        for pn, _ in model.named_parameters():
            if "dict" in pn and "bias" in pn:
                no_decay.add(pn)
            if "per_dim_encoding" in pn or "mask_token_dict" in pn:
                no_decay.add(pn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
