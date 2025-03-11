# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, MochiAttnProcessor2_0
from diffusers.models.embeddings import MochiCombinedTimestepCaptionEmbedding, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, LuminaLayerNormContinuous, MochiRMSNormZero, RMSNorm

from ditango.logger import init_logger
from ditango.timer import get_timer
from ditango.core.parallel_state import get_usp_group
from ditango.executor.utils import split_tensor_uneven, remove_padding_after_gather
# from ditango.core.attention import oAttention
from ditango.executor.model.attention_processor import Mochi_RingAttnProcessor, Mochi_FakeUlyssesAttnProcessor, Mochi_oCacheAttnProcessor

logger = init_logger(__name__)    # pylint: disable=invalid-name

import nvtx
from contextlib import contextmanager

@contextmanager
def nvtx_range(name, domain="CUSTOM"):
    nvtx.push_range(name, domain=domain)
    try:
        yield
    finally:
        nvtx.pop_range(domain=domain)

@maybe_allow_in_graph
class MochiTransformerBlock(nn.Module):
    r"""
    Transformer block used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        context_pre_only (`bool`, defaults to `False`):
            Whether or not to process context-related conditions with additional layers.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        qk_norm: str = "rms_norm",
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
        layer_id: int = -1,  # DoIT
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3

        self.norm1 = MochiRMSNormZero(dim, 4 * dim, eps=eps, elementwise_affine=False)
        # # ===================== DoIT =====================
        self.layer_id = layer_id
        # ===================== DoIT =====================
        if not context_pre_only:
            self.norm1_context = MochiRMSNormZero(dim, 4 * pooled_projection_dim, eps=eps, elementwise_affine=False)
        else:
            self.norm1_context = LuminaLayerNormContinuous(
                embedding_dim=pooled_projection_dim,
                conditioning_embedding_dim=dim,
                eps=eps,
                elementwise_affine=False,
                norm_type="rms_norm",
                out_dim=None,
            )
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
            qk_norm=qk_norm,
            added_kv_proj_dim=pooled_projection_dim,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            # processor=MochiAttnProcessor2_0(),
            # processor=Mochi_RingAttnProcessor(),
            # processor=Mochi_FakeUlyssesAttnProcessor(),
            processor=Mochi_oCacheAttnProcessor(self.layer_id),
            eps=eps,
            elementwise_affine=True,
        )

        # TODO(aryan): norm_context layers are not needed when `context_pre_only` is True
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2_context = RMSNorm(pooled_projection_dim, eps=eps, elementwise_affine=False)

        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3_context = RMSNorm(pooled_projection_dim, eps=eps, elementwise_affine=False)

        self.ff = FeedForward(dim, inner_dim=self.ff_inner_dim, activation_fn=activation_fn, bias=False)
        self.ff_context = None
        if not context_pre_only:
            self.ff_context = FeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
            )

        self.norm4 = RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.norm4_context = RMSNorm(pooled_projection_dim, eps=eps, elementwise_affine=False)

    @get_timer("block")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if not self.context_pre_only:
            norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            # logger.info(f"{hidden_states.shape=} {encoder_hidden_states.shape=} {image_rotary_emb[0].shape=} {image_rotary_emb[1].shape=}")
            # exit()
        
        attn_hidden_states, context_attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + self.norm2(attn_hidden_states) * torch.tanh(gate_msa).unsqueeze(1)
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp.unsqueeze(1))
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + self.norm4(ff_output) * torch.tanh(gate_mlp).unsqueeze(1)

        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + self.norm2_context(
                context_attn_hidden_states
            ) * torch.tanh(enc_gate_msa).unsqueeze(1)
            norm_encoder_hidden_states = self.norm3_context(encoder_hidden_states) * (1 + enc_scale_mlp.unsqueeze(1))
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + self.norm4_context(context_ff_output) * torch.tanh(
                enc_gate_mlp
            ).unsqueeze(1)

        return hidden_states, encoder_hidden_states


class MochiRoPE(nn.Module):
    r"""
    RoPE implementation used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        base_height (`int`, defaults to `192`):
            Base height used to compute interpolation scale for rotary positional embeddings.
        base_width (`int`, defaults to `192`):
            Base width used to compute interpolation scale for rotary positional embeddings.
    """

    def __init__(self, base_height: int = 192, base_width: int = 192) -> None:
        super().__init__()

        self.target_area = base_height * base_width

    def _centers(self, start, stop, num, device, dtype) -> torch.Tensor:
        edges = torch.linspace(start, stop, num + 1, device=device, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2

    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        scale = (self.target_area / (height * width)) ** 0.5

        t = torch.arange(num_frames, device=device, dtype=dtype)
        h = self._centers(-height * scale / 2, height * scale / 2, height, device, dtype)
        w = self._centers(-width * scale / 2, width * scale / 2, width, device, dtype)

        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        positions = torch.stack([grid_t, grid_h, grid_w], dim=-1).view(-1, 3)
        return positions

    def _create_rope(self, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("nd,dhf->nhf", pos, freqs.float())
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def forward(
        self,
        pos_frequencies: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self._get_positions(num_frames, height, width, device, dtype)
        rope_cos, rope_sin = self._create_rope(pos_frequencies, pos)
        return rope_cos, rope_sin


@maybe_allow_in_graph
class MochiTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    A Transformer model for video-like data introduced in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `48`):
            The number of layers of Transformer blocks to use.
        in_channels (`int`, defaults to `12`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `256`):
            Output dimension of timestep embeddings.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        max_sequence_length (`int`, defaults to `256`):
            The maximum sequence length of text embeddings supported.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_type=None,
        )

        # 2. Time embeddings
        self.time_embed = MochiCombinedTimestepCaptionEmbedding(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )

        self.pos_frequencies = nn.Parameter(torch.full((3, num_attention_heads, attention_head_dim // 2), 0.0))
        self.rope = MochiRoPE()
        
        # 3. Define Spatio-Temporal Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MochiTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    pooled_projection_dim=pooled_projection_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    context_pre_only=i == num_layers - 1,
                    layer_id=i, # DoIT
                )
                for i in range(num_layers)
            ]
        )

        # 4. Output Blocks
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6, norm_type="layer_norm"
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @get_timer("DiT")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        temb, encoder_hidden_states = self.time_embed(
            timestep, encoder_hidden_states, encoder_attention_mask, hidden_dtype=hidden_states.dtype
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )
    
        # ===================== DoIT - usp split =====================
        if get_usp_group().world_size > 1:
            hidden_states = split_tensor_uneven(hidden_states, get_usp_group().world_size, dim=1, tensor_name="hidden")[get_usp_group().rank_in_group]
            encoder_hidden_states = split_tensor_uneven(encoder_hidden_states, get_usp_group().world_size, dim=1, tensor_name='encoder_hidden')[get_usp_group().rank_in_group]
            if image_rotary_emb is not None:
                    cos, sin = image_rotary_emb
                    cos_splits = split_tensor_uneven(cos, get_usp_group().world_size, dim=0)
                    sin_splits = split_tensor_uneven(sin, get_usp_group().world_size, dim=0)
                    image_rotary_emb = (cos_splits[get_usp_group().rank_in_group], sin_splits[get_usp_group().rank_in_group])
        # ===================== DoIT =====================
        # logger.debug(f"{hidden_states.shape=}")
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
        
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        
        # ===================== DoIT - usp gather =====================
        if get_usp_group().world_size > 1:
            hidden_states = get_usp_group().all_gather(hidden_states, dim=1)
            encoder_hidden_states = get_usp_group().all_gather(encoder_hidden_states, dim=1)
            hidden_states = remove_padding_after_gather(hidden_states, "hidden")
            encoder_hidden_states = remove_padding_after_gather(encoder_hidden_states, "encoder_hidden")
        # ===================== DoIT =====================
        
        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)