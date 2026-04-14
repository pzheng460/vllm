# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiMo-V2-Flash MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mimo_v2_flash import (
    MiMoV2Attention,
    MiMoV2MLP,
)
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix


class MiMoV2FlashMTPLayer(nn.Module):
    """MTP layer for MiMo-V2-Flash, using SWA attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        eps = getattr(config, "layernorm_epsilon", 1e-5)
        self.enorm = RMSNorm(config.hidden_size, eps=eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(
            config, "max_position_embeddings", 32768
        )
        v_scale = getattr(config, "attention_value_scale", None)
        layer_id = int(prefix.rsplit(".", 1)[-1]) if prefix else 0

        self.self_attn = MiMoV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            head_dim=config.swa_head_dim,
            v_head_dim=getattr(config, "swa_v_head_dim", None),
            v_scale=v_scale,
            sliding_window_size=getattr(config, "sliding_window_size", -1),
            attention_bias=getattr(config, "attention_bias", False),
            add_swa_attention_sink_bias=getattr(
                config, "add_swa_attention_sink_bias", False
            ),
            layer_id=layer_id,
            rope_theta=getattr(config, "swa_rope_theta",
                               getattr(config, "rope_theta", 1000000)),
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            partial_rotary_factor=getattr(
                config, "partial_rotary_factor", 1.0
            ),
            prefix=f"{prefix}.self_attn",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = MiMoV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        inputs_embeds[positions == 0] = 0
        normed_embeds = self.enorm(inputs_embeds)
        normed_hidden = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([normed_hidden, normed_embeds], dim=-1)
        )

        # Decoder block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return self.final_layernorm(hidden_states)


class MiMoV2FlashMTP(nn.Module):
    """MTP model for MiMo-V2-Flash."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.mtp_start_layer = config.num_hidden_layers
        num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.mtp_layers = nn.ModuleDict({
            str(idx): MiMoV2FlashMTPLayer(
                config=config,
                prefix=f"{maybe_prefix(prefix, 'model')}.mtp_layers.{idx}",
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(
                config.num_hidden_layers,
                config.num_hidden_layers + num_mtp_layers,
            )
        })

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.num_mtp_layers = num_mtp_layers

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        layer_idx = self.mtp_start_layer + spec_step_idx
        hidden_states = self.mtp_layers[str(layer_idx)](
            inputs_embeds, positions, hidden_states, spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import regex as re

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Map: model.mtp.layers.{i}.* -> mtp_layers.{i + offset}.*
            name = self._map_weight_name(name, re)
            if name is None:
                continue

            # Stacked params (qkv_proj, gate_up_proj)
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mtp_layers" not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)
                if name_rewritten not in params_dict:
                    continue
                param = params_dict[name_rewritten]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_rewritten)
                matched = True
                break
            if matched:
                continue

            if name not in params_dict:
                continue
            param = params_dict[name]

            # TP shard attention_sink_bias
            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                param.data.copy_(loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank))
                loaded_params.add(name)
                continue

            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

    def _map_weight_name(self, name: str, re) -> str | None:
        # Only load MTP, embed_tokens, or lm_head weights
        if ("mtp" not in name and "embed_tokens" not in name
                and "lm_head" not in name):
            return None

        # model.mtp.layers.{i}.* -> mtp_layers.{i + offset}.*
        pattern = r"model\.mtp\.layers\.(\d+)\."
        match = re.match(pattern, name)
        if match:
            idx = int(match.group(1))
            new_idx = idx + self.config.num_hidden_layers
            name = re.sub(pattern, f"model.mtp_layers.{new_idx}.", name)
            # Remove "model." prefix since mtp_layers is directly on self
            name = name.replace("model.mtp_layers.", "mtp_layers.", 1)

        if name.startswith("model.embed_tokens."):
            name = name.replace("model.", "", 1)
        if name.startswith("model.lm_head."):
            name = name.replace("model.", "", 1)

        return name
