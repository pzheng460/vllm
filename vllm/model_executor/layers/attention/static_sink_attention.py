# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy

import torch

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    subclass_attention_backend,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    SinkFullAttentionSpec,
)

logger = init_logger(__name__)


@functools.lru_cache
def create_static_sink_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    sink_len: int = 0,
) -> type[AttentionBackend]:
    prefix = "StaticSink_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class StaticSinkAttentionBuilder(underlying_builder):  # type: ignore
        supports_update_block_table: bool = False

        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            model_config = vllm_config.model_config
            scheduler_config = vllm_config.scheduler_config
            self.sink_len = sink_len
            self.block_size = vllm_config.cache_config.block_size
            self.num_sink_blocks = self.sink_len // vllm_config.cache_config.block_size
            self.max_num_blocks = cdiv(
                model_config.max_model_len, vllm_config.cache_config.block_size
            )
            self.block_table_with_sink = torch.zeros(
                (
                    scheduler_config.max_num_seqs,
                    self.max_num_blocks + self.num_sink_blocks,
                ),
                device=device,
                dtype=torch.int32,
            )
            self.block_table_with_sink[:, : self.num_sink_blocks] = torch.arange(
                1,
                self.num_sink_blocks + 1,
                device=device,
                dtype=torch.int32,
            )
            # Separate buffer for build_for_drafting to avoid corrupting
            # main model's FlashAttentionMetadata during Parallel-SD
            # async CUDA stream execution.
            self.block_table_with_sink_draft = torch.zeros(
                (
                    scheduler_config.max_num_seqs,
                    self.max_num_blocks + self.num_sink_blocks,
                ),
                device=device,
                dtype=torch.int32,
            )
            self.block_table_with_sink_draft[
                :, : self.num_sink_blocks
            ] = torch.arange(
                1,
                self.num_sink_blocks + 1,
                device=device,
                dtype=torch.int32,
            )

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            common_attn_metadata.seq_lens[:] = (
                common_attn_metadata.seq_lens + self.sink_len
            )
            common_attn_metadata.seq_lens[
                common_attn_metadata.seq_lens == self.sink_len
            ] = 0
            common_attn_metadata.max_seq_len = (
                common_attn_metadata.max_seq_len + self.sink_len
            )
            max_num_blocks = cdiv(common_attn_metadata.max_seq_len, self.block_size)
            num_reqs = common_attn_metadata.num_reqs
            self.block_table_with_sink[
                :num_reqs, self.num_sink_blocks : self.num_sink_blocks + max_num_blocks
            ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
            common_attn_metadata.block_table_tensor = self.block_table_with_sink[
                :num_reqs
            ]

            return super().build(common_prefix_len, common_attn_metadata, fast_build)

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ) -> AttentionMetadata:
            # Parallel-SD safe: uses copies instead of in-place modification.
            #
            # After main model's build(), the shared seq_lens tensor is L+S
            # (from in-place [:] modification). Adding another S gives L+2S,
            # matching the double-adjustment behavior the drafter needs.
            #
            # We use a separate block_table buffer and new seq_lens tensor
            # to avoid race conditions when Parallel-SD runs the drafter
            # concurrently on a separate CUDA stream.
            draft_cm = copy(common_attn_metadata)

            sink_seq_lens = common_attn_metadata.seq_lens + self.sink_len
            sink_seq_lens[sink_seq_lens == self.sink_len] = 0
            draft_cm.seq_lens = sink_seq_lens

            draft_cm.max_seq_len = (
                common_attn_metadata.max_seq_len + self.sink_len
            )

            max_num_blocks = cdiv(draft_cm.max_seq_len, self.block_size)
            num_reqs = draft_cm.num_reqs

            self.block_table_with_sink_draft[
                :num_reqs,
                self.num_sink_blocks : self.num_sink_blocks + max_num_blocks,
            ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
            draft_cm.block_table_tensor = self.block_table_with_sink_draft[
                :num_reqs
            ]

            # Call underlying builder's build() directly (skip our build()
            # which does in-place modification).
            return super().build(
                common_prefix_len=0,
                common_attn_metadata=draft_cm,
                fast_build=True,
            )

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=StaticSinkAttentionBuilder,
    )

    return attn_backend


@CustomOp.register("static_sink_attention")
class StaticSinkAttention(Attention, CustomOp):
    """
    Attention with static sink tokens
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sink_len: int,
        attn_backend: type[AttentionBackend] | None = None,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ):
        CustomOp.__init__(self)
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if attn_backend is not None:
            underlying_attn_backend = attn_backend
        else:
            underlying_attn_backend = get_attn_backend(
                head_size, dtype, kv_cache_dtype, block_size
            )
        attn_backend = create_static_sink_attention_backend(
            underlying_attn_backend,  # type: ignore[arg-type]
            sink_len=sink_len,
        )
        Attention.__init__(
            self=self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            **kwargs,
        )

        self.sink_len = sink_len
        self.block_size = block_size
        self.sink_populated = False
        self.sink_key = None
        self.sink_value = None

    def update_sink_kv(self, sink_key, sink_value) -> None:
        self.sink_key = sink_key
        self.sink_value = sink_value

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        assert self.sink_key is not None and self.sink_value is not None, (
            "sink_key and sink_value have not been prepared"
        )
        if not self.sink_populated:
            forward_context: ForwardContext = get_forward_context()
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            torch.ops.vllm.maybe_populate_sink(self_kv_cache, self.layer_name)

        return super().forward(query, key, value, output_shape)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, output_shape)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def populate_sink_kv(self, self_kv_cache):
        sink_kv_slot_mapping = torch.arange(
            self.block_size,
            self.sink_len + self.block_size,
            device=torch.cuda.current_device(),
            dtype=torch.long,
        )
        triton_reshape_and_cache_flash_diffkv(
            self.sink_key,
            self.sink_value,
            self_kv_cache,
            sink_kv_slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
            self._v_scale,
        )
        # We only populate the sink_key and sink_value once
        self.sink_populated = True

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER

        return SinkFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            sink_len=self.sink_len,
            dtype=self.kv_cache_torch_dtype,
        )


def maybe_populate_sink(
    self_kv_cache: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if self.sink_populated or self_kv_cache.numel() == 0:
        return
    self.populate_sink_kv(self_kv_cache)


def maybe_populate_sink_fake(
    self_kv_cache: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_populate_sink",
    op_func=maybe_populate_sink,
    mutates_args=["self_kv_cache"],
    fake_impl=maybe_populate_sink_fake,
)
