# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from typing import Optional
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    TENorm = None  # Will be set to LNImpl below

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f"Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False

# Set TENorm to LNImpl if TransformerEngine is not available
if TENorm is None:
    TENorm = LNImpl


def encoder_model_with_transformer_engine_default_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    """T5 encoder TE spec (uses Transformer Engine components).

    num_experts, moe_grouped_gemm and moe_use_legacy_grouped_gemm are
    optional hooks to create MoE MLP specs when requested by the model
    config. Defaults maintain backward compatibility.
    """

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=get_mlp_module_spec(
                use_te=True,
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def decoder_model_with_transformer_engine_default_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    """T5 decoder TE spec (uses Transformer Engine components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=TENorm,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            mlp=get_mlp_module_spec(
                use_te=True,
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def encoder_model_with_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    """T5 encoder local spec (uses Megatron-Core components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=get_mlp_module_spec(
                use_te=False,
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )


def decoder_model_with_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    """T5 decoder local spec (uses Megatron-Core components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=LNImpl,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=CrossAttentionSubmodules(
                    linear_q=ColumnParallelLinear,
                    linear_kv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=get_mlp_module_spec(
                use_te=False,
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )


def get_t5_encoder_with_transformer_engine_block_spec(
    num_layers: int,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> TransformerBlockSubmodules:
    """T5 encoder block spec for Transformer Engine

    Args:
      config (TransformerConfig): config, containing number of layers for encoder
    """

    layer_spec = encoder_model_with_transformer_engine_default_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers, layer_norm=TENorm)
    return block_spec


def get_t5_decoder_with_transformer_engine_block_spec(
    num_layers: int,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> TransformerBlockSubmodules:
    """T5 decoder block spec for Transformer Engine

    Args:
      config (TransformerConfig): config, containing number of layers for decoder
    """

    layer_spec = decoder_model_with_transformer_engine_default_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers, layer_norm=TENorm)
    return block_spec


def get_t5_encoder_with_local_block_spec(
    num_layers: int,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> TransformerBlockSubmodules:
    """T5 encoder block spec for local (uses Megatron-Core components)

    Args:
      num_layers (int): number of encoder layers
    """

    layer_spec = encoder_model_with_local_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers, layer_norm=TENorm)
    return block_spec


def get_t5_decoder_with_local_block_spec(
    num_layers: int,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> TransformerBlockSubmodules:
    """T5 decoder block spec for local (uses Megatron-Core components)

    Args:
      num_layers (int): number of decoder layers
    """

    layer_spec = decoder_model_with_local_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers, layer_norm=TENorm)
    return block_spec
