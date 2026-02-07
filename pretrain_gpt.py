# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import json
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import traceback

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, get_tokenizer, inprocess_restart, pretrain, print_rank_0
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report

# def inspect_gpt_moe_model(model: GPTModel):
#     """Print parameter counts and MoE layer statistics."""

#     from megatron.core.utils import get_attr_wrapped_model
#     from megatron.training import print_rank_0

#     # unwrap DDP / FSDP / pipeline wrappers
#     model = get_attr_wrapped_model(model, "module")

#     total_params = 0
#     moe_params = 0
#     dense_params = 0

#     moe_layers = []
#     total_layers = 0

#     for name, module in model.named_modules():
#         # Transformer layers
#         if hasattr(module, "mlp"):
#             total_layers += 1

#             mlp = module.mlp
#             is_moe = hasattr(mlp, "experts") or hasattr(mlp, "expert_parallel_group")

#             if is_moe:
#                 moe_layers.append(total_layers - 1)

#             for p in mlp.parameters(recurse=True):
#                 n = p.numel()
#                 total_params += n
#                 if is_moe:
#                     moe_params += n
#                 else:
#                     dense_params += n

#         # count non-MLP parameters once
#     for p in model.parameters():
#         total_params += 0  # already counted above, avoid double count

#     print_rank_0("========== GPT MoE Model Inspection ==========")
#     print_rank_0(f"Total Transformer layers     : {total_layers}")
#     print_rank_0(f"MoE layers indices (0-based) : {moe_layers}")
#     print_rank_0(f"Number of MoE layers         : {len(moe_layers)}")
#     print_rank_0(f"Total parameters             : {total_params:,}")
#     print_rank_0(f"MoE parameters               : {moe_params:,}")
#     print_rank_0(f"Dense MLP parameters         : {dense_params:,}")
#     print_rank_0("=============================================")

def inspect_gpt_moe_model(model: GPTModel):
    """Print parameter counts and MoE layer statistics (correct version)."""

    from megatron.core.utils import get_attr_wrapped_model
    from megatron.training import print_rank_0

    # unwrap DDP / FSDP / pipeline wrappers
    model = get_attr_wrapped_model(model, "module")

    # ---------- 1. total parameters ----------
    total_params = sum(p.numel() for p in model.parameters())

    moe_params = 0
    moe_layers = []
    total_layers = 0

    # ---------- 2. MoE layer detection ----------
    for _, module in model.named_modules():
        if hasattr(module, "mlp"):
            total_layers += 1
            mlp = module.mlp

            is_moe = hasattr(mlp, "experts")
            if is_moe:
                moe_layers.append(total_layers - 1)

                # only experts count as MoE parameters
                for p in mlp.experts.parameters():
                    moe_params += p.numel()

    non_moe_params = total_params - moe_params

    # ---------- 3. pretty print ----------
    def human(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)

    print_rank_0("========== GPT MoE Model Inspection ==========")
    print_rank_0(f"Total Transformer layers     : {total_layers}")
    print_rank_0(f"MoE layers indices (0-based) : {moe_layers}")
    print_rank_0(f"Number of MoE layers         : {len(moe_layers)}")
    print_rank_0(
        f"Total parameters             : {human(total_params)} ({total_params:,})"
    )
    print_rank_0(
        f"MoE expert parameters        : {human(moe_params)} ({moe_params:,})"
    )
    print_rank_0(
        f"Non-MoE parameters           : {human(non_moe_params)} ({non_moe_params:,})"
    )
    print_rank_0("=============================================")

# # Global flag to indicate backward phase
# _IN_BACKWARD = False

# def _mark_backward_hook(grad):
#     global _IN_BACKWARD
#     _IN_BACKWARD = True
#     return grad

# ============================================================
# Zero-intrusion patch for torch.autograd.backward
# Ensures _IN_BACKWARD is reset strictly after backward finishes
# ============================================================

# _orig_autograd_backward = None

# def install_backward_reset_patch_once():
#     global _orig_autograd_backward

#     if getattr(install_backward_reset_patch_once, "_installed", False):
#         return

#     # Save original backward
#     _orig_autograd_backward = torch.autograd.backward

#     def wrapped_autograd_backward(*args, **kwargs):
#         global _IN_BACKWARD
#         try:
#             return _orig_autograd_backward(*args, **kwargs)
#         finally:
#             # This is the ONLY place guaranteed to run
#             # after the entire backward graph has finished
#             _IN_BACKWARD = False

#     torch.autograd.backward = wrapped_autograd_backward
#     install_backward_reset_patch_once._installed = True

#     print_rank_0("[DEBUG][AUTOGRAD] torch.autograd.backward patched (reset _IN_BACKWARD)")

def install_alltoall_debug_patch_once():
    """Monkey-patch torch.distributed.all_to_all_single to log every call.

    This captures both forward and backward all-to-all(v) communications,
    including those hidden by autograd/kernel fusion.
    """
    if getattr(install_alltoall_debug_patch_once, "_installed", False):
        return

    if not (dist.is_available() and dist.is_initialized()):
        # In Megatron, dist should be initialized before forward_step is ever called.
        # If this triggers, your call site is too early.
        # print_rank_0("[DEBUG][NCCL] dist not initialized; skip patch for now.")
        return

    orig = dist.all_to_all_single

    def wrapped_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        # Best-effort phase tagging: during backward, autograd is executing and grad mode is disabled.
        # phase = "BW" if _IN_BACKWARD else "FW"

        # Basic tensor stats
        try:
            in_shape = tuple(input.shape)
            in_bytes = input.numel() * input.element_size()
        except Exception:
            in_shape, in_bytes = ("<unknown>",), -1

        # Rank info
        try:
            rank = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rank = -1

        # print_rank_0(
        #     f"[DEBUG][NCCL][rank={rank}] all_to_all_single "
        #     f"in_shape={in_shape} bytes={in_bytes/1e6:.2f}MB "
        #     f"out_splits={output_split_sizes} in_splits={input_split_sizes} async={async_op}"
        # )

        return orig(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op
        )

    dist.all_to_all_single = wrapped_all_to_all_single
    install_alltoall_debug_patch_once._installed = True
    # print_rank_0("[DEBUG][NCCL] all_to_all_single monkey-patch installed.")

def _unwrap_model(m):
    # unwrap DDP / FSDP / Float16Module / pipeline wrappers
    from megatron.core.utils import get_attr_wrapped_model
    for attr in ["module", "model", "language_model"]:
        try:
            m = get_attr_wrapped_model(m, attr)
        except Exception:
            pass
    # 有些 wrapper 仍然用 .module
    while hasattr(m, "module"):
        m = m.module
    return m


def _get_any_moe_dispatcher(model):
    """Return (dispatcher, moe_layer_id) if found, else (None, None)."""
    m = _unwrap_model(model)

    # 方式A：直接遍历所有子模块，找 token_dispatcher（最稳）
    for name, mod in m.named_modules():
        # MoE MLP 通常会挂 token_dispatcher
        if hasattr(mod, "token_dispatcher"):
            return getattr(mod, "token_dispatcher"), name

    return None, None


def _get_all_moe_dispatchers(model):
    """Return list of (dispatcher, moe_block_id)."""
    m = _unwrap_model(model)
    dispatchers = []
    for name, mod in m.named_modules():
        if hasattr(mod, "token_dispatcher"):
            dispatchers.append((mod.token_dispatcher, name))
    return dispatchers


def _next_trace_step_id():
    if not hasattr(_next_trace_step_id, "i"):
        _next_trace_step_id.i = 0
    i = _next_trace_step_id.i
    _next_trace_step_id.i += 1
    return i

def _install_moe_layer_id_hooks_once(model):
    m = _unwrap_model(model)
    installed = getattr(_install_moe_layer_id_hooks_once, "_installed", False)
    if installed:
        return

    for name, mod in m.named_modules():
        if hasattr(mod, "token_dispatcher"):
            disp = mod.token_dispatcher

            # forward_pre_hook：进入该 MoE 模块时，设置当前 dispatcher 的层 id
            def _pre_hook(module, inputs, _disp=disp, _name=name):
                try:
                    _disp._current_moe_block_id = _name
                except Exception:
                    pass

            # forward_hook：离开该 MoE 模块时，可选清理（不清也行，但清更干净）
            def _post_hook(module, inputs, outputs, _disp=disp):
                try:
                    _disp._current_moe_block_id = None
                except Exception:
                    pass

            mod.register_forward_pre_hook(_pre_hook)
            mod.register_forward_hook(_post_hook)

    _install_moe_layer_id_hooks_once._installed = True
    print_rank_0("[TRACE] MoE dispatcher layer-id hooks installed.")

def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()

    # if not hasattr(forward_step, "_printed_iter_fields"):
    #     cand = ["iteration", "train_iteration", "consumed_train_samples", "consumed_train_tokens"]
    #     for k in cand:
    #         if hasattr(args, k):
    #             print_rank_0(f"[TRACE] args.{k} = {getattr(args, k)}")
    #     forward_step._printed_iter_fields = True

    install_alltoall_debug_patch_once()
    # install_backward_reset_patch_once()

     # >>>>>>> 新增 BEGIN <<<<<<<<
    if not hasattr(forward_step, "_moe_inspected"):
        inspect_gpt_moe_model(model)
        forward_step._moe_inspected = True
    # >>>>>>> 新增 END <<<<<<<<

    _install_moe_layer_id_hooks_once(model)

    # # -------- 新增：trace batch begin --------
    # dispatcher, moe_layer_id = _get_any_moe_dispatcher(model)
    # if not hasattr(forward_step, "_moe_dispatcher_checked"):
    #     print_rank_0(f"[TRACE] moe dispatcher found? {dispatcher is not None}, id={moe_layer_id}")
    #     forward_step._moe_dispatcher_checked = True

    # if dispatcher is not None:
    #     # iteration 在 Megatron 中等价于 batch_id
    #     batch_id = _next_trace_step_id()
    #     dispatcher.start_trace_batch(
    #         batch_id=batch_id,
    #         moe_block_id=moe_layer_id,
    #     )

    # -------- trace: batch begin (ONCE) --------
    dispatchers = _get_all_moe_dispatchers(model)
    if dispatchers:
        # batch_id = _next_trace_step_id()
        # batch_id = getattr(args, "iteration", None)
        # if batch_id is None:
        #     batch_id = _next_trace_step_id()
        batch_id = _next_trace_step_id()

        # 任取一个 dispatcher 启动 batch（它们共享 recorder）
        dispatchers[0][0].start_trace_batch(
            batch_id=batch_id,
            moe_block_id=None,  # batch 级，不绑定具体 MoE 层
        )

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # # ---- BEGIN: register backward marker hook ----
    # if not hasattr(forward_step, "_bw_hook_registered"):
    #     # output_tensor 一定会参与 loss.backward()
    #     output_tensor.register_hook(_mark_backward_hook)
    #     forward_step._bw_hook_registered = True
    # # ---- END: register backward marker hook ----

    # # -------- 新增：trace batch end --------
    # if dispatcher is not None:
    #     dispatcher.end_trace_batch()
         # -------- trace: batch end (ONCE) --------
    if dispatchers:
        dispatchers[0][0].end_trace_batch()

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
