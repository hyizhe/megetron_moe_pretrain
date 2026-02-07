# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain BERT"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.bert.bert_model import BertModel
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.bert.bert_layer_specs import bert_layer_local_spec, get_bert_layer_local_spec
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core import mpu, tensor_parallel
from megatron.core.tokenizers import MegatronTokenizer
 
# ---- MoE trace helpers (ported from pretrain_gpt) ----
def _unwrap_model(m):
    # unwrap DDP / FSDP / pipeline wrappers
    from megatron.core.utils import get_attr_wrapped_model
    for attr in ["module", "model", "language_model"]:
        try:
            m = get_attr_wrapped_model(m, attr)
        except Exception:
            pass
    while hasattr(m, "module"):
        m = m.module
    return m


def _get_all_moe_dispatchers(model):
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

            def _pre_hook(module, inputs, _disp=disp, _name=name):
                try:
                    _disp._current_moe_block_id = _name
                except Exception:
                    pass

            def _post_hook(module, inputs, outputs, _disp=disp):
                try:
                    _disp._current_moe_block_id = None
                except Exception:
                    pass

            mod.register_forward_pre_hook(_pre_hook)
            mod.register_forward_hook(_post_hook)

    _install_moe_layer_id_hooks_once._installed = True
    print_rank_0("[TRACE] MoE dispatcher layer-id hooks installed.")

# ---- end MoE trace helpers ----


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    num_tokentypes = 2 if args.bert_binary_head else 0

    if args.use_legacy_models:
        model = megatron.legacy.model.BertModel(
            config=config,
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)
    else:
        if args.spec is None:
            # Use local bert layer spec; if num_experts is provided, enable MoE
            transformer_layer_spec = get_bert_layer_local_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=getattr(args, 'moe_grouped_gemm', False),
                moe_use_legacy_grouped_gemm=getattr(args, 'moe_use_legacy_grouped_gemm', False),
            )
        elif args.spec[0] == 'local':
            print_rank_0('Using Local spec for transformer layers')
            transformer_layer_spec = get_bert_layer_local_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=getattr(args, 'moe_grouped_gemm', False),
                moe_use_legacy_grouped_gemm=getattr(args, 'moe_use_legacy_grouped_gemm', False),
            )
        else :
            transformer_layer_spec = import_module(args.spec)

        model = BertModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels',
            'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}
    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    # Install layer-id hooks and start batch trace (if any MoE dispatchers exist)
    _install_moe_layer_id_hooks_once(model)
    dispatchers = _get_all_moe_dispatchers(model)
    if dispatchers:
        batch_id = _next_trace_step_id()
        try:
            dispatchers[0][0].start_trace_batch(batch_id, moe_block_id=None)
        except Exception:
            pass

    output_tensor = model(tokens, padding_mask,
                          tokentype_ids=types, lm_labels=lm_labels)

    # End batch trace if started
    if dispatchers:
        try:
            dispatchers[0][0].end_trace_batch()
        except Exception:
            pass

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build train, valid, and test datasets."""
    args = get_args()

    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    config = BERTMaskedWordPieceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        masking_probability=args.mask_prob,
        short_sequence_probability=args.short_seq_prob,
        masking_max_ngram=3,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=True,
        classification_head=args.bert_binary_head,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
    )

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        BERTMaskedWordPieceDataset,
        train_val_test_num_samples,
        lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()
    
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
