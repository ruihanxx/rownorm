import torch
import torch.nn as nn


def _getattr_nested(obj, path: str, default=None):
    """Safely get nested attribute by dot-path, e.g. 'model.embed_tokens'."""
    cur = obj
    for part in path.split('.'):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _set_param_fans(param: torch.nn.Parameter, fin: int, fout: int):
    """Attach fan-in / fan-out info to a Parameter (as attributes)."""
    if hasattr(param, "data"):
        param.fan_in_override = int(fin)
        param.fan_out_override = int(fout)


def _conv_fans_from_weight(w: torch.Tensor):
    """Compute (fan_in, fan_out) for ConvNd weight.

    Torch ConvNd weight shapes:
      Conv1d: (out_channels, in_channels/groups, kW)
      Conv2d: (out_channels, in_channels/groups, kH, kW)
      Conv3d: (out_channels, in_channels/groups, kD, kH, kW)

    We follow PyTorch init convention: fan = channels * receptive_field.
    """
    if w is None or w.numel() == 0:
        return None
    if w.dim() < 3:
        return None
    out_channels = int(w.shape[0])
    in_channels_per_group = int(w.shape[1])
    receptive_field = 1
    for d in w.shape[2:]:
        receptive_field *= int(d)
    fan_in = in_channels_per_group * receptive_field
    fan_out = out_channels * receptive_field
    return fan_in, fan_out


def set_fans_llama(model: nn.Module):
    """Annotate LLaMA(-like) models' parameters with fan_in_override/fan_out_override.

    - nn.Linear: weight shape (out_features, in_features) => fin=in_features, fout=out_features
    - torch.nn.ConvNd: uses channels * receptive_field
    - transformers.pytorch_utils.Conv1D (GPT-style): weight shape (in_features, out_features)
    - nn.Embedding: weight shape (num_embeddings, embedding_dim) => fin=num_embeddings, fout=embedding_dim

    Additionally attaches lightweight tags on parameters (is_linear/is_ebd/is_conv/is_ln/is_qkv/is_proj...).

    This function is written for HuggingFace Transformers LLaMA-family models (LlamaForCausalLM,
    CodeLlama, Llama-2/3, etc.). It is defensive and will still work on "LLaMA-like" modules
    that follow similar naming conventions.
    """

    # Optional: HF Conv1D (mainly for GPT2; kept here for completeness)
    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D  # type: ignore
    except Exception:
        HFConv1D = type("___DummyConv1D___", (), {})

    # Optional: Llama RMSNorm class (Transformers)
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFLlamaRMSNorm  # type: ignore
    except Exception:
        HFLlamaRMSNorm = type("___DummyLlamaRMSNorm___", (), {})

    # --- detect token embedding + lm_head, and break tie if needed (so we can tag them differently) ---
    embed = None
    for p in (
        "model.embed_tokens",        # HF LlamaForCausalLM -> model (LlamaModel) -> embed_tokens
        "transformer.wte",           # compatibility with GPT-like wrappers
        "embed_tokens",              # bare LlamaModel
    ):
        cand = _getattr_nested(model, p, None)
        if isinstance(cand, nn.Embedding):
            embed = cand
            break

    lm_head = getattr(model, "lm_head", None)

    try:
        if (
            lm_head is not None
            and hasattr(lm_head, "weight")
            and embed is not None
            and hasattr(embed, "weight")
            and lm_head.weight is embed.weight
        ):
            # Clone lm_head weight to decouple (otherwise it can't be both embedding and linear).
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    except Exception:
        pass

    # --- main walk ---
    for mod_name, m in model.named_modules():

        # =============== Embedding ===============
        if isinstance(m, nn.Embedding):
            if getattr(m, "weight", None) is not None and m.weight.dim() == 2:
                W = m.weight
                # role tags
                W.is_ebd = 1
                W.is_linear = 0
                W.is_conv = 0
                # common LLaMA naming: model.embed_tokens
                if mod_name.endswith("embed_tokens"):
                    W.is_token_ebd = 1
                # fin/fout
                # weight: (num_embeddings, embedding_dim)  => vocab -> hidden
                fin = int(W.shape[0])
                fout = int(W.shape[1])
                _set_param_fans(W, fin=fin, fout=fout)
            continue

        # =============== Linear ===============
        # Note: quantized linears may not be nn.Linear; handle nn.Linear first.
        if isinstance(m, nn.Linear):
            if getattr(m, "weight", None) is not None and m.weight is not None and m.weight.dim() == 2:
                W = m.weight
                # role tags
                W.is_ebd = 0
                W.is_linear = 1
                W.is_conv = 0
                W.is_head = 1  # Match your GPT-2 code: mark all 2D linear weights with is_head=1

                # LLaMA attention projections
                if mod_name.endswith(".q_proj"):
                    W.is_q = 1
                    W.is_qkv = 1
                elif mod_name.endswith(".k_proj"):
                    W.is_k = 1
                    W.is_qkv = 1
                elif mod_name.endswith(".v_proj"):
                    W.is_v = 1
                    W.is_qkv = 1
                elif mod_name.endswith(".o_proj"):
                    W.is_proj = 1
                    W.is_attn_out = 1

                # LLaMA MLP projections
                if mod_name.endswith(".gate_proj"):
                    W.is_proj = 1
                    W.is_mlp_gate = 1
                elif mod_name.endswith(".up_proj"):
                    W.is_proj = 1
                    W.is_mlp_up = 1
                elif mod_name.endswith(".down_proj"):
                    W.is_proj = 1
                    W.is_mlp_down = 1

                # LM head
                if mod_name == "lm_head" or mod_name.endswith(".lm_head"):
                    W.is_lm_head = 1

                # nn.Linear weight: (out_features, in_features)
                out_features, in_features = W.shape
                _set_param_fans(W, fin=int(in_features), fout=int(out_features))

            if getattr(m, "bias", None) is not None and m.bias is not None and m.bias.dim() == 1:
                b = m.bias
                _set_param_fans(b, fin=1, fout=int(b.shape[0]))
            continue

        # =============== ConvNd ===============
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if getattr(m, "weight", None) is not None and m.weight is not None:
                W = m.weight
                W.is_conv = 1
                W.is_linear = 0
                W.is_ebd = 0
                fans = _conv_fans_from_weight(W)
                if fans is not None:
                    fin, fout = fans
                    _set_param_fans(W, fin=fin, fout=fout)
            if getattr(m, "bias", None) is not None and m.bias is not None and m.bias.dim() == 1:
                _set_param_fans(m.bias, fin=1, fout=int(m.bias.shape[0]))
            continue

        # =============== HF Conv1D (GPT-style) ===============
        if isinstance(m, HFConv1D):
            if getattr(m, "weight", None) is not None and m.weight is not None and m.weight.dim() == 2:
                W = m.weight
                W.is_conv = 1
                W.is_linear = 0
                W.is_ebd = 0
                # HF Conv1D: (in_features, out_features)
                in_features, out_features = W.shape
                _set_param_fans(W, fin=int(in_features), fout=int(out_features))
            if getattr(m, "bias", None) is not None and m.bias is not None and m.bias.dim() == 1:
                _set_param_fans(m.bias, fin=1, fout=int(m.bias.shape[0]))
            continue

        # =============== Norms (RMSNorm / LayerNorm) ===============
        # LLaMA uses RMSNorm; also be compatible with LayerNorm.
        if isinstance(m, (HFLlamaRMSNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None and m.weight is not None:
                m.weight.is_ln = 1
            continue
        # Name-based fallback for other RMSNorm implementations
        if mod_name.endswith(("input_layernorm", "post_attention_layernorm", ".norm", "norm")):
            if hasattr(m, "weight") and getattr(m, "weight") is not None and getattr(m, "weight").dim() == 1:
                m.weight.is_ln = 1

        # =============== Fallback for "Linear-like" modules ===============
        # Some low-bit / custom linear layers are not nn.Linear but expose weight + in/out features.
        # If it looks like a 2D weight matrix, we still set fans.
        if hasattr(m, "weight") and getattr(m, "weight") is not None:
            W = getattr(m, "weight")
            if isinstance(W, torch.Tensor) and W.dim() == 2:
                # Try to infer in/out features
                if hasattr(m, "in_features") and hasattr(m, "out_features"):
                    fin = int(getattr(m, "in_features"))
                    fout = int(getattr(m, "out_features"))
                else:
                    # assume (out, in) like nn.Linear
                    fout, fin = int(W.shape[0]), int(W.shape[1])
                _set_param_fans(W, fin=fin, fout=fout)


__all__ = ["set_fans_llama"]
