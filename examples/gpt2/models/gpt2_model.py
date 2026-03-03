
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
import math

try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
except Exception:
    # Fallback placeholder that will not match any real module
    HFConv1D = type("___DummyConv1D___", (), {})

import types
import torch.nn.functional as F

# —— 1) Patch entry: walk modules and replace forward for each GPT2Attention —— #
def apply_attn_logit_scaling(model: torch.nn.Module, mode: str = "d"):
    """
    Change GPT-2 attention logits scaling from 1/sqrt(d) to 1/d.
    Usage: call once after the model has been constructed.
    mode:
        - "d": use 1/d as the scaling factor
        - "sqrt": restore the default 1/sqrt(d) scaling
    """
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not find transformers.models.gpt2.modeling_gpt2.GPT2Attention; "
            "please check your transformers version or import path."
        ) from e

    if mode not in {"d", "sqrt"}:
        raise ValueError("mode must be {'d', 'sqrt'}")

    for m in model.modules():
        if isinstance(m, GPT2Attention):
            _patch_single_gpt2_attention(m, mode=mode)


# —— 2) Replace forward for a single GPT2Attention module —— #
def _patch_single_gpt2_attention(attn_mod, mode: str = "d"):
    """
    For HF's GPT2Attention, replace its forward so that attention logits are
    scaled by 1/d (or fall back to 1/sqrt(d)).
    """
    # Record head_dim
    head_dim = getattr(attn_mod, "head_dim", None)
    if head_dim is None:
        # Support some older versions: head_dim = embed_dim // num_heads
        embed_dim = getattr(attn_mod, "embed_dim", None) or getattr(attn_mod, "embed_dim", None)
        num_heads = getattr(attn_mod, "num_heads", None)
        if embed_dim is not None and num_heads is not None:
            head_dim = embed_dim // num_heads
        else:
            raise RuntimeError("Could not determine head_dim.")

    # Target scaling factor
    if mode == "d":
        scale_factor = 1.0 / float(head_dim)
    else:  # "sqrt"
        scale_factor = 1.0 / math.sqrt(float(head_dim))

    # Keep a reference to the original forward if needed (we no longer call it to avoid double scaling)
    # orig_forward = attn_mod.forward

    # —— Decide whether to use SDPA path (newer transformers + PyTorch) —— #
    use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def _split_heads(x, num_heads, head_dim):
        # (B, T, C) -> (B, num_heads, T, head_dim)
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(*new_shape)  # (B, T, num_heads, head_dim)
        return x.permute(0, 2, 1, 3)  # (B, num_heads, T, head_dim)

    def _merge_heads(x):
        # (B, num_heads, T, head_dim) -> (B, T, num_heads*head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    @torch.no_grad()
    def _warn_once(msg):
        if not hasattr(attn_mod, "_scale_warned"):
            attn_mod._scale_warned = True
            print(f"[GPT2Attention patch] {msg}")

    def patched_forward(self, hidden_states, **kwargs):
        """
        Compatible wrapper around different HF GPT2Attention.forward variants:
        - supports both layer_past= and past_key_values=
        - pulls other arguments from **kwargs as needed
        Only changes the logits scaling to scale_factor (1/d or 1/sqrt(d)); all
        other logic is kept identical to HF.
        """
        # Read arguments (compatible with different naming conventions)
        layer_past = kwargs.get("layer_past", None)
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is not None and layer_past is None:
            layer_past = past_key_values

        attention_mask         = kwargs.get("attention_mask", None)
        head_mask              = kwargs.get("head_mask", None)
        encoder_hidden_states  = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        use_cache              = kwargs.get("use_cache", False)
        output_attentions      = kwargs.get("output_attentions", False)

        c_attn = self.c_attn
        c_proj = self.c_proj
        num_heads = self.num_heads
        head_dim  = self.head_dim
        dropout_p = self.attn_dropout.p if hasattr(self.attn_dropout, "p") else 0.0

        # Target scaling factor (computed in _patch_single_gpt2_attention and attached to the instance)
        scale_factor = getattr(self, "_patched_scale_factor", None)
        if scale_factor is None:
            # Fallback to default 1/sqrt(d)
            scale_factor = 1.0 / math.sqrt(float(head_dim))

        def _split_heads(x, num_heads, head_dim):
            new_shape = x.size()[:-1] + (num_heads, head_dim)
            x = x.view(*new_shape)             # (B, T, H, D)
            return x.permute(0, 2, 1, 3)       # (B, H, T, D)

        def _merge_heads(x):
            # (B, H, T, D) -> (B, T, H*D)
            x = x.permute(0, 2, 1, 3).contiguous()
            new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
            return x.view(*new_shape)

        is_cross_attn = encoder_hidden_states is not None

        # 1) Q,K,V
        if is_cross_attn:
            qkv = c_attn(hidden_states)
            query, _, _ = qkv.split(self.split_size, dim=2)
            kv = c_attn(encoder_hidden_states)
            key, value, _ = kv.split(self.split_size, dim=2)
        else:
            x = c_attn(hidden_states)  # (B, T, 3*C)
            query, key, value = x.split(self.split_size, dim=2)

        # 2) reshape to (B, H, T, D)
        query = _split_heads(query, num_heads, head_dim)
        key   = _split_heads(key,   num_heads, head_dim)
        value = _split_heads(value, num_heads, head_dim)

        # 3) Handle kv cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # 4) Attention (prefer SDPA when available)
        if hasattr(F, "scaled_dot_product_attention"):
            # SDPA does not directly support head_mask; multiply it on the output (head_mask is usually None in training)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.to(torch.bool)
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=key_padding_mask,           # (B,T) Bool, True = mask
                dropout_p=dropout_p if self.training else 0.0,
                is_causal=True,                       # Enable causal masking; SDPA creates the upper-triangular mask
                scale=scale_factor
            )
            attn_weights_to_return = None
        else:
            # Fallback attention path using explicit matmul and custom scaling
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale_factor
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            if head_mask is not None:
                # head_mask has shape like (H,) or (B,H,1,1); scale each head as HF does
                attn_weights = attn_weights * head_mask
            if dropout_p and self.training:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            attn_output = torch.matmul(attn_weights, value)
            attn_weights_to_return = attn_weights if output_attentions else None

        # 5) Merge heads + output projection
        attn_output = _merge_heads(attn_output)  # (B, T, C)
        attn_output = c_proj(attn_output)

        if output_attentions:
            # SDPA path does not return weights (for efficiency), matching some HF implementations
            return attn_output, present, attn_weights_to_return
        return attn_output, present


    # Bind the new forward
    attn_mod.forward = types.MethodType(patched_forward, attn_mod)
    # Attach the scaling factor on the instance for patched_forward to read
    attn_mod._patched_scale_factor = scale_factor
    # Disable the default sqrt scaling to avoid double scaling
    if hasattr(attn_mod, "scale_attn"):
        attn_mod.scale_attn = False





__all__ = [
    "load_tokenizer",
    "build_gpt2_small",
    "build_model_from_name",
]


@torch.no_grad()
def _normalize_rows_dual(A: torch.Tensor, p: float, eps: float = 1e-7, weight_decay = 0.1) -> torch.Tensor:
    """
    Row-wise: row <- sign(row) * |row|^{p-1} / ||row||_p^{p-1} / din^{1/p}
    A: (..., m, n)  where the last two dims are (m, n); works both for strictly 2D
       matrices and Conv weights that have been flattened to 2D.
    """
    assert A.dim() >= 2, "expects a tensor whose last two dims are (m, n)"
    p = float(p)
    X = A
    m, n = A.shape
    # Generic p ∈ [1, ∞) case (commented-out alternative formulation)
    # sign(x) * |x|^{p-1}
    # V = torch.sign(X) * X.abs().clamp_min(eps).pow(max(p - 1.0, 0.0))
    # # ||row||_p   -> (B, m, 1)
    # lp = X.abs().clamp_min(eps).pow(p).sum(dim=1, keepdim=True).pow(1.0 / p)
    # # ||row||_p^{p-1} * n^{1/p}
    # scale = lp.clamp_min(eps).pow(max(p - 1.0, 0.0)) * (n ** (1.0 / p))
    # # scale = lp.clamp_min(eps).pow(max(p - 1.0, 0.0)) 
    # Y = V / scale
    if p>1:
        q = p/(p-1)
        lq = X.abs().clamp_min(eps).pow(q).sum(dim=1, keepdim=True).pow(1.0 / q)
        return X/lq
    elif p==1:
        Y = torch.sign(X)
        return Y

def _normalize_rows_lmo(A: torch.Tensor, p: float, eps: float = 1e-7, weight_decay = 0.1) -> torch.Tensor:
    """
    Row-wise: row <- sign(row) * |row|^{p-1} / ||row||_p^{p-1} / din^{1/p}
    A: (..., m, n)  where the last two dims are (m, n); works both for strictly 2D
       matrices and Conv weights that have been flattened to 2D.
    """
    assert A.dim() >= 2, "expects a tensor whose last two dims are (m, n)"
    p = float(p)
    X = A
    m, n = A.shape
    # Generic p ∈ [1, ∞) case (commented-out alternative formulation)
    # sign(x) * |x|^{p-1}
    # V = torch.sign(X) * X.abs().clamp_min(eps).pow(max(p - 1.0, 0.0))
    # # ||row||_p   -> (B, m, 1)
    # lp = X.abs().clamp_min(eps).pow(p).sum(dim=1, keepdim=True).pow(1.0 / p)
    # # ||row||_p^{p-1} * n^{1/p}
    # scale = lp.clamp_min(eps).pow(max(p - 1.0, 0.0)) * (n ** (1.0 / p))
    # # scale = lp.clamp_min(eps).pow(max(p - 1.0, 0.0)) 
    # Y = V / scale
    if p>1:
        norm = X.abs().clamp_min(eps).pow(p).sum(dim=1, keepdim=True).pow(1 / p).pow(p-1)
        lq = X.abs().pow(p-1) * X.sign()
        return lq/norm
    elif p==1:
        Y = torch.sign(X)
        return Y

@torch.no_grad()
def _mixnormalize(G: torch.Tensor, eps: float = 1e-12):
    """
    Inputs:
        G: arbitrary floating tensor; only its shape/device/dtype are used
        eps: small constant to avoid division by zero
        generator: optional RNG for reproducibility
    Outputs:
        signs: Rademacher(±1) tensor with the same shape as G
        Z_norm: tensor with the same shape as G. Each row is ~N(0,1) and then
                L1-normalized so that the row L1 norm is approximately 1.
    Notes:
        - Runs entirely on GPU when G is on a CUDA device.
        - Uses float32 for the random and normalization steps for better numerical
          stability, then casts back to G.dtype.
    """
    if not G.is_floating_point():
        raise TypeError("G must be a floating-point tensor (e.g. float32/float16/bfloat16).")

    out_dtype = G.dtype

    m, n = G.shape
    # --- 1) ±1 with equal probability (Rademacher) ---
    # Use rand_like + threshold; can work with a generator for reproducibility.
    # Do the work in float32 for stability, then cast back to out_dtype.
    u = torch.rand_like(G, dtype=torch.float32)
    signs = (u.ge(0.5).to(torch.float32) * 2.0 - 1.0).to(out_dtype)

    # --- 2) Row-wise L1-normalized Gaussian noise ---
    Z = torch.randn_like(G, dtype=torch.float32)  # sample in float32 first
    l1 = Z.abs().sum(dim=1, keepdim=True)                                           # L1 norm per row
    Z_norm = (Z / torch.clamp(l1, min=eps)).to(out_dtype)

    return signs.mul_(1/n)+ Z_norm
    

    



@torch.no_grad()
def init_model_params_rowlmo(model: nn.Module, p: float = 2.0, weight_decay = 0.1) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight is not None:
            m.weight.copy_(_normalize_rows_lmo(m.weight, p, weight_decay=weight_decay)).mul_(1/(m.weight.shape[1])**(1/p))
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.Embedding) and m.weight is not None: 
            m.weight.copy_(_normalize_rows_lmo(m.weight.T, p, weight_decay=weight_decay).T).mul_(1/(m.weight.shape[0])**(1/p))
        elif isinstance(m, HFConv1D) and m.weight is not None:
            m.weight.copy_(_normalize_rows_lmo(m.weight.T, p, weight_decay=weight_decay).T).mul_(1/(m.weight.shape[0])**(1/p))

@torch.no_grad()
def init_model_params_rowlmo_rowmix(model: nn.Module, p: float = 2.0, weight_decay = 0.1) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight is not None:
            m.weight.copy_(_mixnormalize(m.weight))
            
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.Embedding) and m.weight is not None:
            m.weight.copy_(_mixnormalize(m.weight.T).T)
            
           
        elif isinstance(m, HFConv1D) and m.weight is not None:
            m.weight.copy_(_mixnormalize(m.weight.T).T)
            


def load_tokenizer(model_name: str):
    """
    Load a GPT-2 tokenizer and ensure pad_token exists.
    """
    
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _post_init_resize(model: GPT2LMHeadModel, tokenizer):
    # Ensure model knows pad token and embedding size matches tokenizer size
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

def build_model_from_name(model_name: str, tokenizer=None, device: str = "cpu", data_parallel: bool = True, use_d=0):
    """
    Generic builder for GPT-2 family (gpt2, gpt2-medium, gpt2-large).
    Returns (model, tokenizer).
    """
   
    if tokenizer is None:
        tokenizer = load_tokenizer('large')
    config_0 = GPT2Config(
    vocab_size=len(tokenizer),   # determined by the tokenizer
    n_positions=1024,            # context length (can be changed, e.g. 2048, but position embeddings then need retraining/interpolation)
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=None,                # defaults to 4 * n_embd when unset
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,      # GPT-2 default initialization std
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,             # usually disable cache during training to save memory
    tie_word_embeddings=True     # GPT-2 typically ties token embeddings and LM head weights
)
    config_l = GPT2Config(
    vocab_size=len(tokenizer),   # same as tokenizer (GPT-2 default is 50257)
    n_positions=1024,            # GPT-2 Large default is 1024
    n_ctx=1024,
    n_embd=1280,                 # Large hidden size
    n_layer=36,                  # number of layers
    n_head=20,                   # number of attention heads (1280/20=64 dims per head)
    n_inner=None,                # default 4 * n_embd (=5120), as in GPT-2
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,  
    use_cache=False,
    tie_word_embeddings=True      
)
    config_m = GPT2Config(
    vocab_size=len(tokenizer),   # same as tokenizer (GPT-2 default is 50257)
    n_positions=1024,            # GPT-2 Medium default is 1024
    n_ctx=1024,
    n_embd=1024,                 # Medium hidden size
    n_layer=24,                  # number of layers (Medium)
    n_head=16,                   # number of attention heads (1024/16=64 dims per head)
    n_inner=None,                # default 4 * n_embd (=4096), as in GPT-2
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
    tie_word_embeddings=True
)
    config_xl = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_layer=48,
        n_head=25,
        n_embd=1600,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        initializer_range=0.02,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  
        use_cache=False,
        tie_word_embeddings=True
    )
    config_xs = GPT2Config(
    vocab_size=50257,   # determined by the tokenizer (typically 24k–32k)
    n_positions=1024,            # context length
    n_embd=192,                  # d_model
    n_layer=8,                   # number of layers
    n_head=3,                    # 192/3=64 dims per head; could also use 4 heads (48 dims/head)
    n_inner=None,                # None means 4 * n_embd (=768)
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,             # disable cache during training to save memory
    tie_word_embeddings=True
)

    if model_name == 'small':
        model = GPT2LMHeadModel(config_0)
    elif model_name == 'medium':
        model = GPT2LMHeadModel(config_m)
    elif model_name == 'large':
        model = GPT2LMHeadModel(config_l)
    elif model_name == 'xl':
        model = GPT2LMHeadModel(config_xl)
    elif model_name == 'xs':
        model = GPT2LMHeadModel(config_xs)
    _post_init_resize(model, tokenizer)
    if use_d:
        print('use d in logit')
        apply_attn_logit_scaling(model, mode="d")
    # if data_parallel and (device == "cuda") and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(device)
    return model, tokenizer

def build_gpt2(device: str = "cpu", data_parallel: bool = False, gpt2name = 'small', use_d = 0):
    """
    Paper-compatible GPT-2 small (124M): 'gpt2' in HF hub.
    Returns (model, tokenizer).
    """
    tokenizer = load_tokenizer('large')
    return build_model_from_name(gpt2name, tokenizer=tokenizer, device=device, data_parallel=data_parallel, use_d = use_d)
