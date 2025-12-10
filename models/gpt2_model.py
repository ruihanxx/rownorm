
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
import math

try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
except Exception:
    HFConv1D = type("___DummyConv1D___", (), {})  # 兜底占位，不会匹配任何模块

import types
import torch.nn.functional as F

# —— 1) 打补丁的入口：遍历模块，给每个 GPT2Attention 换 forward —— #
def apply_attn_logit_scaling(model: torch.nn.Module, mode: str = "d"):
    """
    将 GPT-2 注意力 logits 的缩放从 1/sqrt(d) 改为 1/d。
    用法：模型构建完成后调用一次即可。
    mode:
        - "d": 使用 1/d 作为缩放（你需要的）
        - "sqrt": 恢复为 1/sqrt(d)
    """
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未找到 transformers.models.gpt2.modeling_gpt2.GPT2Attention，"
            "请确认 transformers 版本或导入路径。"
        ) from e

    if mode not in {"d", "sqrt"}:
        raise ValueError("mode 必须是 {'d', 'sqrt'}")

    for m in model.modules():
        if isinstance(m, GPT2Attention):
            _patch_single_gpt2_attention(m, mode=mode)


# —— 2) 给单个 GPT2Attention 模块换 forward —— #
def _patch_single_gpt2_attention(attn_mod, mode: str = "d"):
    """
    针对 HF 的 GPT2Attention，替换 forward，使其在 logits 缩放时
    使用 1/d（或回退为 1/sqrt(d)）。
    """
    # 记录 head_dim
    head_dim = getattr(attn_mod, "head_dim", None)
    if head_dim is None:
        # 兼容一些老版本：head_dim = embed_dim // num_heads
        embed_dim = getattr(attn_mod, "embed_dim", None) or getattr(attn_mod, "embed_dim", None)
        num_heads = getattr(attn_mod, "num_heads", None)
        if embed_dim is not None and num_heads is not None:
            head_dim = embed_dim // num_heads
        else:
            raise RuntimeError("无法确定 head_dim。")

    # 目标缩放因子
    if mode == "d":
        scale_factor = 1.0 / float(head_dim)
    else:  # "sqrt"
        scale_factor = 1.0 / math.sqrt(float(head_dim))

    # 保存原始 forward 以便需要时参考（这里不再调用它，避免重复缩放）
    # orig_forward = attn_mod.forward

    # —— 判断是否走 SDPA 路径（新版本 transformers + PyTorch）—— #
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
        兼容不同 HF 版本的 GPT2Attention.forward：
        - 既支持 layer_past= 也支持 past_key_values=
        - 其它可能经由 **kwargs 传入的参数也统一从 kwargs 读取
        仅修改 logits 缩放为 scale_factor（1/d 或 1/sqrt(d)），其余逻辑保持一致
        """
        # 取参数（兼容不同命名）
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

        # 目标缩放（由外层 _patch_single_gpt2_attention 计算好并绑定在实例上）
        scale_factor = getattr(self, "_patched_scale_factor", None)
        if scale_factor is None:
            # 兜底：按默认 1/sqrt(d)
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

        # 3) 处理 kv cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # 4) 注意力（优先走 SDPA）
        if hasattr(F, "scaled_dot_product_attention"):
            # SDPA 不直接支持 head_mask，对输出再乘也可（大多训练里 head_mask=None）
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.to(torch.bool)
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=key_padding_mask,           # (B,T) Bool，True=mask
                dropout_p=dropout_p if self.training else 0.0,
                is_causal=True,                       # ★ 打开因果，上三角由 SDPA 生成
                scale=scale_factor
            )
            attn_weights_to_return = None
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale_factor  # ★ 关键替换
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            if head_mask is not None:
                # head_mask 形如 (H,) 或 (B,H,1,1)，与 HF 行为一致地逐头缩放
                attn_weights = attn_weights * head_mask
            if dropout_p and self.training:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            attn_output = torch.matmul(attn_weights, value)
            attn_weights_to_return = attn_weights if output_attentions else None

        # 5) 合并 heads + 输出映射
        attn_output = _merge_heads(attn_output)  # (B, T, C)
        attn_output = c_proj(attn_output)

        if output_attentions:
            # SDPA 路径下不返回权重（效率原因），与 HF 部分实现一致
            return attn_output, present, attn_weights_to_return
        return attn_output, present


    # 绑定新 forward
    attn_mod.forward = types.MethodType(patched_forward, attn_mod)
    # 把缩放系数挂在实例上，供 patched_forward 读取
    attn_mod._patched_scale_factor = scale_factor
    # 禁止默认的 sqrt 缩放（避免重复缩放）
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
    A: (..., m, n)  允许最后两维是 (m, n)，便于Conv展平后还原；若是严格2D也OK
    """
    assert A.dim() >= 2, "expects a tensor whose last two dims are (m, n)"
    p = float(p)
    X = A
    m, n = A.shape
    # 通用 p ∈ [1, ∞) 情形
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
    A: (..., m, n)  允许最后两维是 (m, n)，便于Conv展平后还原；若是严格2D也OK
    """
    assert A.dim() >= 2, "expects a tensor whose last two dims are (m, n)"
    p = float(p)
    X = A
    m, n = A.shape
    # 通用 p ∈ [1, ∞) 情形
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
    输入:
        G: 任意浮点类型张量，仅使用其 shape/device/dtype
        eps: 防止除零的微小常数
        generator: 可选的随机数发生器，用于可复现性
    输出:
        signs: 与 G 同形状的 Rademacher(±1 等概率) 张量
        Z_norm: 与 G 同形状的张量。每行独立 ~N(0,1)，再按该行 L1 范数归一化，使得每行的 L1 范数≈1
    说明:
        - 完全在 GPU 上运行（如果 G 在 CUDA 设备上）。
        - 使用 float32 做随机与归一化的中间计算，数值更稳健，然后再转回 G 的 dtype。
    """
    if not G.is_floating_point():
        raise TypeError("G 必须是浮点类型张量（如 float32/float16/bfloat16）。")

    out_dtype = G.dtype

    m, n = G.shape
    # --- 1) ±1 等概率（Rademacher）---
    # 用 rand_like + 阈值，可配合 generator 复现；中间用 float32 更稳健，再转回 dtype
    u = torch.rand_like(G, dtype=torch.float32)
    signs = (u.ge(0.5).to(torch.float32) * 2.0 - 1.0).to(out_dtype)

    # --- 2) 行 L1 归一化的高斯噪声 ---
    Z = torch.randn_like(G, dtype=torch.float32)  # 先以 float32 采样
    l1 = Z.abs().sum(dim=1, keepdim=True)                                           # 每行 L1 范数
    Z_norm = (Z / torch.clamp(l1, min=eps)).to(out_dtype)

    return signs.mul_(1/n)+ Z_norm
    

    



@torch.no_grad()
def init_model_params_rowlmo(model: nn.Module, p: float = 2.0, weight_decay = 0.1) -> None:
    # for m in model.modules():
    #     if isinstance(m, nn.Linear) and m.weight is not None:
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight, 1, weight_decay=weight_decay))
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight, p, weight_decay=weight_decay)).mul_(1/(m.weight.shape[1]))
    #         m.weight.copy_(_normalize_rows_dual(m.weight, p, weight_decay=weight_decay)).mul_(1/(m.weight.shape[1])**(1/p))
    #         if m.bias is not None:
    #             m.bias.zero_()
    #     elif isinstance(m, nn.Embedding) and m.weight is not None:
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight.T, 1, weight_decay=weight_decay).T)
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight.T, p, weight_decay=weight_decay).T).mul_(1/(m.weight.shape[1])**0.5)
    #         m.weight.copy_(_normalize_rows_dual(m.weight.T, p, weight_decay=weight_decay).T).mul_(1/(m.weight.shape[0])**(1/p))
    #         # print(m.weight.pow(2).sum(dim=0,keepdim=True).pow(1/2))
           
    #     elif isinstance(m, HFConv1D) and m.weight is not None:
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight.T, 1, weight_decay=weight_decay).T)
    #         # m.weight.copy_(_normalize_rows_lmo(m.weight.T, p, weight_decay=weight_decay).T)
    #         m.weight.copy_(_normalize_rows_dual(m.weight.T, p, weight_decay=weight_decay).T).mul_(1/(m.weight.shape[0])**(1/p))
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
    if model_name == "small":
        tok = GPT2TokenizerFast.from_pretrained("./gpt2")
    else:
        tok = GPT2TokenizerFast.from_pretrained("./gpt2_l")
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
    vocab_size=len(tokenizer),   # 由分词器决定
    n_positions=1024,            # 上下文长度（可改，例如 2048，但位置嵌入需重训/插值）
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=None,                # 不设时默认 4*n_embd
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,      # GPT-2 默认初始化 std
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,             # 训练时一般关掉 cache 省显存
    tie_word_embeddings=True     # GPT-2 通常词嵌入与LM头权重共享
)
    config_l = GPT2Config(
    vocab_size=len(tokenizer),   # 与分词器一致（GPT-2缺省50257）
    n_positions=1024,            # GPT-2 Large 默认 1024
    n_ctx=1024,
    n_embd=1280,                 # 关键：Large 的隐藏维度
    n_layer=36,                  # 关键：层数
    n_head=20,                   # 关键：注意力头数（1280/20=64 每头维度）
    n_inner=None,                # 留空=默认4*n_embd（即5120）；与GPT-2一致
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
    vocab_size=len(tokenizer),   # 与分词器一致（GPT-2缺省50257）
    n_positions=1024,            # GPT-2 Medium 默认 1024
    n_ctx=1024,
    n_embd=1024,                 # 关键：Medium 的隐藏维度
    n_layer=24,                  # 关键：层数（Medium）
    n_head=16,                   # 关键：注意力头数（1024/16=64 每头维度）
    n_inner=None,                # 留空=默认4*n_embd（即4096）；与GPT-2一致
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
    vocab_size=50257,   # 由分词器决定（建议 24k–32k）
    n_positions=1024,            # 上下文长度
    n_embd=192,                  # d_model
    n_layer=8,                   # 层数
    n_head=3,                    # 192/3=64 维每头；也可用 4 头(48维/头)
    n_inner=None,                # None 表示 4 * n_embd (=768)
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,             # 训练时关掉 cache 省显存
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
