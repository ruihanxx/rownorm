
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
import math

try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
except Exception:
    HFConv1D = type("___DummyConv1D___", (), {})  # 兜底占位，不会匹配任何模块



__all__ = [
    "load_tokenizer",
    "build_gpt2_small",
    "build_model_from_name",
]


@torch.no_grad()
def _normalize_rows_lmo(A: torch.Tensor, p: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Row-wise: row <- sign(row) * |row|^{p-1} / ||row||_p^{p-1} / din^{1/p}
    A: (..., m, n)  允许最后两维是 (m, n)，便于Conv展平后还原；若是严格2D也OK
    """
    assert A.dim() >= 2, "expects a tensor whose last two dims are (m, n)"
    p = float(p)
    
    X = A
    m, n = A.shape
    if math.isinf(p) or p >= 1e5:
        # —— 可选的 p = +∞ 极限：每行仅保留 |x| 最大位置，置为 -sign(max)（你之前的特例）
        # 若不需要这个分支，可以删除
        Y = torch.zeros_like(X)
        idx = X.abs().argmax(dim=1, keepdim=True)                 # (B,1,n) 每行最大位置
        signs = -torch.sign(torch.gather(X, 1, idx)).clamp(min=-1, max=1)
        Y.scatter_(1, idx, signs)                                 # 赋值 -sign(max)
        return Y

    # 通用 p ∈ [1, ∞) 情形
    # sign(x) * |x|^{p-1}
    V = torch.sign(X) * X.abs().clamp_min(eps).pow(max(p - 1.0, 0.0))
    # ||row||_p   -> (B, m, 1)
    lp = X.abs().clamp_min(eps).pow(p).sum(dim=1, keepdim=True).pow(1.0 / p)
    # ||row||_p^{p-1} * n^{1/p}
    scale = lp.clamp_min(eps).pow(max(p - 1.0, 0.0)) * (n ** (1.0 / p))
    Y = V / scale
    return Y

@torch.no_grad()
def init_param_matrix_like(A: torch.Tensor, p: float) -> torch.Tensor:
    
    return _normalize_rows_lmo(A, p)

@torch.no_grad()
def init_model_params_rowlmo(model: nn.Module, p: float = 2.0) -> None:
    print(p)
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight is not None:
            m.weight.copy_(init_param_matrix_like(m.weight, p))
            if m.bias is not None:
                m.bias.zero_()
        # elif isinstance(m, nn.Embedding) and m.weight is not None:
            
            
        #     m.weight.copy_(init_param_matrix_like(m.weight.T, p).T)
           
        elif isinstance(m, HFConv1D) and m.weight is not None:
            m.weight.copy_(init_param_matrix_like(m.weight.T, p).T)
        


def load_tokenizer(model_name: str):
    """
    Load a GPT-2 tokenizer and ensure pad_token exists.
    """
    if model_name == "small":
        tok = GPT2TokenizerFast.from_pretrained("./gpt2")
    elif model_name == "medium":
        tok = GPT2TokenizerFast.from_pretrained("./gpt2_m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _post_init_resize(model: GPT2LMHeadModel, tokenizer):
    # Ensure model knows pad token and embedding size matches tokenizer size
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

def build_model_from_name(model_name: str, tokenizer=None, device: str = "cpu", data_parallel: bool = True):
    """
    Generic builder for GPT-2 family (gpt2, gpt2-medium, gpt2-large).
    Returns (model, tokenizer).
    """
   
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    config1 = GPT2Config(
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
    if model_name == 'small':
        model = GPT2LMHeadModel(config1)
    elif model_name == 'medium':
        model = GPT2LMHeadModel.from_pretrained("./gpt2_m")
    _post_init_resize(model, tokenizer)
    model = model.to(device)
    if data_parallel and (device == "cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model, tokenizer

def build_gpt2(device: str = "cpu", data_parallel: bool = True, gpt2name = 'small'):
    """
    Paper-compatible GPT-2 small (124M): 'gpt2' in HF hub.
    Returns (model, tokenizer).
    """
    tokenizer = load_tokenizer(gpt2name)
    return build_model_from_name(gpt2name, tokenizer=tokenizer, device=device, data_parallel=data_parallel)
