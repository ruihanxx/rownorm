import torch.nn as nn
import torch
def set_fans_gpt2(model: nn.Module):

    # Try to get HF's Conv1D type (GPT-2 uses it to implement Linear)
    try:
        from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
    except Exception:
        HFConv1D = type("___DummyConv1D___", (), {})  # Fallback placeholder that will not match any module

    # First check if weights are tied: whether lm_head.weight and wte.weight are shared
    tied_lm_head_with_wte = False
    vocab_size_for_tied = None
    embed_dim_for_tied = None
    try:
        wte = model.transformer.wte  # nn.Embedding
        embed_dim_for_tied = wte.embedding_dim
    except Exception:
        wte = None

    try:
        lm_head = model.lm_head  # nn.Linear(bias=False) in GPT2LMHeadModel
        vocab_size_for_tied = getattr(lm_head, "out_features", None)
        if wte is not None and hasattr(lm_head, "weight") and hasattr(wte, "weight"):
            tied_lm_head_with_wte = (lm_head.weight is wte.weight)
    except Exception:
        lm_head = None
    if tied_lm_head_with_wte:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    # Small helper: write fan attributes onto the param
    def _set_param_fans(param, fin, fout):
        if hasattr(param, "data"):
            param.fan_in_override = int(fin)
            param.fan_out_override = int(fout)

    # Record lm_head weight objects for easier identification
    lm_head_weight_obj = getattr(lm_head, "weight", None) if lm_head is not None else None
    wte_weight_obj = getattr(wte, "weight", None) if wte is not None else None
    for mod_name, m in model.named_modules():
        
        if isinstance(m, nn.Linear):
            m.weight.is_ebd = 0
            
            
            # HFConv1D weight has shape (in_features, out_features)
            # nn.Linear weight has shape (out_features, in_features)
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2:
                
                out_features, in_features = m.weight.shape
                m.weight.is_head = 1
                _set_param_fans(m.weight, fin=in_features, fout=out_features)
                # print('linear',m.weight.fan_in_override,m.weight.fan_out_override)
            # bias (if present)
            if hasattr(m, "bias") and m.bias is not None and m.bias.dim() == 1:
                out_features = m.bias.shape[0]
                _set_param_fans(m.bias, fin=1, fout=out_features)
        elif isinstance(m, HFConv1D):
            m.weight.is_conv = 1
            
            
            if mod_name.endswith(".attn.c_attn"):
                m.weight.is_qkv = 1
                
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2:
                
                in_features, out_features = m.weight.shape
                _set_param_fans(m.weight, fin=in_features, fout=out_features)
            # bias (if present)
            if hasattr(m, "bias") and m.bias is not None and m.bias.dim() == 1:
                out_features = m.bias.shape[0]
                _set_param_fans(m.bias, fin=1, fout=out_features)

        # ---- Embedding（wte / wpe）----
        elif isinstance(m, nn.Embedding):
            # print(m.weight.shape)
            m.weight.is_ebd = 1
            num_embeddings = m.num_embeddings
            embed_dim = m.embedding_dim

            # Default strategy: input/positional embeddings -> (fin=embed_dim, fout=1)
            
            if mod_name in ['transformer.wte','transformer.wpe']:
                fin = m.weight.shape[0]
                fout = m.weight.shape[1]
                _set_param_fans(m.weight, fin=fin, fout=fout)
                m.weight.is_ebd = 1
                
                
                # print('emb',m.weight.fan_in_override,m.weight.fan_out_override)
        else:
            
            if mod_name.endswith(".ln_1") or mod_name.endswith(".ln_2") or mod_name.endswith(".ln_f"):
                m.weight.is_ln  = 1
                
            # Other modules (such as LayerNorm) usually only have 1D weights; nothing to set
            pass
    
    # Fallback: if the model has lm_head whose weight has not been handled above as (embed_dim, vocab) fan values,
    # set it once more explicitly (in case module pattern checks missed it)
    # if lm_head is not None and hasattr(lm_head, "weight") and lm_head.weight is not None:
        
    #     W = lm_head.weight
    #     if W.dim() == 2:
    #         out_features, in_features = W.shape  # (vocab_size, embed_dim)
    #         _set_param_fans(W, fin=in_features, fout=out_features)
    #     # lm_head usually has no bias; if there is one, handle it as well
    #     if hasattr(lm_head, "bias") and lm_head.bias is not None and lm_head.bias.dim() == 1:
    #         _set_param_fans(lm_head.bias, fin=1, fout=lm_head.bias.shape[0])
