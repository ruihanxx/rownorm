import torch.nn as nn
import torch
def set_fans_gpt2(model: nn.Module):

    # 尝试拿到 HF 的 Conv1D 类型（GPT-2 用它实现 Linear）
    try:
        from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
    except Exception:
        HFConv1D = type("___DummyConv1D___", (), {})  # 兜底占位，不会匹配任何模块

    # 先探测是否存在绑权：lm_head.weight 与 wte.weight 是否共享
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
    # 一个小工具：给 param 写入 fan 属性
    def _set_param_fans(param, fin, fout):
        if hasattr(param, "data"):
            param.fan_in_override = int(fin)
            param.fan_out_override = int(fout)

    # 记录 lm_head 权重对象，便于识别
    lm_head_weight_obj = getattr(lm_head, "weight", None) if lm_head is not None else None
    wte_weight_obj = getattr(wte, "weight", None) if wte is not None else None
    for mod_name, m in model.named_modules():
        
        if isinstance(m, nn.Linear):
            m.weight.is_ebd = 0
            
            
            # HFConv1D 的 weight 形状是 (in_features, out_features)
            # nn.Linear 的 weight 形状是 (out_features, in_features)
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2:
                
                out_features, in_features = m.weight.shape
                m.weight.is_head = 1
                _set_param_fans(m.weight, fin=in_features, fout=out_features)
                # print('linear',m.weight.fan_in_override,m.weight.fan_out_override)
            # bias（若存在）
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
            # bias（若存在）
            if hasattr(m, "bias") and m.bias is not None and m.bias.dim() == 1:
                out_features = m.bias.shape[0]
                _set_param_fans(m.bias, fin=1, fout=out_features)

        # ---- Embedding（wte / wpe）----
        elif isinstance(m, nn.Embedding):
            # print(m.weight.shape)
            m.weight.is_ebd = 1
            num_embeddings = m.num_embeddings
            embed_dim = m.embedding_dim

            # 缺省策略：输入嵌入 / 位置嵌入 -> (fin=embed_dim, fout=1)
            
            if mod_name in ['transformer.wte','transformer.wpe']:
                fin = m.weight.shape[0]
                fout = m.weight.shape[1]
                _set_param_fans(m.weight, fin=fin, fout=fout)
                m.weight.is_ebd = 1
                
                
                # print('emb',m.weight.fan_in_override,m.weight.fan_out_override)
        else:
            
            if mod_name.endswith(".ln_1") or mod_name.endswith(".ln_2") or mod_name.endswith(".ln_f"):
                m.weight.is_ln  = 1
                
            # 其他模块（如 LayerNorm）通常只有 1D 权重，无需设置
            pass
    
    # 兜底：如果模型里有 lm_head 且其 weight 未在上面被处理成 (embed_dim, vocab) 的 fan，
    # 再单独设置一次（防止因为模块判断路径漏掉）
    # if lm_head is not None and hasattr(lm_head, "weight") and lm_head.weight is not None:
        
    #     W = lm_head.weight
    #     if W.dim() == 2:
    #         out_features, in_features = W.shape  # (vocab_size, embed_dim)
    #         _set_param_fans(W, fin=in_features, fout=out_features)
    #     # lm_head 通常无 bias；若有则一并处理
    #     if hasattr(lm_head, "bias") and lm_head.bias is not None and lm_head.bias.dim() == 1:
    #         _set_param_fans(lm_head.bias, fin=1, fout=lm_head.bias.shape[0])
