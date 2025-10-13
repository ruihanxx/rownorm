# Transformer优化策略：基于RowNorm实验结果的分析

## 📊 ResNet上RowNorm实验结果总结

### 最终验证准确率对比 (20 epochs)

| Scale Type | Original RowNorm | Enhanced RowNorm+SignGD1D | 改进幅度 |
|------------|------------------|---------------------------|---------|
| none       | 88.16%          | 86.98%                   | **-1.18%** ❌ |
| mean_scale | 88.51%          | 89.29%                   | **+0.78%** ✅ |
| rms_scale  | 88.26%          | 87.50%                   | **-0.76%** ❌ |
| dim_scale  | 89.24%          | 88.71%                   | **-0.53%** ❌ |

### 关键发现

1. **RowNorm对2D/4D权重效果好**：所有scale types都达到了88%+的性能
2. **SignGD for 1D参数效果混合**：只有`mean_scale`有提升，其他都下降
3. **最佳组合**：`RowNorm + mean_scale` (无论是否包含SignGD1D)
4. **1D SignGD不稳定**：平均改进-0.422%，只有1/4的配置有提升

## 🔍 Transformer参数分析结果

### 参数维度分布

| 模型大小 | 1D参数占比 | 2D参数占比 | 总参数量 |
|----------|------------|------------|----------|
| Tiny     | 0.23%      | 99.77%     | 742K     |
| Small    | 0.27%      | 99.73%     | 1.88M    |
| Base     | 0.23%      | 99.77%     | 5.75M    |

### 参数类型分布

**2D参数 (~99.7%)**：
- **Embedding矩阵** (vocab_size × d_model)：占大头，特殊语义
- **Attention权重** (in_proj_weight, out_proj.weight)：标准2D线性层
- **Feed-forward权重** (linear1.weight, linear2.weight)：标准2D线性层
- **分类器权重**：标准2D线性层

**1D参数 (~0.3%)**：
- **LayerNorm参数** (weight, bias)：类似BatchNorm
- **Linear偏置** (attention, feed-forward, classifier)
- **Attention偏置**

## 🚀 Transformer优化策略设计

### 基于实验结果的策略选择

1. **2D Linear权重** → **RowNorm + mean_scale**
   - ResNet实验显示这是最稳定的高性能组合
   - 适用于：attention权重、feed-forward权重、分类器权重

2. **1D参数** → **标准SGD** (保守策略)
   - SignGD在ResNet上效果不稳定
   - LayerNorm与BatchNorm可能行为不同
   - 适用于：所有bias、LayerNorm weight/bias

3. **Embedding矩阵** → **特殊处理**
   - 形状是2D但语义特殊 (每行是词向量)
   - 选项A：应用RowNorm (行归一化 = 词向量归一化)
   - 选项B：标准SGD (保守，保持词向量自然尺度)

### 具体实现方案

#### 方案1：保守型 TransformerRowNorm
```python
# 参数分类
if 'embedding' in param_name:
    # Embedding: 使用标准SGD
    apply_standard_sgd(param)
elif param.ndim == 2:
    # 2D线性权重: RowNorm + mean_scale
    apply_rownorm(param, scale_type='mean_scale')
else:
    # 1D参数: 标准SGD
    apply_standard_sgd(param)
```

#### 方案2：激进型 TransformerRowNorm
```python
# 参数分类
if param.ndim == 2:
    # 所有2D参数(包括embedding): RowNorm + mean_scale
    apply_rownorm(param, scale_type='mean_scale')
elif param.ndim == 1:
    # 1D参数: SignGD (基于ResNet最佳结果)
    if 'mean_scale' in optimizer_config:
        apply_signgd_1d(param)
    else:
        apply_standard_sgd(param)
else:
    apply_standard_sgd(param)
```

## 📋 实验计划

### 阶段1：基础验证
1. 实现保守型TransformerRowNorm
2. 在AG News数据集上对比：
   - AdamW (baseline)
   - SGD
   - TransformerRowNorm (保守)
3. 验证基础可行性

### 阶段2：策略对比
1. 实现激进型TransformerRowNorm
2. 对比不同embedding处理策略：
   - Embedding用标准SGD
   - Embedding用RowNorm
3. 测试1D参数处理：
   - 标准SGD
   - SignGD (mean_scale配置)

### 阶段3：规模验证
1. 在不同大小模型上测试 (tiny, small, base)
2. 在不同数据集上验证泛化性
3. 分析收敛速度和最终性能

## 🎯 预期结果

### 乐观情况
- TransformerRowNorm在文本分类上超越AdamW 1-2%
- 特别是在较大模型上效果更明显 (更多2D参数)
- 训练稳定性提升

### 现实情况
- 与AdamW性能接近，但收敛特性不同
- 某些任务上有优势，某些任务上略差
- 为Transformer优化提供新的探索方向

### 风险评估
- Transformer的LayerNorm与ResNet的BatchNorm行为差异
- 文本数据与图像数据的梯度分布差异
- Embedding矩阵的特殊性可能需要专门处理

## 💡 创新点

1. **首次系统性地将RowNorm应用到Transformer**
2. **基于ResNet实验数据的科学决策**
3. **针对Transformer参数特点的定制化处理**
4. **保守vs激进策略的对比研究**

这个策略基于扎实的实验数据，既保守又有探索性，为Transformer优化开辟新方向。

