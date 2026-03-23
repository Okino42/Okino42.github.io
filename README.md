<div align="center">

# 🚀 Okino42's LLM Universe

### 从 Token 到 Agent，系统拆解大语言模型的核心模块

<p>
  <img src="https://img.shields.io/badge/LLM-Architecture-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Transformer-Core-00C2FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RAG-Agent-Memory-00D084?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Fine--Tuning-PEFT%20%7C%20RLHF-orange?style=for-the-badge" />
</p>

<p>
  <b>✨ 一个面向大模型学习者 / 开发者 / 面试准备者的知识仓库</b><br>
  聚焦大语言模型（LLM）的模块化理解、训练逻辑、推理优化与应用系统设计
</p>

</div>

---

## 🌌 项目简介

这个仓库不是单纯堆概念，而是试图回答一个核心问题：

> **一个大模型，到底是由哪些模块组成的？这些模块分别在做什么？它们为什么要这样设计？**

这里会围绕大模型的完整链路展开：

- **输入层**：文本如何变成模型能处理的向量
- **建模层**：Attention、FFN、Norm、Positional Encoding 如何协同工作
- **训练层**：预训练、监督微调、偏好对齐是怎么连接起来的
- **增强层**：RAG、Tool Use、Agent、Memory 如何让模型从“会说”变成“会做”
- **推理层**：KV Cache、MQA/GQA、量化、并行推理如何提升性能
- **评估层**：怎么判断一个模型真的更强，而不是只是“更会说”

---

# 🧠 LLM 知识地图

## 1. Tokenization：一切的起点

自然语言不能直接输入神经网络，必须先切分成 **Token**。

### 核心作用
- 把文本转成离散符号序列
- 建立文本与词表 ID 的映射
- 平衡词表大小、表达能力、长度效率

### 常见方法
- **BPE**
- **WordPiece**
- **SentencePiece**
- **Unigram**

### 关键问题
- 为什么中文、英文、代码的 token 切分方式不一样？
- 为什么同一句话在不同模型里 token 数不同？
- token 长度为什么会影响推理成本？

> **一句话理解：** Tokenization 决定了模型“看见世界”的最小单位。

---

## 2. Embedding：把符号映射成向量

Token ID 本身没有语义，Embedding 层负责把离散 ID 映射到连续向量空间。

### 它做了什么？
- 把每个 token 变成一个高维向量
- 让语义相近的词在向量空间中更接近
- 为后续 Transformer 提供可学习表示

### 你需要理解
- **Token Embedding**
- **Position Embedding / RoPE**
- **输入嵌入与输出权重共享（Weight Tying）**

> **一句话理解：** Embedding 是把“字面符号”翻译成“神经网络语言”。

---

## 3. Positional Encoding：告诉模型顺序

Attention 本身不理解顺序，因此必须显式加入位置信息。

### 常见方案
- **绝对位置编码**
- **相对位置编码**
- **RoPE（Rotary Position Embedding）**
- **ALiBi**

### 为什么重要？
因为：

- “我喜欢你” 和 “你喜欢我” token 很像
- 但语义完全不同
- 没有位置编码，模型只会“看集合”，不会“看序列”

### 热门重点
- 为什么现在很多大模型更偏向 **RoPE**
- RoPE 为什么对长上下文更友好
- 外推长度为什么会失稳

> **一句话理解：** Position 是模型理解“前后关系”的坐标系。

---

## 4. Attention：大模型的核心发动机

Attention 是 Transformer 最关键的模块，它让模型能够根据当前 token 动态关注上下文中的其他 token。

## 标准公式

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### 三个核心对象
- **Q（Query）**：我要找什么
- **K（Key）**：我这里有什么信息
- **V（Value）**：真正被取出的内容

### 你必须搞懂的点
- 为什么要点积
- 为什么要除以 \(\sqrt{d_k}\)
- softmax 的作用是什么
- causal mask 为什么能实现自回归生成

### 常见变体
- **MHA（Multi-Head Attention）**
- **MQA（Multi-Query Attention）**
- **GQA（Grouped Query Attention）**
- **MLA / Latent Attention 相关结构**

> **一句话理解：** Attention 让模型在每一步都能“动态查资料”。

---

## 5. Multi-Head Attention：一个头不够，就开多个头

单个注意力头表达能力有限，所以 Transformer 引入多头机制。

### 多头的意义
- 不同头学习不同关系
- 有的头关注语法
- 有的头关注长距离依赖
- 有的头关注实体、指代、格式结构

### 核心理解
多头不是简单重复，而是：
- 在不同子空间投影
- 并行提取不同模式
- 最后再拼接融合

### 延伸问题
- head 数量怎么影响性能？
- head_dim 为什么不能太小？
- 多头一定越多越好吗？

> **一句话理解：** 多头机制相当于让模型用多种视角同时理解上下文。

---

## 6. Feed Forward Network：每个 token 的局部计算器

Attention 负责“信息交互”，FFN 负责“非线性变换”。

### 结构通常是
\[
\text{FFN}(x)=W_2 \sigma(W_1x)
\]

现代模型中常见：
- **GELU**
- **SwiGLU**
- **GeGLU**

### 作用
- 提升表达能力
- 让模型不仅能“聚合信息”
- 还能对信息做复杂变换

### 为什么 FFN 很重要？
很多人以为 Transformer 主要靠 Attention，实际上：
- Attention 更像路由器
- FFN 更像计算核心

> **一句话理解：** FFN 负责把“看到的信息”加工成更强的表示。

---

## 7. Residual + LayerNorm / RMSNorm：训练稳定器

大模型足够深，不加稳定结构几乎无法训练。

### Residual Connection（残差连接）
作用：
- 防止深层网络梯度消失
- 保留原始信息通路
- 让优化更稳定

### Normalization
常见有：
- **LayerNorm**
- **RMSNorm**

### 热门对比
- **Pre-Norm**
- **Post-Norm**

### 为什么这些很关键？
因为深层 Transformer 的训练成败，往往不取决于“概念是否高级”，而取决于“数值是否稳定”。

> **一句话理解：** 残差和归一化，是大模型能训起来的底层保障。

---

## 8. Transformer Block：模块化堆叠的基本单元

一个标准 Transformer Block 往往包含：

1. Attention 子层  
2. Add & Norm  
3. FFN 子层  
4. Add & Norm  

很多层 Block 叠起来，形成最终的大模型。

### 你需要建立的直觉
- 每一层都在重写 token 表示
- 低层偏局部模式
- 中层偏结构关系
- 高层偏抽象语义与任务输出

> **一句话理解：** 大模型不是“一个大模块”，而是很多标准积木叠出来的。

---

# 🏗️ 训练体系

## 9. Pretraining：先学语言世界

预训练阶段的目标是让模型从海量语料中学习语言规律、知识关联和推理模式。

### 常见目标
- **Causal Language Modeling**
- 预测下一个 token

### 学到什么？
- 语法
- 常识
- 世界知识
- 模式补全能力
- 初步推理能力

### 本质
预训练不是“背答案”，而是在压缩世界中的统计结构。

> **一句话理解：** 预训练决定模型的知识底座和泛化上限。

---

## 10. SFT：让模型学会“按要求说话”

SFT（Supervised Fine-Tuning）通常使用指令数据，让模型从“会续写”变成“会回答”。

### 它解决什么问题？
- 提升指令跟随能力
- 改善对话格式
- 对齐人类偏好的输出风格

### 常见数据类型
- 指令-回答
- 多轮对话
- Chain-of-Thought
- 任务型样本

> **一句话理解：** SFT 让模型从语言引擎变成助手雏形。

---

## 11. RLHF / DPO：让模型更像“人想要的样子”

模型会回答，不代表回答“更符合人类偏好”。

所以就有了：
- **RLHF**
- **RLAIF**
- **DPO**
- **IPO / ORPO / KTO** 等偏好优化方法

### 目标
- 更安全
- 更有帮助
- 更符合指令
- 更少胡说八道

### 经典三段式
1. SFT
2. Reward Model
3. Preference Optimization

> **一句话理解：** 对齐阶段不是教模型知识，而是教模型“怎么用知识”。

---

# ⚡ 推理优化

## 12. KV Cache：为什么第二个 token 更快

在自回归生成时，历史 token 的 K/V 没必要重复计算。

### KV Cache 的价值
- 避免重复计算
- 降低延迟
- 提高长序列生成效率

### 为什么它重要？
因为实际推理的瓶颈很多时候不是参数量本身，而是：
- 显存
- 带宽
- 序列长度
- cache 管理

> **一句话理解：** KV Cache 是 LLM 推理提速的核心工程优化之一。

---

## 13. MQA / GQA：为了更省显存、更快推理

多头注意力性能强，但 KV 开销大，于是出现：
- **MQA**：多个 Query 共享一组 K/V
- **GQA**：多组 Query 共享分组 K/V

### 优势
- 减少 KV Cache 占用
- 提升推理吞吐
- 适合长上下文与高并发服务

### 权衡
- 节省资源
- 但可能损失少量表达能力

> **一句话理解：** MQA/GQA 是大模型从“能跑”到“跑得动”的关键工程设计。

---

## 14. Quantization：让大模型更便宜

量化的核心思想是：
把高精度参数（FP16/BF16）压缩成低精度表示（INT8/INT4 等）。

### 常见方向
- **PTQ**
- **QAT**
- **GPTQ**
- **AWQ**
- **GGUF / llama.cpp 生态**

### 目标
- 降低显存占用
- 提高部署可行性
- 在尽量保留效果的前提下压缩模型

> **一句话理解：** 量化是在性能、成本、精度之间做工程平衡。

---

# 🔧 参数高效微调

## 15. LoRA / QLoRA / PEFT：不重新训练整个模型

全参数微调成本很高，因此出现参数高效微调。

### 代表方法
- **LoRA**
- **QLoRA**
- **AdaLoRA**
- **Prefix Tuning**
- **P-Tuning**

### LoRA 的核心思想
不是直接修改原权重，而是学习一个低秩增量：

\[
W' = W + BA
\]

### 优点
- 参数少
- 显存省
- 易于迁移
- 可叠加不同任务适配器

> **一句话理解：** PEFT 让“训练大模型”从大厂专属变成普通开发者也能尝试的事。

---

# 🧩 系统增强层

## 16. RAG：让模型学会“查资料再回答”

RAG（Retrieval-Augmented Generation）是把检索系统接到大模型前面。

### 标准流程
1. 用户提问
2. 检索相关文档
3. 拼接上下文
4. 让模型基于外部知识生成答案

### 解决的问题
- 减少幻觉
- 引入私有知识
- 支持动态更新
- 降低对参数记忆的依赖

### 关键模块
- Embedding Model
- Vector Database
- Retriever
- Reranker
- Generator

> **一句话理解：** RAG 让模型从“靠记忆回答”升级为“先找资料再回答”。

---

## 17. Tool Use：让模型具备行动能力

纯语言模型只能“说”，工具调用让它开始“做”。

### 常见工具
- 搜索
- 计算器
- 数据库查询
- 代码执行
- API 调用
- 浏览器操作

### 本质变化
模型不再只是输出文本，而是：
- 规划步骤
- 决定调用什么工具
- 读取工具结果
- 再组织最终回答

> **一句话理解：** Tool Use 是 LLM 走向智能体的第一步。

---

## 18. Agent：从单轮回答到任务执行

Agent 通常包含：
- 目标
- 规划
- 记忆
- 工具
- 反思
- 多步执行

### Agent 关注什么？
- 如何拆任务
- 如何使用外部环境
- 如何根据反馈修正计划
- 如何在长任务中维持状态

### 相关能力
- Planning
- Reflection
- Multi-step Reasoning
- Long-term Memory

> **一句话理解：** Agent 让模型从“回答器”变成“执行者”。

---

# 📏 评估体系

## 19. Evaluation：怎么知道模型真的更强？

模型效果不能只靠“感觉不错”。

### 常见评估维度
- **Perplexity**
- **Accuracy / F1**
- **BLEU / ROUGE**
- **MMLU**
- **Human Eval**
- **Arena / 偏好对比**
- **事实性 / 安全性 / 鲁棒性**

### 评估难点
- benchmark 可能刷分
- 通用能力与专业能力不一致
- 离线指标不等于真实用户体验

> **一句话理解：** 评估不是给模型打分，而是界定模型真正的能力边界。

---

# 🪐 一个完整的大模型系统长什么样？

```text
User Query
   ↓
Tokenizer
   ↓
Embedding + Positional Encoding
   ↓
Stacked Transformer Blocks
   ├─ Attention
   ├─ FFN
   ├─ Residual
   └─ Norm
   ↓
Next Token Prediction / Decoding
   ↓
(可选增强)
   ├─ RAG
   ├─ Tool Use
   ├─ Agent
   └─ Memory
   ↓
Final Response
