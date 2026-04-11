# BEDR Framework (Final Version)

这是论文 "BEDR: Boundary Entropy-Driven Resampling" 的官方代码实现。

## 目录结构说明

1. **s1_desc2graph.py**
   - 功能：将原始 ATT&CK 文本描述转化为结构化图谱 (Intent/Action/Entity)。
   
2. **s2_graph_embed.py**
   - 功能：构建混合向量空间 (Hybrid Vector Space)，计算并保存用于熵计算的参考向量。

3. **s3_fourth_pass_augmentation.py** (核心代码)
   - 功能：BEDR 的主程序。执行多轮自适应过采样，利用边界熵过滤生成的样本。
   
4. **organize_and_generate_datasets.py**
   - 功能：执行边界熵驱动的欠采样 (Undersampling)，清洗多数类，并生成最终的训练集。

5. **generate_native_llm_direct.py**
   - 功能：对照组实验代码。直接使用 LLM 生成样本，不进行边界熵过滤。

6. **utils_llm.py / utils_embedding.py**
   - 功能：封装好的 OpenAI/DeepSeek API 调用接口和 Embedding 计算接口。

## 运行顺序

1. 运行 `s1_desc2graph.py` 构建图谱。
2. 运行 `s2_graph_embed.py` 准备向量空间。
3. 运行 `s3_fourth_pass_augmentation.py` 生成增强数据。
4. 运行 `organize_and_generate_datasets.py` 整理最终数据集。

