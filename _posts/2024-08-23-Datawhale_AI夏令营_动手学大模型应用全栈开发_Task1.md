---
title: "2024-08-23-Datawhale_AI夏令营_动手学大模型应用全栈开发_Task1"
author: "BlackJack0083"
date: "2024-08-23"
toc: true
tags: ["大模型"]
comments: true
---

# 基本步骤

### step0
- 开通阿里云 PAI-DSW试用[https://free.aliyun.com/?productCode=learn](https://free.aliyun.com/?productCode=learn)
- 在魔搭社区授权 https://www.modelscope.cn/my/mynotebook/authorization

### step1
在魔搭社区创建PAI实例 https://www.modelscope.cn/my/mynotebook/authorization

### step2 Demo 搭建

#### 文件下载

在终端输入如下指令：

```Bash
git lfs install
git clone https://www.modelscope.cn/datasets/Datawhale/AICamp_yuan_baseline.git
```

#### 环境安装

```Bash
pip install streamlit==1.24.0
```

#### 启动Demo

```Bash
streamlit run AICamp_yuan_baseline/Task\ 1：零基础玩转源大模型/web_demo_2b.py --server.address 127.0.0.1 --server.port 6006
```

### 对话体验

模型加载完后即可进行对话

# baseline 精读

```Python
# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# 创建一个标题和一个副标题
st.title("💬 Yuan2.0 智能编程助手")

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')
```
**模型下载：**
Yuan2-2B-Mars支持通过多个平台进行下载，包括魔搭、HuggingFace、OpenXlab、百度网盘、WiseModel等。因为我们的机器就在魔搭，所以这里我们直接选择通过魔搭进行下载。模型在魔搭平台的地址为 [IEITYuan/Yuan2-2B-Mars-hf](https://modelscope.cn/models/IEITYuan/Yuan2-2B-Mars-hf)。

模型下载使用的是 modelscope 中的 snapshot_download 函数，第一个参数为模型名称 `IEITYuan/Yuan2-2B-Mars-hf`，第二个参数 `cache_dir` 为模型保存路径，这里`.`表示当前路径。

模型大小约为4.1G，由于是从魔搭直接进行下载，速度会非常快。下载完成后，会在当前目录增加一个名为 `IEITYuan` 的文件夹，其中 `Yuan2-2B-Mars-hf` 里面保存着我们下载好的源大模型。

```python
# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'
# path = './IEITYuan/Yuan2-2B-July-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100
```

基本类型、路径确定与加载

```python
# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
# 装饰器，缓存加载好的模型和tokenizer
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()

    print("Done.")
    return tokenizer, model
```

使用 `transformers` 中的 `from_pretrained` 函数来加载下载好的模型和tokenizer，并通过 `.cuda()` 将模型放置在GPU上。另外，这里额外使用了 `streamlit` 提供的一个装饰器 `@st.cache_resource` ，它可以用于缓存加载好的模型和tokenizer。

```python
# 加载model和tokenizer
tokenizer, model = get_model()

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
```

初始化和不断更新聊天界面

```python
# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 调用模型
    prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>" # 拼接对话历史
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
```

这段代码的目的是在一个聊天应用中，处理用户的输入，调用模型生成回复，并将双方的对话显示在界面上。下面是代码逐步解释：

1. **用户输入检测**：
    ```python
    if prompt := st.chat_input():
    ```
    这一行使用了海象运算符 `:=`，表示如果用户在聊天输入框中输入了内容（即 `st.chat_input()` 返回了非空字符串），则将输入赋值给 `prompt` 变量，并进入代码块继续执行。

2. **存储用户输入**：
    ```python
    st.session_state.messages.append({"role": "user", "content": prompt})
    ```
    这行代码将用户的输入保存到 `st.session_state.messages` 列表中，`messages` 列表中的每一条消息是一个字典，包含 `role`（用户身份）和 `content`（消息内容）的键值对。在这里，`role` 被设置为 `"user"`，内容是用户输入的 `prompt`。

3. **在界面显示用户输入**：
    ```python
    st.chat_message("user").write(prompt)
    ```
    使用 `st.chat_message("user").write(prompt)` 在聊天界面上显示用户输入的内容。

4. **拼接对话历史**：
    ```python
    prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
    ```
    这一行代码通过遍历 `st.session_state.messages` 列表，将所有消息的 `content`（消息内容）拼接起来，使用 `<n>` 作为消息之间的分隔符，并在最后附加上 `<sep>`，表示结束符号。这一步是为了构建完整的对话上下文，供模型使用。

5. **准备输入数据**：
    ```python
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    ```
    将上一步拼接的 `prompt` 通过 `tokenizer` 进行分词处理（`tokenizer` 是一个分词器对象），并返回张量形式的输入数据（`return_tensors="pt"` 表示返回 PyTorch 张量），然后将其移动到 GPU 上（通过 `.cuda()`）。

6. **模型生成回复**：
    ```python
    outputs = model.generate(inputs, do_sample=False, max_length=1024)
    ```
    调用模型的 `generate` 方法生成回复。`do_sample=False` 表示使用贪心搜索（不进行采样），`max_length=1024` 设置生成的最大长度为 1024 个 token。

7. **处理模型输出**：
    ```python
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')
    ```
    模型的输出是一个 token 序列，首先使用 `tokenizer.decode` 将其解码为可读的文本。接下来，`output.split("<sep>")[-1]` 从生成的文本中提取出 `<sep>` 后的内容，表示最终回复，`replace("<eod>", '')` 则去除结束标志 `<eod>`。

8. **存储和显示模型的回复**：
    ```python
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    ```
    将模型生成的回复以类似于用户输入的方式保存到 `st.session_state.messages` 列表中，`role` 被设置为 `"assistant"`，`content` 是模型生成的 `response`。随后使用 `st.chat_message("assistant").write(response)` 在界面上显示助手的回复。

---

总结：
- 这段代码的核心流程是：当用户输入内容时，系统会保存用户的输入并在界面上显示，然后使用模型生成回复，最终保存并显示模型的输出。
- 该代码依赖于一个分词器（`tokenizer`）和模型（`model`），并使用 `streamlit`（简称为 `st`）进行界面的交互和显示。

# RAG

## 1 引言

### 1.1 什么是RAG

在上一节，我们成功搭建了一个源大模型智能对话Demo，亲身体验到了大模型出色的能力。然而，在实际业务场景中，通用的基础大模型可能存在无法满足我们需求的情况，主要有以下几方面原因：

- 知识局限性：大模型的知识来源于训练数据，而这些数据主要来自于互联网上已经公开的资源，对于一些实时性的或者非公开的，由于大模型没有获取到相关数据，这部分知识也就无法被掌握。
- 数据安全性：为了使得大模型能够具备相应的知识，就需要将数据纳入到训练集进行训练。然而，对于企业来说，数据的安全性至关重要，任何形式的数据泄露都可能对企业构成致命的威胁。
- 大模型幻觉：由于大模型是基于概率统计进行构建的，其输出本质上是一系列数值运算。因此，有时会出现模型“一本正经地胡说八道”的情况，尤其是在大模型不具备的知识或不擅长的场景中。

为了上述这些问题，研究人员提出了检索增强生成（Retrieval Augmented Generation, **RAG**）的方法。这种方法通过引入外部知识，使大模型能够生成准确且符合上下文的答案，同时能够减少模型幻觉的出现。

由于RAG简单有效，它已经成为主流的大模型应用方案之一。
如下图所示，RAG通常包括以下三个基本步骤：

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825130004.png)

- 索引：将文档库分割成较短的 **Chunk**，即文本块或文档片段，然后构建成向量索引。
- 检索：计算问题和 Chunks 的相似度，检索出若干个相关的 Chunk。
- 生成：将检索到的Chunks作为背景信息，生成问题的回答。

### 1.2 一个完整的RAG链路

本小节我们将带大家构建一个完整的RAG链路。

![image.png|500](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825130228.png)
从上图可以看到，线上接收到用户`query`后，RAG会先进行检索，然后将检索到的 **`Chunks`** 和 **`query`** 一并输入到大模型，进而回答用户的问题。

为了完成检索，需要离线将文档（ppt、word、pdf等）经过解析、切割甚至OCR转写，然后进行向量化存入数据库中。

接下来，我们将分别介绍离线计算和在线计算流程。

#### 1.2.1 离线计算

首先，知识库中包含了多种类型的文件，如pdf、word、ppt等，这些 `文档`（Documents）需要提前被解析，然后切割成若干个较短的 `Chunk`，并且进行清洗和去重。

由于知识库中知识的数量和质量决定了RAG的效果，因此这是非常关键且必不可少的环节。

然后，我们会将知识库中的所有 `Chunk` 都转成向量，这一步也称为 `向量化`（Vectorization）或者 `索引`（Indexing）。

`向量化` 需要事先构建一个 `向量模型`（Embedding Model），它的作用就是将一段 `Chunk` 转成 `向量`（Embedding）。如下图所示：

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825130327.png)

一个好的向量模型，会使得具有相同语义的文本的向量表示在语义空间中的距离会比较近，而语义不同的文本在语义空间中的距离会比较远。

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825130350.png)
由于知识库中的所有 `Chunk` 都需要进行 `向量化`，这会使得计算量非常大，因此这一过程通常是离线完成的。

随着新知识的不断存储，向量的数量也会不断增加。这就需要将这些向量存储到 `数据库` （DataBase）中进行管理，例如 [Milvus](https://milvus.io/) 中。

至此，离线计算就完成了。

#### 1.2.2 在线计算

在实际使用RAG系统时，当给定一条用户 `查询`（Query），需要先从知识库中找到所需的知识，这一步称为 `检索`（Retrieval）。

在 `检索` 过程中，用户查询首先会经过向量模型得到相应的向量，然后与 `数据库` 中所有 `Chunk` 的向量计算相似度，最简单的例如 `余弦相似度`，然后得到最相近的一系列 `Chunk` 。

由于向量相似度的计算过程需要一定的时间，尤其是 `数据库` 非常大的时候。

这时，可以在检索之前进行 `召回`（Recall），即从 `数据库` 中快速获得大量大概率相关的 `Chunk`，然后只有这些 `Chunk` 会参与计算向量相似度。这样，计算的复杂度就从整个知识库降到了非常低。

`召回` 步骤不要求非常高的准确性，因此通常采用简单的基于字符串的匹配算法。由于这些算法不需要任何模型，速度会非常快，常用的算法有 `TF-IDF`，`BM25` 等。

另外，也有很多工作致力于实现更快的 `向量检索` ，例如 [faiss](https://github.com/facebookresearch/faiss)，[annoy](https://github.com/spotify/annoy)。 

另一方面，人们发现，随着知识库的增大，除了检索的速度变慢外，检索的效果也会出现退化，如下图中绿线所示：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825130653.png)

这是由于 `向量模型` 能力有限，而随着知识库的增大，已经超出了其容量，因此准确性就会下降。在这种情况下，相似度最高的结果可能并不是最优的。

为了解决这一问题，提升RAG效果，研究者提出增加一个二阶段检索——`重排` (Rerank)，即利用 `重排模型`（Reranker），使得越相似的结果排名更靠前。这样就能实现准确率稳定增长，即数据越多，效果越好（如上图中紫线所示）。

通常，为了与 `重排` 进行区分，一阶段检索有时也被称为 `精排` 。而在一些更复杂的系统中，在 `召回` 和 `精排` 之间还会添加一个 `粗排` 步骤，这里不再展开，感兴趣的同学可以自行搜索。

综上所述，在整个 `检索` 过程中，计算量的顺序是 `召回` > `精排` > `重排`，而检索效果的顺序则是 `召回` < `精排` < `重排` 。

当这一复杂的 `检索` 过程完成后，我们就会得到排好序的一系列 `检索文档`（Retrieval Documents）。

然后我们会从其中挑选最相似的 `k` 个结果，将它们和用户查询拼接成prompt的形式，输入到大模型。

最后，大型模型就能够依据所提供的知识来生成回复，从而更有效地解答用户的问题

至此，一个完整的RAG链路就构建完毕了。

### 1.2 开源RAG框架

目前，开源社区中已经涌现出了众多RAG框架，例如：

- [TinyRAG](https://github.com/KMnO4-zx/TinyRAG)：DataWhale成员宋志学精心打造的纯手工搭建RAG框架。
- [LlamaIndex](https://github.com/run-llama/llama_index)：一个用于构建大语言模型应用程序的数据框架，包括数据摄取、数据索引和查询引擎等功能
- [LangChain](https://github.com/langchain-ai/langchain)：一个专为开发大语言模型应用程序而设计的框架，提供了构建所需的模块和工具。
- [QAnything](https://github.com/netease-youdao/QAnything)：网易有道开发的本地知识库问答系统，支持任意格式文件或数据库。
- [RAGFlow](https://github.com/infiniflow/ragflow)：InfiniFlow开发的基于深度文档理解的RAG引擎。
- ···

这些开源项目各具优势，功能丰富，极大的推动了RAG技术的发展。
然而，随着这些框架功能的不断扩展，学习者不可避免地需要承担较高的学习成本。

因此，本节课将以 `Yuan2-2B-Mars` 模型为基础，进行RAG实战。希望通过构建一个简化版的RAG系统，来帮助大家掌握RAG的核心技术，从而进一步了解一个完整的RAG链路。

## 尝试实例

### 下载文件

```Bash
git lfs install
git clone https://www.modelscope.cn/datasets/Datawhale/AICamp_yuan_baseline.git
cp AICamp_yuan_baseline/Task\ 3：源大模型RAG实战/* .
pip install streamlit == 1.24.0
```

### 2.2 模型下载

在RAG实战中，我们需要构建一个向量模型。

向量模型通常采用BERT架构，它是一个Transformer Encoder。

输入向量模型前，首先会在文本的最前面额外加一个 `[CLS]` token，然后将该token最后一层的隐藏层向量作为文本的表示。如下图所示：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825140714.png)

目前，开源的基于BERT架构的向量模型有如下：
- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)：智源通用embedding（BAAI general embedding, BGE）
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)：网易有道训练的Bilingual and Crosslingual Embedding
- [jina-embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh)：Jina AI训练的text embedding
- [M3E](https://huggingface.co/moka-ai/m3e-large)：MokaAI训练的 Massive Mixed Embedding
- ···

除了BERT架构之外，还有基于LLM的向量模型有如下：
- [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)：智源LLM-Embedder
- ···

其次，还有API:
- [OpenAI API](https://platform.openai.com/docs/guides/embeddings)
- [Jina AI API](https://jina.ai/embeddings/)
- [ZhipuAI API](https://open.bigmodel.cn/dev/api#text_embedding)
- ···

在本次学习中，我们选用基于BERT架构的向量模型 `bge-small-zh-v1.5`，它是一个4层的BERT模型，最大输入长度512，输出的向量维度也为512。

`bge-small-zh-v1.5` 支持通过多个平台进行下载，因为我们的机器就在魔搭，所以这里我们直接选择通过魔搭进行下载。

模型在魔搭平台的地址为 [AI-ModelScope/bge-small-zh-v1.5](https://modelscope.cn/models/AI-ModelScope/bge-small-zh-v1.5)。

```Python
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
```

### 2.3 RAG实战

模型下载完成后，就可以开始RAG实战啦！

#### 2.3.1 **索引**

为了构造索引，这里我们封装了一个向量模型类 `EmbeddingModel`：

```Python
# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        # 初始化函数，接收模型的路径作为参数
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # 加载与模型路径对应的Tokenizer

        self.model = AutoModel.from_pretrained(path).cuda()
        # 加载与模型路径对应的模型，并将其移动到GPU上
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List) -> List[float]:
        """
        计算文本列表的嵌入向量
        参数:
            texts: 要处理的文本列表
        返回:
            sentence_embeddings: 文本的嵌入向量列表
        """
        # 使用tokenizer处理文本，进行分词、填充(padding)和截断(truncation)
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # 将编码后的输入数据移动到GPU上
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        
        with torch.no_grad():
            # 不计算梯度，以加快计算速度
            model_output = self.model(**encoded_input)
            # 从模型输出中获取最后一层的隐藏状态
            sentence_embeddings = model_output[0][:, 0]
        # 将句子的嵌入向量标准化，使其具有单位范数
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # 将嵌入向量转换为列表格式并返回
        return sentence_embeddings.tolist()
```

通过传入模型路径，新建一个 `EmbeddingModel` 对象 `embed_model`。

初始化时自动加载向量模型的tokenizer和模型参数。

> 在Python中，`->`符号通常用于类型注解，表示函数返回值的预期类型。它是类型提示（Type Hints）的一部分，用于告诉解释器函数的参数和返回值应该是什么类型。类型提示可以帮助开发者避免类型错误，并且可以被IDE和静态类型检查工具使用来提高代码的可读性和健壮性。

```Python
print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
embed_model = EmbeddingModel(embed_model_path)
```

#### 2.3.2 检索

为了实现向量检索，我们定义了一个向量库索引类 `VectorStoreIndex`：

```Python
# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        # 初始化函数，接收文档路径和嵌入模型作为参数
        self.documents = []
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)  # 读取文档并去除空白字符，然后添加到文档列表

        self.embed_model = embed_model  # 存储嵌入模型实例
        self.vectors = self.embed_model.get_embeddings(self.documents)  # 为所有文档获取嵌入向量

        print(f'Loading {len(self.documents)} documents for {document_path}.')  # 打印加载文档数量和路径

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        """
        dot_product = np.dot(vector1, vector2)  # 计算两个向量的点积
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  # 计算两个向量的模长乘积
        if not magnitude:  # 如果模长乘积为0，避免除以0
            return 0
        return dot_product / magnitude  # 返回余弦相似度值

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        根据问题查询最相似的文档
        参数:
            question: 输入的问题文本
            k: 返回最相似的文档数量，默认为1
        返回:
            查询结果的文档列表
        """
        question_vector = self.embed_model.get_embeddings([question])[0]  # 为问题文本获取嵌入向量
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  # 计算问题向量与所有文档向量的相似度
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()  # 返回最相似的k个文档

# 注意：代码中使用了numpy库来计算向量操作，需要导入numpy库
# 例如：import numpy as np
# 同时，List类型需要从typing模块导入
# 例如：from typing import List

print("> Create index...")
doecment_path = './knowledge.txt'
index = VectorStoreIndex(doecment_path, embed_model)
```

上文提到 `get_embeddings()` 函数支持一次性传入多条文本，但由于GPU的显存有限，输入的文本不宜太多。
所以，如果知识库很大，需要将知识库切分成多个batch，然后分批次送入向量模型。
这里，因为我们的知识库比较小，所以就直接传到了 `get_embeddings()` 函数。

其次，`VectorStoreIndex` 类还有一个 `get_similarity()` 函数，它用于计算两个向量之间的相似度，这里采用了余弦相似度。

最后，我们介绍一下 `VectorStoreIndex` 类的入口，即查询函数 `query()`。传入用户的提问后，首先会送入向量模型获得其向量表示，然后与知识库中的所有向量计算相似度，最后将 `k` 个最相似的文档按顺序返回，`k`默认为1。

这里我们传入用户问题 `介绍一下广州大学`，可以看到，准确地返回了知识库中的第一条知识。
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825144237.png)

#### 2.3.3 生成

为了实现基于RAG的生成，我们还需要定义一个大语言模型类 `LLM`：

```Python
# 定义大语言模型类
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Creating tokenizer...")
        # 初始化Tokenizer，不添加EOS和BOS标记，而是使用自定义的EOS标记
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        # 添加特殊标记到Tokenizer中，这些标记将用于模型的不同部分
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
                                   '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>',
                                   '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creating model...")
        # 初始化模型，使用半精度浮点数来减少内存使用，并将模型移动到GPU上
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List):
        # 根据问题和上下文生成回答
        if context:
            # 如果提供了上下文，则构建一个提示，包括背景和问题
            prompt = f'背景：{context}\n问题：{question}\n请基于背景，回答问题。'
        else:
            # 如果没有提供上下文，只使用问题作为提示
            prompt = question

        # 将提示添加分隔符<sep>，并使用Tokenizer进行编码
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        
        # 使用模型生成回答，不进行采样（do_sample=False），最大长度限制为1024
        outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        
        # 将生成的token解码回文本
        output = self.tokenizer.decode(outputs[0])
        
        # 打印输出文本，只显示分隔符<sep>之后的部分
        print(output.split("<sep>")[-1])

# 注意：代码中使用了AutoTokenizer, AutoModelForCausalLM等类，这些类通常来自Hugging Face的Transformers库。
# 此外，List类型注解需要从typing模块导入，torch.bfloat16是PyTorch中的数据类型，用于指定模型使用的精度。


print("> Create Yuan2.0 LLM...")
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
llm = LLM(model_path)

```

1. `__init__`：构造函数，用于初始化Tokenizer和模型，并将模型移动到GPU上以加速计算。此外，还添加了一系列特殊标记到Tokenizer中。
2. `generate`：该方法接受一个问题和可选的上下文作为输入，构建一个提示，然后使用模型生成回答。生成的回答将被解码并打印出来，只显示分隔符`<sep>`之后的部分。

```Python
print('> Without RAG:')
llm.generate(question, [])

print('> With RAG:')
# 根据问题内容找到最匹配的文本contents，将其附加在prompt里面
llm.generate(question, context)
```
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240825145556.png)

## 3 课后作业

构建一个知识库，使用`Yuan2-2B`模型进行RAG实战，对比使用和不使用RAG的效果。
### 3.1 知识库构建
- 确定想要开发应用所在的领域：大模型在哪些领域还存在问题？能否通过知识库进行解决？
- 收集领域数据：该领域的数据有哪些来源？百度百科？书籍？论文？
- 构建知识库：收集好的数据需要哪些预处理步骤？数据清洗？去重？
### 3.2 RAG实战
- 知识库索引：知识库怎么切分效果最好？考虑到效果和效率等因素，哪个向量模型更适配？
- 检索：如果知识库比较大或者为了实现更快的检索，需要哪些工具？
- 生成：检索出内容怎么能被大模型利用好？prompt怎么调优？

### 3.3 其他
参考下面资料，探索RAG框架的使用方法：
- langchain：[https://github.com/datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook/tree/main/content/%E9%80%89%E4%BF%AE-Building%20and%20Evaluating%20Advanced%20RAG%20Applications)
- llamaIndex：[https://github.com/datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe/tree/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8)

# AI 科研助手

