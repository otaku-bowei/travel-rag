# 🌍 旅游行程推荐 RAG 系统

> 基于 LangChain + Chroma + MiniMax 的旅游问答系统

## 📁 项目结构

```
travel-rag/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── config.py           # 配置模块
│   ├── loader.py         # 文档加载器
│   ├── rag.py           # RAG 核心（向量化+检索）
│   ├── chat.py          # 对话模块（LLM+QA链）
│   └── server.py        # LangChain Server
├── data/
│   ├── knowledge/       # 知识库（小红书攻略）
│   │   └── kyoto_osaka.md
│   └── vectorstore/    # Chroma 向量库
├── run.py              # 启动脚本
├── requirements.txt   # 依赖
├── .env               # 环境变量
└── README.md
```

---

## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户提问                               │
│                  "京都三日游怎么安排"                       │
└─────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  1. Embedding 向量化                         │
│         (BAAI/bge-small-zh-v1.5, 384维向量)                │
│                      rag.py                                 │
└─────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  2. Chroma 向量库检索                      │
│              在 725 个文本块中搜索相似内容                    │
│                      rag.py                                 │
└─────────────────────────┬───────────────────────────────────┘
                         │ 返回相关文档
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. MiniMax LLM 生成答案                     │
│         (MiniMax-M2.1 / MiniMax-M2.5)                       │
│                      chat.py                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔗 如何对接 LangChain RAG

### 核心代码文件

| 文件 | 作用 | LangChain 组件 |
|------|------|---------------|
| `rag.py` | 向量化 + 检索 | `HuggingFaceEmbeddings` + `Chroma` |
| `chat.py` | 对话 + LLM | `ChatOpenAI` + `ChatPromptTemplate` |
| `server.py` | Web 服务 | LangChain Serve |

---

### 1. 对接 Chroma 向量库 (`rag.py`)

**核心代码**：

```python
# src/rag.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TravelRAG:
    def __init__(self, data_dir):
        # 1. 初始化 Embedding 模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",  # 中文 Embedding
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def build_vectorstore(self):
        # 3. 加载文档
        docs = self.load_and_split()

        # 4. 构建 Chroma 向量库
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="travel_guide",
            persist_directory="data/vectorstore",
        )

    def get_retriever(self, top_k=5):
        # 5. 获取检索器
        return self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
```

**LangChain 组件说明**：

| 组件 | 作用 |
|------|------|
| `HuggingFaceEmbeddings` | 调用 BGE 模型将文本转为向量 |
| `Chroma` | 本地向量数据库，存储和检索向量 |
| `RecursiveCharacterTextSplitter` | 将长文档分割成小块 |

---

### 2. 对接 MiniMax LLM (`chat.py`)

**核心代码**：

```python
# src/chat.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TravelChat:
    def __init__(self, rag):
        # 1. 初始化 MiniMax (兼容 OpenAI 格式)
        self.llm = ChatOpenAI(
            model="MiniMax-M2.1",
            base_url="https://api.minimax.chat/v1",
            api_key="your_api_key",
            temperature=0.7,
        )

    def chat(self, question):
        # 2. RAG 检索
        retriever = self.rag.get_retriever(top_k=5)
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. 判断检索是否成功
        if not docs:
            # RAG 未命中，使用通用提示
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个旅游顾问..."),
                ("user", "用户问题：{question}"),
            ])
        else:
            # RAG 命中，使用知识库
            prompt = ChatPromptTemplate.from_messages([
                ("system", "基于旅游攻略信息回答..."),
                ("system", "旅游攻略信息：\n{context}"),
                ("user", "用户问题：{question}"),
            ])

        # 4. 调用 LLM
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {"answer": answer, "rag_hit": bool(docs)}
```

**LangChain 组件说明**：

| 组件 | 作用 |
|------|------|
| `ChatOpenAI` | 调用 MiniMax API（兼容 OpenAI 格式） |
| `ChatPromptTemplate` | 构建提示词模板 |
| `StrOutputParser` | 解析 LLM 输出 |

---

### 3. LangChain Server (`server.py`)

用于启动 Web UI：

```python
# src/server.py

from langchain_core.runnables import Runnable

def create_chain() -> Runnable:
    """创建 LCEL chain"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "旅游攻略：\n{context}"),
        ("user", "{question}"),
    ])

    chain = (
        {"context": retriever, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# 启动 Web UI
# langchain serve
```

---

## 🔑 关键配置 (`.env`)

```bash
# LLM 配置
LLM_PROVIDER=minimax
MINIMAX_API_KEY=sk-xxx
MINIMAX_MODEL=MiniMax-M2.5

# Embedding 配置
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
HF_ENDPOINT=https://hf-mirror.com

# RAG 配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5

# 向量库
VECTORSTORE_TYPE=chroma
```

---

## 📊 数据流向

```
┌────────────────────────────────────────────────────────────────┐
│                    知识库构建                              │
├────────────────────────────────────────────────────────────────┤
│  kyoto_osaka.md (50万字)                                  │
│        ↓                                                  │
│  loader.py (TextLoader)                                    │
│        ↓                                                  │
│  分割成 725 个文本块 (500字/块, 50字重叠)                   │
│        ↓                                                  │
│  rag.py (HuggingFaceEmbeddings)                            │
│        ↓                                                  │
│  转为 384维向量                                          │
│        ↓                                                  │
│  存入 Chroma 向量库 (chroma.sqlite3)                     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    问答流程                                │
├────────────────────────────────────────────────────────────────┤
│  用户: "京都有什么好吃的"                                   │
│        ↓                                                  │
│  Embedding 模型 → 384维向量                                │
│        ↓                                                  │
│  Chroma 相似度搜索 → 返回 Top 5 相关文档                    │
│        ↓                                                  │
│  构建上下文 (相关文档拼接)                                  │
│        ↓                                                  │
│  MiniMax LLM + Prompt → 生成答案                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Volumes/zhitai2/git/python/travel-rag
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 MiniMax API Key
```

### 3. 运行

```bash
# 命令行问答
python run.py "京都有什么好吃的？"

# Web UI
langchain serve
# 访问 http://localhost:8000
```

---

## 🧪 调试命令

```bash
# 查看向量库内容
python src/inspect_vector.py

# 测试检索
python -c "
from src.rag import TravelRAG
rag = TravelRAG('data/knowledge')
docs = rag.similarity_search('京都美食', top_k=3)
for doc in docs:
    print(doc.page_content[:200])
"

# 测试对话
python -c "
from src.rag import TravelRAG
from src.chat import TravelChat
rag = TravelRAG('data/knowledge')
chat = TravelChat(rag)
result = chat.chat('京都三日游怎么安排')
print(result['answer'])
"
```

---

## 📝 笔记

- **Embedding 模型**: BAAI/bge-small-zh-v1.5 (384维)
- **向量库**: Chroma (SQLite 文件存储)
- **LLM**: MiniMax-M2.1 或 MiniMax-M2.5
- 首次运行需构建向量库，约需 1-2 分钟
- 向量库构建后，查询速度 < 100ms