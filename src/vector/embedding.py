from typing import List
import os

from transformers import AutoModel, AutoTokenizer
import torch

# ⭐ 强制离线：避免 hf-mirror.com 不可用 / 代理连不上 huggingface.co
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 清空 HF_ENDPOINT，避免走 hf-mirror.com
os.environ["HF_ENDPOINT"] = ""

# BGE-M3：多语言 / 1024 维 / max_seq_length=8192 / CLS pooling
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(DEVICE).eval()


def embedding_by_baai(query: List[str]):
    """返回 numpy.ndarray，shape=(n, 1024)，L2 归一化（cosine 距离需要）

    加载说明：用 transformers 直接加载，绕过 sentence-transformers 5.x
    的 processor class 要求（bge-m3 是 XLMRobertaForMaskedLM 衍生，
    没有 processor_config.json）。
    """
    inputs = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=8192,  # bge-m3 支持 8192 token
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # ⭐ CLS token pooling（sentence-transformers bge-m3 配置）
    embeddings = outputs.last_hidden_state[:, 0]
    # L2 归一化（cosine 距离等价于 dot product）
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()