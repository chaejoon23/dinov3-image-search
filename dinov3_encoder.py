"""DINOv3 인코더 — 사전학습 가중치를 실제로 로드하는 드롭인 교체본.

## 왜 이 파일이 있나

기존 코드는 `torch.hub.load("./dinov3", ..., pretrained=False)`로 모델을 올리고 있었다.
`pretrained=False`는 **랜덤 초기화**를 뜻하므로, 뽑힌 임베딩은 DINOv3의 자기지도 특징이
아니다. 이 파일은 Hugging Face `transformers`로 공개 가중치를 받아 쓰는 방식으로 바꾼다.

`torch.hub` + 로컬 `dinov3/` 체크아웃 방식보다 이게 나은 이유:

- 가중치 파일을 따로 구해 경로를 맞출 필요가 없다 (허브에서 자동 캐시)
- `pretrained` 플래그로 실수할 여지가 없다 — `from_pretrained`는 이름 그대로 동작한다
- 랜덤 초기화가 필요할 때는 `from_config`로 **명시적으로만** 만들 수 있다(ablation용)

## 사용법

    from dinov3_encoder import Dinov3Encoder

    encoder = Dinov3Encoder()                 # 사전학습 가중치
    emb = encoder.embed(list_of_pil_images)   # (N, 384) L2 정규화

    # ablation 용도로만
    random_encoder = Dinov3Encoder(pretrained=False)

## 모델

`facebook/dinov3-vits16-pretrain-lvd1689m` — ViT-S/16, 21M 파라미터, 384차원.
Siméoni et al., "DINOv3", arXiv:2508.10104.

라이선스 주의: DINOv3는 Apache 2.0이 **아니다**(그건 DINOv2다). Meta의 별도 DINOv3
License로 배포되므로 상용 배포 전 조건을 확인해야 한다.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

Embedding = NDArray[np.float32]


def pick_device(prefer: str | None = None) -> torch.device:
    """MPS > CUDA > CPU 순으로 사용 가능한 장치를 고른다."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Dinov3Encoder:
    """이미지 → L2 정규화된 DINOv3 CLS 임베딩."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        pretrained: bool = True,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """인코더를 초기화한다.

        Args:
            model_id: Hugging Face 모델 ID.
            pretrained: True면 공개 가중치를 로드한다. **기본값이 True인 것이 중요하다** —
                False는 랜덤 초기화이며 ablation 실험 용도로만 쓴다.
            device: "mps" / "cuda" / "cpu". None이면 자동 선택.
            batch_size: 추론 배치 크기.
        """
        self.model_id = model_id
        self.pretrained = pretrained
        self.device = pick_device(device)
        self.batch_size = batch_size

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        if pretrained:
            model = AutoModel.from_pretrained(model_id)
        else:
            # 명시적으로 요청했을 때만 랜덤 초기화. 실수로 이 경로를 타지 않도록
            # 경고를 남긴다.
            print(
                "[경고] pretrained=False — 랜덤 초기화 모델입니다. "
                "이 임베딩은 DINOv3의 자기지도 특징이 아닙니다(ablation 전용)."
            )
            config = AutoModel.from_pretrained(model_id).config
            model = AutoModel.from_config(config)

        self.model = model.to(self.device).eval()

    @property
    def name(self) -> str:
        """로그·표에 찍히는 식별자."""
        suffix = "pretrained" if self.pretrained else "random-init"
        return f"dinov3({self.model_id.split('/')[-1]}, {suffix})"

    @property
    def embedding_dim(self) -> int:
        """임베딩 차원 (ViT-S/16 = 384)."""
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> Embedding:
        """(N, D) float32 행렬. 각 행은 L2 정규화되어 코사인 유사도를 내적으로 계산할 수 있다."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        rows: list[NDArray[np.float32]] = []
        for start in range(0, len(images), self.batch_size):
            chunk = [image.convert("RGB") for image in images[start : start + self.batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            cls = self.model(**inputs).last_hidden_state[:, 0, :]  # CLS 토큰
            rows.append(cls.float().cpu().numpy().astype(np.float32))

        matrix = np.vstack(rows).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized: Embedding = (matrix / np.maximum(norms, 1e-8)).astype(np.float32)
        return normalized

    def embed_one(self, image: Image.Image) -> Embedding:
        """이미지 1장의 임베딩 (D,)."""
        result: Embedding = self.embed([image])[0]
        return result


def search(
    database: Embedding, query: Embedding, top_k: int = 5
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """코사인 유사도 검색.

    두 입력이 모두 L2 정규화되어 있으므로 내적이 곧 코사인 유사도다.

    Args:
        database: (N, D) 정규화된 임베딩.
        query: (D,) 정규화된 질의 임베딩.
        top_k: 반환 개수.

    Returns:
        (인덱스, 유사도) — 유사도 내림차순.
    """
    similarities = (database @ query).astype(np.float32)
    order = np.argsort(-similarities)[:top_k].astype(np.int64)
    return order, similarities[order]
