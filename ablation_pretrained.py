"""사전학습 가중치 유무에 따른 검색 품질 비교.

## 왜 이 스크립트가 있나

이 레포의 임베딩 추출 코드는 `pretrained=False`로 DINOv3를 로드하고 있었다.

    # image_search_app.py:34, mapillary_search_app.py:37
    model = torch.hub.load("./dinov3", model_name, source="local", pretrained=False)

즉 **랜덤 초기화된 ViT**로 임베딩을 뽑고 있었다. 랜덤 초기화 ViT도 무작위 투영처럼
동작해 저수준 구조(색 분포, 대략적 배치)는 어느 정도 보존하므로 검색 결과가 "그럭저럭
비슷해 보이는" 일이 생긴다. 그래서 눈으로는 버그를 알아채기 어렵다. 하지만 그건
DINOv3의 자기지도 특징이 아니다.

이 스크립트는 그 차이를 정량화한다.

## 지표

Precision@k — 질의 이미지의 상위 k개 이웃 중 같은 클래스인 비율. 클래스 레이블은
**평가에만** 쓰고 임베딩 추출에는 쓰지 않는다(자기지도 특징의 선형 분리성을 재는
표준 방식인 kNN 평가와 같은 취지).

## 사용법

    pip install torch transformers pillow numpy

    # 클래스별 하위 폴더 구조: images_dir/<class>/<image>.jpg
    python ablation_pretrained.py --images-dir ./sample_images

출력된 표를 README에 그대로 붙이면 된다.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_labeled_images(images_dir: Path, limit: int) -> tuple[list[Image.Image], np.ndarray]:
    """`images_dir/<class>/<image>` 구조에서 이미지와 클래스 인덱스를 읽는다."""
    paths: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append((path, class_dir.name))

    if not paths:
        raise SystemExit(
            f"{images_dir} 에서 클래스 하위 폴더를 찾지 못했습니다.\n"
            "구조: images_dir/<class_name>/<image>.jpg"
        )

    rng = np.random.default_rng(0)
    if len(paths) > limit:
        chosen = rng.choice(len(paths), size=limit, replace=False)
        paths = [paths[int(i)] for i in sorted(chosen)]

    classes = sorted({name for _, name in paths})
    class_to_index = {name: index for index, name in enumerate(classes)}

    images = [Image.open(path).convert("RGB") for path, _ in paths]
    labels = np.array([class_to_index[name] for _, name in paths])

    counts = Counter(name for _, name in paths)
    print(f"이미지 {len(images)}장, 클래스 {len(classes)}개")
    print("  " + ", ".join(f"{name}:{counts[name]}" for name in classes))
    return images, labels


@torch.inference_mode()
def embed(model: nn.Module, processor: object, images: list[Image.Image], batch: int) -> np.ndarray:
    """CLS 토큰 임베딩을 뽑아 L2 정규화해서 반환한다."""
    rows: list[np.ndarray] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        inputs = processor(images=chunk, return_tensors="pt")  # type: ignore[operator]
        cls = model(**inputs).last_hidden_state[:, 0, :]
        rows.append(cls.float().numpy())
        print(f"  {min(start + batch, len(images))}/{len(images)}", end="\r")
    print()
    matrix = np.vstack(rows).astype(np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def precision_at_k(emb: np.ndarray, labels: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    """상위 k 이웃의 클래스 일치율. 자기 자신은 제외한다."""
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)
    return {k: float((labels[order[:, :k]] == labels[:, None]).mean()) for k in ks}


def main() -> None:
    parser = argparse.ArgumentParser(description="사전학습 가중치 유무 비교")
    parser.add_argument("--images-dir", type=Path, required=True, help="클래스별 하위 폴더 구조")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(0)
    images, labels = load_labeled_images(args.images_dir, args.limit)
    ks = (1, 5, 10)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    print(f"\n모델: {args.model_id}")

    print("\n[1/2] 사전학습 가중치 (정상)")
    pretrained = AutoModel.from_pretrained(args.model_id).eval()
    scores_pretrained = precision_at_k(embed(pretrained, processor, images, args.batch), labels, ks)

    print("[2/2] 랜덤 초기화 (pretrained=False 상태 재현)")
    random_init = AutoModel.from_config(pretrained.config).eval()
    scores_random = precision_at_k(embed(random_init, processor, images, args.batch), labels, ks)

    # 같은 클래스 쌍의 비율 = 무작위로 골랐을 때의 기대 정확도
    chance = float((labels[:, None] == labels[None, :]).mean())

    header = " | ".join(f"Precision@{k}" for k in ks)
    print(f"\n| 설정 | {header} |")
    print("|---|" + "---:|" * len(ks))
    pretrained_row = " | ".join(f"{scores_pretrained[k] * 100:.1f}%" for k in ks)
    print(f"| DINOv3 사전학습 | {pretrained_row} |")
    random_row = " | ".join(f"{scores_random[k] * 100:.1f}%" for k in ks)
    print(f"| 랜덤 초기화 | {random_row} |")
    chance_row = " | ".join(f"{chance * 100:.1f}%" for _ in ks)
    print(f"| 무작위 기대값 | {chance_row} |")

    gain = scores_pretrained[5] - scores_random[5]
    ratio = scores_pretrained[5] / max(scores_random[5], 1e-9)
    print(f"\nPrecision@5: {gain * 100:+.1f}%p ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
