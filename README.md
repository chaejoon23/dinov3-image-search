# DINOv3 Image Search

**거리 사진 한 장으로 같은 장소를 찾는다.** DINOv3의 자기지도 특징으로 이미지를 384차원
벡터로 바꿔, 코사인 유사도로 시각적으로 닮은 이미지를 검색합니다. 후쿠오카 거리영상
243장에는 GPS 좌표·촬영시각·나침반 방위가 붙어 있어, 검색 결과가 곧 위치 추정이 됩니다.


---

## 사전학습 가중치가 실제로 얼마나 기여하는가

이 레포의 핵심 실험입니다. 처음 구현에서 저는 모델을 이렇게 올리고 있었습니다.

```python
model = torch.hub.load("./dinov3", model_name, source="local", pretrained=False)
#                                                              ^^^^^^^^^^^^^^^^^
```

`pretrained=False`는 **랜덤 초기화**입니다. 즉 DINOv3의 구조만 쓰고 자기지도 학습으로
얻은 가중치는 전혀 쓰지 않은 상태로 임베딩을 뽑고 있었습니다.

**그런데 데모는 돌아가는 것처럼 보였습니다.** 랜덤 초기화 ViT도 무작위 투영처럼 작동해서
색 분포나 대략적인 구도 같은 저수준 구조는 어느 정도 보존합니다. 그래서 검색 결과가
"그럭저럭 비슷해 보이는" 이미지를 돌려주고, 눈으로는 버그를 알아채기 어렵습니다.

측정해서 확인했습니다.

| 설정 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---:|---:|
| DINOv3 사전학습 | 89.6% | 86.1% | 83.3% |
| 랜덤 초기화 | 21.7% | 19.5% | 18.8% |
| 무작위 기대값 | 10.1% | 10.1% | 10.1% |

CIFAR-10 테스트셋 1,000장(10개 클래스), `facebook/dinov3-vits16-pretrain-lvd1689m` 기준.
Precision@5에서 **+66.6%p (4.41배)** 차이가 납니다.

Precision@k는 질의 이미지의 상위 k개 이웃 중 같은 클래스인 비율입니다. **클래스 레이블은
평가에만 쓰고 임베딩 추출에는 쓰지 않습니다** — 자기지도 특징의 품질을 재는 kNN 평가와
같은 방식입니다.

재현:

```bash
python ablation_pretrained.py --images-dir ./sample_images
```

교훈은 두 가지였습니다. **"돌아간다"가 "맞게 돌아간다"는 뜻이 아니고**, 무작위 기대값을
기준선으로 깔아두지 않으면 얼마나 잘하는지 알 수 없다는 것.

---

## 왜 이 문제인가

영상·사진에서 **장소를 알아내는 문제**를 계속 보고 있습니다.

이 레포는 그중 "한 장의 사진이 어디인가"를 임베딩 유사도로 푸는 쪽입니다. 같은 관심의
다른 축 — 영상에서 어떤 프레임을 골라 봐야 하는가, 어두운 프레임을 어떻게 읽을 수 있게
만드는가 — 는 [**Bin_pind**](https://github.com/chaejoon23/Bin_pind)의 비전 프론트엔드에
있습니다. 그쪽은 DINOv3를 **장면 중복 제거**에 쓰고, 이 레포는 **검색**에 씁니다.
같은 모델을 서로 다른 목적에 붙여본 셈입니다.

---

## 구조

```
질의 이미지
    │
    ├─ 전처리: Resize → CenterCrop → Normalize
    ├─ DINOv3 ViT-S/16 → CLS 토큰 384차원
    └─ L2 정규화
    │
    ▼
코사인 유사도 = 정규화된 DB 행렬 @ 질의 벡터   (내적 한 번)
    │
    ▼
상위 K개 + 메타데이터(GPS·촬영시각·방위)
```

L2 정규화를 미리 해두면 코사인 유사도가 그냥 내적이 됩니다. 그래서 1,000장 검색이
행렬-벡터 곱 한 번으로 끝나고, 임베딩만 미리 계산해두면 검색은 1ms 미만입니다.

| | |
|---|---|
| 모델 | DINOv3 ViT-S/16 (21M 파라미터, 384차원) |
| 임베딩 추출 | 약 55–77 img/s (Apple Silicon MPS) |
| 검색 | 1,000장 < 1ms |
| 저장 | 이미지당 약 1.5KB |
| 장치 | MPS / CUDA / CPU 자동 선택 |

---

## 데이터셋

**후쿠오카 Mapillary** (243장) — 크라우드소싱 거리영상. GPS 좌표, 촬영 시각, 나침반 방위,
Mapillary 원본 링크가 결과에 함께 표시됩니다. 도시 전역을 약 0.8km 타일 격자로 훑어
수집했습니다.

**CIFAR-10** (1,000장) — 10개 카테고리. 빠른 데모와 위 ablation 평가용. 클래스 레이블이
있어서 Precision@k를 계산할 수 있습니다.

---

## 실행

```bash
pip install torch transformers gradio pillow numpy tqdm

# 1. 데이터 준비
python download_dataset.py                      # CIFAR-10 1,000장
python download_mapillary_fukuoka.py \           # Mapillary (API 토큰 필요)
    --token "MLY|..." --count 500

# 2. 임베딩 DB 생성
python build_embedding_database.py --images-dir ./sample_images

# 3. 검색 UI
python image_search_app.py                      # CIFAR-10
python mapillary_search_app.py                  # 후쿠오카
```

Mapillary API 토큰은 https://www.mapillary.com/dashboard/developers 에서 발급합니다.

### 더 큰 모델

```bash
python build_embedding_database.py --model-id facebook/dinov3-vitb16-pretrain-lvd1689m
```

| 모델 | 파라미터 | 차원 |
|---|---|---|
| `dinov3-vits16` | 21M | 384 ← 기본 |
| `dinov3-vitb16` | 86M | 768 |
| `dinov3-vitl16` | 304M | 1024 |

---

## 한계

- **243장은 작습니다.** 후쿠오카 전역을 덮지 못하므로, 질의 사진과 같은 거리의 이미지가
  DB에 없으면 엉뚱한 결과가 나옵니다. 검색 품질과 커버리지를 분리해서 봐야 합니다
- **전수 검색**입니다. 1만 장을 넘기면 FAISS 같은 근사 최근접 탐색이 필요합니다
- **CIFAR-10 Precision@k가 거리영상 성능을 대표하지 않습니다.** 32×32 물체 분류와
  거리 장면 매칭은 다른 문제입니다. 거리영상 쪽 정답 레이블이 없어 아직 정량 평가를
  못 했고, GPS 좌표로 "상위 결과가 질의 위치에서 몇 m 안에 있는가"를 재는 것이 다음 작업입니다

---

## 라이선스

**DINOv3 가중치는 Apache 2.0이 아닙니다.** Apache 2.0은 DINOv2이고, DINOv3는 Meta의 별도
DINOv3 License로 배포됩니다. 상용 목적이면 조건을 먼저 확인하세요.

- 논문: Siméoni et al., "DINOv3", [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- 모델: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- CIFAR-10: Krizhevsky, 2009
- Mapillary: 크라우드소싱 거리영상, [API 문서](https://www.mapillary.com/developer/api-documentation)

이 구현은 학습·연구 목적입니다.
