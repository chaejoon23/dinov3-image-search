import torch
import time


def check_device():
    """Apple Silicon(MPS)을 최우선으로 체크하고, 없으면 CPU를 사용합니다."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon (MPS) 사용 가능. 디바이스로 설정합니다.")
    elif torch.cuda.is_available():
        # 혹시 CUDA가 가능한 다른 환경일 경우
        device = torch.device("cuda")
        print("✅ CUDA 사용 가능. 디바이스로 설정합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS/CUDA 사용 불가. CPU로 실행합니다. (속도가 매우 느릴 수 있습니다)")
    return device


def run_dinov3_smoke_test(device):
    """
    DinoV3 모델 로딩 및 포워드 패스(추론) 스모크 테스트
    """
    print("\n--- DinoV3 스모크 테스트 시작 ---")

    try:
        # 1. 모델 로드 (로컬 DINOv3 리포지토리 사용)
        # pretrained=False로 설정하여 가중치 없이 모델 구조만 로드합니다.
        print("1. DinoV3 (ViT-Small/16) 모델 로드 중... (가중치 없이 초기화)")
        start_load = time.time()
        model = torch.hub.load(
            "./dinov3",
            "dinov3_vits16",
            source="local",
            pretrained=False  # 사전 학습된 가중치 없이 모델 구조만 로드
        )
        print(f"   ... 로드 완료! ({time.time() - start_load:.2f}초 소요)")

        # 2. 모델을 MPS 디바이스로 이동
        model.to(device)
        model.eval()  # 추론 모드로 설정 (필수)
        print(f"2. 모델을 '{device}' 디바이스로 이동 완료.")

        # 3. 가상 이미지(더미 텐서) 생성
        # (배치크기, 채널, 높이, 너비)
        # DinoV3는 224x224 입력을 처리할 수 있습니다.
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        print("3. (1, 3, 224, 224) 크기의 가상 이미지 텐서 생성 완료.")

        # 4. 포워드 패스(추론) 실행
        # torch.no_grad()는 그래디언트 계산을 비활성화하여 메모리와 속도를 최적화합니다.
        print("4. 모델 포워드 패스(추론) 실행 중...")
        start_inference = time.time()
        with torch.no_grad():
            features = model(dummy_input)
        print(f"   ... 추론 완료! ({time.time() - start_inference:.2f}초 소요)")

        # 5. 결과 확인
        print("\n--- 🚀 스모크 테스트 성공! ---")
        print(f"DinoV3 모델이 '{device}' 디바이스에서 성공적으로 실행되었습니다.")
        print(f"최종 출력(특징 벡터) Shape: {features.shape}")

    except Exception as e:
        print("\n--- ❌ 스모크 테스트 실패 ---")
        print(f"오류 발생: {e}")
        print("네트워크 연결, PyTorch 설치 상태 또는 torch.hub 캐시를 확인하세요.")


if __name__ == "__main__":
    target_device = check_device()
    run_dinov3_smoke_test(target_device)
