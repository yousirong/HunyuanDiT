# Waymo 자율주행 데이터셋 HunyuanDiT 훈련 가이드

이 가이드는 256×256 Waymo 자율주행 데이터셋을 사용하여 HunyuanDiT 모델을 훈련하는 방법을 설명합니다.

## 📁 데이터셋 구조

```
/mnt/ssd/HunyuanDiT/dataset/waymo/
├── images/                     # 256x256 자율주행 이미지
├── arrows/                     # Arrow 형식 데이터 파일 (39개)
├── csvfile/
│   └── image_text.csv         # 이미지-텍스트 쌍
├── depth_maps/                # ControlNet용 깊이 맵 (선택사항)
└── waymo_index.json           # 훈련용 인덱스 파일
```

## 🚀 빠른 시작

### 1. 소규모 테스트 (100개 샘플, 50 스텝)
```bash
bash scripts/test_waymo_small.sh
```

### 2. 기본 모델 훈련 (1000개 샘플 기본)
```bash
# 1000개 샘플로 훈련
bash scripts/train_waymo_base.sh

# 5000개 샘플로 훈련
bash scripts/train_waymo_base.sh 5000
```

### 3. ControlNet 훈련 (깊이 제어)
```bash
# 먼저 깊이 맵 생성 (선택사항)
python scripts/generate_depth_maps.py --limit 1000

# ControlNet 훈련
bash scripts/train_waymo_controlnet.sh 1000
```

## 📊 샘플 개수 제한 기능

모든 훈련 스크립트에서 `--sample-limit` 파라미터로 사용할 샘플 개수를 제한할 수 있습니다:

```bash
# 방법 1: 스크립트 인수로 전달
bash scripts/train_waymo_base.sh 500

# 방법 2: 직접 파라미터 지정
python hydit/train_deepspeed.py \
    --sample-limit 500 \
    [other parameters...]
```

## 🛠️ 훈련 설정

### 기본 모델 훈련 설정
- **이미지 크기**: 256×256
- **배치 사이즈**: 4 (기본)
- **학습률**: 1e-5
- **최대 스텝**: 10,000
- **모델**: DiT-g/2
- **정밀도**: FP16

### ControlNet 훈련 설정  
- **이미지 크기**: 256×256
- **배치 사이즈**: 2 (기본)
- **ControlNet 모드**: depth
- **LoRA 랭크**: 32
- **훈련 부분**: controlnet만

## 📝 데이터 포맷

### CSV 파일 형식 (image_text.csv)
```csv
image_path,text_en
dataset/waymo/images/frame_1.jpg,"Residential area with wet roads, one parked vehicle..."
dataset/waymo/images/frame_2.jpg,"Daytime scene with lane markings, caution needed..."
```

### 필수 컬럼
- `image_path`: 이미지 파일 경로
- `text_en`: 영어 캡션 (자율주행 상황 설명)

## 🎯 사용 예시

### 예시 1: 빠른 테스트
```bash
# 100개 샘플로 50스텝 훈련
bash scripts/test_waymo_small.sh
```

### 예시 2: 중간 규모 훈련
```bash  
# 2000개 샘플로 기본 모델 훈련
bash scripts/train_waymo_base.sh 2000
```

### 예시 3: ControlNet 깊이 제어
```bash
# 1. 1000개 이미지에 대한 깊이 맵 생성
python scripts/generate_depth_maps.py --limit 1000

# 2. ControlNet 훈련
bash scripts/train_waymo_controlnet.sh 1000
```

## 📈 모니터링

### 로그 확인
```bash
# 훈련 로그
tail -f results/waymo_*/logs/train.log

# TensorBoard (있는 경우)
tensorboard --logdir results/waymo_*
```

### 체크포인트 위치
- **기본 모델**: `results/waymo_base/`
- **ControlNet**: `results/waymo_controlnet_depth/`
- **테스트**: `results/waymo_test/`

## ⚙️ 고급 설정

### GPU 메모리 최적화
```bash
# 작은 배치 사이즈 사용
bash scripts/train_waymo_base.sh 1000
# 스크립트 내 BATCH_SIZE=2 로 수정

# ZeRO Stage 3 사용 (더 많은 메모리 절약)
# 스크립트 내 --use-zero-stage 3 으로 수정
```

### 다중 해상도 훈련
```bash
# YAML 설정 파일 사용
python hydit/train_deepspeed.py \
    --multireso \
    --index-file dataset/yamls/waymo.yaml \
    [other parameters...]
```

## 🔧 문제 해결

### 메모리 부족
1. 배치 사이즈 줄이기: `BATCH_SIZE=1`
2. 샘플 제한: `--sample-limit 100`
3. ZeRO Stage 3 사용

### 데이터 로딩 오류
1. 인덱스 파일 재생성: `dataset/waymo/waymo_index.json`
2. Arrow 파일 확인: `dataset/waymo/arrows/`
3. 이미지 경로 확인: `dataset/waymo/images/`

## 📚 참고사항

- **총 샘플 수**: 약 192,548개
- **Arrow 파일**: 39개 (각 ~5000개 샘플)
- **권장 테스트**: 100-1000 샘플로 시작
- **풀 데이터셋**: 모든 샘플 사용시 `--sample-limit` 제거

## 🎬 자율주행 시나리오 특화

이 설정은 다음과 같은 자율주행 시나리오에 최적화되어 있습니다:
- **현재 장면** → **텍스트 캡션** → **미래 프레임 생성**
- **ControlNet**을 통한 도로 구조 유지
- **깊이 정보**를 활용한 3D 일관성
- **다중 제어** (Canny + Depth) 지원

훈련 완료 후 생성된 모델로 자율주행 상황의 미래 프레임을 생성할 수 있습니다.