# Waymo Dataset Auto Inference Guide

이 가이드는 훈련된 HunyuanDiT 모델을 사용하여 Waymo 데이터셋의 텍스트 프롬프트로부터 자동으로 이미지를 생성하는 방법을 설명합니다.

## 개요

`waymo_inference.py` 스크립트는 다음 작업을 수행합니다:

1. `/mnt/ssd/HunyuanDiT/dataset/waymo/csvfile/image_text.csv`에서 텍스트 프롬프트를 읽습니다
2. 훈련된 HunyuanDiT 모델을 로드합니다
3. 각 텍스트 프롬프트로부터 이미지를 생성합니다
4. 원본 이미지와 생성된 이미지를 나란히 비교할 수 있는 이미지를 생성합니다
5. 결과를 JSON 메타데이터와 함께 저장합니다

## 사용 방법

### 1. 빠른 실행 (쉘 스크립트 사용)

```bash
# 기본 설정으로 10개 샘플 처리
./scripts/run_waymo_inference.sh

# 50개 샘플 처리
./scripts/run_waymo_inference.sh --limit 50

# 모든 샘플 처리
./scripts/run_waymo_inference.sh --all

# 특정 체크포인트 사용
./scripts/run_waymo_inference.sh --load-key e0084 --limit 20

# 도움말 보기
./scripts/run_waymo_inference.sh --help
```

### 2. 직접 Python 스크립트 실행

```bash
# 기본 설정
cd /mnt/ssd/HunyuanDiT
python scripts/waymo_inference.py

# 사용자 정의 설정
python scripts/waymo_inference.py \
    --model-root /mnt/ssd/HunyuanDiT/results/waymo_base/001-DiT-g-2 \
    --load-key latest \
    --limit 10 \
    --seed 42 \
    --extra-fp16 \
    --random-sample
```

## 주요 매개변수

### 필수 매개변수
- `--model-root`: 훈련된 모델이 저장된 디렉토리 경로
- `--load-key`: 로드할 체크포인트 키 (예: `latest`, `e0084`, `final`)

### 선택적 매개변수
- `--csv-path`: Waymo CSV 파일 경로 (기본값: dataset/waymo/csvfile/image_text.csv)
- `--output-dir`: 결과를 저장할 디렉토리 (기본값: 자동 생성)
- `--limit`: 처리할 샘플 수 제한 (기본값: 전체)
- `--seed`: 재현 가능한 결과를 위한 랜덤 시드
- `--random-sample`: 데이터셋에서 랜덤 샘플링
- `--extra-fp16`: FP16 정밀도 사용 (메모리 절약)

### 모델 관련 매개변수
- `--model`: 모델 아키텍처 (기본값: DiT-g/2)
- `--image-size`: 생성 이미지 크기 (기본값: 1024)
- `--text-len`: BERT 텍스트 시퀀스 길이 (기본값: 77)
- `--text-len-t5`: T5 텍스트 시퀀스 길이 (기본값: 256)

## 출력 결과

실행 후 출력 디렉토리에 다음 파일들이 생성됩니다:

### 이미지 파일
- `generated_XXXXXX.jpg`: 생성된 이미지들
- `comparison_XXXXXX.jpg`: 원본과 생성 이미지 비교 (원본 이미지가 있는 경우)

### 메타데이터
- `inference_results.json`: 실행 설정과 결과 정보

```json
{
  "args": { ... },
  "total_samples": 10,
  "successful_generations": 10,
  "total_time": 120.5,
  "results": [
    {
      "index": 0,
      "original_path": "dataset/waymo/images/...",
      "text_prompt": "The scene is set during the day...",
      "generated_path": "/path/to/generated_000000.jpg",
      "comparison_path": "/path/to/comparison_000000.jpg"
    }
  ]
}
```

## 예제 실행

### 예제 1: 빠른 테스트
```bash
# 5개 샘플로 빠른 테스트
./scripts/run_waymo_inference.sh --limit 5 --seed 42
```

### 예제 2: 특정 체크포인트로 품질 평가
```bash
# 특정 epoch의 체크포인트로 20개 샘플 생성
./scripts/run_waymo_inference.sh --load-key e0100 --limit 20
```

### 예제 3: 전체 데이터셋 처리
```bash
# 모든 샘플 처리 (시간이 오래 걸림)
./scripts/run_waymo_inference.sh --all
```

## 문제 해결

### 메모리 부족
```bash
# FP16 사용으로 메모리 사용량 줄이기
python scripts/waymo_inference.py --extra-fp16
```

### 체크포인트를 찾을 수 없음
```bash
# 사용 가능한 체크포인트 확인
ls -la /mnt/ssd/HunyuanDiT/results/waymo_base/001-DiT-g-2/checkpoints/
```

### CUDA 메모리 오류
- 배치 크기를 줄이거나 `--extra-fp16` 옵션을 사용하세요
- 다른 GPU 프로세스가 실행 중인지 확인하세요

## 결과 분석

생성된 이미지들을 평가하기 위해:

1. **시각적 품질**: `comparison_*.jpg` 파일들로 원본과 비교
2. **텍스트 일치도**: 생성된 이미지가 프롬프트와 얼마나 잘 맞는지 확인
3. **일관성**: 비슷한 프롬프트들이 일관된 스타일로 생성되는지 확인

## 추가 기능

향후 업데이트에서 다음 기능들이 추가될 예정입니다:

- [ ] 자동 품질 메트릭 계산 (FID, LPIPS 등)
- [ ] 배치 처리 최적화
- [ ] 다양한 이미지 크기 지원
- [ ] ControlNet 통합
- [ ] 웹 인터페이스

## 문의사항

문제가 발생하거나 개선사항이 있으면 GitHub Issues에 등록해주세요.