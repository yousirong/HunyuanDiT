# Best Checkpoint Tracking Guide

이 가이드는 HunyuanDiT 훈련 중에 자동으로 최고 성능의 체크포인트를 추적하고 저장하는 새로운 기능을 설명합니다.

## 🌟 새로운 기능

### 자동 성능 평가
- **CLIP Score**: 텍스트-이미지 일치도 측정
- **LPIPS**: 지각적 이미지 유사성 측정
- **Inception Score**: 이미지 품질과 다양성 측정
- **FID**: Frechet Inception Distance (참조 이미지가 있을 때)
- **Composite Score**: 모든 메트릭을 종합한 점수

### 자동 Best 체크포인트 관리
- 매 2번째 체크포인트마다 성능 평가 실행
- 상위 3개 체크포인트 자동 유지
- `best.pt` 심볼릭 링크 자동 업데이트
- 오래된 체크포인트 자동 정리 (매 10k 스텝마다 하나씩 보존)

## 📦 설치

평가 메트릭을 위한 추가 패키지 설치:

```bash
# 평가용 패키지 설치
pip install -r requirements_evaluation.txt

# 또는 개별 설치
pip install lpips clip-by-openai pytorch-fid scipy scikit-image
```

## 🚀 사용 방법

기존 훈련 스크립트를 그대로 사용하면 자동으로 best checkpoint tracking이 작동합니다:

```bash
# 기본 훈련 (자동 평가 포함)
./scripts/train_waymo_base.sh

# 또는 직접 실행
python hydit/train_deepspeed.py \
    --model DiT-g/2 \
    --task-flag waymo_training \
    # ... 기타 옵션들
```

## 📊 출력 결과

### 훈련 중 로그 예시
```
[Step 2000] Evaluating model performance at step 2000
🏆 New best checkpoint! Score: 0.7245
=== Best Checkpoint Summary ===
Tracking metric: composite_score
Saved checkpoints: 3/3
  #1: Step 2000, Epoch 15
      Path: 0002000.pt
      Metrics: clip_score: 0.8123, lpips: 0.1234, inception_score: 3.45, composite_score: 0.7245
  #2: Step 1600, Epoch 12
      ...
================================
```

### 생성되는 파일들

#### 체크포인트 디렉토리에서:
- `best.pt` - 최고 성능 체크포인트로의 심볼릭 링크
- `best_checkpoints.json` - Best 체크포인트 추적 상태
- `best_checkpoint_results.json` - 최종 결과 요약

#### 실험 디렉토리에서:
- `evaluation_samples/` - 평가용 샘플 이미지들
  - `eval_generated_XXX.jpg` - 생성된 이미지
  - `eval_comparison_XXX.jpg` - 원본-생성 비교 이미지
  - `eval_prompt_XXX.txt` - 사용된 텍스트 프롬프트

## 🔧 설정 커스터마이징

### Best Tracker 설정 변경

`hydit/train_deepspeed.py`에서 설정을 수정할 수 있습니다:

```python
# Best Checkpoint Tracker 초기화 부분
best_tracker = BestCheckpointTracker(
    checkpoint_dir=checkpoint_dir,
    metric_name='composite_score',  # 추적할 메트릭 변경
    higher_is_better=True,          # 점수 방향
    save_top_k=3                    # 유지할 체크포인트 수
)
```

### 평가 빈도 조정

```python
# save_checkpoint 함수 내부
if by != "final" and train_steps % (args.ckpt_every * 2) == 0:  # 매 2번째 체크포인트
    # 평가 빈도를 변경하려면 이 조건을 수정
    # 예: args.ckpt_every * 1 (매번), args.ckpt_every * 4 (4번에 1번)
```

### 사용할 메트릭 선택

주요 메트릭별 특징:

- **`composite_score`**: 모든 메트릭을 종합 (권장)
- **`clip_score`**: 텍스트-이미지 일치도만 고려
- **`inception_score`**: 이미지 품질과 다양성만 고려
- **`lpips`**: 지각적 유사성 (낮을수록 좋음, `higher_is_better=False`)
- **`fid`**: Frechet Inception Distance (낮을수록 좋음, `higher_is_better=False`)

## 📈 평가 메트릭 이해하기

### CLIP Score (0.0 ~ 1.0)
- **0.8+**: 텍스트와 이미지가 매우 잘 일치
- **0.6-0.8**: 적절한 일치도
- **0.6 미만**: 일치도가 낮음

### LPIPS (0.0 ~ 2.0+)
- **0.0-0.2**: 매우 유사 (낮을수록 좋음)
- **0.2-0.5**: 적당히 유사
- **0.5+**: 큰 차이

### Inception Score (1.0+)
- **5.0+**: 매우 좋은 품질
- **3.0-5.0**: 적절한 품질
- **3.0 미만**: 개선 필요

### Composite Score
- 모든 메트릭을 가중평균한 점수 (높을수록 좋음)
- 가중치: CLIP Score (40%), IS (30%), LPIPS (-20%), FID (-10%)

## 🛠️ 문제 해결

### 평가 패키지 설치 오류
```bash
# CLIP 설치 문제
pip install git+https://github.com/openai/CLIP.git

# FID 설치 문제  
pip install clean-fid

# LPIPS 설치 문제
pip install lpips
```

### 메모리 부족 오류
평가 시 메모리 사용량을 줄이려면:

```python
# evaluation_sampling.py에서 num_evaluation_samples 줄이기
num_evaluation_samples=10,  # 기본값 20에서 10으로 줄임
```

### 평가 비활성화
평가 없이 훈련하려면:

```python
# save_checkpoint 함수에서 평가 조건을 False로 설정
if False:  # by != "final" and train_steps % (args.ckpt_every * 2) == 0:
    # 평가 코드...
```

## 📊 TensorBoard 통합

Best checkpoint 정보도 TensorBoard에서 확인할 수 있습니다:

```bash
tensorboard --logdir results/waymo_base/001-DiT-g-2/tensorboard_logs
```

다음 메트릭들이 추가로 기록됩니다:
- `Evaluation/CLIP_Score`
- `Evaluation/LPIPS`
- `Evaluation/Inception_Score`
- `Evaluation/FID`
- `Evaluation/Composite_Score`
- `Evaluation/IsBest` (새로운 best인지 여부)

## 🎯 최적 활용법

1. **훈련 초기**: Composite score를 모니터링하여 전반적인 성능 향상 확인
2. **훈련 중기**: 특정 메트릭(예: CLIP score)에 집중하여 텍스트 일치도 개선
3. **훈련 후기**: Best 체크포인트들로 추가 평가 및 선택

## 📝 사용자 정의 메트릭

새로운 평가 메트릭을 추가하려면:

1. `hydit/evaluation_metrics.py`에 메트릭 함수 추가
2. `evaluate_sample_quality` 메서드에 통합
3. `best_checkpoint_tracker.py`에서 가중치 설정

예시:
```python
# evaluation_metrics.py에 새 메트릭 추가
def calculate_custom_metric(self, images):
    # 사용자 정의 메트릭 계산
    return score

# evaluate_sample_quality에 통합  
metrics['custom_metric'] = self.calculate_custom_metric(generated_images)
```

이제 훈련을 시작하면 자동으로 최고 성능의 모델이 `best.pt`로 저장됩니다! 🚀