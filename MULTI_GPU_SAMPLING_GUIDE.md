# 🖼️ 멀티 GPU 샘플링 기능 가이드

HunyuanDiT 훈련 중 체크포인트 저장 시 모든 GPU에서 자동으로 샘플 이미지를 생성하는 기능을 추가했습니다.

## ✨ 주요 특징

- **멀티 GPU 샘플링**: 체크포인트 저장 시 모든 GPU(최대 8개)에서 동시에 1개씩 샘플 생성
- **자동 실행**: 체크포인트 저장 시 자동으로 실행
- **다양한 프롬프트**: 각 GPU마다 서로 다른 자율주행 시나리오 프롬프트 사용
- **빠른 샘플링**: 10 스텝의 단순화된 디노이징으로 빠른 생성

## 🔧 주요 변경사항

### 1. 새로운 샘플링 함수 추가
```python
@torch.no_grad()
def sample_images_multi_gpu(
    model, diffusion, vae, text_encoder, tokenizer, text_encoder_t5, tokenizer_t5, 
    rank, world_size, args, logger, train_steps
):
```

### 2. 체크포인트 저장 시 자동 샘플링
- `save_checkpoint()` 함수에 샘플링 기능 통합
- 모든 체크포인트 저장 시점에서 샘플 생성:
  - `--ckpt-every` 스텝마다
  - `--ckpt-latest-every` 스텝마다  
  - 에포크 종료 시
  - 최종 훈련 완료 시

### 3. GPU별 프롬프트
각 GPU는 서로 다른 자율주행 시나리오를 사용합니다:
- GPU 0: "A residential street scene with wet roads and parked cars during daytime"
- GPU 1: "An urban intersection with traffic lights and multiple vehicles"
- GPU 2: "A highway scene with clear lane markings and distant cars"
- GPU 3: "A rainy street with reflections on the pavement and buildings"
- GPU 4: "A suburban road with trees and houses on both sides"
- GPU 5: "A parking lot scene with various parked vehicles"
- GPU 6: "A city street with pedestrian crossings and traffic signs"
- GPU 7: "A rural road with natural scenery and minimal traffic"

## 📁 샘플 저장 구조

```
results/
├── waymo_base/
│   ├── samples/
│   │   ├── step_0000010/
│   │   │   ├── gpu_0_sample.png
│   │   │   ├── gpu_0_prompt.txt
│   │   │   ├── gpu_1_sample.png
│   │   │   ├── gpu_1_prompt.txt
│   │   │   ├── ...
│   │   │   ├── gpu_7_sample.png
│   │   │   └── gpu_7_prompt.txt
│   │   ├── step_0000020/
│   │   └── ...
│   └── checkpoints/
```

## 🚀 사용 방법

### 1. 일반 훈련 (샘플링 자동 포함)
```bash
# 8 GPU 훈련 (각 체크포인트마다 8개 샘플 생성)
bash scripts/train_waymo_base.sh 5000

# 4 GPU 훈련 (각 체크포인트마다 4개 샘플 생성)
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_waymo_base.sh 1000
```

### 2. 샘플링 기능 테스트
```bash
# 4 GPU로 빠른 샘플링 테스트
bash scripts/test_sampling.sh
```

### 3. 체크포인트 빈도 조정
샘플링 빈도를 조절하려면 체크포인트 저장 주기를 수정하세요:

```bash
python hydit/train_deepspeed.py \
    --ckpt-every 500 \      # 500 스텝마다 샘플링
    --ckpt-latest-every 100 # 100 스텝마다 latest 샘플링
```

## ⚡ 성능 고려사항

### 샘플링 최적화
- **빠른 스텝 수**: 10 스텝으로 단순화된 디노이징
- **메모리 효율적**: `@torch.no_grad()` 사용으로 gradient 계산 제외
- **병렬 처리**: 모든 GPU에서 동시에 생성

### 저장 공간
- 각 샘플: ~100KB (256x256 PNG)
- 8 GPU × 100 체크포인트 = ~80MB 추가 저장 공간

## 🛠️ 커스터마이징

### 1. 프롬프트 변경
`sample_images_multi_gpu` 함수의 `prompts` 리스트를 수정:

```python
prompts = [
    "Your custom prompt 1",
    "Your custom prompt 2", 
    # ... 최대 8개
]
```

### 2. 샘플링 품질 조정
샘플링 스텝 수나 알파 값 조정:

```python
num_steps = 20  # 더 높은 품질을 위해 스텝 수 증가
alpha_t = 0.95 - i * 0.05  # 디노이징 강도 조절
```

### 3. 해상도 변경
이미지 크기는 훈련 설정을 따라갑니다:
```bash
--image-size 512 512  # 512x512 샘플 생성
```

## 🔍 문제 해결

### 샘플링 실패
```
Error during sampling on GPU X: ...
```
- 메모리 부족: 배치 사이즈를 1로 유지
- 모델 호환성: 모델이 올바르게 로드되었는지 확인

### 저장 실패
```
OSError: cannot write to directory...
```
- 디스크 공간 확인
- 권한 확인: `chmod 755 results/`

### 동기화 문제
```
NCCL timeout...
```
- `dist.barrier()` 호출 순서 확인
- 모든 GPU가 동일한 진행 상태인지 확인

## 📊 모니터링

### 로그 확인
```bash
# 샘플링 상태 확인
tail -f results/waymo_base/logs/train.log | grep "Generating samples"

# 저장된 샘플 확인
ls -la results/waymo_base/samples/step_*/
```

### TensorBoard 통합 (선택사항)
향후 업데이트에서 TensorBoard 이미지 로깅 추가 예정.

---

이 기능으로 훈련 중 실시간으로 모델의 성능과 생성 품질을 모니터링할 수 있습니다! 🎯