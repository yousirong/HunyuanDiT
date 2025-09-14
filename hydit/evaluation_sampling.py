"""
평가용 샘플링 및 메트릭 계산 통합 함수

훈련 중에 모델의 성능을 평가하기 위해 샘플링과 메트릭 계산을 통합하여 수행합니다.
"""

import os
import random
import torch
import torch.distributed as dist
from PIL import Image
import numpy as np
from pathlib import Path
import csv
from typing import List, Dict, Any, Optional, Tuple

from .evaluation_metrics import ImageEvaluationMetrics


def load_evaluation_dataset(csv_path: str, limit: int = 50, random_sample: bool = True) -> List[Dict[str, str]]:
    """평가용 데이터셋을 로드합니다."""
    evaluation_data = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_data = list(reader)
            
        if random_sample and len(all_data) > limit:
            evaluation_data = random.sample(all_data, limit)
        else:
            evaluation_data = all_data[:limit]
            
        print(f"Loaded {len(evaluation_data)} samples for evaluation from {csv_path}")
        
    except Exception as e:
        print(f"Warning: Could not load evaluation dataset: {e}")
        # 기본 프롬프트들로 폴백
        default_prompts = [
            "A high quality image of a cat.",
            "A beautiful landscape painting.",
            "A modern building with a unique design.",
            "A bustling street scene in a futuristic city.",
            "A photorealistic portrait of a person.",
            "An abstract artwork with vibrant colors.",
            "A delicious-looking plate of food.",
            "A cute cartoon character.",
            "A serene lake surrounded by mountains.",
            "A vintage car parked on a city street."
        ]
        
        evaluation_data = [
            {'text_en': prompt, 'image_path': '', 'index': i} 
            for i, prompt in enumerate(default_prompts[:limit])
        ]
    
    return evaluation_data


def sample_images_for_evaluation(
    model, diffusion, vae, text_encoder, text_encoder_t5, tokenizer, tokenizer_t5,
    freqs_cis_img, evaluation_data: List[Dict[str, str]], args, device, rank: int = 0
) -> List[Image.Image]:
    """평가용 이미지들을 샘플링합니다."""
    
    model.eval()
    generated_images = []
    
    try:
        from hydit.constants import T5_ENCODER
        
        with torch.no_grad():
            for i, sample in enumerate(evaluation_data):
                if rank == 0:  # 로깅은 rank 0에서만
                    if i % 10 == 0:
                        print(f"  Generating evaluation sample {i+1}/{len(evaluation_data)}")
                
                prompt = sample['text_en']
                
                # 텍스트 임베딩 준비
                text_inputs = tokenizer(
                    [prompt], max_length=args.text_len, padding="max_length",
                    truncation=True, return_tensors="pt"
                ).to(device)
                
                encoder_hidden_states = text_encoder(
                    text_inputs.input_ids, attention_mask=text_inputs.attention_mask
                )[0]
                
                text_inputs_t5 = tokenizer_t5(
                    [prompt], max_length=args.text_len_t5, padding="max_length",
                    truncation=True, return_tensors="pt"
                ).to(device)
                
                output_t5 = text_encoder_t5(
                    input_ids=text_inputs_t5.input_ids,
                    attention_mask=text_inputs_t5.attention_mask if T5_ENCODER["attention_mask"] else None,
                    output_hidden_states=True,
                )
                encoder_hidden_states_t5 = output_t5["hidden_states"][T5_ENCODER["layer_index"]].detach()
                
                # 이미지 생성
                image_size = args.image_size[0] if isinstance(args.image_size, (list, tuple)) else args.image_size
                shape = (1, 4, image_size // 8, image_size // 8)
                noise = torch.randn(shape, device=device, dtype=torch.float16 if args.extra_fp16 else torch.float32)
                
                # Positional embeddings
                reso_key = f"{image_size}x{image_size}"
                cos_cis_img, sin_cis_img = freqs_cis_img[reso_key]
                
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_hidden_states_t5": encoder_hidden_states_t5,
                    "text_embedding_mask": text_inputs.attention_mask,
                    "text_embedding_mask_t5": text_inputs_t5.attention_mask,
                    "image_meta_size": None,
                    "style": None,
                    "cos_cis_img": cos_cis_img,
                    "sin_cis_img": sin_cis_img,
                }
                
                # Diffusion sampling
                samples = diffusion.p_sample_loop(
                    model=model,
                    shape=shape,
                    noise=noise,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    progress=False,
                )
                
                # VAE decoding
                vae_scaling_factor = vae.config.scaling_factor
                samples = samples / vae_scaling_factor
                
                if args.extra_fp16:
                    samples = samples.half()
                images = vae.decode(samples).sample
                
                # Post-processing
                images = (images / 2 + 0.5).clamp(0, 1)
                images = (images * 255).cpu().numpy().astype(np.uint8)
                pil_image = Image.fromarray(images[0].transpose(1, 2, 0))
                
                generated_images.append(pil_image)
                
    except Exception as e:
        print(f"Error during evaluation sampling: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        model.train()
    
    return generated_images


def load_reference_images(evaluation_data: List[Dict[str, str]], 
                         base_path: str = "/mnt/ssd/HunyuanDiT") -> List[Optional[Image.Image]]:
    """참조 이미지들을 로드합니다."""
    reference_images = []
    
    for sample in evaluation_data:
        image_path = sample.get('image_path', '')
        if image_path and image_path.strip():
            full_path = os.path.join(base_path, image_path)
            try:
                if os.path.exists(full_path):
                    ref_img = Image.open(full_path).convert('RGB')
                    reference_images.append(ref_img)
                else:
                    reference_images.append(None)
            except Exception as e:
                print(f"Warning: Could not load reference image {full_path}: {e}")
                reference_images.append(None)
        else:
            reference_images.append(None)
    
    return reference_images


def evaluate_model_performance(
    model, diffusion, vae, text_encoder, text_encoder_t5, tokenizer, tokenizer_t5,
    freqs_cis_img, args, device, rank: int = 0, 
    evaluation_csv: str = "/mnt/ssd/HunyuanDiT/dataset/waymo/csvfile/image_text.csv",
    num_evaluation_samples: int = 20,
    save_samples: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    모델 성능을 종합적으로 평가합니다.
    
    Returns:
        평가 메트릭들의 딕셔너리
    """
    if rank != 0:  # 평가는 rank 0에서만 수행
        return {}
    
    print(f"\n=== Starting Model Performance Evaluation ===")
    print(f"Evaluation samples: {num_evaluation_samples}")
    print(f"Evaluation CSV: {evaluation_csv}")
    
    try:
        # 평가용 데이터셋 로드
        evaluation_data = load_evaluation_dataset(
            evaluation_csv, 
            limit=num_evaluation_samples, 
            random_sample=True
        )
        
        if not evaluation_data:
            print("Warning: No evaluation data available")
            return {}
        
        # 이미지 생성
        print("Generating images for evaluation...")
        generated_images = sample_images_for_evaluation(
            model, diffusion, vae, text_encoder, text_encoder_t5, 
            tokenizer, tokenizer_t5, freqs_cis_img, 
            evaluation_data, args, device, rank
        )
        
        if not generated_images:
            print("Warning: No images generated for evaluation")
            return {}
        
        # 참조 이미지 로드
        reference_images = load_reference_images(evaluation_data)
        
        # 텍스트 프롬프트 추출
        text_prompts = [sample['text_en'] for sample in evaluation_data]
        
        # 평가 메트릭 계산
        print("Calculating evaluation metrics...")
        evaluator = ImageEvaluationMetrics(device=device)
        
        # 참조 이미지가 있는 샘플만 필터링
        valid_references = [ref for ref in reference_images if ref is not None]
        valid_generated = [gen for gen, ref in zip(generated_images, reference_images) if ref is not None]
        
        metrics = evaluator.evaluate_sample_quality(
            generated_images=generated_images,
            reference_images=valid_references if valid_references else None,
            text_prompts=text_prompts
        )
        
        # 결과 출력
        print("\n=== Evaluation Results ===")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        
        # 샘플 이미지 저장
        if save_samples and output_dir:
            save_evaluation_samples(
                generated_images, reference_images, text_prompts, 
                output_dir, evaluation_data
            )
        
        print("=== Evaluation Complete ===\n")
        
        return metrics
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_evaluation_samples(
    generated_images: List[Image.Image],
    reference_images: List[Optional[Image.Image]], 
    text_prompts: List[str],
    output_dir: str,
    evaluation_data: List[Dict[str, str]]
):
    """평가 샘플들을 저장합니다."""
    
    try:
        eval_dir = Path(output_dir) / "evaluation_samples"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (gen_img, ref_img, prompt) in enumerate(zip(generated_images, reference_images, text_prompts)):
            # 생성 이미지 저장
            gen_path = eval_dir / f"eval_generated_{i:03d}.jpg"
            gen_img.save(gen_path, quality=95)
            
            # 비교 이미지 생성 (참조 이미지가 있는 경우)
            if ref_img is not None:
                # 크기 맞추기
                ref_resized = ref_img.resize(gen_img.size, Image.Resampling.LANCZOS)
                
                # 나란히 배치
                comparison = Image.new('RGB', (gen_img.width * 2, gen_img.height))
                comparison.paste(ref_resized, (0, 0))
                comparison.paste(gen_img, (gen_img.width, 0))
                
                comp_path = eval_dir / f"eval_comparison_{i:03d}.jpg"
                comparison.save(comp_path, quality=95)
            
            # 텍스트 파일 저장
            text_path = eval_dir / f"eval_prompt_{i:03d}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        print(f"Evaluation samples saved to {eval_dir}")
        
    except Exception as e:
        print(f"Warning: Could not save evaluation samples: {e}")


def run_distributed_evaluation(
    model, diffusion, vae, text_encoder, text_encoder_t5, tokenizer, tokenizer_t5,
    freqs_cis_img, args, device, world_size: int, rank: int,
    **evaluation_kwargs
) -> Dict[str, float]:
    """분산 훈련 환경에서 평가를 실행합니다."""
    
    # rank 0에서만 평가 수행
    if rank == 0:
        metrics = evaluate_model_performance(
            model, diffusion, vae, text_encoder, text_encoder_t5, 
            tokenizer, tokenizer_t5, freqs_cis_img, args, device, 
            rank, **evaluation_kwargs
        )
    else:
        metrics = {}
    
    # 모든 프로세스에서 동기화
    if world_size > 1:
        # 메트릭을 모든 프로세스에 브로드캐스트
        if rank == 0:
            metrics_tensor = torch.tensor(list(metrics.values()), device=device)
            keys = list(metrics.keys())
        else:
            # 다른 rank에서는 빈 텐서 준비
            metrics_tensor = torch.zeros(len(ImageEvaluationMetrics(device).evaluate_sample_quality([], [], [])), device=device)
            keys = []
        
        # 브로드캐스트
        dist.broadcast(metrics_tensor, src=0)
        
        # rank 0이 아닌 프로세스에서도 메트릭 재구성
        if rank != 0:
            # 표준 메트릭 키들
            default_keys = ['clip_score', 'lpips', 'inception_score', 'fid', 'composite_score']
            metrics = {key: val.item() for key, val in zip(default_keys[:len(metrics_tensor)], metrics_tensor)}
    
    return metrics