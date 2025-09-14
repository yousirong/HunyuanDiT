"""
이미지 생성 모델을 위한 평가 메트릭 함수들

주요 메트릭:
- FID (Frechet Inception Distance): 생성 이미지의 전반적인 품질
- LPIPS (Learned Perceptual Image Patch Similarity): 지각적 유사성  
- CLIP Score: 텍스트-이미지 일치도
- IS (Inception Score): 이미지의 다양성과 품질
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy import linalg
import clip
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import lpips

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. FID calculation will be disabled.")

class ImageEvaluationMetrics:
    """이미지 생성 모델 평가를 위한 메트릭 클래스"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """평가에 필요한 모델들을 초기화합니다."""
        
        # CLIP 모델 로드
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            self.clip_model = None
            
        # LPIPS 모델 로드
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print("LPIPS model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load LPIPS model: {e}")
            self.lpips_model = None
            
        # Inception 모델 (IS 계산용)
        try:
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()
            print("Inception model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Inception model: {e}")
            self.inception_model = None
            
        # FID용 Inception 모델
        if FID_AVAILABLE:
            try:
                self.fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device)
                self.fid_model.eval()
                print("FID Inception model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load FID Inception model: {e}")
                self.fid_model = None
        else:
            self.fid_model = None
            
        # 이미지 전처리 변환
        self.image_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_images(self, images):
        """이미지들을 평가용으로 전처리합니다."""
        if isinstance(images, torch.Tensor):
            # 이미 텐서인 경우
            if images.dim() == 4 and images.shape[1] == 3:
                # (B, C, H, W) 형태
                return images
        
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = self.image_transform(img)
                processed.append(img_tensor)
            elif isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img_tensor = self.image_transform(transforms.ToPILImage()(img))
                    processed.append(img_tensor)
                else:
                    processed.append(img)
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
                img_tensor = self.image_transform(pil_img)
                processed.append(img_tensor)
        
        return torch.stack(processed).to(self.device)
    
    def calculate_clip_score(self, generated_images, text_prompts):
        """CLIP Score를 계산합니다 (텍스트-이미지 일치도)."""
        if self.clip_model is None:
            return 0.0
            
        try:
            # 텍스트 인코딩
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 이미지 전처리 및 인코딩
            clip_scores = []
            for img in generated_images:
                if isinstance(img, Image.Image):
                    img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                elif isinstance(img, torch.Tensor):
                    # PIL Image로 변환 후 전처리
                    img_pil = transforms.ToPILImage()(img.cpu())
                    img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
                else:
                    continue
                    
                img_features = self.clip_model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                
                # 코사인 유사도 계산
                similarity = torch.cosine_similarity(text_features, img_features, dim=-1)
                clip_scores.append(similarity.item())
            
            return np.mean(clip_scores) if clip_scores else 0.0
            
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            return 0.0
    
    def calculate_lpips(self, generated_images, reference_images):
        """LPIPS를 계산합니다 (지각적 유사성)."""
        if self.lpips_model is None:
            return float('inf')
            
        try:
            lpips_scores = []
            
            for gen_img, ref_img in zip(generated_images, reference_images):
                # 이미지를 [-1, 1] 범위로 정규화
                if isinstance(gen_img, Image.Image):
                    gen_tensor = transforms.ToTensor()(gen_img) * 2 - 1
                elif isinstance(gen_img, torch.Tensor):
                    gen_tensor = gen_img * 2 - 1
                else:
                    continue
                    
                if isinstance(ref_img, Image.Image):
                    ref_tensor = transforms.ToTensor()(ref_img) * 2 - 1  
                elif isinstance(ref_img, torch.Tensor):
                    ref_tensor = ref_img * 2 - 1
                else:
                    continue
                
                gen_tensor = gen_tensor.unsqueeze(0).to(self.device)
                ref_tensor = ref_tensor.unsqueeze(0).to(self.device)
                
                # LPIPS 계산
                lpips_val = self.lpips_model(gen_tensor, ref_tensor)
                lpips_scores.append(lpips_val.item())
            
            return np.mean(lpips_scores) if lpips_scores else float('inf')
            
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            return float('inf')
    
    def calculate_inception_score(self, generated_images, splits=10):
        """Inception Score를 계산합니다."""
        if self.inception_model is None:
            return 0.0
            
        try:
            # 이미지 전처리
            processed_imgs = self.preprocess_images(generated_images)
            
            # Inception 모델로 예측
            with torch.no_grad():
                preds = self.inception_model(processed_imgs)
                preds = F.softmax(preds, dim=1)
            
            # IS 계산
            scores = []
            N = len(preds)
            for i in range(splits):
                part = preds[i * N // splits:(i + 1) * N // splits]
                p_y = part.mean(dim=0)
                kl_div = part * (torch.log(part) - torch.log(p_y))
                kl_div = kl_div.sum(dim=1).mean()
                scores.append(torch.exp(kl_div).item())
                
            return np.mean(scores)
            
        except Exception as e:
            print(f"Error calculating Inception Score: {e}")
            return 0.0
    
    def calculate_fid_features(self, images):
        """FID 계산을 위한 특징 추출"""
        if self.fid_model is None:
            return None
            
        try:
            processed_imgs = self.preprocess_images(images)
            
            with torch.no_grad():
                features = self.fid_model(processed_imgs)[0]
                
            # Flatten features
            if features.size(2) != 1 or features.size(3) != 1:
                features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze(3).squeeze(2)
            
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting FID features: {e}")
            return None
    
    def calculate_fid(self, generated_images, reference_images):
        """FID를 계산합니다."""
        try:
            gen_features = self.calculate_fid_features(generated_images)
            ref_features = self.calculate_fid_features(reference_images)
            
            if gen_features is None or ref_features is None:
                return float('inf')
            
            # 평균과 공분산 계산
            mu1, sigma1 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
            mu2, sigma2 = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
            
            # FID 계산
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * 1e-6
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
                
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            return fid
            
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return float('inf')
    
    def evaluate_sample_quality(self, generated_images, reference_images=None, text_prompts=None):
        """종합적인 샘플 품질 평가"""
        metrics = {}
        
        # CLIP Score (텍스트가 있는 경우)
        if text_prompts and len(text_prompts) > 0:
            clip_score = self.calculate_clip_score(generated_images, text_prompts)
            metrics['clip_score'] = clip_score
            
        # LPIPS (참조 이미지가 있는 경우)
        if reference_images:
            lpips_score = self.calculate_lpips(generated_images, reference_images)
            metrics['lpips'] = lpips_score
            
        # Inception Score
        is_score = self.calculate_inception_score(generated_images)
        metrics['inception_score'] = is_score
        
        # FID (참조 이미지가 있는 경우)
        if reference_images:
            fid_score = self.calculate_fid(generated_images, reference_images)
            metrics['fid'] = fid_score
            
        # 종합 점수 계산 (높을수록 좋음)
        composite_score = 0.0
        weights = {'clip_score': 0.4, 'inception_score': 0.3, 'lpips': -0.2, 'fid': -0.1}
        
        for metric, value in metrics.items():
            if metric in weights and not np.isnan(value) and np.isfinite(value):
                if metric in ['lpips', 'fid']:  # 낮을수록 좋은 메트릭
                    composite_score += weights[metric] * (1.0 / (1.0 + value))
                else:  # 높을수록 좋은 메트릭
                    composite_score += weights[metric] * value
                    
        metrics['composite_score'] = composite_score
        
        return metrics

def create_evaluation_metrics(device='cuda'):
    """평가 메트릭 인스턴스를 생성합니다."""
    return ImageEvaluationMetrics(device=device)