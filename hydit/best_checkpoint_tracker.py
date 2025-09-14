"""
Best Checkpoint Tracker

훈련 중에 최고 성능의 체크포인트를 추적하고 관리하는 시스템
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch

class BestCheckpointTracker:
    """최고 성능 체크포인트를 추적하는 클래스"""
    
    def __init__(self, checkpoint_dir: str, metric_name: str = 'composite_score', 
                 higher_is_better: bool = True, save_top_k: int = 3):
        """
        Args:
            checkpoint_dir: 체크포인트가 저장되는 디렉토리
            metric_name: 추적할 주요 메트릭 이름
            higher_is_better: True면 높은 값이 좋음, False면 낮은 값이 좋음
            save_top_k: 상위 K개의 체크포인트를 유지
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.save_top_k = save_top_k
        
        # 상태 파일 경로
        self.tracker_file = self.checkpoint_dir / 'best_checkpoints.json'
        
        # 최고 성능 기록 로드
        self.best_checkpoints = self.load_tracker_state()
        
        print(f"Best checkpoint tracker initialized:")
        print(f"  Metric: {metric_name} ({'higher is better' if higher_is_better else 'lower is better'})")
        print(f"  Save top {save_top_k} checkpoints")
        print(f"  Current best: {self.get_best_score()}")
    
    def load_tracker_state(self) -> Dict[str, Any]:
        """추적 상태를 로드합니다."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                    return data.get('checkpoints', [])
            except Exception as e:
                print(f"Warning: Could not load tracker state: {e}")
        
        return []
    
    def save_tracker_state(self):
        """추적 상태를 저장합니다."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            state = {
                'metric_name': self.metric_name,
                'higher_is_better': self.higher_is_better,
                'save_top_k': self.save_top_k,
                'checkpoints': self.best_checkpoints
            }
            
            with open(self.tracker_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"Error saving tracker state: {e}")
    
    def get_best_score(self) -> Optional[float]:
        """현재 최고 점수를 반환합니다."""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0]['metrics'][self.metric_name]
    
    def is_better(self, new_score: float, current_best: float) -> bool:
        """새 점수가 현재 최고 점수보다 좋은지 판단합니다."""
        if self.higher_is_better:
            return new_score > current_best
        else:
            return new_score < current_best
    
    def should_save_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """새 체크포인트를 저장할지 결정합니다."""
        if self.metric_name not in metrics:
            print(f"Warning: Metric '{self.metric_name}' not found in metrics")
            return False
        
        new_score = metrics[self.metric_name]
        
        # 아직 체크포인트가 없거나, top_k 미만인 경우
        if len(self.best_checkpoints) < self.save_top_k:
            return True
        
        # 최악의 점수와 비교
        worst_score = self.best_checkpoints[-1]['metrics'][self.metric_name]
        return self.is_better(new_score, worst_score)
    
    def update_checkpoint(self, step: int, epoch: int, checkpoint_path: str, 
                         metrics: Dict[str, float]) -> bool:
        """새 체크포인트 정보로 추적 상태를 업데이트합니다."""
        
        if not self.should_save_checkpoint(metrics):
            return False
        
        new_checkpoint = {
            'step': step,
            'epoch': epoch,
            'checkpoint_path': str(checkpoint_path),
            'metrics': metrics,
            'timestamp': Path(checkpoint_path).stat().st_mtime if os.path.exists(checkpoint_path) else 0
        }
        
        # 기존 리스트에 추가
        self.best_checkpoints.append(new_checkpoint)
        
        # 주요 메트릭으로 정렬
        self.best_checkpoints.sort(
            key=lambda x: x['metrics'][self.metric_name],
            reverse=self.higher_is_better
        )
        
        # top_k만 유지
        if len(self.best_checkpoints) > self.save_top_k:
            # 제거될 체크포인트들
            removed_checkpoints = self.best_checkpoints[self.save_top_k:]
            self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
            
            # 실제 파일 삭제 (선택적)
            self.cleanup_old_checkpoints(removed_checkpoints)
        
        # 상태 저장
        self.save_tracker_state()
        
        # best.pt 링크 업데이트
        self.update_best_link()
        
        return True
    
    def cleanup_old_checkpoints(self, removed_checkpoints: list, 
                               keep_every_n: int = 10):
        """제거된 체크포인트 파일들을 정리합니다."""
        for checkpoint in removed_checkpoints:
            checkpoint_path = Path(checkpoint['checkpoint_path'])
            
            # 매 N번째 체크포인트는 보존
            if checkpoint['step'] % (keep_every_n * 1000) == 0:  # 매 10k 스텝마다 보존
                continue
                
            try:
                if checkpoint_path.exists() and checkpoint_path.is_dir():
                    # 디렉토리 삭제 (DeepSpeed 체크포인트)
                    shutil.rmtree(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint_path}")
                elif checkpoint_path.exists():
                    # 단일 파일 삭제
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {checkpoint_path}: {e}")
    
    def update_best_link(self):
        """best.pt 링크를 최고 성능 체크포인트로 업데이트합니다."""
        if not self.best_checkpoints:
            return
        
        best_checkpoint_path = Path(self.best_checkpoints[0]['checkpoint_path'])
        best_link_path = self.checkpoint_dir / 'best.pt'
        
        try:
            # 기존 링크 제거
            if best_link_path.exists() or best_link_path.is_symlink():
                best_link_path.unlink()
            
            # 새로운 심볼릭 링크 생성 (상대 경로 사용)
            if best_checkpoint_path.exists():
                relative_path = os.path.relpath(best_checkpoint_path, self.checkpoint_dir)
                best_link_path.symlink_to(relative_path)
                print(f"Updated best.pt -> {best_checkpoint_path.name}")
            else:
                # 심볼릭 링크가 지원되지 않는 경우 복사
                if best_checkpoint_path.is_dir():
                    shutil.copytree(best_checkpoint_path, best_link_path)
                else:
                    shutil.copy2(best_checkpoint_path, best_link_path)
                print(f"Copied best checkpoint to best.pt")
                
        except Exception as e:
            print(f"Warning: Could not update best.pt link: {e}")
            # 링크 생성 실패 시 복사로 폴백
            try:
                if best_link_path.exists():
                    if best_link_path.is_dir():
                        shutil.rmtree(best_link_path)
                    else:
                        best_link_path.unlink()
                        
                if best_checkpoint_path.is_dir():
                    shutil.copytree(best_checkpoint_path, best_link_path)
                else:
                    shutil.copy2(best_checkpoint_path, best_link_path)
                print(f"Copied best checkpoint to best.pt (fallback)")
            except Exception as e2:
                print(f"Error: Could not create best.pt: {e2}")
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """최고 성능 체크포인트 정보를 반환합니다."""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0]
    
    def print_summary(self):
        """현재 상태 요약을 출력합니다."""
        print(f"\n=== Best Checkpoint Summary ===")
        print(f"Tracking metric: {self.metric_name}")
        print(f"Saved checkpoints: {len(self.best_checkpoints)}/{self.save_top_k}")
        
        if self.best_checkpoints:
            for i, checkpoint in enumerate(self.best_checkpoints):
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in checkpoint['metrics'].items()])
                print(f"  #{i+1}: Step {checkpoint['step']}, Epoch {checkpoint['epoch']}")
                print(f"      Path: {Path(checkpoint['checkpoint_path']).name}")
                print(f"      Metrics: {metrics_str}")
        else:
            print("  No checkpoints saved yet")
        print("=" * 32)
    
    def export_results(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """결과를 JSON 파일로 내보냅니다."""
        if output_file is None:
            output_file = self.checkpoint_dir / 'best_checkpoint_results.json'
        
        results = {
            'tracker_config': {
                'metric_name': self.metric_name,
                'higher_is_better': self.higher_is_better,
                'save_top_k': self.save_top_k
            },
            'best_checkpoints': self.best_checkpoints,
            'summary': {
                'total_tracked': len(self.best_checkpoints),
                'best_score': self.get_best_score(),
                'best_checkpoint': self.get_best_checkpoint_info()
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results exported to {output_file}")
        except Exception as e:
            print(f"Error exporting results: {e}")
        
        return results