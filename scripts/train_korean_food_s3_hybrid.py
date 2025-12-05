"""
S3 하이브리드 YOLO 학습 (클래스별 전체 학습)
- 각 클래스의 Train/Val 데이터를 모두 다운로드하여 학습
- 한 클래스 학습 완료 후 모델 저장 (버전 관리)
- 다음 클래스 학습 시 이전 모델 로드하여 연속 학습
"""
import sys
import os
import logging
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.nn import Sequential
from ultralytics import YOLO
from ultralytics.nn import ClassificationModel

sys.path.append(str(Path(__file__).parent))

from utils.s3_loader import S3ImageLoader
from config.s3_config import get_s3_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# 신뢰할수있는 클래스 등록
torch.serialization.add_safe_globals([ClassificationModel, Sequential])


def get_version_from_class_count(class_count: int) -> str:
    """클래스 개수에 따라 버전 문자열 생성 (예: 1.0.1, 1.1.0)"""
    if class_count <= 0:
        return "1.0.0"
    patch = class_count % 10
    minor = (class_count // 10) % 10
    major = class_count // 100
    return f"{major + 1}.{minor}.{patch}"


def save_progress(progress_file: Path, progress_data: Dict):
    """학습 진행 상황 저장"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)


def load_progress(progress_file: Path) -> Dict:
    """학습 진행 상황 로드"""
    if not progress_file.exists():
        return {'last_class_idx': 0}
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'last_class_idx': 0}


def download_class_data(
    s3_loader: S3ImageLoader,
    class_name: str,
    train_keys: List[str],
    val_keys: List[str],
    temp_dir: Path
) -> bool:
    """
    특정 클래스의 Train/Val 데이터를 모두 다운로드
    """
    try:
        # Train 다운로드
        train_dir = temp_dir / 'train' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"   ⬇️  Train 데이터 다운로드 중... ({len(train_keys)}장)")
        for key in train_keys:
            try:
                image = s3_loader.load_image_from_s3(key)
                if image:
                    save_path = train_dir / Path(key).name
                    image.save(save_path)
            except Exception as e:
                logger.warning(f"      ⚠️ 이미지 다운로드 실패: {key} ({e})")

        # Val 다운로드
        val_dir = temp_dir / 'val' / class_name
        val_dir.mkdir(parents=True, exist_ok=True)
        
        if val_keys:
            logger.info(f"   ⬇️  Val 데이터 다운로드 중... ({len(val_keys)}장)")
            for key in val_keys:
                try:
                    image = s3_loader.load_image_from_s3(key)
                    if image:
                        save_path = val_dir / Path(key).name
                        image.save(save_path)
                except Exception as e:
                    logger.warning(f"      ⚠️ 이미지 다운로드 실패: {key} ({e})")
        else:
            # Val 데이터가 없는 경우 더미 데이터 생성 (에러 방지)
            logger.info(f"   ⚠️ Val 데이터가 없습니다. 더미 데이터를 생성합니다.")
            dummy_dir = temp_dir / 'val' / 'dummy_class'
            dummy_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.new('RGB', (64, 64), color='black').save(dummy_dir / 'dummy.jpg')

        return True
    except Exception as e:
        logger.error(f"   ❌ 데이터 다운로드 중 치명적 오류: {e}")
        return False


def train_single_class(
    model: YOLO,
    class_name: str,
    temp_dir: Path,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    project_name: str,
    run_name: str
) -> Tuple[YOLO, Optional[Path]]:
    """
    단일 클래스에 대해 학습 수행
    """
    logger.info(f"\n🚀 클래스 '{class_name}' 학습 시작...")
    
    try:
        # 학습 실행
        results = model.train(
            data=str(temp_dir),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project_name,
            name=run_name,
            exist_ok=True,
            patience=50,
            save=True,
            plots=True,
            verbose=True
        )
        
        # 결과 모델 경로 찾기
        script_dir = Path(__file__).parent.parent
        weights_dir = script_dir / project_name / run_name / 'weights'
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        # best.pt가 없으면 last.pt 사용
        target_pt = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)
        
        if target_pt:
            logger.info(f"   ✅ 학습 완료. 모델 경로: {target_pt}")
            return YOLO(str(target_pt)), target_pt
        else:
            logger.error("   ❌ 학습 결과 모델 파일을 찾을 수 없습니다.")
            return model, None
            
    except Exception as e:
        logger.error(f"   ❌ 학습 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return model, None


def main():
    print("\n" + "="*70)
    print("🚀 S3 하이브리드 YOLO 클래스별 순차 학습")
    print("   (클래스 단위 전체 다운로드 → 학습 → 모델 업데이트)")
    print("="*70 + "\n")

    # 설정
    MODEL_SIZE = "11n"
    MODEL_TYPE = "-cls"
    EPOCHS = 20  # 클래스당 에포크 수
    BATCH_SIZE = 50
    IMG_SIZE = 1280
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        s3_config = get_s3_config()
        s3_params = s3_config.get_s3_params()
        # cache_size 인자 제거 (S3ImageLoader에서 지원하지 않음)
        if 'cache_size' in s3_params:
            del s3_params['cache_size']
            
        s3_loader = S3ImageLoader(**s3_params)
        
        if not s3_loader.check_bucket_access():
            logger.error("S3 버킷 접근 실패!")
            return

        # S3 클래스 구조 분석
        logger.info("📊 S3 클래스 목록 가져오는 중...")
        train_structure = s3_loader.get_class_structure(s3_config.train_prefix)
        val_structure = s3_loader.get_class_structure(s3_config.val_prefix)
        
        class_names = sorted(train_structure.keys())
        logger.info(f"📚 총 {len(class_names)}개 클래스 발견")
        
        # 진행 상황 로드
        script_dir = Path(__file__).parent.parent
        progress_file = script_dir / 'progress_class_based.json'
        progress_data = load_progress(progress_file)
        start_idx = progress_data.get('last_class_idx', 0)
        
        # 초기 모델 로드
        project_name = 'runs/classify'
        run_name = 's3_korean_food_sequential'
        weights_dir = script_dir / project_name / run_name / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # 이전에 학습된 최신 모델이 있으면 로드, 없으면 기본 모델
        # PyTorch 호환성 설정은 필요시에만 추가 (모델 로드 에러 발생 시)

        # 이전에 학습된 최신 모델이 있으면 로드, 없으면 기본 모델
        last_model_path = None

        # 1. 버전 관리 파일 찾기 (start_idx > 0일 때만)
        if start_idx > 0:
            last_saved_idx = (start_idx // 100) * 100
            if last_saved_idx > 0:
                version = get_version_from_class_count(last_saved_idx)
                versioned_path = weights_dir / f'best_v{version}.pt'
                if versioned_path.exists():
                    last_model_path = versioned_path

        # 2. 버전 관리 파일이 없으면 best.pt 사용 (항상 확인)
        if not (last_model_path and last_model_path.exists()):
            best_pt_path = weights_dir / 'best.pt'
            if best_pt_path.exists():
                last_model_path = best_pt_path

        # 3. 모델 로드
        if last_model_path and last_model_path.exists():
            logger.info(f"📌 이전 학습 모델 로드: {last_model_path}")
            model = YOLO(str(last_model_path))
        else:
            logger.info(f"🎯 초기 모델 로드: yolo{MODEL_SIZE}{MODEL_TYPE}.pt")
            model = YOLO(f'yolo{MODEL_SIZE}{MODEL_TYPE}.pt')

        # 임시 디렉토리
        temp_dir = script_dir / 'temp' / 'class_learning'
        
        # 클래스별 순차 학습
        for i, class_name in enumerate(class_names[start_idx:], start=start_idx + 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"📦 [{i}/{len(class_names)}] 클래스 처리 중: {class_name}")
            logger.info(f"{'='*70}")
            
            # 임시 디렉토리 초기화
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 데이터 다운로드
            train_keys = train_structure.get(class_name, [])
            val_keys = val_structure.get(class_name, [])
            
            if not train_keys:
                logger.warning(f"   ⚠️ Train 데이터가 없습니다. 건너뜁니다.")
                continue
                
            success = download_class_data(s3_loader, class_name, train_keys, val_keys, temp_dir)
            if not success:
                logger.error("   ❌ 데이터 다운로드 실패. 중단합니다.")
                break
            
            # 학습 수행
            model, new_model_path = train_single_class(
                model=model,
                class_name=class_name,
                temp_dir=temp_dir,
                epochs=EPOCHS,
                img_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                project_name=project_name,
                run_name=run_name
            )
            
            if new_model_path:
                # 100개 클래스마다 모델 저장
                SAVE_INTERVAL = 100  # 저장 간격 설정
                
                if i % SAVE_INTERVAL == 0 or i == len(class_names):
                    # 버전 관리된 이름으로 복사
                    version = get_version_from_class_count(i)
                    versioned_name = f'best_v{version}.pt'
                    dest_path = weights_dir / versioned_name
                    shutil.copy2(new_model_path, dest_path)
                    
                    logger.info(f"   💾 모델 저장 완료 ({i}개 클래스 완료): {dest_path}")
                    logger.info(f"   📊 다음 저장 시점: {((i // SAVE_INTERVAL) + 1) * SAVE_INTERVAL}개 클래스")
                else:
                    next_save = ((i // SAVE_INTERVAL) + 1) * SAVE_INTERVAL
                    logger.info(f"   ✅ 클래스 학습 완료 ({i}/{len(class_names)}) - 다음 저장: {next_save}개 클래스")
                
                # 진행 상황은 항상 저장 (중단 시 복구용)
                progress_data['last_class_idx'] = i
                save_progress(progress_file, progress_data)
            else:
                logger.error("   ❌ 모델 생성 실패. 학습을 중단합니다.")
                break
                
            # 디스크 공간 확보를 위해 임시 파일 즉시 삭제
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
        logger.info("\n✅ 모든 학습이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
