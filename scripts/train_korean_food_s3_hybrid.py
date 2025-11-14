"""
S3 하이브리드 YOLO 학습
- S3에서 배치 단위로 이미지를 임시 다운로드하여 학습
- 학습 후 임시 파일 자동 삭제
- 서버 디스크 사용 최소화
"""
import sys
import logging
import tempfile
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent))

from utils.s3_loader import S3ImageLoader
from config.s3_config import get_s3_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_s3_dataset_to_temp(
    s3_loader: S3ImageLoader,
    train_prefix: str,
    val_prefix: str,
    temp_dir: Path,
    sample_limit: int = None
) -> tuple:
    """
    S3에서 데이터셋을 임시 디렉토리로 다운로드
    
    Args:
        s3_loader: S3ImageLoader 인스턴스
        train_prefix: 학습 데이터 prefix
        val_prefix: 검증 데이터 prefix
        temp_dir: 임시 디렉토리 경로
        sample_limit: 클래스당 최대 샘플 수 (None이면 전체)
        
    Returns:
        (train_count, val_count) 튜플
    """
    logger.info("="*70)
    logger.info("📥 S3에서 임시 디렉토리로 데이터 다운로드 중...")
    logger.info("="*70)
    
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    # 학습 데이터 다운로드
    logger.info(f"\n📦 학습 데이터 다운로드: {train_prefix}")
    train_structure = s3_loader.get_class_structure(train_prefix)
    
    for class_name, image_keys in train_structure.items():
        class_dir = train_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # 샘플 제한 적용
        if sample_limit:
            image_keys = image_keys[:sample_limit]
        
        logger.info(f"   {class_name}: {len(image_keys)}개 이미지 다운로드 중...")
        
        for img_key in image_keys:
            try:
                image = s3_loader.load_image_from_s3(img_key)
                if image:
                    # 파일명 추출
                    filename = Path(img_key).name
                    save_path = class_dir / filename
                    image.save(save_path)
                    train_count += 1
            except Exception as e:
                logger.warning(f"      ⚠️ {img_key} 다운로드 실패: {e}")
    
    # 검증 데이터 다운로드
    logger.info(f"\n📦 검증 데이터 다운로드: {val_prefix}")
    val_structure = s3_loader.get_class_structure(val_prefix)
    
    for class_name, image_keys in val_structure.items():
        class_dir = val_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # 샘플 제한 적용
        if sample_limit:
            image_keys = image_keys[:sample_limit]
        
        logger.info(f"   {class_name}: {len(image_keys)}개 이미지 다운로드 중...")
        
        for img_key in image_keys:
            try:
                image = s3_loader.load_image_from_s3(img_key)
                if image:
                    filename = Path(img_key).name
                    save_path = class_dir / filename
                    image.save(save_path)
                    val_count += 1
            except Exception as e:
                logger.warning(f"      ⚠️ {img_key} 다운로드 실패: {e}")
    
    logger.info("\n✅ 다운로드 완료!")
    logger.info(f"   학습: {train_count:,}장")
    logger.info(f"   검증: {val_count:,}장")
    logger.info(f"   임시 디렉토리: {temp_dir}")
    logger.info("="*70)
    
    return train_count, val_count


def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("🚀 S3 하이브리드 YOLO 한국 음식 분류 학습")
    print("   (임시 디렉토리 사용 → 학습 후 자동 삭제)")
    print("="*70 + "\n")
    
    # ==================== S3 설정 로드 ====================
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 .env 파일을 생성하고 설정하세요.")
        return
    
    # ==================== 학습 파라미터 ====================
    MODEL_SIZE = "11n"
    EPOCHS = 5
    BATCH_SIZE = 4
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 샘플 제한 (테스트용 - None이면 전체 다운로드)
    SAMPLE_LIMIT_PER_CLASS = None  # 예: 100으로 설정하면 클래스당 100개만
    
    # ==================== GPU 확인 ====================
    print("\n" + "="*70)
    print("🖥️  하드웨어 정보")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f} GB")
        
        if gpu_memory <= 3:
            BATCH_SIZE = 4
            print(f"⚙️  배치 크기: {BATCH_SIZE} (3GB GPU 최적화)")
    else:
        print("❌ CUDA 사용 불가! CPU로 학습합니다.")
        DEVICE = 'cpu'
    
    # ==================== 임시 디렉토리 생성 ====================
    temp_dir = Path(tempfile.mkdtemp(prefix='s3_yolo_'))
    logger.info(f"\n📁 임시 디렉토리 생성: {temp_dir}")
    
    try:
        # ==================== S3 로더 초기화 ====================
        s3_params = s3_config.get_s3_params()
        s3_loader = S3ImageLoader(
            bucket_name=s3_params['bucket_name'],
            aws_access_key=s3_params.get('aws_access_key'),
            aws_secret_key=s3_params.get('aws_secret_key'),
            region_name=s3_params['region_name']
        )
        
        # 버킷 접근 확인
        if not s3_loader.check_bucket_access():
            logger.error("S3 버킷 접근 실패!")
            return
        
        # ==================== S3 → 임시 디렉토리 다운로드 ====================
        train_count, val_count = download_s3_dataset_to_temp(
            s3_loader=s3_loader,
            train_prefix=s3_config.train_prefix,
            val_prefix=s3_config.val_prefix,
            temp_dir=temp_dir,
            sample_limit=SAMPLE_LIMIT_PER_CLASS
        )
        
        if train_count == 0:
            logger.error("학습 데이터가 없습니다!")
            return
        
        # ==================== 학습 설정 ====================
        print("\n" + "="*70)
        print("⚙️  학습 설정")
        print("="*70)
        print(f"모델: YOLO{MODEL_SIZE}-cls")
        print(f"에포크: {EPOCHS}")
        print(f"배치 크기: {BATCH_SIZE}")
        print(f"이미지 크기: {IMG_SIZE}")
        print(f"장치: {DEVICE.upper()}")
        print(f"학습 샘플: {train_count:,}개")
        print(f"검증 샘플: {val_count:,}개")
        
        # ==================== YOLO 학습 ====================
        print("\n" + "="*70)
        print("🚀 YOLO 학습 시작")
        print("="*70)
        print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
        
        # YOLO 모델 로드
        model = YOLO(f'yolo{MODEL_SIZE}-cls.pt')
        
        # 학습 실행
        results = model.train(
            data=str(temp_dir),  # 임시 디렉토리 경로
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            patience=50,
            save=True,
            plots=True,
            verbose=True,
            workers=4,
            project='runs/classify',
            name='s3_korean_food',
            exist_ok=True,
        )
        
        print("\n" + "="*70)
        print("✅ 학습 완료!")
        print("="*70)
        
        # 평가
        best_model = YOLO('runs/classify/s3_korean_food/weights/best.pt')
        metrics = best_model.val()
        
        print(f"\n📈 최종 성능:")
        print(f"   Top-1 정확도: {metrics.top1:.2%}")
        print(f"   Top-5 정확도: {metrics.top5:.2%}")
        
        print(f"\n💾 저장 위치:")
        print(f"   🎯 모델: runs/classify/s3_korean_food/weights/best.pt")
        print(f"   📊 그래프: runs/classify/s3_korean_food/results.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        print("중간 모델은 저장되었습니다: runs/classify/s3_korean_food/weights/last.pt")
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n\n❌ GPU 메모리 부족!")
            print(f"현재 배치: {BATCH_SIZE}")
            print(f"해결: BATCH_SIZE를 2로 줄이고 재실행")
        else:
            raise
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    
    finally:
        # ==================== 임시 디렉토리 삭제 ====================
        logger.info("\n" + "="*70)
        logger.info("🗑️  임시 디렉토리 정리 중...")
        logger.info("="*70)
        
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"✅ 임시 디렉토리 삭제 완료: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 임시 디렉토리 삭제 실패: {e}")
                logger.warning(f"   수동 삭제 필요: {temp_dir}")
        
        print("\n" + "="*70)
        print("✅ 모든 작업 완료!")
        print("="*70)


if __name__ == "__main__":
    main()

