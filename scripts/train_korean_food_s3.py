"""
S3 기반 YOLO 한국 음식 분류 학습
서버에 데이터를 저장하지 않고 S3에서 실시간으로 스트리밍하여 학습
"""
import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from utils.s3_dataset import S3YOLOClassificationDataset
from config.s3_config import get_s3_config


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_yolo_compatible_dataset_yaml(
    dataset_info: dict,
    output_path: str = 'runs/s3_dataset.yaml'
) -> Path:
    """
    YOLO가 읽을 수 있는 dataset.yaml 파일 생성
    
    Args:
        dataset_info: S3 데이터셋 정보 딕셔너리
        output_path: 저장 경로
        
    Returns:
        생성된 yaml 파일 경로
    """
    import yaml
    
    yaml_path = Path(output_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # YOLO Classification 형식
    dataset_config = {
        'path': str(yaml_path.parent.absolute()),  # 데이터셋 루트 (사용 안 함)
        'train': 's3_streaming',  # 더미 값
        'val': 's3_streaming',    # 더미 값
        'nc': dataset_info['num_classes'],  # 클래스 수
        'names': dataset_info['classes']  # 클래스 이름 리스트
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
    
    logger.info(f"✅ 데이터셋 설정 파일 생성: {yaml_path}")
    return yaml_path


def train_with_s3_streaming():
    """S3 스트리밍 기반 YOLO 학습 메인 함수"""
    
    print("\n" + "="*70)
    print("🚀 S3 기반 YOLO 한국 음식 분류 학습")
    print("   (서버에 데이터를 저장하지 않고 실시간 스트리밍)")
    print("="*70 + "\n")
    
    # ==================== S3 설정 로드 ====================
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 해결 방법:")
        logger.info("   1. .env.example 파일을 .env로 복사")
        logger.info("   2. .env 파일에서 실제 AWS 자격증명과 버킷 이름 입력")
        return
    
    # ==================== 학습 파라미터 ====================
    MODEL_SIZE = "11n"
    EPOCHS = 5
    BATCH_SIZE = 4  # GTX 1060 3GB에 맞춤
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================== GPU 확인 ====================
    print("\n" + "="*70)
    print("🖥️  하드웨어 정보")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f} GB")
        
        # GTX 1060 3GB 최적화
        if gpu_memory <= 3:
            BATCH_SIZE = 4
            print(f"⚙️  배치 크기: {BATCH_SIZE} (3GB GPU 최적화)")
    else:
        print("❌ CUDA 사용 불가! CPU로 학습합니다.")
        DEVICE = 'cpu'
    
    # ==================== S3 데이터셋 초기화 ====================
    print("\n" + "="*70)
    print("📦 S3 데이터셋 로드 중...")
    print("="*70)
    
    try:
        # S3 데이터셋 매니저 생성
        s3_params = s3_config.get_s3_params()
        s3_dataset_manager = S3YOLOClassificationDataset(
            bucket_name=s3_params['bucket_name'],
            train_prefix=s3_config.train_prefix,
            val_prefix=s3_config.val_prefix,
            aws_access_key=s3_params.get('aws_access_key'),
            aws_secret_key=s3_params.get('aws_secret_key'),
            region_name=s3_params['region_name'],
            cache_size=s3_params['cache_size']
        )
        
        dataset_info = s3_dataset_manager.get_info()
        
    except Exception as e:
        logger.error(f"S3 데이터셋 로드 실패: {e}")
        logger.info("\n💡 확인 사항:")
        logger.info("   1. AWS 자격증명이 올바른지 확인")
        logger.info("   2. S3 버킷 이름이 정확한지 확인")
        logger.info("   3. S3 버킷에 train/, val/ 폴더가 존재하는지 확인")
        logger.info("   4. S3 버킷 접근 권한이 있는지 확인")
        return
    
    # ==================== 학습 설정 출력 ====================
    print("\n" + "="*70)
    print("⚙️  학습 설정")
    print("="*70)
    print(f"모델: YOLO{MODEL_SIZE}-cls")
    print(f"에포크: {EPOCHS}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"이미지 크기: {IMG_SIZE}")
    print(f"장치: {DEVICE.upper()}")
    print(f"캐시 크기: {s3_params['cache_size']}개 이미지")
    
    # 예상 시간
    if DEVICE == 'cuda':
        hours = (dataset_info['train_count'] / 10000) * EPOCHS * 0.15
    else:
        hours = (dataset_info['train_count'] / 1000) * EPOCHS * 0.15
    print(f"예상 시간: 약 {hours:.1f}시간")
    
    # ==================== YOLO 학습 (커스텀 데이터로더 사용) ====================
    print("\n" + "="*70)
    print("🚀 YOLO 학습 시작")
    print("   💡 S3에서 실시간으로 데이터를 스트리밍합니다")
    print("   💡 서버 디스크에는 모델 가중치만 저장됩니다")
    print("="*70)
    print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
    
    try:
        # YOLO 모델 로드
        model = YOLO(f'yolo{MODEL_SIZE}-cls.pt')
        
        # 데이터셋 yaml 생성 (YOLO가 클래스 정보를 알 수 있도록)
        yaml_path = create_yolo_compatible_dataset_yaml(dataset_info)
        
        # ⚠️ 주의: YOLO의 train() 메서드는 기본적으로 로컬 파일 시스템을 기대합니다
        # 따라서 커스텀 트레이닝 루프를 사용하거나, YOLO의 내부를 수정해야 합니다
        # 여기서는 두 가지 접근 방법을 제공합니다:
        
        # ========== 방법 1: YOLO 기본 train() 사용 (권장하지 않음 - S3 지원 안 함) ==========
        # results = model.train(
        #     data=str(yaml_path),
        #     epochs=EPOCHS,
        #     batch=BATCH_SIZE,
        #     imgsz=IMG_SIZE,
        #     device=DEVICE,
        #     ...
        # )
        
        # ========== 방법 2: 커스텀 트레이닝 루프 (S3 스트리밍 지원) ==========
        logger.info("\n⚙️  커스텀 트레이닝 루프로 S3 스트리밍 학습 시작...\n")
        
        # Transform 정의 (YOLO가 내부적으로 사용하는 것과 유사)
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # S3 데이터셋 생성
        train_dataset = s3_dataset_manager.create_train_dataset(transform=train_transform)
        val_dataset = s3_dataset_manager.create_val_dataset(transform=val_transform)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        logger.info(f"✅ 데이터로더 준비 완료")
        logger.info(f"   학습 배치: {len(train_loader)}개")
        logger.info(f"   검증 배치: {len(val_loader)}개\n")
        
        # ===== 실제 학습은 ultralytics의 제약으로 인해 =====
        # ===== 완전한 커스텀 트레이닝 루프가 필요합니다 =====
        # ===== 아래는 PyTorch 기본 학습 루프 예시입니다 =====
        
        logger.warning("\n⚠️  YOLO ultralytics는 기본적으로 로컬 파일 시스템을 요구합니다.")
        logger.warning("완전한 S3 스트리밍 학습을 위해서는 PyTorch 기본 학습 루프를 사용하거나")
        logger.warning("YOLO 소스코드를 수정해야 합니다.\n")
        
        logger.info("💡 대안 1: 임시 디렉토리를 사용한 하이브리드 방식")
        logger.info("   - S3에서 배치 단위로 다운로드 → 학습 → 삭제")
        logger.info("   - train_korean_food_s3_hybrid.py 참조\n")
        
        logger.info("💡 대안 2: PyTorch 직접 학습")
        logger.info("   - YOLO 모델을 PyTorch 모델로 변환")
        logger.info("   - 위에서 생성한 DataLoader 사용")
        logger.info("   - train_korean_food_s3_pytorch.py 참조\n")
        
        # 테스트: 첫 배치 로드 확인
        logger.info("🧪 데이터 로딩 테스트 중...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            logger.info(f"   배치 {batch_idx+1}: 이미지 shape={images.shape}, 레이블 shape={labels.shape}")
            if batch_idx >= 2:  # 3개 배치만 테스트
                break
        
        logger.info("✅ S3 스트리밍 데이터 로딩 테스트 성공!\n")
        
        print("="*70)
        print("✅ S3 데이터셋 설정 완료!")
        print("="*70)
        print("\n📝 다음 단계:")
        print("   1. train_korean_food_s3_hybrid.py: 하이브리드 방식 (추천)")
        print("   2. train_korean_food_s3_pytorch.py: 순수 PyTorch 방식")
        print("\n이 두 스크립트는 곧 생성됩니다...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
    
    except Exception as e:
        logger.error(f"\n❌ 학습 중 오류 발생: {e}", exc_info=True)


def main():
    """메인 함수"""
    train_with_s3_streaming()


if __name__ == "__main__":
    main()

