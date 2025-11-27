"""
로컬 데이터를 S3 버킷에 업로드하는 스크립트
train/val 폴더 구조를 유지하면서 S3에 업로드
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config.s3_config import get_s3_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']


class S3DataUploader:
    """로컬 데이터를 S3에 업로드하는 클래스"""
    
    def __init__(self, bucket_name: str, access_key=None, secret_key=None, region='ap-northeast-2', domain=None):
        """
        Args:
            bucket_name: S3 버킷 이름
            access_key: Access Key
            secret_key: Secret Key
            region: 리전
            domain: S3 엔드포인트 URL (NCP 등 커스텀 S3용)
        """
        self.bucket_name = bucket_name
        
        # S3 클라이언트 초기화
        client_kwargs = {'region_name': region}
        
        if access_key and secret_key:
            client_kwargs['aws_access_key_id'] = access_key
            client_kwargs['aws_secret_access_key'] = secret_key
        
        if domain:
            client_kwargs['endpoint_url'] = domain
        
        self.s3_client = boto3.client('s3', **client_kwargs)
        
        endpoint_info = f", 엔드포인트: {domain}" if domain else ""
        logger.info(f"S3 클라이언트 초기화 완료 - 버킷: {bucket_name}{endpoint_info}")
    
    def check_bucket_exists(self) -> bool:
        """버킷 존재 여부 확인"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ 버킷 '{self.bucket_name}' 접근 확인")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"❌ 버킷 '{self.bucket_name}'을 찾을 수 없습니다.")
            elif error_code == '403':
                logger.error(f"❌ 버킷 '{self.bucket_name}'에 대한 접근 권한이 없습니다.")
            else:
                logger.error(f"❌ 버킷 확인 오류: {e}")
            return False
    
    def create_bucket_if_not_exists(self, region_name='ap-northeast-2'):
        """버킷이 없으면 생성"""
        try:
            if not self.check_bucket_exists():
                logger.info(f"버킷 생성 중: {self.bucket_name}")
                
                if region_name == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region_name}
                    )
                
                logger.info(f"✅ 버킷 생성 완료: {self.bucket_name}")
                return True
            return True
        except Exception as e:
            logger.error(f"❌ 버킷 생성 실패: {e}")
            return False
    
    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        단일 파일을 S3에 업로드
        
        Args:
            local_path: 로컬 파일 경로
            s3_key: S3 객체 키
            
        Returns:
            성공 여부
        """
        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            return True
        except Exception as e:
            logger.error(f"업로드 실패 ({local_path}): {e}")
            return False
    
    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        extensions: list = None
    ) -> tuple:
        """
        디렉토리를 S3에 업로드 (하위 폴더 구조 유지)
        
        Args:
            local_dir: 로컬 디렉토리 경로
            s3_prefix: S3 prefix (예: 'train/' or 'val/')
            extensions: 업로드할 파일 확장자 리스트
            
        Returns:
            (성공 수, 실패 수) 튜플
        """
        if extensions is None:
            extensions = IMAGE_EXTENSIONS.copy()
        
        # 모든 이미지 파일 찾기
        image_files = []
        for ext in extensions:
            image_files.extend(local_dir.rglob(ext))
        
        if not image_files:
            logger.warning(f"'{local_dir}' 하위에 이미지 파일이 없습니다.")
            return 0, 0
        
        logger.info(f"\n📤 업로드 시작: {local_dir} → s3://{self.bucket_name}/{s3_prefix}")
        logger.info(f"   총 {len(image_files):,}개 파일")
        
        success_count = 0
        fail_count = 0
        
        # tqdm을 사용한 진행률 표시
        with tqdm(total=len(image_files), desc=f"Uploading {s3_prefix}", unit="file") as pbar:
            for image_path in image_files:
                # 상대 경로 계산
                relative_path = image_path.relative_to(local_dir)
                
                # S3 키 생성
                s3_key = s3_prefix + str(relative_path).replace('\\', '/')
                
                # 업로드
                if self.upload_file(image_path, s3_key):
                    success_count += 1
                else:
                    fail_count += 1
                
                pbar.update(1)
        
        logger.info(f"✅ 업로드 완료: 성공 {success_count:,}개, 실패 {fail_count}개\n")
        return success_count, fail_count
    
    def list_s3_structure(self, prefix=''):
        """S3 버킷의 구조 출력"""
        logger.info(f"\n📊 S3 버킷 구조: s3://{self.bucket_name}/{prefix}")
        
        try:
            # 상위 폴더 리스트
            result = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' in result:
                logger.info("\n📁 폴더:")
                for prefix_info in result['CommonPrefixes']:
                    folder_name = prefix_info['Prefix']
                    logger.info(f"   {folder_name}")
                    
                    # 하위 폴더 (클래스) 리스트
                    sub_result = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=folder_name,
                        Delimiter='/'
                    )
                    
                    if 'CommonPrefixes' in sub_result:
                        for sub_prefix in sub_result['CommonPrefixes']:
                            sub_folder = sub_prefix['Prefix']
                            
                            # 해당 폴더의 파일 수 계산
                            count_result = self.s3_client.list_objects_v2(
                                Bucket=self.bucket_name,
                                Prefix=sub_folder
                            )
                            
                            file_count = count_result.get('KeyCount', 0)
                            logger.info(f"      └─ {sub_folder}: {file_count:,}개 파일")
        
        except Exception as e:
            logger.error(f"S3 구조 조회 실패: {e}")


def _count_images(root_dir: Path) -> Tuple[int, int]:
    """클래스 개수와 전체 이미지를 계산한다."""
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    total_images = 0
    for class_dir in class_dirs:
        for pattern in IMAGE_EXTENSIONS:
            total_images += len(list(class_dir.glob(pattern)))
    return len(class_dirs), total_images


def upload_local_dataset_to_s3(
    local_dataset_dir: str,
    create_bucket_if_missing: bool = False,
) -> Dict[str, int]:
    """
    로컬 train/val 데이터를 S3 버킷으로 업로드한다.

    Args:
        local_dataset_dir: train/val 구조가 포함된 로컬 디렉터리
        create_bucket_if_missing: True면 버킷이 없을 경우 자동 생성

    Returns:
        업로드 통계 정보
    """
    logger.info("📤 S3 업로드 요청 - local_dir=%s", local_dataset_dir)
    dataset_dir = Path(local_dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"로컬 데이터셋을 찾을 수 없습니다: {dataset_dir}")

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError("train/val 폴더가 모두 존재해야 합니다.")

    train_classes, train_count = _count_images(train_dir)
    val_classes, val_count = _count_images(val_dir)

    s3_config = get_s3_config()
    s3_params = s3_config.get_s3_params()
    uploader = S3DataUploader(
        bucket_name=s3_params['bucket_name'],
        access_key=s3_params.get('access_key'),
        secret_key=s3_params.get('secret_key'),
        region=s3_params['region'],
        domain=s3_params.get('domain'),
    )

    if not uploader.check_bucket_exists():
        if not create_bucket_if_missing:
            raise ValueError(
                f"버킷 '{s3_config.bucket_name}'을 찾을 수 없습니다. "
                "create_bucket_if_missing=True로 요청하거나 버킷을 생성하세요."
            )
        if not uploader.create_bucket_if_not_exists(s3_config.region):
            raise RuntimeError("버킷 생성에 실패했습니다.")

    train_success, train_fail = uploader.upload_directory(
        local_dir=train_dir,
        s3_prefix=s3_config.train_prefix,
    )
    val_success, val_fail = uploader.upload_directory(
        local_dir=val_dir,
        s3_prefix=s3_config.val_prefix,
    )

    summary = {
        "bucket": s3_config.bucket_name,
        "train_prefix": s3_config.train_prefix,
        "val_prefix": s3_config.val_prefix,
        "train_classes": train_classes,
        "val_classes": val_classes,
        "train_images": train_count,
        "val_images": val_count,
        "uploaded_train": train_success,
        "uploaded_val": val_success,
        "upload_failures": train_fail + val_fail,
    }
    logger.info("✅ S3 업로드 완료 - %s", summary)
    return summary


def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("📤 로컬 데이터 → S3 업로드")
    print("="*70 + "\n")
    
    # ==================== S3 설정 로드 ====================
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 .env 파일을 생성하고 설정하세요.")
        return
    
    # ==================== 로컬 데이터 경로 ====================
    LOCAL_DATASET_DIR = Path(r"D:\lck_data\dataset\kfood-yolo")
    
    if not LOCAL_DATASET_DIR.exists():
        logger.error(f"❌ 로컬 데이터셋을 찾을 수 없습니다: {LOCAL_DATASET_DIR}")
        logger.info("\n💡 먼저 'prepare_data.py'를 실행하여 로컬 데이터셋을 생성하세요.")
        return
    
    train_dir = LOCAL_DATASET_DIR / "train"
    val_dir = LOCAL_DATASET_DIR / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"❌ train 또는 val 폴더를 찾을 수 없습니다.")
        return
    
    train_classes, train_count = _count_images(train_dir)
    val_classes, val_count = _count_images(val_dir)
    
    logger.info(f"\n📊 로컬 데이터셋 정보:")
    logger.info(f"   경로: {LOCAL_DATASET_DIR}")
    logger.info(f"   카테고리(Train/Val): {train_classes}/{val_classes}")
    logger.info(f"   학습: {train_count:,}장")
    logger.info(f"   검증: {val_count:,}장")
    logger.info(f"   전체: {train_count + val_count:,}장")
    
    # ==================== 업로드 확인 ====================
    print("\n" + "="*70)
    print("⚠️  주의: 대용량 데이터 업로드는 시간이 오래 걸릴 수 있습니다.")
    print(f"   업로드할 파일: {train_count + val_count:,}개")
    print(f"   대상 버킷: {s3_config.bucket_name}")
    print("="*70)
    
    response = input("\n계속 진행하시겠습니까? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ 업로드 취소됨")
        return
    
    create_bucket_if_missing = False
    print("\n버킷이 존재하지 않을 경우 자동으로 생성할까요? (yes/no): ", end="")
    if input().strip().lower() == 'yes':
        create_bucket_if_missing = True
    
    try:
        summary = upload_local_dataset_to_s3(
            local_dataset_dir=str(LOCAL_DATASET_DIR),
            create_bucket_if_missing=create_bucket_if_missing,
        )
    except Exception as exc:
        logger.error(f"업로드 실패: {exc}")
        return
    
    print("\n" + "="*70)
    print("✅ 업로드 완료!")
    print("="*70)
    print(f"버킷: s3://{summary['bucket']}/")
    print(f"Train 업로드: {summary['uploaded_train']:,}개")
    print(f"Val 업로드: {summary['uploaded_val']:,}개")
    print(f"실패: {summary['upload_failures']}개")
    
    # S3 구조 출력
    s3_params = s3_config.get_s3_params()
    uploader = S3DataUploader(
        bucket_name=s3_params['bucket_name'],
        access_key=s3_params.get('access_key'),
        secret_key=s3_params.get('secret_key'),
        region=s3_params['region'],
        domain=s3_params.get('domain')
    )
    uploader.list_s3_structure()
    
    print("\n" + "="*70)
    print("📝 다음 단계:")
    print("   python train_korean_food_s3_hybrid.py")
    print("="*70)


if __name__ == "__main__":
    main()

