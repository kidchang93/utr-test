"""
S3의 신규 추가 데이터를 train/val로 증분 업데이트하는 스크립트
foods/에 새로운 클래스나 이미지가 추가되었을 때 사용

사용 시나리오:
1. foods/새로운클래스/ 폴더가 추가됨
2. 기존 클래스에 이미지가 추가됨
3. 전체 재분할 없이 증분 업데이트
"""
import sys
import logging
from pathlib import Path
import random
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config.s3_config import get_s3_config
from utils.log_utils import LoggerFactory


# 전역 로거 (이 스크립트 전용 이름으로 생성)
logger = LoggerFactory.get_logger(__name__, log_name="update_s3_data")


class S3IncrementalUpdater:
    """S3 데이터 증분 업데이트 클래스"""
    
    def __init__(self, bucket_name: str, access_key=None, secret_key=None, region='ap-northeast-2', domain=None):
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
    
    def list_images_in_prefix(self, prefix: str) -> list:
        """특정 prefix 하위의 모든 이미지 키 리스트"""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_keys = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext.lower()) for ext in extensions):
                        image_keys.append(key)
            return image_keys
        except Exception as e:
            logger.error(f"리스트 조회 오류: {e}")
            return []
    
    def get_class_structure(self, base_prefix: str) -> dict:
        """클래스 구조 분석"""
        class_images = {}
        
        try:
            result = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=base_prefix,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' not in result:
                return class_images
            
            for prefix_info in result['CommonPrefixes']:
                class_prefix = prefix_info['Prefix']
                class_name = class_prefix.rstrip('/').split('/')[-1]
                images = self.list_images_in_prefix(class_prefix)
                if images:
                    class_images[class_name] = images
            
            return class_images
        except Exception as e:
            logger.error(f"클래스 구조 분석 오류: {e}")
            return {}
    
    def copy_object(self, source_key: str, dest_key: str) -> bool:
        """S3 내에서 객체 복사"""
        try:
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            return True
        except Exception as e:
            logger.error(f"복사 실패: {e}")
            return False
    
    def find_new_classes(self, raw_prefix: str, train_prefix: str) -> list:
        """foods/에는 있지만 train/에는 없는 새로운 클래스 찾기"""
        raw_structure = self.get_class_structure(raw_prefix)
        train_structure = self.get_class_structure(train_prefix)
        
        raw_classes = set(raw_structure.keys())
        train_classes = set(train_structure.keys())
        
        new_classes = raw_classes - train_classes
        return list(new_classes)
    
    def find_new_images(self, raw_prefix: str, train_prefix: str, val_prefix: str, class_name: str) -> list:
        """특정 클래스에서 새로 추가된 이미지 찾기"""
        raw_images = set(self.list_images_in_prefix(f"{raw_prefix}{class_name}/"))
        train_images = set(self.list_images_in_prefix(f"{train_prefix}{class_name}/"))
        val_images = set(self.list_images_in_prefix(f"{val_prefix}{class_name}/"))
        
        # 파일명만 비교 (경로 제거)
        train_filenames = {img.split('/')[-1] for img in train_images}
        val_filenames = {img.split('/')[-1] for img in val_images}
        existing_filenames = train_filenames | val_filenames
        
        # 새로운 이미지 찾기
        new_images = []
        for raw_img in raw_images:
            filename = raw_img.split('/')[-1]
            if filename not in existing_filenames:
                new_images.append(raw_img)
        
        return new_images
    
    def update_incremental(
        self,
        raw_prefix: str = 'foods/',
        train_prefix: str = 'train/',
        val_prefix: str = 'val/',
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        증분 업데이트 실행
        
        1. 새로운 클래스 찾기 → 전체 분할하여 추가
        2. 기존 클래스의 새 이미지 찾기 → 기존 비율 유지하며 추가
        """
        logger.info("="*70)
        logger.info("🔄 S3 증분 업데이트 시작")
        logger.info("="*70)
        
        random.seed(random_seed)
        
        # 1. 새로운 클래스 처리
        logger.info("\n1️⃣ 새로운 클래스 확인 중...")
        new_classes = self.find_new_classes(raw_prefix, train_prefix)
        
        if new_classes:
            logger.info(f"✨ 새로운 클래스 발견: {len(new_classes)}개")
            for class_name in new_classes:
                logger.info(f"   - {class_name}")
            
            logger.info("\n새 클래스 분할 및 추가 중...")
            raw_structure = self.get_class_structure(raw_prefix)
            
            for class_name in new_classes:
                images = raw_structure[class_name]
                shuffled = images.copy()
                random.shuffle(shuffled)
                
                split_idx = int(len(shuffled) * train_ratio)
                train_keys = shuffled[:split_idx]
                val_keys = shuffled[split_idx:]
                
                logger.info(f"\n   {class_name}: 학습 {len(train_keys)}, 검증 {len(val_keys)}")
                
                # Train 복사
                for source_key in tqdm(train_keys, desc=f"     Train {class_name}"):
                    filename = source_key.split('/')[-1]
                    dest_key = f"{train_prefix}{class_name}/{filename}"
                    self.copy_object(source_key, dest_key)
                
                # Val 복사
                for source_key in tqdm(val_keys, desc=f"     Val {class_name}"):
                    filename = source_key.split('/')[-1]
                    dest_key = f"{val_prefix}{class_name}/{filename}"
                    self.copy_object(source_key, dest_key)
        else:
            logger.info("✅ 새로운 클래스 없음")
        
        # 2. 기존 클래스의 새 이미지 처리
        logger.info("\n2️⃣ 기존 클래스의 새 이미지 확인 중...")
        train_structure = self.get_class_structure(train_prefix)
        
        total_new_images = 0
        classes_with_new_images = []
        
        for class_name in train_structure.keys():
            new_images = self.find_new_images(raw_prefix, train_prefix, val_prefix, class_name)
            if new_images:
                total_new_images += len(new_images)
                classes_with_new_images.append((class_name, new_images))
        
        if classes_with_new_images:
            logger.info(f"✨ 새 이미지 발견: {len(classes_with_new_images)}개 클래스, 총 {total_new_images}개 이미지")
            
            for class_name, new_images in classes_with_new_images:
                logger.info(f"\n   {class_name}: {len(new_images)}개 새 이미지")
                
                # 랜덤 셔플 후 분할
                shuffled = new_images.copy()
                random.shuffle(shuffled)
                
                split_idx = int(len(shuffled) * train_ratio)
                train_keys = shuffled[:split_idx]
                val_keys = shuffled[split_idx:]
                
                # Train 복사
                for source_key in tqdm(train_keys, desc=f"     Train {class_name}"):
                    filename = source_key.split('/')[-1]
                    dest_key = f"{train_prefix}{class_name}/{filename}"
                    self.copy_object(source_key, dest_key)
                
                # Val 복사
                for source_key in tqdm(val_keys, desc=f"     Val {class_name}"):
                    filename = source_key.split('/')[-1]
                    dest_key = f"{val_prefix}{class_name}/{filename}"
                    self.copy_object(source_key, dest_key)
        else:
            logger.info("✅ 새 이미지 없음")
        
        # 3. 결과 요약
        logger.info("\n" + "="*70)
        logger.info("✅ 증분 업데이트 완료!")
        logger.info("="*70)
        logger.info(f"새 클래스: {len(new_classes)}개")
        logger.info(f"업데이트된 클래스: {len(classes_with_new_images)}개")
        logger.info(f"총 새 이미지: {total_new_images}개")
        logger.info("="*70)


def main():
    """메인 함수"""
    logger.info("=" * 70)
    logger.info("🚀 update_s3_data.py 실행 시작")
    logger.info("=" * 70)

    print("\n" + "="*70)
    print("🔄 S3 증분 업데이트 (foods/ → train/, val/)")
    print("="*70 + "\n")
    
    # S3 설정 로드
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        return
    
    # 설정
    RAW_PREFIX = s3_config.raw_prefix
    TRAIN_PREFIX = s3_config.train_prefix
    VAL_PREFIX = s3_config.val_prefix
    TRAIN_RATIO = 0.8
    
    print("\n" + "="*70)
    print("⚙️  업데이트 설정")
    print("="*70)
    print(f"Raw 경로: {RAW_PREFIX}")
    print(f"학습 경로: {TRAIN_PREFIX}")
    print(f"검증 경로: {VAL_PREFIX}")
    print(f"분할 비율: {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    print("\n💡 이 스크립트는:")
    print("   1. foods/에 새로 추가된 클래스를 train/val로 분할")
    print("   2. 기존 클래스에 추가된 이미지를 train/val로 분할")
    print("   3. 기존 데이터는 그대로 유지")
    print("="*70)
    
    response = input("\n계속 진행하시겠습니까? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ 작업 취소됨")
        return
    
    # S3 업데이터 초기화
    s3_params = s3_config.get_s3_params()
    updater = S3IncrementalUpdater(
        bucket_name=s3_params['bucket_name'],
        access_key=s3_params.get('access_key'),
        secret_key=s3_params.get('secret_key'),
        region=s3_params['region'],
        domain=s3_params.get('domain')
    )
    
    # 증분 업데이트 실행
    try:
        updater.update_incremental(
            raw_prefix=RAW_PREFIX,
            train_prefix=TRAIN_PREFIX,
            val_prefix=VAL_PREFIX,
            train_ratio=TRAIN_RATIO
        )
        logger.info("✅ update_s3_data.py 정상 종료")

        print("\n" + "="*70)
        print("📝 다음 단계:")
        print("="*70)
        print("\n1. 연결 테스트:")
        print("   python scripts/test_s3_connection.py")
        print("\n2. 학습 시작:")
        print("   python scripts/train_korean_food_s3_hybrid.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자가 작업을 중단했습니다.")
        print("\n\n⚠️  사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    main()

