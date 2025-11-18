"""
S3의 raw 데이터를 train/val로 분할하는 스크립트
원본 foods/{label}/ 구조는 그대로 유지하고
train/{label}/, val/{label}/ 구조를 새로 생성
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3DataSplitter:
    """S3 내에서 데이터를 train/val로 분할하는 클래스"""
    
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
    
    def check_bucket_access(self) -> bool:
        """버킷 접근 확인"""
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
    
    def list_images_in_prefix(self, prefix: str, extensions=None) -> list:
        """
        특정 prefix 하위의 모든 이미지 키 리스트 반환
        
        Args:
            prefix: S3 prefix (예: 'foods/김치찌개/')
            extensions: 이미지 확장자 리스트
            
        Returns:
            이미지 키 리스트
        """
        if extensions is None:
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
            logger.error(f"S3 리스트 조회 중 오류: {e}")
            return []
    
    def get_class_structure(self, base_prefix: str = 'foods/') -> dict:
        """
        S3 버킷의 클래스 구조 분석
        
        Args:
            base_prefix: 기본 prefix (예: 'foods/')
            
        Returns:
            {클래스명: [이미지_키_리스트]} 딕셔너리
        """
        class_images = {}
        
        try:
            # 클래스 폴더 리스트 가져오기
            result = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=base_prefix,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' not in result:
                logger.warning(f"'{base_prefix}' 하위에 폴더가 없습니다.")
                return class_images
            
            # 각 클래스별로 이미지 리스트 조회
            for prefix_info in result['CommonPrefixes']:
                class_prefix = prefix_info['Prefix']
                class_name = class_prefix.rstrip('/').split('/')[-1]
                
                # 해당 클래스의 이미지 리스트 가져오기
                images = self.list_images_in_prefix(class_prefix)
                
                if images:
                    class_images[class_name] = images
                    logger.info(f"클래스 '{class_name}': {len(images)}개 이미지")
            
            return class_images
            
        except Exception as e:
            logger.error(f"클래스 구조 분석 중 오류: {e}")
            return {}
    
    def object_exists(self, key: str) -> bool:
        """
        S3 객체 존재 여부 확인
        
        Args:
            key: S3 객체 키
            
        Returns:
            존재 여부
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                # 404가 아닌 다른 오류는 로그만 남기고 False 반환
                logger.warning(f"객체 확인 중 오류 ({key}): {e}")
                return False
    
    def copy_object(self, source_key: str, dest_key: str) -> bool:
        """
        S3 내에서 객체 복사
        
        Args:
            source_key: 원본 키
            dest_key: 대상 키
            
        Returns:
            성공 여부
        """
        try:
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            return True
        except Exception as e:
            logger.error(f"복사 실패 ({source_key} → {dest_key}): {e}")
            return False
    
    def split_and_copy(
        self,
        source_prefix: str = 'foods/',
        train_prefix: str = 'train/',
        val_prefix: str = 'val/',
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> tuple:
        """
        S3 데이터를 train/val로 분할하여 복사
        
        Args:
            source_prefix: 원본 데이터 prefix
            train_prefix: 학습 데이터 대상 prefix
            val_prefix: 검증 데이터 대상 prefix
            train_ratio: 학습 데이터 비율 (0~1)
            random_seed: 랜덤 시드
            
        Returns:
            (train_count, val_count) 튜플
        """
        logger.info("="*70)
        logger.info(f"📊 S3 데이터 분할 시작")
        logger.info("="*70)
        logger.info(f"원본: {source_prefix}")
        logger.info(f"학습: {train_prefix}")
        logger.info(f"검증: {val_prefix}")
        logger.info(f"비율: {train_ratio:.0%} / {1-train_ratio:.0%}")
        logger.info("="*70 + "\n")
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        
        # 클래스 구조 분석
        class_structure = self.get_class_structure(source_prefix)
        
        if not class_structure:
            logger.error("❌ 원본 데이터를 찾을 수 없습니다!")
            return 0, 0
        
        total_train = 0
        total_val = 0
        skipped_train = 0
        skipped_val = 0
        
        # 각 클래스별로 분할 및 복사
        for class_idx, (class_name, image_keys) in enumerate(class_structure.items(), 1):
            logger.info(f"\n[{class_idx}/{len(class_structure)}] {class_name} 처리 중...")
            logger.info(f"   전체: {len(image_keys)}개")
            
            # 랜덤 셔플
            shuffled_keys = image_keys.copy()
            random.shuffle(shuffled_keys)
            
            # Train/Val 분할
            split_idx = int(len(shuffled_keys) * train_ratio)
            train_keys = shuffled_keys[:split_idx]
            val_keys = shuffled_keys[split_idx:]
            
            logger.info(f"   학습: {len(train_keys)}개, 검증: {len(val_keys)}개")
            
            # Train 복사
            train_success = 0
            train_skipped = 0
            with tqdm(total=len(train_keys), desc=f"  Train {class_name}", unit="file") as pbar:
                for source_key in train_keys:
                    # 파일명 추출
                    filename = source_key.split('/')[-1]
                    dest_key = f"{train_prefix}{class_name}/{filename}"
                    
                    # 이미 존재하는지 확인
                    if self.object_exists(dest_key):
                        train_skipped += 1
                        pbar.update(1)
                        continue
                    
                    if self.copy_object(source_key, dest_key):
                        train_success += 1
                    
                    pbar.update(1)
            
            # Val 복사
            val_success = 0
            val_skipped = 0
            with tqdm(total=len(val_keys), desc=f"  Val {class_name}", unit="file") as pbar:
                for source_key in val_keys:
                    filename = source_key.split('/')[-1]
                    dest_key = f"{val_prefix}{class_name}/{filename}"
                    
                    # 이미 존재하는지 확인
                    if self.object_exists(dest_key):
                        val_skipped += 1
                        pbar.update(1)
                        continue
                    
                    if self.copy_object(source_key, dest_key):
                        val_success += 1
                    
                    pbar.update(1)
            
            total_train += train_success
            total_val += val_success
            skipped_train += train_skipped
            skipped_val += val_skipped
            
            logger.info(f"   ✅ 완료: 학습 {train_success}개 복사, {train_skipped}개 스킵 / 검증 {val_success}개 복사, {val_skipped}개 스킵")
        
        logger.info("\n" + "="*70)
        logger.info("✅ 전체 분할 완료!")
        logger.info("="*70)
        logger.info(f"학습 데이터: {total_train:,}개 복사, {skipped_train:,}개 스킵")
        logger.info(f"검증 데이터: {total_val:,}개 복사, {skipped_val:,}개 스킵")
        logger.info(f"전체: {total_train + total_val:,}개 복사, {skipped_train + skipped_val:,}개 스킵")
        logger.info("="*70)
        
        return total_train, total_val
    
    def verify_structure(self, train_prefix='train/', val_prefix='val/'):
        """생성된 train/val 구조 확인"""
        logger.info("\n" + "="*70)
        logger.info("🔍 생성된 구조 확인")
        logger.info("="*70)
        
        # Train 구조
        logger.info(f"\n📁 {train_prefix}")
        train_structure = self.get_class_structure(train_prefix)
        train_total = sum(len(imgs) for imgs in train_structure.values())
        logger.info(f"   클래스: {len(train_structure)}개")
        logger.info(f"   이미지: {train_total:,}개")
        
        # Val 구조
        logger.info(f"\n📁 {val_prefix}")
        val_structure = self.get_class_structure(val_prefix)
        val_total = sum(len(imgs) for imgs in val_structure.values())
        logger.info(f"   클래스: {len(val_structure)}개")
        logger.info(f"   이미지: {val_total:,}개")
        
        # 클래스별 분포 (처음 5개만)
        logger.info(f"\n📈 클래스별 분포 (처음 5개):")
        for i, class_name in enumerate(sorted(train_structure.keys())[:5], 1):
            train_count = len(train_structure.get(class_name, []))
            val_count = len(val_structure.get(class_name, []))
            total = train_count + val_count
            train_pct = train_count / total * 100 if total > 0 else 0
            logger.info(f"   {i}. {class_name}: 학습 {train_count}개 ({train_pct:.1f}%), 검증 {val_count}개")
        
        if len(train_structure) > 5:
            logger.info(f"   ... 외 {len(train_structure) - 5}개 클래스")
        
        logger.info("="*70)


def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("🔄 S3 데이터 분할 (foods/ → train/, val/)")
    print("="*70 + "\n")
    
    # ==================== S3 설정 로드 ====================
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 .env 파일을 생성하고 설정하세요.")
        return
    
    # ==================== 사용자 확인 ====================
    print("\n" + "="*70)
    print("⚙️  분할 설정")
    print("="*70)
    
    SOURCE_PREFIX = 'foods/'
    TRAIN_PREFIX = 'train/'
    VAL_PREFIX = 'val/'
    TRAIN_RATIO = 0.8
    
    print(f"원본 경로: {SOURCE_PREFIX}")
    print(f"학습 경로: {TRAIN_PREFIX}")
    print(f"검증 경로: {VAL_PREFIX}")
    print(f"분할 비율: {TRAIN_RATIO:.0%} (학습) / {1-TRAIN_RATIO:.0%} (검증)")
    print("\n⚠️  주의:")
    print("   - 원본 foods/ 데이터는 그대로 유지됩니다")
    print("   - train/, val/ 경로에 복사본이 생성됩니다")
    print("   - 기존에 train/, val/ 데이터가 있으면 중복될 수 있습니다")
    print("="*70)
    
    response = input("\n계속 진행하시겠습니까? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ 작업 취소됨")
        return
    
    # ==================== S3 분할기 초기화 ====================
    s3_params = s3_config.get_s3_params()
    splitter = S3DataSplitter(
        bucket_name=s3_params['bucket_name'],
        access_key=s3_params.get('access_key'),
        secret_key=s3_params.get('secret_key'),
        region=s3_params['region'],
        domain=s3_params.get('domain')
    )
    
    # 버킷 접근 확인
    if not splitter.check_bucket_access():
        logger.error("버킷 접근 실패. 종료합니다.")
        return
    
    # ==================== 분할 실행 ====================
    try:
        train_count, val_count = splitter.split_and_copy(
            source_prefix=SOURCE_PREFIX,
            train_prefix=TRAIN_PREFIX,
            val_prefix=VAL_PREFIX,
            train_ratio=TRAIN_RATIO
        )
        
        if train_count == 0 and val_count == 0:
            logger.error("❌ 분할 실패!")
            return
        
        # ==================== 결과 확인 ====================
        splitter.verify_structure(TRAIN_PREFIX, VAL_PREFIX)
        
        # ==================== 다음 단계 안내 ====================
        print("\n" + "="*70)
        print("📝 다음 단계:")
        print("="*70)
        print("\n1. 연결 테스트:")
        print("   python test_s3_connection.py")
        print("\n2. 학습 시작:")
        print("   python train_korean_food_s3_hybrid.py")
        print("\n💡 자세한 사용법은 QUICK_START_S3.md를 참고하세요.")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    main()

