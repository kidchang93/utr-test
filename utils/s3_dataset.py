"""
S3 기반 커스텀 PyTorch Dataset
YOLO 학습을 위해 S3에서 직접 이미지를 로드
"""
import random
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import logging

from .s3_loader import S3ImageLoader

logger = logging.getLogger(__name__)


class S3ClassificationDataset(Dataset):
    """
    S3에서 이미지를 스트리밍으로 로드하는 분류용 Dataset
    YOLO 분류 모델 학습에 사용
    """
    
    def __init__(
        self, 
        bucket_name: str,
        prefix: str,
        transform: Optional[Callable] = None,
        cache_size: int = 100,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = 'ap-northeast-2',
        domain: Optional[str] = None
    ):
        """
        Args:
            bucket_name: S3 버킷 이름
            prefix: S3 prefix (예: 'train/' or 'val/')
            transform: 이미지 변환 함수 (torchvision transforms)
            cache_size: 메모리에 캐싱할 이미지 수 (0이면 캐싱 안 함)
            access_key: Access Key
            secret_key: Secret Key
            region: 리전
            domain: S3 엔드포인트 URL (NCP 등 커스텀 S3용)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.cache_size = cache_size
        
        # S3 로더 초기화
        self.s3_loader = S3ImageLoader(
            bucket_name=bucket_name,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            domain=domain
        )
        
        # 클래스 구조 로드
        logger.info(f"S3 '{prefix}' 데이터셋 구조 분석 중...")
        self.class_to_images = self.s3_loader.get_class_structure(prefix)
        
        if not self.class_to_images:
            raise ValueError(f"'{prefix}' 하위에 이미지를 찾을 수 없습니다.")
        
        # 클래스명 정렬 및 인덱스 매핑
        self.classes = sorted(self.class_to_images.keys())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 전체 샘플 리스트 생성 [(image_key, class_idx), ...]
        self.samples = []
        for class_name, image_keys in self.class_to_images.items():
            class_idx = self.class_to_idx[class_name]
            for image_key in image_keys:
                self.samples.append((image_key, class_idx))
        
        # 샘플 섞기 (선택사항)
        random.shuffle(self.samples)
        
        # 간단한 LRU 캐시 (딕셔너리 기반)
        self.cache = {}
        self.cache_keys = []
        
        logger.info(f"✅ 데이터셋 로드 완료:")
        logger.info(f"   클래스: {len(self.classes)}개")
        logger.info(f"   샘플: {len(self.samples):,}개")
        logger.info(f"   캐시 크기: {cache_size}")
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        인덱스에 해당하는 이미지와 레이블 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            (image, label) 튜플
        """
        image_key, class_idx = self.samples[idx]
        
        # 캐시 확인
        if image_key in self.cache:
            image = self.cache[image_key].copy()
        else:
            # S3에서 이미지 로드
            image = self.s3_loader.load_image_from_s3(image_key)
            
            if image is None:
                # 로드 실패시 검은 이미지 반환
                logger.warning(f"이미지 로드 실패: {image_key}, 검은 이미지로 대체")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            # 캐시에 저장 (LRU 방식)
            if self.cache_size > 0:
                self._add_to_cache(image_key, image)
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx
    
    def _add_to_cache(self, key: str, image: Image.Image):
        """LRU 캐시에 이미지 추가"""
        if len(self.cache) >= self.cache_size:
            # 가장 오래된 항목 제거
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = image.copy()
        self.cache_keys.append(key)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """클래스별 샘플 수 반환"""
        distribution = {}
        for class_name, images in self.class_to_images.items():
            distribution[class_name] = len(images)
        return distribution
    
    def get_sample_info(self, idx: int) -> Dict:
        """특정 인덱스의 샘플 정보 반환"""
        image_key, class_idx = self.samples[idx]
        class_name = self.classes[class_idx]
        return {
            'index': idx,
            'image_key': image_key,
            'class_name': class_name,
            'class_idx': class_idx
        }


class S3YOLOClassificationDataset:
    """
    YOLO Classification 학습을 위한 S3 데이터셋 래퍼
    train/val 데이터셋을 모두 관리
    """
    
    def __init__(
        self,
        bucket_name: str,
        train_prefix: str = 'train/',
        val_prefix: str = 'val/',
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = 'ap-northeast-2',
        domain: Optional[str] = None,
        cache_size: int = 100
    ):
        """
        Args:
            bucket_name: S3 버킷 이름
            train_prefix: 학습 데이터 prefix
            val_prefix: 검증 데이터 prefix
            access_key: Access Key
            secret_key: Secret Key
            region: 리전
            domain: S3 엔드포인트 URL (NCP 등 커스텀 S3용)
            cache_size: 캐시 크기
        """
        self.bucket_name = bucket_name
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        
        # S3 로더 (구조 확인용)
        self.s3_loader = S3ImageLoader(
            bucket_name=bucket_name,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            domain=domain
        )
        
        # 버킷 접근 확인
        if not self.s3_loader.check_bucket_access():
            raise ConnectionError(f"S3 버킷 '{bucket_name}'에 접근할 수 없습니다.")
        
        # 클래스 구조 로드
        logger.info("="*70)
        logger.info("📊 S3 데이터셋 정보 로드 중...")
        logger.info("="*70)
        
        train_structure = self.s3_loader.get_class_structure(train_prefix)
        val_structure = self.s3_loader.get_class_structure(val_prefix)
        
        if not train_structure:
            raise ValueError(f"'{train_prefix}' 하위에 학습 데이터를 찾을 수 없습니다.")
        
        if not val_structure:
            raise ValueError(f"'{val_prefix}' 하위에 검증 데이터를 찾을 수 없습니다.")
        
        self.classes = sorted(train_structure.keys())
        self.num_classes = len(self.classes)
        
        # 통계 계산
        self.train_count = sum(len(imgs) for imgs in train_structure.values())
        self.val_count = sum(len(imgs) for imgs in val_structure.values())
        
        # 정보 출력
        logger.info(f"\n✅ S3 데이터셋 정보:")
        logger.info(f"   버킷: {bucket_name}")
        logger.info(f"   리전: {region_name}")
        logger.info(f"   카테고리: {self.num_classes}개")
        logger.info(f"   학습: {self.train_count:,}장")
        logger.info(f"   검증: {self.val_count:,}장")
        logger.info(f"   전체: {self.train_count + self.val_count:,}장")
        
        # 클래스별 분포 (처음 5개만)
        logger.info(f"\n📈 클래스 분포 (처음 5개):")
        for i, class_name in enumerate(self.classes[:5], 1):
            train_imgs = len(train_structure.get(class_name, []))
            val_imgs = len(val_structure.get(class_name, []))
            logger.info(f"   {i}. {class_name}: 학습 {train_imgs}장, 검증 {val_imgs}장")
        
        if len(self.classes) > 5:
            logger.info(f"   ... 외 {len(self.classes) - 5}개 클래스")
        
        logger.info("="*70)
        
        # 데이터셋 파라미터 저장 (나중에 생성시 사용)
        self.dataset_params = {
            'bucket_name': bucket_name,
            'access_key': access_key,
            'secret_key': secret_key,
            'region': region,
            'domain': domain,
            'cache_size': cache_size
        }
    
    def create_train_dataset(self, transform=None) -> S3ClassificationDataset:
        """학습 데이터셋 생성"""
        return S3ClassificationDataset(
            prefix=self.train_prefix,
            transform=transform,
            **self.dataset_params
        )
    
    def create_val_dataset(self, transform=None) -> S3ClassificationDataset:
        """검증 데이터셋 생성"""
        return S3ClassificationDataset(
            prefix=self.val_prefix,
            transform=transform,
            **self.dataset_params
        )
    
    def get_info(self) -> Dict:
        """데이터셋 정보 딕셔너리 반환"""
        return {
            'bucket_name': self.bucket_name,
            'train_prefix': self.train_prefix,
            'val_prefix': self.val_prefix,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'train_count': self.train_count,
            'val_count': self.val_count,
            'total_count': self.train_count + self.val_count
        }


if __name__ == "__main__":
    # 사용 예시
    logging.basicConfig(level=logging.INFO)
    
    print("S3 데이터셋 테스트:")
    print("dataset = S3YOLOClassificationDataset(")
    print("    bucket_name='your-bucket-name',")
    print("    train_prefix='train/',")
    print("    val_prefix='val/'")
    print(")")

