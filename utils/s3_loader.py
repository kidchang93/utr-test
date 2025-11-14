"""
S3에서 이미지를 스트리밍으로 읽어오는 유틸리티 모듈
서버에 파일을 저장하지 않고 메모리에서 직접 처리
"""
import io
import boto3
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3ImageLoader:
    """S3 버킷에서 이미지를 스트리밍으로 로드하는 클래스"""
    
    def __init__(self, bucket_name: str, access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None, region: str = 'ap-northeast-2',
                 domain: Optional[str] = None):
        """
        Args:
            bucket_name: S3 버킷 이름
            access_key: Access Key (None이면 환경변수나 IAM Role 사용)
            secret_key: Secret Key
            region: 리전 (기본값: ap-northeast-2)
            domain: S3 엔드포인트 URL (NCP 등 커스텀 S3용, None이면 AWS S3 기본)
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
        logger.info(f"S3ImageLoader 초기화 완료 - 버킷: {bucket_name}, 리전: {region}{endpoint_info}")
    
    def load_image_from_s3(self, s3_key: str) -> Optional[Image.Image]:
        """
        S3에서 이미지를 직접 메모리로 로드
        
        Args:
            s3_key: S3 객체 키 (예: 'train/김치찌개/001.jpg')
            
        Returns:
            PIL Image 객체 또는 None (실패시)
        """
        try:
            # S3에서 객체 가져오기
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            # 바이트 데이터를 PIL Image로 변환
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환 (YOLO는 RGB 필요)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"S3 키를 찾을 수 없음: {s3_key}")
            else:
                logger.error(f"S3 클라이언트 오류: {e}")
            return None
            
        except NoCredentialsError:
            logger.error("AWS 자격증명을 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            logger.error(f"이미지 로드 중 오류 ({s3_key}): {e}")
            return None
    
    def list_images_in_prefix(self, prefix: str, extensions: List[str] = None) -> List[str]:
        """
        S3 버킷의 특정 prefix 하위의 모든 이미지 키 리스트 반환
        
        Args:
            prefix: S3 prefix (예: 'train/' 또는 'train/김치찌개/')
            extensions: 이미지 확장자 리스트 (기본값: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            이미지 키 리스트
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_keys = []
        
        try:
            # S3 객체 리스트 가져오기 (페이지네이션 처리)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    # 확장자 체크
                    if any(key.lower().endswith(ext.lower()) for ext in extensions):
                        image_keys.append(key)
            
            logger.info(f"'{prefix}' 하위에서 {len(image_keys)}개의 이미지 발견")
            return image_keys
            
        except Exception as e:
            logger.error(f"S3 리스트 조회 중 오류: {e}")
            return []
    
    def get_class_structure(self, base_prefix: str = '') -> Dict[str, List[str]]:
        """
        S3 버킷의 폴더 구조를 분석하여 클래스별 이미지 키를 반환
        
        구조 예시:
        train/
          김치찌개/
            001.jpg
            002.jpg
          비빔밥/
            001.jpg
            
        Args:
            base_prefix: 기본 prefix (예: 'train/' 또는 'val/')
            
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
    
    def check_bucket_access(self) -> bool:
        """
        S3 버킷 접근 권한 확인
        
        Returns:
            접근 가능하면 True, 아니면 False
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 버킷 '{self.bucket_name}' 접근 확인 완료")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"버킷 '{self.bucket_name}'을 찾을 수 없습니다.")
            elif error_code == '403':
                logger.error(f"버킷 '{self.bucket_name}'에 대한 접근 권한이 없습니다.")
            else:
                logger.error(f"버킷 접근 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"버킷 확인 중 오류: {e}")
            return False


def test_s3_connection(bucket_name: str, prefix: str = 'train/'):
    """
    S3 연결 테스트 함수
    
    Args:
        bucket_name: 테스트할 S3 버킷 이름
        prefix: 테스트할 prefix
    """
    print("="*70)
    print("🔍 S3 연결 테스트")
    print("="*70)
    
    # 로거 설정
    logging.basicConfig(level=logging.INFO)
    
    # S3 로더 초기화
    loader = S3ImageLoader(bucket_name=bucket_name)
    
    # 1. 버킷 접근 테스트
    print("\n1️⃣ 버킷 접근 테스트...")
    if not loader.check_bucket_access():
        print("❌ 버킷 접근 실패!")
        return
    print("✅ 버킷 접근 성공!")
    
    # 2. 클래스 구조 분석
    print(f"\n2️⃣ '{prefix}' 클래스 구조 분석...")
    class_structure = loader.get_class_structure(prefix)
    
    if not class_structure:
        print("❌ 클래스를 찾을 수 없습니다!")
        return
    
    print(f"✅ {len(class_structure)}개 클래스 발견:")
    for class_name, images in list(class_structure.items())[:5]:  # 처음 5개만 출력
        print(f"   - {class_name}: {len(images)}개 이미지")
    
    # 3. 샘플 이미지 로드 테스트
    print(f"\n3️⃣ 샘플 이미지 로드 테스트...")
    first_class = list(class_structure.keys())[0]
    first_image_key = class_structure[first_class][0]
    
    image = loader.load_image_from_s3(first_image_key)
    if image:
        print(f"✅ 이미지 로드 성공!")
        print(f"   키: {first_image_key}")
        print(f"   크기: {image.size}")
        print(f"   모드: {image.mode}")
    else:
        print("❌ 이미지 로드 실패!")
        return
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)


if __name__ == "__main__":
    # 사용 예시
    # test_s3_connection('your-bucket-name', 'train/')
    print("S3 연결 테스트를 실행하려면:")
    print("test_s3_connection('your-bucket-name', 'train/')")

