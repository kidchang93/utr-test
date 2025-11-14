"""
S3 학습 설정 파일
환경 변수를 읽어서 S3 설정을 제공
"""
import os
from pathlib import Path
from typing import Optional
from decouple import config as env_config, UndefinedValueError


class S3Config:
    """S3 설정 관리 클래스"""
    
    def __init__(self):
        """환경 변수에서 S3 설정 로드"""
        
        # AWS 자격증명 (선택사항 - IAM Role 사용 가능)
        self.access_key: Optional[str] = env_config('ACCESS_KEY', default=None)
        self.secret_key: Optional[str] = env_config('SECRET_KEY', default=None)
        self.region: str = env_config('REGION', default='ap-northeast-2')
        
        # S3 엔드포인트 (선택사항 - 커스텀 S3 호환 스토리지용)
        self.domain: Optional[str] = env_config('DOMAIN', default=None)
        if self.domain and self.domain.strip() == '':
            self.domain = None
        
        # S3 버킷 정보 (필수)
        try:
            self.bucket_name: str = env_config('BUCKET_NAME')
        except UndefinedValueError:
            raise ValueError(
                "BUCKET_NAME 환경 변수가 설정되지 않았습니다.\n"
                ".env 파일을 생성하거나 환경 변수를 설정하세요."
            )
        
        # S3 Prefix
        self.train_prefix: str = env_config('S3_TRAIN_PREFIX', default='train/')
        self.val_prefix: str = env_config('S3_VAL_PREFIX', default='val/')
        self.raw_prefix: str = env_config('S3_RAW_PREFIX', default='foods/')  # Raw 데이터 경로
        
        # 캐시 설정
        self.cache_size: int = int(env_config('IMAGE_CACHE_SIZE', default=100))
    
    def get_s3_params(self) -> dict:
        """S3 연결 파라미터 딕셔너리 반환"""
        return {
            'bucket_name': self.bucket_name,
            'access_key': self.access_key,
            'secret_key': self.secret_key,
            'region': self.region,
            'domain': self.domain,
            'cache_size': self.cache_size
        }
    
    def print_config(self):
        """설정 정보 출력 (보안 정보는 마스킹)"""
        print("="*70)
        print("🔧 S3 설정 정보")
        print("="*70)
        
        # 자격증명 마스킹
        if self.access_key:
            masked_key = self.access_key[:4] + '*' * (len(self.access_key) - 8) + self.access_key[-4:]
        else:
            masked_key = "IAM Role 사용"
        
        if self.secret_key:
            masked_secret = '*' * len(self.secret_key)
        else:
            masked_secret = "IAM Role 사용"
        
        print(f"Access Key: {masked_key}")
        print(f"Secret Key: {masked_secret}")
        print(f"Region: {self.region}")
        
        if self.domain:
            print(f"Domain: {self.domain}")
        else:
            print(f"Domain: AWS S3 기본")
        
        print(f"\nS3 버킷: {self.bucket_name}")
        print(f"Raw Prefix: {self.raw_prefix}")
        print(f"학습 Prefix: {self.train_prefix}")
        print(f"검증 Prefix: {self.val_prefix}")
        print(f"\n캐시 크기: {self.cache_size}개")
        print("="*70)
    
    @staticmethod
    def create_example_env():
        """예제 .env 파일 생성"""
        env_example = """# AWS S3 설정
ACCESS_KEY=your_access_key_here
SECRET_KEY=your_secret_key_here
REGION=ap-northeast-2

# S3 버킷 정보
BUCKET_NAME=your-bucket-name

# S3 엔드포인트 (선택사항)
DOMAIN=

# S3 경로 설정
S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/

# 캐시 설정
IMAGE_CACHE_SIZE=100
"""
        env_path = Path('.env')
        if env_path.exists():
            print("⚠️ .env 파일이 이미 존재합니다.")
            return False
        
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_example)
        
        print("✅ .env 파일이 생성되었습니다.")
        print("   파일을 열어서 실제 값으로 수정하세요.")
        return True


# 글로벌 설정 인스턴스
_config_instance: Optional[S3Config] = None


def get_s3_config() -> S3Config:
    """S3 설정 싱글톤 인스턴스 반환"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = S3Config()
    
    return _config_instance


if __name__ == "__main__":
    # 테스트
    try:
        config = S3Config()
        config.print_config()
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        print("\n.env 파일 예제를 생성하시겠습니까? (yes/no): ", end='')
        response = input().strip().lower()
        if response == 'yes':
            S3Config.create_example_env()

