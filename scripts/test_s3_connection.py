"""
S3 연결 및 설정 테스트 스크립트
학습 전에 S3 연결이 제대로 되는지 확인
"""
import sys
import logging
from pathlib import Path

# scripts 디렉터리에서 최상위 패키지까지 경로를 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.s3_config import get_s3_config
from utils.s3_loader import S3ImageLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """S3 연결 테스트 메인 함수"""
    
    print("\n" + "="*70)
    print("🔍 S3 연결 테스트")
    print("="*70 + "\n")
    
    # ==================== 1. 설정 파일 확인 ====================
    print("1️⃣ .env 설정 파일 확인 중...")
    
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
        print("✅ 설정 파일 로드 성공!\n")
    except ValueError as e:
        logger.error(f"❌ 설정 오류: {e}")
        logger.info("\n💡 해결 방법:")
        logger.info("   1. .env.example을 .env로 복사")
        logger.info("   2. .env 파일에서 AWS 정보 입력")
        return False
    
    # ==================== 2. S3 버킷 접근 확인 ====================
    print("\n2️⃣ S3 버킷 접근 테스트...")
    
    try:
        s3_params = s3_config.get_s3_params()
        s3_loader = S3ImageLoader(
            bucket_name=s3_params['bucket_name'],
            access_key=s3_params.get('access_key'),
            secret_key=s3_params.get('secret_key'),
            region=s3_params['region'],
            domain=s3_params.get('domain')
        )
        
        if not s3_loader.check_bucket_access():
            logger.error("❌ S3 버킷 접근 실패!")
            logger.info("\n💡 확인 사항:")
            logger.info("   1. 버킷 이름이 정확한지 확인")
            logger.info("   2. AWS 자격증명이 올바른지 확인")
            logger.info("   3. 버킷에 대한 읽기 권한이 있는지 확인")
            return False
        
        print("✅ S3 버킷 접근 성공!\n")
        
    except Exception as e:
        logger.error(f"❌ S3 연결 오류: {e}")
        return False
    
    # ==================== 3. 학습 데이터 구조 확인 ====================
    print("\n3️⃣ 학습 데이터 구조 확인...")
    
    try:
        # Train 데이터
        print(f"\n📁 {s3_config.train_prefix}")
        train_structure = s3_loader.get_class_structure(s3_config.train_prefix)
        
        if not train_structure:
            logger.error(f"❌ '{s3_config.train_prefix}' 하위에 데이터가 없습니다!")
            logger.info("\n💡 먼저 prepares/prepare_data_s3.py를 실행하여 데이터를 업로드하세요.")
            return False
        
        train_total = sum(len(imgs) for imgs in train_structure.values())
        print(f"   클래스: {len(train_structure)}개")
        print(f"   이미지: {train_total:,}장")
        
        # 처음 5개 클래스 출력
        for i, (class_name, images) in enumerate(list(train_structure.items())[:5], 1):
            print(f"      {i}. {class_name}: {len(images)}장")
        
        if len(train_structure) > 5:
            print(f"      ... 외 {len(train_structure) - 5}개 클래스")
        
        # Val 데이터
        print(f"\n📁 {s3_config.val_prefix}")
        val_structure = s3_loader.get_class_structure(s3_config.val_prefix)
        
        if not val_structure:
            logger.warning(f"⚠️  '{s3_config.val_prefix}' 하위에 데이터가 없습니다!")
        else:
            val_total = sum(len(imgs) for imgs in val_structure.values())
            print(f"   클래스: {len(val_structure)}개")
            print(f"   이미지: {val_total:,}장")
        
        print("\n✅ 데이터 구조 확인 완료!\n")
        
    except Exception as e:
        logger.error(f"❌ 데이터 구조 확인 실패: {e}")
        return False
    
    # ==================== 4. 샘플 이미지 로드 테스트 ====================
    print("\n4️⃣ 샘플 이미지 로드 테스트...")
    
    try:
        # 첫 번째 클래스의 첫 번째 이미지
        first_class = list(train_structure.keys())[0]
        first_image_key = train_structure[first_class][0]
        
        print(f"   로딩 중: {first_image_key}")
        image = s3_loader.load_image_from_s3(first_image_key)
        
        if image is None:
            logger.error("❌ 이미지 로드 실패!")
            return False
        
        print(f"   ✅ 로드 성공!")
        print(f"      크기: {image.size}")
        print(f"      모드: {image.mode}")
        print(f"      포맷: {image.format if hasattr(image, 'format') else 'PIL Image'}")
        
    except Exception as e:
        logger.error(f"❌ 이미지 로드 실패: {e}")
        return False
    
    # ==================== 5. 최종 결과 ====================
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)
    
    print("\n📝 다음 단계:")
    print("\n1. 하이브리드 방식 학습 (추천):")
    print("   python train_korean_food_s3_hybrid.py")
    
    print("\n2. 순수 스트리밍 방식 학습:")
    print("   python train_korean_food_s3.py")
    
    print("\n💡 자세한 사용법은 S3_TRAINING_GUIDE.md를 참고하세요.")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

