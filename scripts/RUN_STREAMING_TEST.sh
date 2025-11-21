#!/bin/bash
# 진짜 S3 스트리밍 테스트 (메모리만 사용, 디스크 저장 ❌)

echo "🌊 진짜 S3 스트리밍 학습 테스트"
echo "================================"
echo "💡 S3 → 메모리로 직접 로드 (디스크 사용 안 함)"
echo ""

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.." || exit

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo "📦 가상환경 활성화 중..."
    source venv/bin/activate
fi

echo "✅ 현재 디렉토리: $(pwd)"
echo ""

# 옵션 선택
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "선택하세요:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) 🌊 전체 스트리밍 테스트 (10장, 3 에포크) - 약 5분"
echo "2) ⚡ 초고속 테스트 (5장, 1 에포크) - 약 2분"
echo "3) 🔍 S3 연결만 테스트"
echo "4) 📦 스트리밍 데이터셋만 생성"
echo "5) 💾 기존 결과 S3 업로드만"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "번호 입력 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🌊 전체 스트리밍 테스트 시작..."
        echo "💡 S3 → 메모리 → GPU 직접 스트리밍"
        python scripts/test_s3_streaming_real.py --num-images 10 --epochs 3
        ;;
    2)
        echo ""
        echo "⚡ 초고속 테스트 시작..."
        python scripts/test_s3_streaming_real.py --num-images 5 --epochs 1
        ;;
    3)
        echo ""
        echo "🔍 S3 연결 테스트..."
        python scripts/test_s3_streaming_real.py --step 1
        ;;
    4)
        echo ""
        echo "📦 스트리밍 데이터셋 생성..."
        python scripts/test_s3_streaming_real.py --step 2 --num-images 10
        ;;
    5)
        echo ""
        echo "💾 S3 업로드..."
        python scripts/test_s3_streaming_real.py --step 4
        ;;
    *)
        echo "❌ 잘못된 입력입니다."
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 작업 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

