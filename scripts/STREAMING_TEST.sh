#!/bin/bash
# 🌊 YOLO + S3 진짜 스트리밍 학습 테스트

echo "🌊 YOLO + S3 스트리밍 학습 테스트"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 특징:"
echo "   ✨ S3 → 메모리 직접 로드 (디스크 저장 ❌)"
echo "   ✨ YOLO11n 모델 사용"
echo "   ✨ LRU 캐시로 메모리 관리"
echo "   ✨ 최종 모델만 디스크 저장"
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
echo "1) 🚀 전체 테스트 (10장, 3 에포크) - 약 5-10분"
echo "2) ⚡ 초고속 테스트 (5장, 1 에포크) - 약 2-3분"
echo "3) 💪 80GB GPU 테스트 (50장, 5 에포크, 1024px, batch=16)"
echo "4) 🔍 S3 연결만 테스트 - 약 10초"
echo "5) 🌊 스트리밍 데이터셋만 테스트 - 약 30초"
echo "6) 💾 기존 결과 S3 업로드만"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "번호 입력 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 전체 스트리밍 테스트 시작..."
        python scripts/test_yolo_s3_streaming.py \
            --num-images 10 \
            --epochs 3 \
            --img-size 640 \
            --batch-size 2
        ;;
    2)
        echo ""
        echo "⚡ 초고속 스트리밍 테스트 시작..."
        python scripts/test_yolo_s3_streaming.py \
            --num-images 5 \
            --epochs 1 \
            --img-size 640 \
            --batch-size 2
        ;;
    3)
        echo ""
        echo "💪 80GB GPU 최적화 테스트 시작..."
        echo "   - 이미지: 50장"
        echo "   - 에포크: 5"
        echo "   - 이미지 크기: 1024px"
        echo "   - 배치 크기: 16"
        echo ""
        python scripts/test_yolo_s3_streaming.py \
            --num-images 50 \
            --epochs 5 \
            --img-size 1024 \
            --batch-size 16
        ;;
    4)
        echo ""
        echo "🔍 S3 연결 테스트 시작..."
        python scripts/test_yolo_s3_streaming.py --step 1
        ;;
    5)
        echo ""
        echo "🌊 스트리밍 데이터셋 테스트 시작..."
        python scripts/test_yolo_s3_streaming.py --step 2 --num-images 10
        ;;
    6)
        echo ""
        echo "💾 S3 업로드 시작..."
        python scripts/test_yolo_s3_streaming.py --step 4
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

