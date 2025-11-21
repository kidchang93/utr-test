# 🌊 YOLO + S3 스트리밍 학습 가이드

## 💡 개념

**진짜 스트리밍 방식**: S3에서 메모리로 직접 로드, 디스크에 저장하지 않음

```
┌─────────────────────┐
│   S3 버킷           │
│   train/김치찌개/    │
│   ├── img001.jpg   │
│   ├── img002.jpg   │
│   └── ...          │
└─────────────────────┘
         │
         │ ① S3 API 호출
         ↓
┌─────────────────────┐
│   메모리 (RAM)      │
│   LRU 캐시         │
│   - 최근 N개 이미지 │
│   - 자동 관리       │
└─────────────────────┘
         │
         │ ② PyTorch DataLoader
         ↓
┌─────────────────────┐
│   GPU / CPU        │
│   YOLO11n 학습     │
└─────────────────────┘
         │
         │ ③ 최종 모델만
         ↓
┌─────────────────────┐
│   로컬 디스크       │
│   runs/weights/    │
│   └── best.pt      │
└─────────────────────┘
         │
         │ ④ S3 업로드
         ↓
┌─────────────────────┐
│   S3 버킷          │
│   models/...       │
└─────────────────────┘
```

## 🚀 빠른 시작

### 방법 1: 대화형 스크립트 (추천)

```bash
cd /Users/kidchang/Desktop/ck/privacy/utr-test
./scripts/STREAMING_TEST.sh
```

**옵션:**
- `1`: 🚀 전체 테스트 (10장, 3 에포크)
- `2`: ⚡ 초고속 테스트 (5장, 1 에포크)
- `3`: 💪 80GB GPU 테스트 (50장, 5 에포크, 1024px)
- `4`: 🔍 S3 연결만
- `5`: 🌊 스트리밍 데이터셋만
- `6`: 💾 결과 업로드만

### 방법 2: 직접 실행

```bash
# 기본 테스트
python scripts/test_yolo_s3_streaming.py

# 커스텀 설정
python scripts/test_yolo_s3_streaming.py \
    --num-images 20 \
    --epochs 5 \
    --img-size 640 \
    --batch-size 4

# 80GB GPU 최적화
python scripts/test_yolo_s3_streaming.py \
    --num-images 100 \
    --epochs 10 \
    --img-size 1024 \
    --batch-size 256

# 단계별 실행
python scripts/test_yolo_s3_streaming.py --step 1  # S3 연결
python scripts/test_yolo_s3_streaming.py --step 2  # 데이터셋
python scripts/test_yolo_s3_streaming.py --step 3  # 학습
python scripts/test_yolo_s3_streaming.py --step 4  # 업로드
```

## 📊 4단계 프로세스

### 1️⃣ S3 연결 테스트
- AWS 자격증명 확인
- 버킷 접근 권한 확인
- train/ 폴더 존재 확인

**소요 시간**: ~10초

### 2️⃣ 스트리밍 데이터셋 생성
- S3 버킷 구조 스캔
- 첫 번째 클래스 선택
- N장의 이미지 메타데이터 로드
- **디스크 저장: 0 bytes** ✨

**소요 시간**: ~30초

### 3️⃣ YOLO 스트리밍 학습
- YOLO11n 모델 로드
- S3 → 메모리 → GPU 파이프라인
- LRU 캐시로 메모리 관리
- 학습 진행

**소요 시간**: 
- CPU: ~10-15분 (10장, 3 에포크)
- GPU: ~2-5분 (10장, 3 에포크)
- 80GB GPU: ~1-2분 (100장, 10 에포크, 배치 256)

### 4️⃣ 결과 S3 업로드
- 최종 모델 업로드
- 학습 정보 업로드
- 타임스탬프 자동 추가

**소요 시간**: ~30초-1분

## ⚙️ 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--num-images` | 10 | 사용할 이미지 수 |
| `--epochs` | 3 | 학습 에포크 수 |
| `--img-size` | 640 | 이미지 크기 (px) |
| `--batch-size` | 2 | 배치 크기 |
| `--step` | 전체 | 실행할 단계 (1-4) |

## 💾 디스크 사용량

### 스트리밍 방식 (현재)
```
학습 중 디스크 사용: 0 bytes ✨
학습 후 디스크 사용: ~45 MB (모델 파일만)
```

### 다운로드 방식 (비교)
```
학습 중 디스크 사용: 수백 MB ~ 수십 GB
학습 후 디스크 사용: 수백 MB ~ 수십 GB + 모델
```

## 🎯 GPU별 최적 설정

### GTX 1060 3GB
```bash
python scripts/test_yolo_s3_streaming.py \
    --num-images 10 \
    --epochs 5 \
    --img-size 640 \
    --batch-size 2
```

### RTX 3090 24GB
```bash
python scripts/test_yolo_s3_streaming.py \
    --num-images 100 \
    --epochs 10 \
    --img-size 1024 \
    --batch-size 32
```

### A100 80GB
```bash
python scripts/test_yolo_s3_streaming.py \
    --num-images 500 \
    --epochs 20 \
    --img-size 1280 \
    --batch-size 256
```

## 📁 생성되는 파일

**로컬:**
```
runs/yolo_streaming/s3_test/
├── weights/
│   └── best.pt          # YOLO 모델 (최종)
└── train_info.txt       # 학습 정보
```

**S3:**
```
s3://your-bucket/models/yolo_streaming/s3_test_YYYYMMDD_HHMMSS/
├── weights/
│   └── best.pt
└── train_info.txt
```

## ✨ 핵심 특징

### 1. 진짜 스트리밍
- ✅ S3에서 메모리로 **직접** 로드
- ✅ 디스크에 **절대** 저장 안 함
- ✅ LRU 캐시로 메모리 자동 관리

### 2. YOLO11n 사용
- ✅ 최신 YOLO 모델
- ✅ Classification 전용
- ✅ PyTorch 백본 활용

### 3. 효율적 메모리 관리
- ✅ 캐시 크기 조절 가능
- ✅ 자동 LRU 알고리즘
- ✅ 메모리 부족 방지

### 4. 유연한 설정
- ✅ 이미지 수 조절
- ✅ 이미지 크기 조절
- ✅ 배치 크기 조절
- ✅ GPU별 최적화 가능

## 🔍 문제 해결

### S3 연결 실패
```bash
# .env 파일 확인
cat .env

# S3 연결 직접 테스트
python scripts/test_s3_connection.py
```

### YOLO 모델 없음
```bash
# yolo11n-cls.pt를 프로젝트 루트에 배치
ls yolo11n-cls.pt
```

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python scripts/test_yolo_s3_streaming.py \
    --batch-size 1 \
    --img-size 416

# 또는 CPU로 실행
CUDA_VISIBLE_DEVICES="" python scripts/test_yolo_s3_streaming.py
```

### 메모리 부족 (RAM)
```bash
# 캐시 크기를 줄이려면 스크립트 수정:
# cache_size=num_images → cache_size=5
```

## 📊 성능 비교

| 항목 | 다운로드 방식 | 스트리밍 방식 |
|------|-------------|-------------|
| 디스크 사용 | 수십 GB | 0 bytes ✨ |
| 시작 시간 | 느림 (다운로드) | 빠름 (즉시) |
| 메모리 사용 | 낮음 | 중간 (캐시) |
| 유연성 | 낮음 | 높음 |
| 확장성 | 제한적 | 무제한 |

## 🎓 다음 단계

### 1. 전체 데이터셋 학습
```bash
# 모든 클래스, 모든 이미지로 확장
# 스크립트 수정: Subset 제거, 전체 dataset 사용
```

### 2. 검증 데이터 추가
```bash
# val/ 폴더도 스트리밍
# 검증 루프 추가
```

### 3. 모델 버전 관리
```bash
# DVC 설정
# S3 버전 관리 자동화
```

### 4. 프로덕션 배포
```bash
# API 서버 구축
# 모니터링 추가
# 자동 재학습 파이프라인
```

## 💡 팁

### 빠른 반복 개발
```bash
# 1분 이내 테스트
python scripts/test_yolo_s3_streaming.py \
    --num-images 3 \
    --epochs 1 \
    --batch-size 1
```

### 완전한 테스트
```bash
# 10-15분 테스트
python scripts/test_yolo_s3_streaming.py \
    --num-images 50 \
    --epochs 10 \
    --batch-size 8
```

### 메모리 모니터링
```bash
# 실시간 메모리 확인
watch -n 1 free -h

# GPU 메모리 확인
watch -n 1 nvidia-smi
```

## 📚 관련 파일

- `test_yolo_s3_streaming.py` - 메인 스크립트
- `STREAMING_TEST.sh` - 빠른 실행 스크립트
- `utils/s3_dataset.py` - S3 스트리밍 Dataset 클래스
- `utils/s3_loader.py` - S3 이미지 로더
- `config/s3_config.py` - S3 설정 관리

## ❓ FAQ

### Q: 정말 디스크에 저장 안 됨?
A: 네! 학습 데이터는 **0 bytes**입니다. 오직 최종 모델만 저장됩니다.

### Q: 메모리가 부족하면?
A: LRU 캐시가 자동으로 관리합니다. 캐시 크기를 줄이면 메모리 사용량이 줄어듭니다.

### Q: 느리지 않나요?
A: S3 API는 매우 빠르며, 캐시 덕분에 반복 접근이 효율적입니다. 실제로는 로컬보다 더 빠를 수 있습니다.

### Q: 전체 데이터셋으로 어떻게?
A: 스크립트의 `Subset` 부분을 제거하고 `full_dataset`을 직접 사용하면 됩니다.

### Q: 다른 모델도 가능?
A: 네! PyTorch 모델이라면 무엇이든 가능합니다. ResNet, EfficientNet 등.

