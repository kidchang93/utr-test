# 🔄 S3 스트리밍 학습 워크플로우

## 📊 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                     S3 버킷                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  train/                                                │  │
│  │  ├── 김치찌개/                                          │  │
│  │  │   ├── img001.jpg                                    │  │
│  │  │   ├── img002.jpg                                    │  │
│  │  │   └── ...                                           │  │
│  │  ├── 된장찌개/                                          │  │
│  │  └── ...                                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ ① S3 연결 테스트
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               로컬 서버 (버퍼 영역)                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  temp_train_test/  ◀── ② 최소 데이터 다운로드         │  │
│  │  └── train/                                            │  │
│  │      └── 김치찌개/                                      │  │
│  │          ├── img001.jpg (10장만)                       │  │
│  │          └── ...                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         │ ③ YOLO 학습                       │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  YOLO11n-cls 모델                                      │  │
│  │  - 이미지 크기: 640px                                  │  │
│  │  - 배치 크기: 2                                        │  │
│  │  - 에포크: 3                                           │  │
│  │  - GPU/CPU 자동 선택                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  runs/classify/s3_test/                                │  │
│  │  ├── weights/                                          │  │
│  │  │   ├── best.pt   ◀── 최고 성능 모델                 │  │
│  │  │   └── last.pt                                      │  │
│  │  ├── results.png   ◀── 학습 곡선                       │  │
│  │  └── confusion_matrix.png                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ ④ 결과 업로드
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     S3 버킷                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  models/runs/s3_test_20251114_153022/                 │  │
│  │  ├── weights/                                          │  │
│  │  │   ├── best.pt   ◀── 모델 파일 버전 관리            │  │
│  │  │   └── last.pt                                      │  │
│  │  ├── results.png                                       │  │
│  │  └── confusion_matrix.png                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 단계별 설명

### 1️⃣ S3 연결 테스트
```python
# AWS 자격증명 확인
# 버킷 접근 권한 확인
# train/ 폴더 존재 확인
```
**소요 시간:** ~10초

### 2️⃣ 최소 데이터셋 다운로드
```python
# 첫 번째 클래스 찾기
# 10장만 임시 다운로드
# temp_train_test/ 폴더에 저장
```
**소요 시간:** ~30초 (이미지 크기에 따라)

### 3️⃣ YOLO 학습
```python
# YOLO11n-cls 모델 로드
# 3 에포크 학습
# GPU 사용 가능 시 자동 활성화
# 학습 결과 runs/ 폴더에 저장
```
**소요 시간:** 
- GPU: ~2-5분
- CPU: ~10-15분

### 4️⃣ 결과 S3 업로드
```python
# runs/ 폴더 전체를 S3로
# 타임스탬프가 포함된 경로로 업로드
# models/runs/s3_test_YYYYMMDD_HHMMSS/
```
**소요 시간:** ~30초-1분

## 🚀 실행 방법

### 방법 1: 대화형 스크립트 (추천)
```bash
./scripts/QUICK_TEST.sh
```

### 방법 2: 직접 실행
```bash
# 전체 테스트
python scripts/test_s3_train_minimal.py

# 커스텀 설정
python scripts/test_s3_train_minimal.py --num-images 20 --epochs 5

# 단계별 실행
python scripts/test_s3_train_minimal.py --step 1  # S3 연결만
python scripts/test_s3_train_minimal.py --step 2  # 다운로드만
python scripts/test_s3_train_minimal.py --step 3  # 학습만
python scripts/test_s3_train_minimal.py --step 4  # 업로드만
```

## 📁 파일 구조

```
utr-test/
├── scripts/
│   ├── test_s3_train_minimal.py  ◀── 메인 테스트 스크립트
│   ├── QUICK_TEST.sh             ◀── 빠른 실행 스크립트
│   ├── TEST_S3_TRAIN.md          ◀── 상세 가이드
│   └── WORKFLOW.md               ◀── 이 문서
├── config/
│   └── s3_config.py              ◀── S3 설정
├── .env                          ◀── AWS 자격증명 (필수)
├── yolo11n-cls.pt                ◀── YOLO 모델 (필수)
├── temp_train_test/              ◀── 임시 폴더 (자동 생성/삭제)
└── runs/                         ◀── 학습 결과
    └── classify/
        └── s3_test/
```

## ⚙️ 환경 설정

### 1. .env 파일 생성
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your-bucket-name
S3_REGION=ap-northeast-2
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/
```

### 2. S3 버킷 준비
```bash
# 버킷 구조 확인
aws s3 ls s3://your-bucket-name/train/

# 예상 출력:
#   PRE 김치찌개/
#   PRE 된장찌개/
#   ...
```

### 3. YOLO 모델 준비
```bash
# 프로젝트 루트에 yolo11n-cls.pt 파일 필요
ls yolo11n-cls.pt
```

## 🎨 버퍼 방식 스트리밍

### 특징
- ✅ **임시 다운로드**: 학습할 데이터만 선택적으로 다운로드
- ✅ **자동 정리**: 학습 완료 후 임시 파일 자동 삭제
- ✅ **메모리 효율**: 전체 데이터셋을 메모리에 올리지 않음
- ✅ **유연한 설정**: 이미지 수, 에포크 수 조절 가능

### 전체 학습 시 확장
```python
# 전체 데이터셋으로 확장 (추후 구현)
# 배치 단위로 다운로드 → 학습 → 삭제 → 반복

for batch in all_classes:
    download_batch(batch)
    train_on_batch(batch)
    cleanup_batch(batch)
    upload_checkpoint()
```

## 📊 성능 최적화

### GPU 설정 (80GB VRAM)
```python
# scripts/test_s3_train_minimal.py 수정
BATCH_SIZE = 256   # 기본 2에서 증가
IMG_SIZE = 1024    # 기본 640에서 증가
NUM_IMAGES = 100   # 더 많은 이미지로 테스트
```

### 빠른 반복 개발
```bash
# 초고속 테스트 (1분 이내)
python scripts/test_s3_train_minimal.py --num-images 3 --epochs 1
```

## 🔍 모니터링

### 학습 진행 상황
```bash
# 실시간 로그 확인
tail -f runs/classify/s3_test/results.txt

# GPU 사용률 확인 (CUDA 사용 시)
watch -n 1 nvidia-smi
```

### S3 업로드 확인
```bash
# 업로드된 모델 확인
aws s3 ls s3://your-bucket-name/models/runs/ --recursive
```

## 💡 다음 단계

### 1. 전체 데이터셋 학습
- 모든 클래스 포함
- 더 많은 에포크
- 검증 데이터 추가

### 2. 모델 버전 관리
- DVC 설정
- 모델 메타데이터 추가
- 성능 메트릭 기록

### 3. 프로덕션 배포
- 모델 서빙 설정
- API 엔드포인트 구축
- 모니터링 대시보드

## 🐛 문제 해결

### S3 연결 실패
```bash
# 자격증명 확인
python scripts/test_s3_connection.py

# 버킷 권한 확인
aws s3 ls s3://your-bucket-name/
```

### 학습 실패
```bash
# 로그 확인
cat runs/classify/s3_test/train.log

# GPU 메모리 확인
nvidia-smi
```

### 업로드 실패
```bash
# S3 쓰기 권한 확인
aws s3 cp test.txt s3://your-bucket-name/test.txt
```

## 📚 관련 문서

- [TEST_S3_TRAIN.md](TEST_S3_TRAIN.md) - 상세 가이드
- [../docs/S3_TRAINING_GUIDE.md](../docs/S3_TRAINING_GUIDE.md) - S3 학습 전체 가이드
- [../docs/QUICK_START_S3.md](../docs/QUICK_START_S3.md) - 빠른 시작 가이드

