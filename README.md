# 🍜 한국 음식 YOLO 분류 - S3 스트리밍 학습

AWS S3에서 실시간으로 데이터를 스트리밍하여 YOLO 모델을 학습하는 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-green.svg)](https://github.com/ultralytics/ultralytics)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange.svg)](https://aws.amazon.com/s3/)

---

## 🚀 빠른 시작 (4단계)

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 환경 설정
cp env_template.txt .env
notepad .env  # AWS 자격증명 입력

# 3. S3 데이터 분할 (최초 1회)
python prepares/split_s3_data.py

# 4. 학습 시작
python scripts/train_korean_food_s3_hybrid.py
```

**더 자세한 가이드**: [docs/SETUP_GUIDE_KR.md](docs/SETUP_GUIDE_KR.md)

---

## 📊 현재 S3 구조

```
s3://your-bucket/
└── foods/              # Raw 데이터 (원본)
    ├── 김치찌개/
    │   ├── _001.jpg
    │   └── ...
    ├── 비빔밥/
    └── ... (150개 클래스)
```

**분할 후:**

```
s3://your-bucket/
├── foods/              # 원본 유지
├── train/              # 80% 학습 데이터
└── val/                # 20% 검증 데이터
```

---

## 📁 프로젝트 구조

```
utr-test/
├── config/                 # 설정 모듈
│   └── s3_config.py
├── utils/                  # S3 로더 & Dataset
│   ├── s3_loader.py
│   └── s3_dataset.py
├── prepares/               # 데이터 준비 스크립트
│   ├── split_s3_data.py       # 최초 분할
│   └── update_s3_data.py      # 증분 업데이트 ⭐
├── scripts/                # 학습 스크립트
│   ├── train_korean_food_s3_hybrid.py  # 추천!
│   ├── train_korean_food_s3.py
│   └── test_s3_connection.py
├── docs/                   # 문서
│   ├── SETUP_GUIDE_KR.md      # 상세 가이드
│   ├── QUICK_START_S3.md      # 빠른 시작
│   └── ...
├── env_template.txt        # 환경 변수 템플릿
├── requirements.txt
└── README.md               # 이 파일
```

---

## 🎯 주요 기능

### 1. 실시간 S3 스트리밍

- 서버 디스크 사용량 99% 감소 (10GB → 50MB)
- 배치 단위로 S3에서 직접 로드
- 학습 후 자동 정리

### 2. 자동 Train/Val 분할

```bash
python prepares/split_s3_data.py
```

- `foods/{label}/` → `train/{label}/`, `val/{label}/`
- 80:20 비율 자동 분할
- 원본 데이터 유지

### 3. 증분 업데이트 ⭐ NEW!

```bash
python prepares/update_s3_data.py
```

**언제 사용?**

- `foods/새로운클래스/` 추가 시
- 기존 클래스에 이미지 추가 시
- 전체 재분할 없이 업데이트

**동작:**

1. `foods/`의 신규 클래스 감지 → train/val로 분할하여 추가
2. 기존 클래스의 신규 이미지 감지 → 기존 비율 유지하며 추가
3. 기존 train/val 데이터는 그대로 유지

**예시:**

```bash
# foods/에 새 클래스 "떡볶이" 추가됨
# foods/김치찌개/에 이미지 50개 추가됨

python prepares/update_s3_data.py

# 결과:
# train/떡볶이/ 생성 (80%)
# val/떡볶이/ 생성 (20%)
# train/김치찌개/에 40개 추가
# val/김치찌개/에 10개 추가
```

---

## 📚 문서

### 한글 가이드

- **[docs/SETUP_GUIDE_KR.md](docs/SETUP_GUIDE_KR.md)** - 상세 설정 가이드
- **[docs/README_S3_SETUP.md](docs/README_S3_SETUP.md)** - 5분 빠른 설정

### 영문 가이드

- **[docs/QUICK_START_S3.md](docs/QUICK_START_S3.md)** - Quick start
- **[docs/S3_TRAINING_GUIDE.md](docs/S3_TRAINING_GUIDE.md)** - Full guide
- **[docs/S3_MIGRATION_SUMMARY.md](docs/S3_MIGRATION_SUMMARY.md)** - Migration summary

---

## ⚙️ 환경 설정

### .env 파일

```env
# AWS 자격증명
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-northeast-2

# S3 버킷
S3_BUCKET_NAME=your-bucket-name

# S3 경로
S3_RAW_PREFIX=foods/     # Raw 데이터
S3_TRAIN_PREFIX=train/   # 학습 데이터
S3_VAL_PREFIX=val/       # 검증 데이터

# 캐시
IMAGE_CACHE_SIZE=100
```

---

## 🔄 워크플로우

### 최초 설정

```bash
1. pip install -r requirements.txt
2. .env 파일 설정
3. python prepares/split_s3_data.py
4. python scripts/test_s3_connection.py
5. python scripts/train_korean_food_s3_hybrid.py
```

### 데이터 추가 시

```bash
# S3의 foods/에 새 데이터 업로드 후
python prepares/update_s3_data.py

# 재학습
python scripts/train_korean_food_s3_hybrid.py
```

---

## 💡 학습 팁

### GPU 메모리에 맞게 조정

`scripts/train_korean_food_s3_hybrid.py`:

```python
BATCH_SIZE = 4      # 3GB GPU → 2, 6GB → 8
MODEL_SIZE = "11n"  # 11n, 11s, 11m, 11l, 11x
EPOCHS = 5          # 에포크 수
```

### 테스트용 소량 학습

```python
SAMPLE_LIMIT_PER_CLASS = 50  # 클래스당 50개만
```

---

## ❓ 문제 해결

### 자격증명 오류

```
NoCredentialsError
```

→ `.env` 파일의 AWS 키 확인

### 버킷 접근 오류

```
403 Forbidden
```

→ IAM 정책에서 S3 읽기/쓰기 권한 확인

### 데이터 없음

```
'train/' 하위에 데이터가 없습니다
```

→ `python prepares/split_s3_data.py` 실행

### GPU 메모리 부족

```
CUDA out of memory
```

→ `BATCH_SIZE` 줄이기 (4 → 2)

**더 자세한 내용**: [docs/SETUP_GUIDE_KR.md](docs/SETUP_GUIDE_KR.md)

---

## 💰 비용 (예시)

**10GB 데이터, 서울 리전:**

- 스토리지: $0.23/월
- 데이터 전송 (5 에포크): $4.5
- 요청 비용: ~$0.2
- **총 약 $5 이하**

---

## 🛡️ 보안

- ⚠️ `.env` 파일은 Git에 커밋하지 마세요
- ⚠️ 최소 권한 IAM 정책 사용
- ⚠️ S3 버킷을 public으로 설정하지 마세요

---

## 📞 지원

문제가 있으면:

1. `python scripts/test_s3_connection.py` 실행
2. [docs/SETUP_GUIDE_KR.md](docs/SETUP_GUIDE_KR.md) 참고
3. 로그 확인

---

**즐거운 딥러닝 되세요! 🚀**
