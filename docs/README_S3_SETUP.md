# 🎯 S3 기반 YOLO 학습 - 빠른 설정

## 📋 현재 상황

- **S3 구조**: `bucket_name/foods/{label}/_001.jpg`
- **목표**: `train/{label}/`, `val/{label}/`로 분할하여 학습
- **원칙**: Raw 데이터(`foods/`)는 그대로 유지

---

## 🚀 5분 설정 가이드

### 1️⃣ 패키지 설치 (1분)

```bash
pip install -r requirements.txt
```

### 2️⃣ AWS 설정 (2분)

```bash
# 환경 변수 파일 생성
cp env_template.txt .env

# .env 파일 편집 (메모장 또는 nano)
notepad .env
```

**입력할 내용:**

```env
AWS_ACCESS_KEY_ID=실제_Access_Key
AWS_SECRET_ACCESS_KEY=실제_Secret_Key
AWS_REGION=ap-northeast-2

S3_BUCKET_NAME=실제_버킷_이름

S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/

IMAGE_CACHE_SIZE=100
```

### 3️⃣ S3 데이터 분할 (5-30분, 데이터 양에 따라)

```bash
python prepares/split_s3_data.py
```

**이 스크립트가 하는 일:**

- `foods/{label}/` 구조를 스캔
- 80:20 비율로 train/val 분할
- S3 내에서 복사 (원본 유지)

### 4️⃣ 연결 테스트 (30초)

```bash
python test_s3_connection.py
```

### 5️⃣ 학습 시작! (수 시간)

```bash
python train_korean_food_s3_hybrid.py
```

---

## 📊 전체 구조

### Before (현재)

```
s3://your-bucket/
└── foods/
    ├── 김치찌개/
    │   ├── _001.jpg
    │   ├── _002.jpg
    │   └── ...
    ├── 비빔밥/
    └── ...
```

### After (3단계 실행 후)

```
s3://your-bucket/
├── foods/              ← 원본 유지
│   ├── 김치찌개/
│   ├── 비빔밥/
│   └── ...
├── train/              ← 새로 생성 (80%)
│   ├── 김치찌개/
│   ├── 비빔밥/
│   └── ...
└── val/                ← 새로 생성 (20%)
    ├── 김치찌개/
    ├── 비빔밥/
    └── ...
```

---

## 🔧 주요 파일 설명

### 설정 및 분할

- `env_template.txt`: 환경 변수 템플릿
- `prepares/split_s3_data.py`: S3 데이터를 train/val로 분할
- `test_s3_connection.py`: S3 연결 테스트

### 학습 스크립트

- `train_korean_food_s3_hybrid.py`: **추천!** 임시 디렉토리 사용
- `train_korean_food_s3.py`: 순수 스트리밍 방식

### 핵심 모듈

- `utils/s3_loader.py`: S3에서 이미지 로드
- `utils/s3_dataset.py`: PyTorch Dataset
- `config/s3_config.py`: 설정 관리

### 문서

- `SETUP_GUIDE_KR.md`: 자세한 설정 가이드
- `S3_TRAINING_GUIDE.md`: 전체 사용 가이드
- `QUICK_START_S3.md`: 빠른 시작

---

## ⚙️ 커스터마이즈

### 분할 비율 변경

`prepares/split_s3_data.py`:

```python
TRAIN_RATIO = 0.7  # 70:30으로 변경
```

### 학습 파라미터 변경

`train_korean_food_s3_hybrid.py`:

```python
EPOCHS = 10              # 에포크 증가
BATCH_SIZE = 8           # 배치 크기 증가
MODEL_SIZE = "11m"       # 더 큰 모델 사용
```

### 테스트용 소량 학습

`train_korean_food_s3_hybrid.py`:

```python
SAMPLE_LIMIT_PER_CLASS = 50  # 클래스당 50개만
```

---

## ❓ 문제 해결

### 문제: 자격증명 오류

```
NoCredentialsError
```

→ `.env` 파일의 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 확인

### 문제: 버킷을 찾을 수 없음

```
404 Not Found
```

→ `.env` 파일의 `S3_BUCKET_NAME` 확인

### 문제: foods/ 폴더가 없음

```
'foods/' 하위에 데이터가 없습니다
```

→ `.env` 파일의 `S3_RAW_PREFIX` 확인 (실제 구조에 맞게)

### 문제: GPU 메모리 부족

```
CUDA out of memory
```

→ `BATCH_SIZE`를 줄이기 (4 → 2)

---

## 📞 추가 도움말

**자세한 가이드:**

```bash
cat SETUP_GUIDE_KR.md      # 한글 상세 가이드
cat S3_TRAINING_GUIDE.md   # 영문 전체 가이드
cat QUICK_START_S3.md      # 빠른 시작
```

**테스트 명령:**

```bash
python test_s3_connection.py              # S3 연결 테스트
python -c "from config.s3_config import get_s3_config; get_s3_config().print_config()"  # 설정 확인
```

---

## 🎉 완료!

설정이 완료되었으면:

```bash
python train_korean_food_s3_hybrid.py
```

**즐거운 학습 되세요! 🚀**
