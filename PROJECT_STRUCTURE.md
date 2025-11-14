# 📁 프로젝트 구조

정리된 프로젝트 구조 및 파일 설명

---

## 🌳 디렉토리 구조

```
utr-test/
│
├── 📄 README.md                    # 프로젝트 메인 README
├── 📄 PROJECT_STRUCTURE.md         # 이 파일 (구조 설명)
├── 📄 requirements.txt             # Python 의존성
├── 📄 env_template.txt             # 환경 변수 템플릿
├── 📄 .env                         # 실제 환경 변수 (생성 필요, Git 제외)
├── 📄 .gitignore                   # Git 제외 목록
├── 📄 main.py                      # FastAPI 메인 (향후 확장용)
│
├── 📁 config/                      # 설정 모듈
│   ├── __init__.py
│   └── s3_config.py                # S3 설정 클래스
│
├── 📁 utils/                       # 유틸리티 모듈
│   ├── __init__.py
│   ├── s3_loader.py                # S3 이미지 로더
│   └── s3_dataset.py               # PyTorch Dataset
│
├── 📁 prepares/                    # 데이터 준비 스크립트
│   ├── README.md                   # 스크립트 가이드
│   ├── split_s3_data.py            # ⭐ 최초 train/val 분할
│   ├── update_s3_data.py           # ⭐ 증분 업데이트 (NEW!)
│   ├── prepare_data_s3.py          # 로컬 → S3 업로드
│   └── prepare_data.py             # 로컬 데이터 준비 (레거시)
│
├── 📁 scripts/                     # 학습 및 테스트 스크립트
│   ├── README.md                   # 스크립트 가이드
│   ├── train_korean_food_s3_hybrid.py  # ⭐ 하이브리드 학습 (추천!)
│   ├── train_korean_food_s3.py     # 순수 스트리밍 학습
│   ├── train_korean_food.py        # 로컬 학습 (레거시)
│   ├── test_s3_connection.py       # S3 연결 테스트
│   └── yolo_test.py                # YOLO 기본 테스트
│
├── 📁 docs/                        # 문서 모음
│   ├── README.md                   # 문서 인덱스
│   ├── SETUP_GUIDE_KR.md           # ⭐ 상세 설정 가이드 (한글)
│   ├── README_S3_SETUP.md          # 빠른 설정 (한글)
│   ├── QUICK_START_S3.md           # 빠른 시작
│   ├── S3_TRAINING_GUIDE.md        # 전체 가이드 (영문)
│   └── S3_MIGRATION_SUMMARY.md     # 마이그레이션 요약
│
└── 📁 runs/                        # 학습 결과 (Git 제외)
    └── classify/
        └── s3_korean_food/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            └── ...
```

---

## 📋 주요 파일 설명

### 🔧 설정 파일

#### `env_template.txt`

환경 변수 템플릿. 이 파일을 `.env`로 복사하여 사용.

```bash
cp env_template.txt .env
```

#### `.env` (생성 필요)

실제 AWS 자격증명과 S3 설정을 포함. **Git에 절대 커밋하지 마세요!**

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your-bucket
S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/
```

#### `config/s3_config.py`

S3 설정을 관리하는 Python 클래스. `.env` 파일을 읽어서 사용.

```python
from config.s3_config import get_s3_config
config = get_s3_config()
```

---

### 📦 데이터 준비 (prepares/)

#### `split_s3_data.py` ⭐

**용도:** S3의 `foods/{label}/` → `train/{label}/`, `val/{label}/` 최초 분할

**실행:** `python prepares/split_s3_data.py`

**언제:** 처음 설정할 때 (1회)

#### `update_s3_data.py` ⭐ NEW!

**용도:** foods/에 추가된 데이터만 train/val로 증분 업데이트

**실행:** `python prepares/update_s3_data.py`

**언제:** 데이터가 추가될 때마다

#### `prepare_data_s3.py`

**용도:** 로컬 데이터를 S3로 업로드

**실행:** `python prepares/prepare_data_s3.py`

**언제:** 로컬 데이터가 있고 S3가 비어있을 때

---

### 🚀 학습 스크립트 (scripts/)

#### `train_korean_food_s3_hybrid.py` ⭐ 추천!

**용도:** S3 → 임시 디렉토리 → 학습 → 자동 정리

**실행:** `python scripts/train_korean_food_s3_hybrid.py`

**장점:** 안정적, 빠름, YOLO 모든 기능 사용 가능

#### `train_korean_food_s3.py`

**용도:** S3에서 배치마다 실시간 스트리밍 학습

**실행:** `python scripts/train_korean_food_s3.py`

**장점:** 디스크 사용 최소화

#### `test_s3_connection.py`

**용도:** S3 연결 및 데이터 구조 테스트

**실행:** `python scripts/test_s3_connection.py`

**언제:** 설정 후, 문제 발생 시

---

### 📚 문서 (docs/)

#### `SETUP_GUIDE_KR.md` ⭐ 추천!

가장 상세한 한글 설정 가이드. 처음 사용자에게 권장.

#### `README_S3_SETUP.md`

5분 빠른 설정 가이드 (한글).

#### `QUICK_START_S3.md`

빠른 시작 가이드 (영한 혼용).

#### `S3_TRAINING_GUIDE.md`

전체 시스템 가이드 (영문).

#### `S3_MIGRATION_SUMMARY.md`

로컬 → S3 마이그레이션 요약 (영문).

---

## 🎯 사용 흐름

### 최초 설정

```
1. env_template.txt → .env 복사 및 편집
2. python prepares/split_s3_data.py
3. python scripts/test_s3_connection.py
4. python scripts/train_korean_food_s3_hybrid.py
```

### 데이터 추가 시

```
1. S3의 foods/에 새 데이터 업로드
2. python prepares/update_s3_data.py
3. python scripts/train_korean_food_s3_hybrid.py
```

### 문제 발생 시

```
1. python scripts/test_s3_connection.py
2. 로그 확인
3. docs/SETUP_GUIDE_KR.md의 "문제 해결" 참고
```

---

## 📊 S3 데이터 구조

### 현재 (Raw)

```
s3://your-bucket/
└── foods/              # 원본 데이터
    ├── 김치찌개/
    │   ├── _001.jpg
    │   └── ...
    ├── 비빔밥/
    └── ...
```

### split_s3_data.py 실행 후

```
s3://your-bucket/
├── foods/              # 원본 유지
├── train/              # 80% 학습 데이터
│   ├── 김치찌개/
│   ├── 비빔밥/
│   └── ...
└── val/                # 20% 검증 데이터
    ├── 김치찌개/
    ├── 비빔밥/
    └── ...
```

### 새 데이터 추가 + update_s3_data.py 실행 후

```
s3://your-bucket/
├── foods/              # 원본 + 새 데이터
│   ├── 김치찌개/       (기존 + 신규)
│   ├── 비빔밥/         (기존)
│   └── 떡볶이/         (신규 클래스)
├── train/              # 기존 + 신규
│   ├── 김치찌개/       (기존 + 신규의 80%)
│   ├── 비빔밥/         (기존)
│   └── 떡볶이/         (신규의 80%)
└── val/                # 기존 + 신규
    ├── 김치찌개/       (기존 + 신규의 20%)
    ├── 비빔밥/         (기존)
    └── 떡볶이/         (신규의 20%)
```

---

## 🔄 증분 업데이트 시나리오

### 시나리오 1: 새 클래스 추가

```bash
# 1. foods/에 새 클래스 업로드
aws s3 cp local/떡볶이/ s3://bucket/foods/떡볶이/ --recursive

# 2. 증분 업데이트
python prepares/update_s3_data.py
# → train/떡볶이/ 자동 생성 (80%)
# → val/떡볶이/ 자동 생성 (20%)

# 3. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 2: 기존 클래스에 이미지 추가

```bash
# 1. foods/김치찌개/에 이미지 추가
aws s3 cp new_images/ s3://bucket/foods/김치찌개/ --recursive

# 2. 증분 업데이트
python prepares/update_s3_data.py
# → train/김치찌개/에 신규 이미지의 80% 추가
# → val/김치찌개/에 신규 이미지의 20% 추가

# 3. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 3: 여러 클래스 동시 업데이트

```bash
# 1. foods/에 여러 변경사항 업로드
# - foods/떡볶이/ (새 클래스)
# - foods/순대/ (새 클래스)
# - foods/김치찌개/에 이미지 50개 추가

# 2. 한 번에 증분 업데이트
python prepares/update_s3_data.py
# → 모든 변경사항을 자동 감지하여 업데이트

# 3. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

---

## 🛠️ 모듈 설명

### `utils/s3_loader.py`

S3에서 이미지를 메모리로 직접 로드하는 클래스.

```python
from utils.s3_loader import S3ImageLoader

loader = S3ImageLoader(bucket_name='my-bucket')
image = loader.load_image_from_s3('train/김치찌개/_001.jpg')
```

### `utils/s3_dataset.py`

PyTorch Dataset 인터페이스로 S3 데이터를 사용.

```python
from utils.s3_dataset import S3ClassificationDataset

dataset = S3ClassificationDataset(
    bucket_name='my-bucket',
    prefix='train/',
    cache_size=100
)
```

---

## 📝 파일 네이밍 규칙

### Python 스크립트

- `train_*.py` - 학습 관련
- `test_*.py` - 테스트 관련
- `prepare_*.py` - 데이터 준비 관련
- `*_s3*.py` - S3 관련 기능

### 문서

- `*_KR.md` - 한글 문서
- `*_GUIDE.md` - 가이드 문서
- `README.md` - 각 폴더의 인덱스

---

## ❓ 자주 묻는 질문

**Q: 어디서부터 시작해야 하나요?**  
A: `README.md` → `docs/SETUP_GUIDE_KR.md` 순서로 읽으세요.

**Q: 스크립트가 너무 많아요.**  
A: 핵심만 사용하세요:

- `prepares/split_s3_data.py` (최초 1회)
- `prepares/update_s3_data.py` (데이터 추가 시)
- `scripts/train_korean_food_s3_hybrid.py` (학습)
- `scripts/test_s3_connection.py` (테스트)

**Q: 문서가 너무 많아요.**  
A: `docs/SETUP_GUIDE_KR.md` 하나면 충분합니다.

**Q: 기존 파일들은 어디갔나요?**  
A: `docs/`, `scripts/` 폴더로 정리되었습니다.

---

**프로젝트 메인으로**: [README.md](README.md)  
**문서 인덱스**: [docs/README.md](docs/README.md)  
**스크립트 가이드**: [scripts/README.md](scripts/README.md)  
**데이터 준비 가이드**: [prepares/README.md](prepares/README.md)
