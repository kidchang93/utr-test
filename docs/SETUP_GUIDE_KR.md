# 🚀 S3 기반 YOLO 학습 초기 설정 가이드

## 현재 S3 구조

```
s3://your-bucket-name/
└── foods/              # Raw 데이터 (원본)
    ├── 김치찌개/
    │   ├── _001.jpg
    │   ├── _002.jpg
    │   └── ...
    ├── 비빔밥/
    │   ├── _001.jpg
    │   └── ...
    └── ...
```

## 목표 구조

```
s3://your-bucket-name/
├── foods/              # Raw 데이터 (원본 유지)
│   ├── 김치찌개/
│   ├── 비빔밥/
│   └── ...
├── train/              # 학습 데이터 (80%)
│   ├── 김치찌개/
│   ├── 비빔밥/
│   └── ...
└── val/                # 검증 데이터 (20%)
    ├── 김치찌개/
    ├── 비빔밥/
    └── ...
```

---

## 📋 전체 설정 단계 (5단계)

### 1단계: 패키지 설치 ⚙️

```bash
pip install -r requirements.txt
```

설치되는 주요 패키지:

- `boto3`: AWS S3 SDK
- `ultralytics`: YOLO 모델
- `torch`, `torchvision`: PyTorch
- `python-decouple`: 환경 변수 관리
- `tqdm`: 진행률 표시

---

### 2단계: AWS 설정 🔐

#### .env 파일 생성

```bash
# .env.template을 .env로 복사
cp .env.template .env
```

#### .env 파일 편집

메모장이나 에디터로 `.env` 파일을 열고 실제 값으로 수정:

```env
# AWS 자격증명
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE        # 실제 Access Key로 변경
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY  # 실제 Secret Key로 변경
AWS_REGION=ap-northeast-2                      # 서울 리전

# S3 버킷 이름
S3_BUCKET_NAME=my-korean-food-dataset          # 실제 버킷 이름으로 변경

# S3 경로 (기본값 사용 가능)
S3_RAW_PREFIX=foods/                           # Raw 데이터 경로
S3_TRAIN_PREFIX=train/                         # 학습 데이터 경로
S3_VAL_PREFIX=val/                             # 검증 데이터 경로

# 캐시 크기
IMAGE_CACHE_SIZE=100                           # 메모리 상황에 맞게 조정
```

**⚠️ 중요:**

- `.env` 파일은 절대 Git에 커밋하지 마세요!
- AWS 자격증명은 안전하게 보관하세요

#### AWS 자격증명 얻는 방법

**방법 1: AWS IAM 사용자 생성**

1. AWS 콘솔 로그인
2. IAM → 사용자 → 사용자 추가
3. 액세스 키 유형: 프로그래밍 방식 액세스
4. 권한: S3 읽기/쓰기 권한 부여
5. Access Key ID와 Secret Access Key 저장

**방법 2: AWS CLI 사용 (이미 설정된 경우)**

```bash
aws configure list
```

**방법 3: EC2 IAM Role (EC2에서 실행하는 경우)**

- `.env`에서 자격증명을 비워두면 자동으로 IAM Role 사용

---

### 3단계: S3 데이터 분할 📊

현재 `foods/{label}/` 구조의 raw 데이터를 `train/`, `val/`로 분할합니다.

```bash
python prepares/split_s3_data.py
```

**이 스크립트가 하는 일:**

1. S3의 `foods/` 하위 모든 클래스 스캔
2. 각 클래스의 이미지를 80:20 비율로 랜덤 분할
3. `train/{label}/`, `val/{label}/`로 S3 내에서 복사
4. **원본 `foods/` 데이터는 그대로 유지**

**실행 예시:**

```
🔄 S3 데이터 분할 (foods/ → train/, val/)
======================================================================

⚙️  분할 설정
======================================================================
원본 경로: foods/
학습 경로: train/
검증 경로: val/
분할 비율: 80% (학습) / 20% (검증)

⚠️  주의:
   - 원본 foods/ 데이터는 그대로 유지됩니다
   - train/, val/ 경로에 복사본이 생성됩니다

계속 진행하시겠습니까? (yes/no): yes

[1/150] 김치찌개 처리 중...
   전체: 500개
   학습: 400개, 검증: 100개
  Train 김치찌개: 100%|████████| 400/400 [00:15<00:00]
  Val 김치찌개: 100%|████████| 100/100 [00:03<00:00]
   ✅ 완료: 학습 400, 검증 100
...
```

**분할 설정 변경하려면:**

`prepares/split_s3_data.py` 파일 수정:

```python
TRAIN_RATIO = 0.8  # 학습 비율 (0.8 = 80%)
SOURCE_PREFIX = 'foods/'  # Raw 데이터 경로
```

---

### 4단계: 연결 테스트 ✅

S3 연결과 데이터 구조를 확인합니다:

```bash
python test_s3_connection.py
```

**출력 예시:**

```
🔍 S3 연결 테스트
======================================================================

1️⃣ .env 설정 파일 확인 중...
✅ 설정 파일 로드 성공!

2️⃣ S3 버킷 접근 테스트...
✅ S3 버킷 접근 성공!

3️⃣ 학습 데이터 구조 확인...
📁 train/
   클래스: 150개
   이미지: 80,000장
   1. 김치찌개: 400장
   2. 비빔밥: 520장
   ...

📁 val/
   클래스: 150개
   이미지: 20,000장

4️⃣ 샘플 이미지 로드 테스트...
   로딩 중: train/김치찌개/_001.jpg
   ✅ 로드 성공!
      크기: (640, 480)
      모드: RGB

======================================================================
✅ 모든 테스트 통과!
======================================================================
```

**문제 발생 시:**

- AWS 자격증명 확인
- S3 버킷 이름 확인
- 3단계(데이터 분할) 실행 여부 확인

---

### 5단계: 학습 시작 🎯

#### 방법 A: 하이브리드 방식 (추천) ⭐

```bash
python train_korean_food_s3_hybrid.py
```

**특징:**

- ✅ 안정적이고 빠름
- ✅ S3 → 임시 디렉토리 → 학습 → 자동 삭제
- ✅ YOLO의 모든 기능 사용 가능
- ✅ GPU 메모리 효율적

**동작 방식:**

```
1. S3에서 전체 데이터를 임시 디렉토리로 다운로드
2. YOLO로 학습 (일반적인 로컬 학습과 동일)
3. 학습 완료 후 임시 디렉토리 자동 삭제
4. 서버에는 모델 가중치만 남음
```

#### 방법 B: 순수 스트리밍 방식

```bash
python train_korean_food_s3.py
```

**특징:**

- ✅ 서버 디스크를 전혀 사용하지 않음
- ✅ 배치마다 실시간으로 S3에서 로드
- ✅ 메모리 캐싱으로 성능 향상

---

## 🎯 학습 파라미터 조정

학습 전에 파라미터를 조정하려면 `train_korean_food_s3_hybrid.py` 파일을 수정하세요:

```python
# 학습 파라미터 (파일 내부)
MODEL_SIZE = "11n"          # 모델 크기: 11n, 11s, 11m, 11l, 11x
EPOCHS = 5                  # 에포크 수
BATCH_SIZE = 4              # 배치 크기 (GPU 메모리에 맞게)
IMG_SIZE = 224              # 이미지 크기
SAMPLE_LIMIT_PER_CLASS = None  # 테스트용: 클래스당 제한 (None = 전체)
```

**GPU 메모리별 권장 배치 크기:**

- 3GB: `BATCH_SIZE = 2`
- 6GB: `BATCH_SIZE = 8`
- 12GB+: `BATCH_SIZE = 16` 이상

---

## 📁 학습 결과 확인

학습 완료 후 다음 위치에 결과가 저장됩니다:

```
runs/classify/s3_korean_food/
├── weights/
│   ├── best.pt              # 최고 성능 모델
│   └── last.pt              # 마지막 에포크 모델
├── results.png              # 학습 그래프
├── confusion_matrix.png     # 혼동 행렬
├── val_batch0_pred.jpg      # 예측 결과 샘플
└── args.yaml                # 학습 설정
```

---

## ❓ 자주 묻는 질문 (FAQ)

### Q1: AWS 자격증명을 어떻게 얻나요?

**A:** AWS 콘솔 → IAM → 사용자 → 사용자 추가:

1. 액세스 유형: 프로그래밍 방식 액세스
2. 권한: `AmazonS3FullAccess` 또는 커스텀 정책
3. Access Key와 Secret Key 저장

### Q2: 버킷이 없는데 어떻게 하나요?

**A:** AWS 콘솔 → S3 → 버킷 만들기:

1. 버킷 이름 입력 (전역 고유)
2. 리전 선택 (ap-northeast-2 = 서울)
3. 모든 퍼블릭 액세스 차단 (권장)
4. 버킷 만들기

### Q3: Raw 데이터를 S3에 어떻게 올리나요?

**A:** 로컬에 데이터가 있다면 AWS CLI 사용:

```bash
aws s3 cp D:\lck_data\dataset\kfood\ s3://your-bucket/foods/ --recursive
```

또는 AWS 콘솔에서 드래그 앤 드롭으로 업로드

### Q4: 분할 비율을 바꾸고 싶어요

**A:** `prepares/split_s3_data.py` 파일에서:

```python
TRAIN_RATIO = 0.7  # 70:30 분할로 변경
```

### Q5: 이미 train/val이 있는데 다시 분할하면?

**A:** S3에 중복 파일이 생성될 수 있습니다. 기존 train/val을 먼저 삭제하거나 다른 prefix를 사용하세요.

### Q6: 학습 중에 중단하면?

**A:** Ctrl+C로 안전하게 중단 가능:

- 중간 모델이 저장됨 (`last.pt`)
- 하이브리드 방식: 임시 파일은 자동 삭제됨
- 다시 실행하면 처음부터 시작

### Q7: 비용이 얼마나 나오나요?

**A:** 예시 (서울 리전, 10GB 데이터):

- 스토리지: $0.23/월
- 데이터 전송 (5 에포크): $4.5
- 요청 비용: ~$0.2
- **총 약 $5 이하**

### Q8: EC2에서 실행하는 경우?

**A:** IAM Role을 EC2에 연결하고 `.env`에서 자격증명을 비워두세요:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

---

## 🛡️ 보안 체크리스트

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는가?
- [ ] AWS 자격증명이 코드에 하드코딩되지 않았는가?
- [ ] IAM 정책이 최소 권한 원칙을 따르는가?
- [ ] S3 버킷이 불필요하게 public이 아닌가?
- [ ] 팀원과 자격증명을 안전하게 공유했는가?

---

## 🔄 전체 워크플로우 요약

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. .env 설정
cp .env.template .env
nano .env  # AWS 자격증명 입력

# 3. S3 데이터 분할 (최초 1회만)
python prepares/split_s3_data.py

# 4. 연결 테스트
python test_s3_connection.py

# 5. 학습 시작!
python train_korean_food_s3_hybrid.py
```

---

## 📞 문제 해결

문제가 발생하면:

1. `test_s3_connection.py` 실행하여 진단
2. `.env` 파일 설정 재확인
3. AWS 콘솔에서 S3 구조 확인
4. 로그 메시지 확인

**더 자세한 정보:**

- [S3_TRAINING_GUIDE.md](S3_TRAINING_GUIDE.md): 전체 가이드
- [QUICK_START_S3.md](QUICK_START_S3.md): 빠른 시작

---

**준비 완료! 성공적인 학습을 기원합니다! 🚀**
