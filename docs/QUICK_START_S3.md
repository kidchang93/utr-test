# 🚀 S3 학습 빠른 시작 가이드

5분 안에 S3 기반 학습을 시작하는 방법입니다.

## 📋 현재 S3 구조

```
s3://your-bucket/
└── foods/              # Raw 데이터
    ├── 김치찌개/
    │   ├── _001.jpg
    │   └── ...
    ├── 비빔밥/
    └── ...
```

## 📦 1단계: 패키지 설치

```bash
pip install -r requirements.txt
```

## ⚙️ 2단계: AWS 설정

### .env 파일 생성

```bash
# env_template.txt를 .env로 복사
cp env_template.txt .env
```

### .env 파일 편집

```env
# AWS 자격증명
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=ap-northeast-2

# S3 버킷 이름 (본인의 버킷 이름으로 변경)
S3_BUCKET_NAME=my-yolo-dataset

# S3 경로 설정
S3_RAW_PREFIX=foods/    # Raw 데이터 경로 (원본)
S3_TRAIN_PREFIX=train/  # 학습 데이터 경로
S3_VAL_PREFIX=val/      # 검증 데이터 경로

# 캐시 크기 (메모리에 캐싱할 이미지 수)
IMAGE_CACHE_SIZE=100
```

## 📊 3단계: S3 데이터 분할

S3의 `foods/{label}/` 구조를 `train/`, `val/`로 분할:

```bash
python prepares/split_s3_data.py
```

**이 스크립트가 하는 일:**
- `foods/` 하위의 모든 클래스 스캔
- 80:20 비율로 train/val 분할
- S3 내에서 복사 (원본 `foods/`는 유지)

**실행 예시:**
```
🔄 S3 데이터 분할 (foods/ → train/, val/)

[1/150] 김치찌개 처리 중...
   전체: 500개
   학습: 400개, 검증: 100개
  Train 김치찌개: 100%|████████| 400/400
  Val 김치찌개: 100%|████████| 100/100
   ✅ 완료: 학습 400, 검증 100
```

## ✅ 4단계: 연결 테스트

```bash
python test_s3_connection.py
```

출력 예시:
```
✅ 설정 파일 로드 성공!
✅ S3 버킷 접근 성공!
✅ 데이터 구조 확인 완료!
✅ 모든 테스트 통과!
```

## 🏃 5단계: 학습 시작

### 방법 A: 하이브리드 방식 (추천)

```bash
python train_korean_food_s3_hybrid.py
```

**장점:**
- ✅ 안정적
- ✅ YOLO 기본 기능 모두 사용 가능
- ✅ 빠른 속도
- ✅ 학습 후 자동으로 임시 파일 삭제

### 방법 B: 순수 스트리밍 방식

```bash
python train_korean_food_s3.py
```

**장점:**
- ✅ 디스크 사용 최소화
- ✅ 실시간 스트리밍

---

## 🎯 전체 워크플로우

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. .env 파일 설정
cp env_template.txt .env
nano .env  # 또는 메모장으로 편집

# 3. S3 데이터 분할 (foods/ → train/, val/)
python prepares/split_s3_data.py

# 4. 연결 테스트
python test_s3_connection.py

# 5. 학습 시작
python train_korean_food_s3_hybrid.py
```

---

## ❓ 문제 해결

### 문제: AWS 자격증명 오류

```
NoCredentialsError: Unable to locate credentials
```

**해결:**
- `.env` 파일이 프로젝트 루트에 있는지 확인
- `AWS_ACCESS_KEY_ID`와 `AWS_SECRET_ACCESS_KEY` 값이 올바른지 확인

### 문제: 버킷을 찾을 수 없음

```
ClientError: The specified bucket does not exist
```

**해결:**
- `.env`의 `S3_BUCKET_NAME`이 정확한지 확인
- AWS 콘솔에서 버킷이 생성되어 있는지 확인
- 리전이 올바른지 확인

### 문제: 데이터가 없음

```
'train/' 하위에 데이터가 없습니다!
```

**해결:**
- 3단계(S3 데이터 분할)를 실행했는지 확인
- AWS 콘솔에서 버킷 구조 확인:
  ```
  my-bucket/
  ├── foods/      ← 원본 (Raw 데이터)
  ├── train/      ← 분할 후 생성
  │   ├── 클래스1/
  │   │   └── 이미지들...
  │   └── 클래스2/
  │       └── 이미지들...
  └── val/        ← 분할 후 생성
      └── ...
  ```

### 문제: foods/ 폴더가 없음

```
'foods/' 하위에 데이터가 없습니다
```

**해결:**
- `.env`의 `S3_RAW_PREFIX` 값 확인
- 실제 S3 구조에 맞게 수정 (예: `S3_RAW_PREFIX=data/` 또는 `images/`)
- AWS 콘솔에서 실제 경로 확인

### 문제: GPU 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결:**
- `train_korean_food_s3_hybrid.py` 파일에서 `BATCH_SIZE` 줄이기:
  ```python
  BATCH_SIZE = 2  # 4에서 2로 줄임
  ```

---

## 📚 추가 정보

자세한 내용은 다음 문서를 참고하세요:

- **[S3_TRAINING_GUIDE.md](S3_TRAINING_GUIDE.md)**: 전체 가이드
- **[.env.example](.env.example)**: 설정 파일 예시

---

## 💡 팁

### 1. 캐시 크기 조정

메모리가 많다면 `.env`에서 캐시 크기를 늘리세요:

```env
IMAGE_CACHE_SIZE=500
```

### 2. 샘플 제한 (테스트용)

전체 데이터로 학습하기 전에 일부만 테스트하려면:

`train_korean_food_s3_hybrid.py` 파일에서:

```python
SAMPLE_LIMIT_PER_CLASS = 100  # 클래스당 100개만
```

### 3. IAM Role 사용 (EC2에서 실행 시)

EC2 인스턴스에 IAM Role이 설정되어 있다면, `.env`에서 자격증명을 비워두세요:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

---

**준비 완료! 즐거운 학습 되세요! 🎉**

