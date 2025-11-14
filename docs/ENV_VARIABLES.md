# 🔧 환경 변수 가이드

`.env` 파일 설정 방법

---

## 📋 환경 변수 목록

### AWS 자격증명

#### `ACCESS_KEY`

- **설명**: AWS Access Key ID
- **필수**: 아니요 (IAM Role 사용 시 생략 가능)
- **예시**: `AKIAIOSFODNN7EXAMPLE`

#### `SECRET_KEY`

- **설명**: AWS Secret Access Key
- **필수**: 아니요 (IAM Role 사용 시 생략 가능)
- **예시**: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
- **⚠️ 주의**: 절대 공개하지 마세요!

#### `REGION`

- **설명**: AWS 리전
- **필수**: 아니요 (기본값: `ap-northeast-2`)
- **예시**: `ap-northeast-2` (서울), `us-east-1` (버지니아), `eu-west-1` (아일랜드)

---

### S3 설정

#### `BUCKET_NAME`

- **설명**: S3 버킷 이름
- **필수**: 예 ✅
- **예시**: `my-korean-food-dataset`

#### `DOMAIN`

- **설명**: S3 커스텀 엔드포인트 URL
- **필수**: 아니요 (AWS S3 사용 시 생략)
- **사용 시나리오**:
  - MinIO, Ceph 등 S3 호환 스토리지 사용 시
  - 프라이빗 S3 엔드포인트 사용 시
- **예시**:
  - `https://s3.custom-domain.com`
  - `http://localhost:9000` (MinIO 로컬)
- **⚠️ 주의**: 비워두면 AWS S3 기본 엔드포인트 사용

---

### S3 경로

#### `S3_RAW_PREFIX`

- **설명**: Raw 데이터 경로 (원본 데이터)
- **필수**: 아니요 (기본값: `foods/`)
- **예시**: `foods/`, `raw/`, `original/`

#### `S3_TRAIN_PREFIX`

- **설명**: 학습 데이터 경로
- **필수**: 아니요 (기본값: `train/`)
- **예시**: `train/`, `training/`

#### `S3_VAL_PREFIX`

- **설명**: 검증 데이터 경로
- **필수**: 아니요 (기본값: `val/`)
- **예시**: `val/`, `validation/`, `test/`

---

### 기타 설정

#### `IMAGE_CACHE_SIZE`

- **설명**: 메모리에 캐싱할 이미지 개수
- **필수**: 아니요 (기본값: `100`)
- **예시**: `100`, `500`, `1000`
- **권장**:
  - 8GB RAM: `100-200`
  - 16GB RAM: `500-1000`
  - 32GB+ RAM: `1000+`

---

## 📝 .env 파일 예시

### AWS S3 사용 (기본)

```env
# AWS 자격증명
ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
REGION=ap-northeast-2

# S3 버킷
BUCKET_NAME=my-korean-food-dataset

# S3 엔드포인트 (AWS S3 사용 시 비워둠)
DOMAIN=

# S3 경로
S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/

# 캐시
IMAGE_CACHE_SIZE=100
```

### MinIO 사용

```env
# MinIO 자격증명
ACCESS_KEY=minioadmin
SECRET_KEY=minioadmin
REGION=us-east-1

# MinIO 버킷
BUCKET_NAME=korean-food

# MinIO 엔드포인트
DOMAIN=http://localhost:9000

# S3 경로
S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/

# 캐시
IMAGE_CACHE_SIZE=200
```

### IAM Role 사용 (EC2)

```env
# IAM Role 사용 (자격증명 생략)
ACCESS_KEY=
SECRET_KEY=
REGION=ap-northeast-2

# S3 버킷
BUCKET_NAME=my-korean-food-dataset

# S3 엔드포인트
DOMAIN=

# S3 경로
S3_RAW_PREFIX=foods/
S3_TRAIN_PREFIX=train/
S3_VAL_PREFIX=val/

# 캐시
IMAGE_CACHE_SIZE=500
```

---

## 🚀 빠른 설정

### 1. 템플릿 복사

```bash
cp env_template.txt .env
```

### 2. .env 파일 편집

```bash
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 3. 필수 값 입력

최소한 다음만 입력하면 됩니다:

```env
ACCESS_KEY=실제_키
SECRET_KEY=실제_비밀키
BUCKET_NAME=실제_버킷_이름
```

나머지는 기본값 사용 가능!

---

## ❓ 자주 묻는 질문

**Q: 모든 값을 입력해야 하나요?**  
A: 아니요. `BUCKET_NAME`만 필수이고, 나머지는 기본값이 있습니다.

**Q: IAM Role을 사용하면?**  
A: `ACCESS_KEY`와 `SECRET_KEY`를 비워두세요.

**Q: AWS S3가 아닌 다른 스토리지는?**  
A: `DOMAIN`에 엔드포인트 URL을 입력하세요.

**Q: .env 파일을 Git에 커밋해도 되나요?**  
A: **절대 안 됩니다!** `.gitignore`에 포함되어 있는지 확인하세요.

**Q: 설정이 제대로 되었는지 확인하려면?**  
A: `python scripts/test_s3_connection.py` 실행하세요.

---

## 🛡️ 보안 팁

1. **자격증명 보호**

   - `.env` 파일을 Git에 커밋하지 마세요
   - 팀원과 공유할 때는 안전한 방법 사용 (AWS Secrets Manager 등)

2. **최소 권한 원칙**

   - IAM 정책에서 S3 읽기/쓰기만 허용
   - 특정 버킷에만 접근 권한 부여

3. **IAM Role 우선**
   - EC2에서 실행 시 IAM Role 사용
   - 자격증명을 파일에 저장하지 않아도 됨

---

**프로젝트 루트로**: [../README.md](../README.md)  
**문서 인덱스**: [README.md](README.md)
