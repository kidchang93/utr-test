# 🎯 로컬 → S3 마이그레이션 완료 요약

## 📊 변경 사항

### 이전 (로컬 기반)
```
로컬 디스크에 전체 데이터셋 저장
→ 서버 디스크 공간 대량 사용
→ 데이터 관리 분산
```

### 이후 (S3 기반)
```
S3에 데이터 저장
→ 서버에는 모델 가중치만 저장
→ 필요할 때만 데이터 스트리밍
→ 중앙화된 데이터 관리
```

---

## 📁 새로 추가된 파일

### 핵심 모듈

```
utils/
├── __init__.py                    # 모듈 초기화
├── s3_loader.py                   # S3 이미지 스트리밍 로더
└── s3_dataset.py                  # PyTorch 커스텀 Dataset

config/
├── __init__.py                    # 모듈 초기화
└── s3_config.py                   # S3 설정 관리
```

### 학습 스크립트

```
train_korean_food_s3.py            # 순수 스트리밍 학습
train_korean_food_s3_hybrid.py     # 하이브리드 학습 (추천!)
```

### 유틸리티

```
prepares/prepare_data_s3.py        # 로컬 → S3 업로드
test_s3_connection.py              # S3 연결 테스트
```

### 설정 파일

```
.env.example                       # 환경 변수 예시
```

### 문서

```
S3_TRAINING_GUIDE.md               # 전체 가이드
QUICK_START_S3.md                  # 빠른 시작 가이드
S3_MIGRATION_SUMMARY.md            # 이 파일
```

### 의존성 업데이트

```diff
requirements.txt
+ boto3==1.34.144          # AWS S3 SDK
+ python-decouple==3.8     # 환경 변수 관리
+ tqdm==4.66.5             # 진행률 표시
```

---

## 🚀 사용 방법 비교

### 기존 방식 (로컬)

```bash
# 1단계: 로컬에 데이터 준비
python prepares/prepare_data.py

# 2단계: 로컬에서 학습
python train_korean_food.py
```

### 새 방식 (S3)

```bash
# 1단계: .env 설정
cp .env.example .env
nano .env

# 2단계: S3에 데이터 업로드 (최초 1회만)
python prepares/prepare_data_s3.py

# 3단계: 연결 테스트
python test_s3_connection.py

# 4단계: S3에서 학습
python train_korean_food_s3_hybrid.py
```

---

## 💰 비용 절감 효과

### 서버 디스크 사용량

**기존:**
- 데이터셋: 10GB
- 학습 중 임시 파일: ~5GB
- **총 15GB 사용**

**S3 방식:**
- 모델 가중치만: ~50MB
- 임시 파일 (학습 후 자동 삭제): 0GB
- **총 50MB 사용** (99.7% 절감!)

### S3 비용 (예시)

- 데이터셋 10GB 저장: 월 **$0.23**
- 5 에포크 학습 (50GB 전송): **$4.5**
- 월 예상 비용: **약 $5 이하**

→ 소형 서버 디스크 업그레이드 비용보다 저렴!

---

## 🎯 주요 기능

### 1. S3ImageLoader (`utils/s3_loader.py`)

```python
from utils.s3_loader import S3ImageLoader

loader = S3ImageLoader(bucket_name='my-bucket')

# 이미지 로드
image = loader.load_image_from_s3('train/김치찌개/001.jpg')

# 클래스 구조 분석
structure = loader.get_class_structure('train/')

# 버킷 접근 확인
loader.check_bucket_access()
```

**특징:**
- ✅ 메모리에서 직접 이미지 로드 (디스크 저장 안 함)
- ✅ 자동 RGB 변환
- ✅ 오류 처리 및 로깅
- ✅ 페이지네이션 지원 (대용량 데이터)

### 2. S3ClassificationDataset (`utils/s3_dataset.py`)

```python
from utils.s3_dataset import S3ClassificationDataset

dataset = S3ClassificationDataset(
    bucket_name='my-bucket',
    prefix='train/',
    cache_size=100
)

# PyTorch DataLoader와 함께 사용
loader = DataLoader(dataset, batch_size=16)
```

**특징:**
- ✅ PyTorch Dataset 표준 인터페이스
- ✅ LRU 캐싱으로 성능 향상
- ✅ 멀티워커 지원
- ✅ 자동 셔플링

### 3. S3Config (`config/s3_config.py`)

```python
from config.s3_config import get_s3_config

config = get_s3_config()
config.print_config()  # 설정 출력 (자격증명 마스킹)

params = config.get_s3_params()  # 딕셔너리로 반환
```

**특징:**
- ✅ 환경 변수 기반 설정
- ✅ 싱글톤 패턴
- ✅ 자격증명 보안 (마스킹 출력)
- ✅ 기본값 지원

---

## 🔄 워크플로우 다이어그램

```
┌─────────────────┐
│  로컬 데이터셋  │
└────────┬────────┘
         │
         │ prepare_data_s3.py
         ▼
┌─────────────────┐
│   S3 버킷       │
│   ├── train/    │
│   └── val/      │
└────────┬────────┘
         │
         │ train_korean_food_s3_hybrid.py
         ▼
┌─────────────────┐
│ 임시 디렉토리   │ ◄─── 배치 단위 다운로드
└────────┬────────┘
         │
         │ YOLO 학습
         ▼
┌─────────────────┐
│  모델 가중치    │
│  runs/classify/ │
└─────────────────┘
         │
         │ 학습 완료 후
         ▼
┌─────────────────┐
│ 임시 파일 삭제  │ ◄─── 자동 정리
└─────────────────┘
```

---

## 🛡️ 보안 고려사항

### 1. 자격증명 관리

**✅ 권장:**
- `.env` 파일 사용 (버전 관리 제외)
- EC2 IAM Role 사용
- AWS Secrets Manager 사용

**❌ 피할 것:**
- 코드에 직접 하드코딩
- 공개 저장소에 업로드
- 팀원과 평문으로 공유

### 2. .gitignore 설정

```gitignore
.env
*.pt
runs/
temp_*
```

### 3. IAM 최소 권한 정책

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

---

## 📈 성능 최적화

### 1. 캐시 크기 조정

```python
# .env
IMAGE_CACHE_SIZE=500  # 메모리가 충분하다면 증가
```

### 2. 멀티워커

```python
DataLoader(..., num_workers=8)  # CPU 코어 수에 맞게
```

### 3. Prefetch Factor

```python
DataLoader(..., prefetch_factor=2)
```

### 4. S3 Transfer Acceleration

AWS 콘솔에서 버킷 속성 → Transfer Acceleration 활성화

---

## 🧪 테스트 가이드

### 1. 연결 테스트

```bash
python test_s3_connection.py
```

### 2. 소량 데이터 테스트

```python
# train_korean_food_s3_hybrid.py
SAMPLE_LIMIT_PER_CLASS = 10  # 클래스당 10개만
```

### 3. 단일 이미지 로드 테스트

```python
from utils.s3_loader import S3ImageLoader

loader = S3ImageLoader(bucket_name='my-bucket')
image = loader.load_image_from_s3('train/김치찌개/001.jpg')
print(f"크기: {image.size}, 모드: {image.mode}")
```

---

## 🔧 문제 해결 체크리스트

### ✅ 설정 확인
- [ ] `.env` 파일이 프로젝트 루트에 있는가?
- [ ] AWS 자격증명이 올바른가?
- [ ] S3 버킷 이름이 정확한가?
- [ ] 리전이 올바른가?

### ✅ 권한 확인
- [ ] IAM 사용자/Role에 S3 읽기 권한이 있는가?
- [ ] 버킷 정책이 올바른가?
- [ ] 버킷이 public이 아닌가? (보안 위험)

### ✅ 데이터 확인
- [ ] S3에 train/, val/ 폴더가 있는가?
- [ ] 각 폴더 하위에 클래스 폴더가 있는가?
- [ ] 이미지 파일이 올바른 형식인가? (jpg, png)

### ✅ 환경 확인
- [ ] 모든 패키지가 설치되었는가?
- [ ] Python 3.8 이상인가?
- [ ] 인터넷 연결이 정상인가?

---

## 📝 다음 단계

### 1. 빠른 시작

```bash
# 5분 안에 시작
cat QUICK_START_S3.md
```

### 2. 전체 가이드

```bash
# 자세한 설명
cat S3_TRAINING_GUIDE.md
```

### 3. 학습 시작

```bash
# 하이브리드 방식 (추천)
python train_korean_food_s3_hybrid.py
```

---

## 🎉 완료!

이제 S3 기반 학습 시스템이 준비되었습니다!

**주요 이점:**
- ✅ 서버 디스크 99% 절감
- ✅ 데이터 중앙화 관리
- ✅ 확장성 향상
- ✅ 비용 효율적

**문의사항:**
- 문제 발생 시 `test_s3_connection.py` 실행
- 로그 확인
- `.env` 설정 재확인

**행복한 딥러닝 되세요! 🚀**

---

**작성일**: 2025-11-11  
**버전**: 1.0

