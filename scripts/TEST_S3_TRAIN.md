# 🧪 S3 스트리밍 학습 테스트 가이드

S3에서 데이터를 스트리밍하여 YOLO 학습하고, 결과를 S3에 업로드하는 전체 파이프라인을 최소 규모로 테스트합니다.

## 📋 테스트 단계

1. **S3 연결 테스트** - AWS 자격증명 및 버킷 접근 확인
2. **최소 데이터셋 다운로드** - 1개 클래스, 10장만 임시 다운로드
3. **YOLO 학습** - 11n 모델로 3 에포크 학습
4. **결과 S3 업로드** - runs/ 폴더를 S3에 자동 업로드

## 🚀 사용법

### 전체 테스트 실행 (권장)

```bash
cd /Users/kidchang/Desktop/ck/privacy/utr-test

# 기본 설정 (10장, 3 에포크)
python scripts/test_s3_train_minimal.py

# 커스텀 설정
python scripts/test_s3_train_minimal.py --num-images 20 --epochs 5

# 임시 파일 유지 (디버깅용)
python scripts/test_s3_train_minimal.py --no-cleanup
```

### 단계별 테스트 실행

```bash
# 단계 1만: S3 연결 테스트
python scripts/test_s3_train_minimal.py --step 1

# 단계 2만: 데이터 다운로드 (10장)
python scripts/test_s3_train_minimal.py --step 2 --num-images 10

# 단계 3만: 학습 (temp_train_test 폴더 사용)
python scripts/test_s3_train_minimal.py --step 3 --epochs 3

# 단계 4만: 기존 결과를 S3에 업로드
python scripts/test_s3_train_minimal.py --step 4
```

## 📊 예상 결과

### 성공 시

```
======================================================================
🎉 모든 단계 성공!
======================================================================

✅ 테스트 완료 요약:
   1. ✅ S3 연결
   2. ✅ 데이터 다운로드
   3. ✅ YOLO 학습
   4. ✅ 결과 S3 업로드
```

### 생성되는 파일

**로컬:**
- `temp_train_test/` - 임시 학습 데이터 (자동 삭제)
- `runs/classify/s3_test/` - 학습 결과
  - `weights/best.pt` - 최고 성능 모델
  - `weights/last.pt` - 마지막 체크포인트
  - `results.png` - 학습 그래프
  - `confusion_matrix.png` - 혼동 행렬

**S3:**
```
s3://your-bucket/models/runs/s3_test_20251114_153022/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── ...
```

## 🎯 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--step` | 전체 | 실행할 단계 (1,2,3,4) |
| `--num-images` | 10 | 다운로드할 이미지 수 |
| `--epochs` | 3 | 학습 에포크 수 |
| `--no-cleanup` | False | 임시 파일 삭제 안 함 |

## ⚙️ 요구사항

### 필수

1. **.env 파일 설정**
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your-bucket
S3_REGION=ap-northeast-2
```

2. **S3 버킷 구조**
```
your-bucket/
└── train/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    └── ...
```

3. **YOLO 모델 파일**
```bash
# 프로젝트 루트에 있어야 함
yolo11n-cls.pt
```

## 🔍 트러블슈팅

### S3 연결 실패

```bash
# .env 파일 확인
cat .env

# S3 연결 직접 테스트
python scripts/test_s3_connection.py
```

### 모델 파일 없음

```bash
# YOLO 모델 다운로드
# yolo11n-cls.pt를 프로젝트 루트에 배치
```

### GPU 메모리 부족

```bash
# CPU로 테스트
CUDA_VISIBLE_DEVICES="" python scripts/test_s3_train_minimal.py

# 또는 에포크 줄이기
python scripts/test_s3_train_minimal.py --epochs 1
```

## 📈 다음 단계

테스트 성공 후:

1. **전체 데이터셋으로 학습**
```bash
python scripts/train_korean_food_s3_hybrid.py
```

2. **모델 버전 관리 설정**
- DVC 설정
- S3 모델 저장소 구조화

3. **프로덕션 파이프라인 구축**
- 자동화 스크립트
- 모니터링 추가

## 💡 팁

- **빠른 테스트**: `--num-images 5 --epochs 1`
- **완전한 테스트**: `--num-images 50 --epochs 10`
- **디버깅**: `--no-cleanup` 으로 임시 파일 확인

## 📝 주의사항

- 임시 다운로드 방식이므로 **실제 스트리밍은 아님**
- 대규모 학습은 `train_korean_food_s3_hybrid.py` 사용
- S3 업로드 비용 주의 (모델 파일이 큰 경우)

