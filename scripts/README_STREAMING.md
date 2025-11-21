# 🌊 S3 메모리 스트리밍 학습 가이드

## 🎯 진짜 스트리밍이란?

```
S3 → 메모리 → GPU
```

- ✅ 디스크에 저장하지 않음
- ✅ LRU 캐시로 메모리 관리
- ✅ 최종 모델만 디스크 저장

## 🚀 빠른 시작

### 1. 대화형 실행 (추천)

```bash
cd /Users/kidchang/Desktop/ck/privacy/utr-test
./scripts/RUN_STREAMING_TEST.sh
```

**메뉴:**
- `1`: 전체 테스트 (10장, 3 에포크) - 약 5분
- `2`: 초고속 테스트 (5장, 1 에포크) - 약 2분
- `3`: S3 연결만
- `4`: 데이터셋 생성만
- `5`: 결과 업로드만

### 2. 직접 실행

```bash
# 전체 파이프라인
python scripts/test_s3_streaming_real.py

# 커스텀 설정
python scripts/test_s3_streaming_real.py --num-images 20 --epochs 5

# 단계별 실행
python scripts/test_s3_streaming_real.py --step 1  # S3 연결
python scripts/test_s3_streaming_real.py --step 2  # 데이터셋
python scripts/test_s3_streaming_real.py --step 3  # 학습
python scripts/test_s3_streaming_real.py --step 4  # 업로드
```

## 📋 4단계 프로세스

### 1️⃣ S3 연결 테스트
```bash
python scripts/test_s3_streaming_real.py --step 1
```
- AWS 자격증명 확인
- 버킷 접근 확인
- train/ 폴더 확인

**소요 시간:** ~10초

### 2️⃣ 스트리밍 데이터셋 생성
```bash
python scripts/test_s3_streaming_real.py --step 2 --num-images 10
```
- S3ClassificationDataset 초기화
- 첫 번째 클래스 선택
- 10장 샘플 선택 (메모리)
- **디스크 사용: 0 bytes** ✨

**소요 시간:** ~30초

### 3️⃣ 스트리밍 학습
```bash
python scripts/test_s3_streaming_real.py --step 3 --epochs 3
```
- PyTorch DataLoader 생성
- ResNet18 백본 로드
- S3 → 메모리 → GPU 스트리밍
- 학습 진행 (tqdm 진행바)

**소요 시간:** 
- GPU: ~3-5분
- CPU: ~10-15분

### 4️⃣ 결과 S3 업로드
```bash
python scripts/test_s3_streaming_real.py --step 4
```
- 학습된 모델 S3 업로드
- `models/streaming/s3_test_YYYYMMDD_HHMMSS/`

**소요 시간:** ~30초

## 🎨 스트리밍 방식 특징

### 메모리 흐름

```
┌─────────────┐
│  S3 버킷    │
│  train/     │
│  └─ 김치찌개/ │
└──────┬──────┘
       │ __getitem__
       ↓
┌─────────────┐
│ 메모리 캐시  │ ← LRU (10개)
│ [img1, ...] │
└──────┬──────┘
       │ transform
       ↓
┌─────────────┐
│    GPU      │
│   학습 중    │
└─────────────┘
```

### LRU 캐시 동작

```python
cache_size = 10  # 메모리에 10개만 보관

# 이미지 요청
img1 → S3에서 로드 → 캐시 [img1]
img2 → S3에서 로드 → 캐시 [img1, img2]
...
img10 → S3에서 로드 → 캐시 [img1, ..., img10]

# 캐시 가득참
img11 → S3에서 로드 → img1 삭제 → 캐시 [img2, ..., img11]

# 캐시 히트 (빠름!)
img2 → 캐시에서 바로 반환 ⚡
```

## 📊 출력 예시

```
======================================================================
🌊 진짜 S3 스트리밍 학습 테스트
======================================================================
💡 특징:
   - S3에서 메모리로 직접 로드
   - 디스크에 데이터 저장 안 함
   - LRU 캐시로 메모리 관리
   - 최종 모델만 디스크 저장
======================================================================

======================================================================
📡 단계 1: S3 연결 테스트
======================================================================
✅ S3 설정 로드 완료
✅ S3 버킷 접근 성공!
✅ train/ 폴더 발견

======================================================================
🌊 단계 2: 스트리밍 데이터셋 생성 (메모리 10장)
======================================================================
💡 디스크에 저장하지 않고 S3 → 메모리로 직접 로드합니다
======================================================================
📦 S3ClassificationDataset 초기화 중...
✅ 데이터셋 로드 완료:
   클래스: 150개
   샘플: 75,000개

✅ 첫 번째 클래스: 김치찌개
✅ 스트리밍 데이터셋 생성 완료
   클래스: 김치찌개
   샘플 수: 10개
   캐시 크기: 10개 (메모리)
   디스크 사용: 0 bytes ✨

🧪 스트리밍 로딩 테스트 중...
   → S3에서 직접 메모리로 로드
✅ 테스트 성공!
   이미지 shape: torch.Size([3, 640, 640])
   레이블: 0

======================================================================
🚀 단계 3: 스트리밍 학습 (3 에포크)
======================================================================
💡 S3 → 메모리 → GPU로 직접 스트리밍 학습
======================================================================

🖥️  디바이스: cuda
   GPU: NVIDIA GeForce RTX 3090
   메모리: 24.0 GB

📊 DataLoader 설정:
   배치 크기: 2
   배치 수: 5
   Workers: 2

🧠 모델 로드 중...
✅ ResNet18 모델 로드 완료

======================================================================
🎯 학습 시작
======================================================================
Epoch 1/3: 100%|████████████| 5/5 [00:15<00:00, loss=0.6932, avg_loss=0.6932]
   Epoch 1/3 - Loss: 0.6932
Epoch 2/3: 100%|████████████| 5/5 [00:12<00:00, loss=0.6845, avg_loss=0.6845]
   Epoch 2/3 - Loss: 0.6845
Epoch 3/3: 100%|████████████| 5/5 [00:12<00:00, loss=0.6701, avg_loss=0.6701]
   Epoch 3/3 - Loss: 0.6701

======================================================================
✅ 학습 완료!
======================================================================

💾 모델 저장:
   경로: runs/streaming/s3_test/streaming_model.pt
   크기: 44.68 MB

======================================================================
☁️  단계 4: 학습 결과 S3 업로드
======================================================================
📤 업로드 시작...
   로컬: runs/streaming/s3_test
   S3: s3://your-bucket/models/streaming/s3_test_20251114_153022/

  [1] streaming_model.pt (44.68 MB)

✅ 업로드 완료: 1개 파일
📍 S3 경로: s3://your-bucket/models/streaming/s3_test_20251114_153022/

======================================================================
🎉 모든 단계 성공!
======================================================================

✅ 테스트 완료 요약:
   1. ✅ S3 연결
   2. ✅ 스트리밍 데이터셋 (메모리)
   3. ✅ 스트리밍 학습
   4. ✅ 결과 S3 업로드

💡 핵심 포인트:
   - 디스크 사용량: 학습 데이터 0 bytes
   - 메모리만 사용하여 학습
   - 최종 모델만 디스크에 저장
```

## 💾 결과 파일

### 로컬
```
runs/streaming/s3_test/
└── streaming_model.pt  # 44MB
```

### S3
```
s3://your-bucket/models/streaming/s3_test_20251114_153022/
└── streaming_model.pt
```

## ⚙️ 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--step` | 전체 | 실행할 단계 (1,2,3,4) |
| `--num-images` | 10 | 사용할 이미지 수 |
| `--epochs` | 3 | 학습 에포크 수 |

## 🎯 80GB GPU 최적화

스크립트 수정:

```python
# scripts/test_s3_streaming_real.py
# line 240-245 수정

train_loader = DataLoader(
    self.dataset,
    batch_size=256,     # 2 → 256
    shuffle=True,
    num_workers=16,     # 2 → 16
    pin_memory=True
)
```

## 🔍 트러블슈팅

### S3 연결 실패
```bash
# .env 확인
cat .env

# 직접 테스트
python scripts/test_s3_connection.py
```

### 메모리 부족
```bash
# 캐시 크기 줄이기
python scripts/test_s3_streaming_real.py --num-images 5
```

### GPU 메모리 부족
```bash
# CPU로 실행
CUDA_VISIBLE_DEVICES="" python scripts/test_s3_streaming_real.py
```

## 📚 다운로드 방식과 비교

| 항목 | 다운로드 | 스트리밍 |
|------|----------|----------|
| 디스크 사용 | 100MB | **0 bytes** ✨ |
| 시작 시간 | 30초 | 3초 |
| 보안 | 낮음 | **높음** 🔒 |
| 스크립트 | test_s3_train_minimal.py | test_s3_streaming_real.py |

상세 비교: [STREAMING_vs_DOWNLOAD.md](STREAMING_vs_DOWNLOAD.md)

## 🧪 테스트 시나리오

### 빠른 검증 (1분)
```bash
python scripts/test_s3_streaming_real.py --num-images 3 --epochs 1
```

### 표준 테스트 (5분)
```bash
python scripts/test_s3_streaming_real.py --num-images 10 --epochs 3
```

### 완전한 테스트 (20분)
```bash
python scripts/test_s3_streaming_real.py --num-images 50 --epochs 10
```

## 💡 핵심 코드

### S3ClassificationDataset 사용

```python
from utils.s3_dataset import S3ClassificationDataset

# 스트리밍 데이터셋 생성
dataset = S3ClassificationDataset(
    bucket_name='your-bucket',
    prefix='train/',
    cache_size=10,  # 메모리 캐시
    ...
)

# __getitem__에서 S3 → 메모리
image, label = dataset[0]  # S3에서 직접 로드
```

### PyTorch 학습 루프

```python
from torch.utils.data import DataLoader

# DataLoader
loader = DataLoader(dataset, batch_size=2)

# 학습
for epoch in range(epochs):
    for images, labels in loader:
        # S3에서 스트리밍된 데이터
        images = images.to(device)
        output = model(images)
        loss.backward()
```

## 🎓 다음 단계

1. **전체 데이터셋으로 확장**
   - 모든 클래스 사용
   - 더 많은 에포크

2. **성능 최적화**
   - 캐시 크기 조정
   - 배치 크기 증가
   - Workers 수 증가

3. **프로덕션 배포**
   - 모델 서빙
   - API 엔드포인트
   - 모니터링

## 📖 관련 문서

- [STREAMING_vs_DOWNLOAD.md](STREAMING_vs_DOWNLOAD.md) - 방식 비교
- [WORKFLOW.md](WORKFLOW.md) - 전체 워크플로우
- [../docs/S3_TRAINING_GUIDE.md](../docs/S3_TRAINING_GUIDE.md) - 상세 가이드

## ✨ 요약

**이 스크립트는:**
- ✅ S3 → 메모리로 직접 스트리밍
- ✅ 디스크 사용 0 bytes
- ✅ LRU 캐시로 메모리 관리
- ✅ 최종 모델만 저장
- ✅ 단계별 독립 실행 가능

**진짜 스트리밍입니다!** 🌊

