# 🚀 스크립트 가이드

학습 및 테스트 스크립트 모음

---

## 📋 스크립트 목록

### 1. test_s3_connection.py ✅

**용도:** S3 연결 및 데이터 구조 테스트

**실행:**

```bash
python scripts/test_s3_connection.py
```

**언제 사용?**

- 처음 설정 후 연결 확인
- 문제 발생 시 진단
- 데이터 구조 확인

**출력 예시:**

```
✅ 설정 파일 로드 성공!
✅ S3 버킷 접근 성공!
✅ 데이터 구조 확인 완료!
✅ 모든 테스트 통과!
```

---

### 2. train_korean_food_s3_hybrid.py ⭐ (추천!)

**용도:** S3 하이브리드 방식 YOLO 학습

**실행:**

```bash
python scripts/train_korean_food_s3_hybrid.py
```

**동작:**

1. S3에서 전체 데이터를 임시 디렉토리로 다운로드
2. YOLO로 학습 (일반 로컬 학습과 동일)
3. 학습 완료 후 임시 파일 자동 삭제
4. 모델 가중치만 서버에 저장

**장점:**

- ✅ 안정적이고 빠름
- ✅ YOLO의 모든 기능 사용 가능
- ✅ GPU 메모리 효율적
- ✅ 자동 정리

**파라미터 조정:**

```python
MODEL_SIZE = "11n"          # 모델 크기
EPOCHS = 5                  # 에포크 수
BATCH_SIZE = 4              # 배치 크기
IMG_SIZE = 224              # 이미지 크기
SAMPLE_LIMIT_PER_CLASS = None  # 테스트용 제한
```

**학습 결과:**

```
runs/classify/s3_korean_food/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
└── confusion_matrix.png
```

---

### 3. train_korean_food_s3.py

**용도:** S3 순수 스트리밍 방식 학습

**실행:**

```bash
python scripts/train_korean_food_s3.py
```

**동작:**

- 배치마다 S3에서 실시간으로 이미지 로드
- 메모리 캐싱 사용
- 디스크를 전혀 사용하지 않음

**장점:**

- ✅ 디스크 사용 최소화
- ✅ 대용량 데이터셋에 적합

**단점:**

- ⚠️ YOLO 기본 기능 일부 제약
- ⚠️ 네트워크 속도에 영향받음

**권장:**

- 대부분의 경우 `train_korean_food_s3_hybrid.py` 추천

---

### 4. train_korean_food.py

**용도:** 로컬 데이터 기반 YOLO 학습 (레거시)

**실행:**

```bash
python scripts/train_korean_food.py
```

**특징:**

- 로컬 디스크의 데이터를 사용
- S3 없이 동작
- 기존 방식

**언제 사용?**

- S3 없이 로컬에서만 학습할 때
- S3 방식과 비교 테스트할 때

---

### 5. yolo_test.py

**용도:** YOLO 기본 테스트

**실행:**

```bash
python scripts/yolo_test.py
```

**특징:**

- YOLO 설치 확인
- 기본 동작 테스트

---

## 🎯 사용 시나리오

### 시나리오 1: 처음 설정 후

```bash
# 1. 연결 테스트
python scripts/test_s3_connection.py

# 2. 학습 시작
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 2: 데이터 추가 후

```bash
# 1. 데이터 업데이트
python prepares/update_s3_data.py

# 2. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 3: 테스트용 소량 학습

```bash
# train_korean_food_s3_hybrid.py 수정
SAMPLE_LIMIT_PER_CLASS = 10  # 클래스당 10개만

python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 4: 문제 발생 시

```bash
# 진단 실행
python scripts/test_s3_connection.py

# 로그 확인 후 재시도
```

---

## ⚙️ 학습 파라미터 가이드

### GPU 메모리별 권장 설정

**3GB GPU (GTX 1060 3GB 등):**

```python
BATCH_SIZE = 2
MODEL_SIZE = "11n"
IMG_SIZE = 224
```

**6GB GPU (GTX 1060 6GB, RTX 2060 등):**

```python
BATCH_SIZE = 8
MODEL_SIZE = "11s"
IMG_SIZE = 224
```

**12GB+ GPU (RTX 3090, 4090 등):**

```python
BATCH_SIZE = 16
MODEL_SIZE = "11m" or "11l"
IMG_SIZE = 320
```

### 모델 크기 비교

| 모델 | 크기 | 속도      | 정확도    | 권장 용도       |
| ---- | ---- | --------- | --------- | --------------- |
| 11n  | 최소 | 매우 빠름 | 낮음      | 테스트, 3GB GPU |
| 11s  | 작음 | 빠름      | 보통      | 일반 학습       |
| 11m  | 중간 | 보통      | 높음      | 고성능 필요 시  |
| 11l  | 큼   | 느림      | 매우 높음 | 최고 성능       |
| 11x  | 최대 | 매우 느림 | 최고      | 연구용          |

---

## 📊 학습 결과 확인

### 학습 완료 후

```
runs/classify/s3_korean_food/
├── weights/
│   ├── best.pt              # 최고 성능 모델
│   └── last.pt              # 마지막 에포크
├── results.png              # 학습 그래프
├── confusion_matrix.png     # 혼동 행렬
├── val_batch0_pred.jpg      # 검증 샘플
└── args.yaml                # 학습 설정
```

### 모델 사용

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/classify/s3_korean_food/weights/best.pt')

# 예측
results = model.predict('test_image.jpg')
print(results[0].probs.top5)  # Top-5 예측
```

---

## ❓ 자주 묻는 질문

**Q: 어떤 스크립트를 사용해야 하나요?**  
A: `train_korean_food_s3_hybrid.py`를 추천합니다. 가장 안정적입니다.

**Q: 학습 중 중단하면 어떻게 되나요?**  
A: Ctrl+C로 안전하게 중단 가능하며, `last.pt`에 중간 모델이 저장됩니다.

**Q: 학습 시간이 얼마나 걸리나요?**  
A: 데이터 양, GPU, 에포크 수에 따라 다릅니다. 스크립트 시작 시 예상 시간이 표시됩니다.

**Q: 여러 모델을 동시에 학습할 수 있나요?**  
A: 가능하지만 GPU 메모리를 고려해야 합니다. `name` 파라미터를 다르게 설정하세요.

**Q: 학습 중 S3 연결이 끊기면?**  
A: 하이브리드 방식은 이미 다운로드되어 영향 없습니다. 순수 스트리밍은 재시도합니다.

---

**프로젝트 루트로 돌아가기**: [../README.md](../README.md)  
**문서 보기**: [../docs/README.md](../docs/README.md)
