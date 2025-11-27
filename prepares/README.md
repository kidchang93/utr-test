# 📦 데이터 준비 스크립트

S3 데이터 분할 및 업데이트 스크립트 모음

---

## 📋 스크립트 목록

### 1. split_s3_data.py - 최초 분할 ⭐

**용도:** S3의 raw 데이터를 train/val로 최초 분할

**실행:**

```bash
python prepares/split_s3_data.py
```

**언제 사용?**

- 처음 설정할 때 (최초 1회)
- train/val을 완전히 재구성하고 싶을 때

**동작:**

```
foods/{label}/_001.jpg    (원본)
         ↓
    80:20 분할
         ↓
train/{label}/_001.jpg    (80%)
val/{label}/_020.jpg      (20%)
```

**특징:**

- ✅ 원본 `foods/` 데이터는 그대로 유지
- ✅ S3 내에서 직접 복사 (다운로드 불필요)
- ✅ 랜덤 시드로 재현 가능
- ✅ 진행률 표시 (tqdm)
- ✅ 클래스별 자동 분할
- ✅ 분할 후 자동 검증

**설정 변경:**

```python
# split_s3_data.py 파일 수정
TRAIN_RATIO = 0.8        # 분할 비율 (80:20)
SOURCE_PREFIX = 'foods/'  # Raw 데이터 경로
TRAIN_PREFIX = 'train/'   # 학습 데이터 경로
VAL_PREFIX = 'val/'       # 검증 데이터 경로
```

**실행 예시:**

```
🔄 S3 데이터 분할 (foods/ → train/, val/)

[1/150] 김치찌개 처리 중...
   전체: 500개
   학습: 400개, 검증: 100개
  Train 김치찌개: 100%|████████| 400/400
  Val 김치찌개: 100%|████████| 100/100
   ✅ 완료: 학습 400, 검증 100

[2/150] 비빔밥 처리 중...
...

✅ 전체 분할 완료!
학습 데이터: 80,000개
검증 데이터: 20,000개
```

---

### 2. update_s3_data.py - 증분 업데이트 ⭐ NEW!

**용도:** foods/에 추가된 데이터만 train/val로 업데이트

**실행:**

```bash
python prepares/update_s3_data.py
```

**언제 사용?**

- `foods/새로운클래스/` 추가 시
- 기존 클래스에 이미지 추가 시
- 전체 재분할 없이 업데이트하고 싶을 때

**동작:**

**케이스 1: 새 클래스 추가**

```
foods/떡볶이/ (신규)
    ├── _001.jpg
    └── _100.jpg (100개)
         ↓
    자동 감지
         ↓
train/떡볶이/ (80개 추가)
val/떡볶이/ (20개 추가)
```

**케이스 2: 기존 클래스에 이미지 추가**

```
foods/김치찌개/
    ├── _001.jpg (기존)
    ├── _500.jpg (기존)
    └── _550.jpg (신규 50개)
         ↓
    신규 50개만 감지
         ↓
train/김치찌개/ (40개 추가)
val/김치찌개/ (10개 추가)
```

**특징:**

- ✅ 기존 train/val 데이터는 그대로 유지
- ✅ 새로운 데이터만 추가
- ✅ 기존 비율(80:20) 유지
- ✅ 전체 재분할 불필요 → 시간 절약
- ✅ 파일명 기반 중복 체크

**실행 예시:**

```
🔄 S3 증분 업데이트 시작

1️⃣ 새로운 클래스 확인 중...
✨ 새로운 클래스 발견: 2개
   - 떡볶이
   - 순대

   떡볶이: 학습 80, 검증 20
     Train 떡볶이: 100%|████████| 80/80
     Val 떡볶이: 100%|████████| 20/20
   ✅ 완료: 학습 80, 검증 20

2️⃣ 기존 클래스의 새 이미지 확인 중...
✨ 새 이미지 발견: 3개 클래스, 총 150개 이미지

   김치찌개: 50개 새 이미지
     Train 김치찌개: 100%|████████| 40/40
     Val 김치찌개: 100%|████████| 10/10

✅ 증분 업데이트 완료!
새 클래스: 2개
업데이트된 클래스: 3개
총 새 이미지: 150개
```

---

### 3. prepare_data_s3.py - 로컬 → S3 업로드

**용도:** 로컬 디스크의 데이터를 S3로 업로드

**실행:**

```bash
python prepares/prepare_data_s3.py
```

**언제 사용?**

- 로컬에 데이터가 있을 때
- S3에 최초로 데이터를 업로드할 때

**동작:**

```
D:\lck_data\dataset\kfood-yolo\
├── train/
│   ├── 김치찌개/
│   └── 비빔밥/
└── val/
         ↓
    S3 업로드
         ↓
s3://bucket/train/
s3://bucket/val/
```

**특징:**

- ✅ 폴더 구조 유지
- ✅ 진행률 표시
- ✅ 업로드 후 검증

---

### 4. prepare_data.py - 로컬 데이터 준비 (레거시)

**용도:** 로컬 디스크에서 train/val 분할

**실행:**

```bash
python prepares/prepare_data.py
```

**언제 사용?**

- S3 없이 로컬에서만 작업할 때
- 기존 방식 사용 시

---

## 🎯 사용 시나리오

### 시나리오 1: 처음 시작 (S3에 foods/ 있음)

```bash
# 1. foods/ → train/, val/ 분할
python prepares/split_s3_data.py

# 2. 학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 2: 새 클래스 추가

```bash
# 1. foods/새클래스/ 업로드 (AWS CLI 또는 콘솔)
aws s3 cp local_folder/ s3://bucket/foods/새클래스/ --recursive

# 2. 증분 업데이트
python prepares/update_s3_data.py

# 3. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 3: 기존 클래스에 이미지 추가

```bash
# 1. foods/기존클래스/에 이미지 추가 업로드
aws s3 cp new_images/ s3://bucket/foods/기존클래스/ --recursive

# 2. 증분 업데이트
python prepares/update_s3_data.py

# 3. 재학습
python scripts/train_korean_food_s3_hybrid.py
```

### 시나리오 4: 전체 재분할 (비율 변경 등)

```bash
# 1. split_s3_data.py에서 TRAIN_RATIO 수정
# 2. 기존 train/val 삭제 (선택사항)
# 3. 재분할
python prepares/split_s3_data.py
```

---

## 🔄 split vs update 비교

| 항목        | split_s3_data.py | update_s3_data.py |
| ----------- | ---------------- | ----------------- |
| 용도        | 최초 분할        | 증분 업데이트     |
| 실행 빈도   | 1회              | 여러 번           |
| 기존 데이터 | 무시             | 유지              |
| 처리 대상   | 전체             | 신규만            |
| 시간        | 오래 걸림        | 빠름              |
| 사용 시점   | 처음 설정        | 데이터 추가 시    |

**예시:**

```
최초: split_s3_data.py → 100,000개 처리 (30분)
추가: update_s3_data.py → 1,000개만 처리 (1분)
```

---

## ⚙️ 고급 설정

### 분할 비율 변경

```python
# split_s3_data.py 또는 update_s3_data.py
TRAIN_RATIO = 0.7  # 70:30 분할
```

### S3 경로 변경

```python
# .env 파일
S3_RAW_PREFIX=data/         # foods/ 대신
S3_TRAIN_PREFIX=training/   # train/ 대신
S3_VAL_PREFIX=validation/   # val/ 대신
```

---

## 📊 업데이트 알고리즘

### update_s3_data.py 동작 원리

**1단계: 새 클래스 감지**

```python
raw_classes = {'김치찌개', '비빔밥', '떡볶이'}
train_classes = {'김치찌개', '비빔밥'}
new_classes = raw_classes - train_classes  # {'떡볶이'}
```

**2단계: 새 이미지 감지**

```python
# 클래스별로
raw_images = {'_001.jpg', '_002.jpg', ..., '_550.jpg'}
existing_images = {'_001.jpg', '_002.jpg', ..., '_500.jpg'}
new_images = raw_images - existing_images  # {'_501.jpg', ..., '_550.jpg'}
```

**3단계: 분할 및 추가**

```python
new_images = ['_501.jpg', ..., '_550.jpg']  # 50개
train_new = new_images[:40]  # 80%
val_new = new_images[40:]     # 20%
```

---

## ❓ 자주 묻는 질문

**Q: split과 update 중 어떤 것을 사용해야 하나요?**  
A: 처음에는 `split_s3_data.py`, 이후 데이터 추가 시에는 `update_s3_data.py`

**Q: update를 여러 번 실행해도 되나요?**  
A: 네, 안전합니다. 이미 추가된 데이터는 건너뜁니다.

**Q: 분할 비율을 바꾸고 싶어요**  
A: `split_s3_data.py`를 재실행하거나 수동으로 이동하세요.

**Q: 특정 클래스만 업데이트할 수 있나요?**  
A: 현재는 전체 클래스를 확인합니다. 필요시 스크립트를 수정하세요.

**Q: 업데이트 중 중단하면?**  
A: 안전합니다. 다시 실행하면 나머지만 처리됩니다.

**Q: 원본 foods/ 데이터가 삭제되나요?**  
A: 아니요, 절대 삭제되지 않습니다. 복사만 합니다.

---

## 🛠️ 문제 해결

### 문제: foods/ 폴더가 없다고 나옴

```
'foods/' 하위에 데이터가 없습니다
```

**해결:** `.env`의 `S3_RAW_PREFIX` 값 확인

### 문제: 이미 train/val이 있음

**해결:**

- 기존 데이터 유지하려면 `update_s3_data.py` 사용
- 재분할하려면 기존 데이터 삭제 후 `split_s3_data.py`

### 문제: 특정 이미지가 누락됨

**해결:**

- 파일 확장자 확인 (.jpg, .png만 지원)
- S3에 실제로 업로드되었는지 확인

---

**프로젝트 루트로 돌아가기**: [../README.md](../README.md)  
**스크립트 보기**: [../scripts/README.md](../scripts/README.md)
