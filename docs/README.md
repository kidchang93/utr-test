# 📚 문서 인덱스

S3 기반 YOLO 학습 시스템 문서 모음

---

## 🚀 시작하기

### 처음 사용자

1. **[SETUP_GUIDE_KR.md](SETUP_GUIDE_KR.md)** ⭐ - 상세한 한글 설정 가이드 (추천!)
2. **[README_S3_SETUP.md](README_S3_SETUP.md)** - 5분 빠른 설정

### 빠른 참고

- **[QUICK_START_S3.md](QUICK_START_S3.md)** - 5단계 빠른 시작

---

## 📖 전체 가이드

### [SETUP_GUIDE_KR.md](SETUP_GUIDE_KR.md) - 상세 설정 가이드 (한글)

**내용:**

- AWS 설정 방법
- S3 버킷 구조
- 데이터 분할 방법
- 학습 실행
- 문제 해결

**대상:** 처음 사용자, 상세한 설명이 필요한 사용자

---

### [README_S3_SETUP.md](README_S3_SETUP.md) - 빠른 설정 (한글)

**내용:**

- 5단계 빠른 설정
- 커스터마이즈 방법
- 주요 문제 해결

**대상:** 빠르게 시작하고 싶은 사용자

---

### [QUICK_START_S3.md](QUICK_START_S3.md) - Quick Start (영한 혼용)

**내용:**

- 5분 빠른 시작
- 문제 해결 팁
- 전체 워크플로우

**대상:** 경험 있는 사용자

---

### [S3_TRAINING_GUIDE.md](S3_TRAINING_GUIDE.md) - Full Guide (영문)

**내용:**

- 전체 시스템 설명
- 아키텍처
- API 문서
- 성능 최적화
- 비용 분석

**대상:** 시스템을 깊이 이해하고 싶은 사용자

---

### [S3_MIGRATION_SUMMARY.md](S3_MIGRATION_SUMMARY.md) - Migration Summary (영문)

**내용:**

- 로컬 → S3 마이그레이션 요약
- 변경 사항
- 비용 절감 효과
- 보안 고려사항

**대상:** 기존 로컬 학습 시스템에서 마이그레이션하는 사용자

---

## 🎯 시나리오별 가이드

### 시나리오 1: 처음 설정하는 경우

```
SETUP_GUIDE_KR.md 읽기
↓
.env 파일 설정
↓
prepares/split_s3_data.py 실행
↓
scripts/train_korean_food_s3_hybrid.py 실행
```

### 시나리오 2: 빠르게 시작하고 싶은 경우

```
README_S3_SETUP.md 읽기
↓
5단계 실행
```

### 시나리오 3: 데이터가 추가된 경우

```
prepares/update_s3_data.py 실행
↓
scripts/train_korean_food_s3_hybrid.py 재실행
```

### 시나리오 4: 문제 발생 시

```
해당 가이드의 "문제 해결" 섹션 참고
↓
scripts/test_s3_connection.py 실행
↓
로그 확인
```

---

## 📝 추가 정보

### 주요 스크립트 위치

**데이터 준비:**

- `prepares/split_s3_data.py` - 최초 train/val 분할
- `prepares/update_s3_data.py` - 증분 업데이트

**학습:**

- `scripts/train_korean_food_s3_hybrid.py` - 하이브리드 방식 (추천!)
- `scripts/train_korean_food_s3.py` - 순수 스트리밍

**테스트:**

- `scripts/test_s3_connection.py` - S3 연결 테스트

### 설정 파일

- `env_template.txt` - 환경 변수 템플릿
- `.env` - 실제 환경 변수 (생성 필요, Git 제외)
- `config/s3_config.py` - S3 설정 클래스

---

## ❓ 자주 묻는 질문

**Q: 어떤 문서부터 읽어야 하나요?**  
A: `SETUP_GUIDE_KR.md`를 추천합니다. 가장 상세하고 한글로 되어 있습니다.

**Q: foods/에 데이터를 추가했는데 어떻게 하나요?**  
A: `prepares/update_s3_data.py`를 실행하세요. 증분 업데이트됩니다.

**Q: 영문 문서가 필요한가요?**  
A: 아니요. 한글 문서만으로도 충분합니다. 영문은 참고용입니다.

**Q: 비용이 얼마나 드나요?**  
A: 약 $5 이하/월 (10GB 기준). 자세한 내용은 `S3_TRAINING_GUIDE.md` 참고.

**Q: 문제가 생겼어요!**  
A: 각 가이드의 "문제 해결" 섹션을 참고하거나 `test_s3_connection.py`를 실행하세요.

---

**프로젝트 루트로 돌아가기**: [../README.md](../README.md)
