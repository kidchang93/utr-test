# Requirements 설치 가이드 (Python 3.13)

이 프로젝트는 Python 3.13을 기준으로 작성되었으며, 로컬 개발(Mac)과 서버 배포(NVIDIA GPU)에 따라 다른 PyTorch 버전을 사용합니다.

## 📋 파일 구조

- `requirements.txt` - 공통 패키지 (PyTorch 제외)
- `requirements-mac.txt` - Apple Silicon Mac용 (MPS GPU)
- `requirements-cuda.txt` - NVIDIA GPU 서버용 (CUDA 12.4)

---

## 🍎 맥북 (Apple Silicon) 설정

### 1. Python 3.13 가상환경 생성
```bash
cd /Users/kidchang/Desktop/ck/privacy/utr-test

# 가상환경 생성
python3.13 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 2. 패키지 설치
```bash
# 공통 패키지 설치
pip install -r requirements.txt

# Mac용 PyTorch 설치 (MPS GPU 지원)
pip install -r requirements-mac.txt
```

### 3. GPU 사용 확인
```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple GPU (MPS) 사용 가능!")
else:
    device = torch.device("cpu")
    print("⚠️ CPU 모드로 실행")

# 테스트
x = torch.rand(5, 3).to(device)
print(f"Device: {x.device}")
```

---

## 🖥️ 서버 (NVIDIA GPU) 설정

### 1. Python 3.13 가상환경 생성
```bash
cd /path/to/project

# 가상환경 생성
python3.13 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 2. 패키지 설치
```bash
# 공통 패키지 설치
pip install -r requirements.txt

# 서버용 PyTorch 설치 (CUDA 12.4)
pip install -r requirements-cuda.txt
```

### 3. GPU 사용 확인
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ NVIDIA GPU 사용 가능! ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("⚠️ CPU 모드로 실행")

# 테스트
x = torch.rand(5, 3).to(device)
print(f"Device: {x.device}")
```

---

## 🐳 Docker 사용 (서버)

### Dockerfile 예시
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Python 3.13 설치
RUN apt-get update && apt-get install -y python3.13 python3.13-venv python3-pip

# 가상환경 생성
RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 패키지 설치
COPY requirements.txt requirements-cuda.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-cuda.txt

COPY . .

CMD ["python", "main.py"]
```

---

## ⚙️ 주요 패키지 버전

| 패키지 | 버전 | 용도 |
|--------|------|------|
| Python | 3.13.5 | 기본 인터프리터 |
| PyTorch | 2.5.1 | 딥러닝 프레임워크 |
| Ultralytics | 8.3.204 | YOLO 모델 |
| FastAPI | 0.118.0 | 웹 서버 |
| OpenCV | 4.12.0 | 이미지 처리 |
| NumPy | 2.2.6 | 수치 계산 |

---

## 🔍 문제 해결

### onnxruntime 설치 실패
- Python 3.14에서는 지원하지 않음
- Python 3.13 이하 사용 필요

### PyTorch CUDA 버전 오류
- CUDA 12.4 설치 확인: `nvcc --version`
- requirements-cuda.txt 사용 확인

### Mac에서 GPU 인식 안됨
- Apple Silicon (M1/M2/M3) 확인: `uname -m` → `arm64`
- requirements-mac.txt 사용 확인
- macOS 12.3 이상 필요

---

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. Python 버전: `python --version` (3.13.x)
2. 가상환경 활성화: 터미널에 `(venv)` 표시
3. 올바른 requirements 파일 사용

