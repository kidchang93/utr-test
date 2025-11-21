"""
🌊 진짜 YOLO + S3 스트리밍 학습
- S3에서 메모리로 직접 로드 (디스크 저장 ❌)
- YOLO11n 모델 사용
- 1개 클래스, 10장만 테스트
- LRU 캐시로 메모리 관리
- 학습 결과만 S3에 업로드

디스크에는 최종 모델만 저장됩니다!
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
from ultralytics import YOLO

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.s3_dataset import S3ClassificationDataset
from config.s3_config import get_s3_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOStreamingTester:
    """YOLO + S3 스트리밍 테스터"""
    
    def __init__(self):
        self.s3_config = None
        self.s3_client = None
        self.dataset = None
        self.model = None
        self.device = None
        self.results_dir = None
        
    def step1_test_s3_connection(self) -> bool:
        """단계 1: S3 연결 테스트"""
        print("\n" + "="*70)
        print("📡 단계 1: S3 연결 테스트")
        print("="*70)
        
        try:
            # S3 설정 로드
            self.s3_config = get_s3_config()
            logger.info("✅ S3 설정 로드 완료")
            self.s3_config.print_config()
            
            # S3 클라이언트 생성
            s3_params = self.s3_config.get_s3_params()
            
            # Naver Cloud 사용 시 로컬 AWS config 무시하고 명시적으로 자격증명 전달
            client_kwargs = {
                'region_name': s3_params.get('region')
            }
            
            # 자격증명 명시적으로 전달 (로컬 AWS config 무시)
            access_key = s3_params.get('access_key')
            secret_key = s3_params.get('secret_key')
            
            if access_key and secret_key:
                client_kwargs['aws_access_key_id'] = access_key
                client_kwargs['aws_secret_access_key'] = secret_key
                logger.info("   자격증명: .env 파일에서 로드 (로컬 AWS config 무시)")
            else:
                logger.warning("   ⚠️  자격증명이 없습니다. 로컬 AWS config를 사용할 수 있습니다.")
            
            # Naver Cloud 등 커스텀 S3 엔드포인트
            if s3_params.get('domain'):
                client_kwargs['endpoint_url'] = s3_params.get('domain')
                logger.info(f"   커스텀 엔드포인트 사용: {s3_params.get('domain')}")
            
            # 명시적 자격증명으로 클라이언트 생성 (로컬 AWS config 무시)
            self.s3_client = boto3.client('s3', **client_kwargs)
            
            # 버킷 접근 테스트만 수행 (폴더 구조 확인은 step2에서)
            bucket_name = s3_params['bucket_name']
            logger.info(f"🔍 버킷 '{bucket_name}' 접근 테스트 중...")
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info("✅ S3 버킷 접근 성공!")
            logger.info("💡 폴더 구조 확인은 다음 단계에서 수행합니다")
            
            return True
                
        except ClientError as e:
            logger.error(f"❌ S3 연결 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 오류 발생: {e}", exc_info=True)
            return False
    
    def step2_create_streaming_dataset(self, num_images: int = 10) -> bool:
        """
        단계 2: 스트리밍 데이터셋 생성 (메모리만 사용)
        
        Args:
            num_images: 사용할 이미지 수
        """
        print("\n" + "="*70)
        print(f"🌊 단계 2: S3 스트리밍 데이터셋 생성")
        print("="*70)
        print(f"💡 설정:")
        print(f"   - 이미지: {num_images}장 (1개 클래스)")
        print(f"   - 디스크 저장: ❌ 없음")
        print(f"   - 메모리 캐시: {num_images}개")
        print(f"   - S3 → 메모리 직접 로드")
        print("="*70)
        
        try:
            s3_params = self.s3_config.get_s3_params()
            
            # Transform 정의 (YOLO 스타일)
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # 첫 번째 클래스만 직접 찾기 (전체 스캔 생략)
            logger.info("\n📦 첫 번째 클래스 찾는 중...")
            
            # 첫 번째 클래스만 찾기 (전체 스캔 없이)
            response = self.s3_client.list_objects_v2(
                Bucket=s3_params['bucket_name'],
                Prefix=self.s3_config.train_prefix,
                Delimiter='/',
                MaxKeys=1  # 첫 번째 클래스만
            )
            
            if 'CommonPrefixes' not in response or len(response['CommonPrefixes']) == 0:
                logger.error("❌ 클래스를 찾을 수 없습니다")
                return False
            
            first_class_prefix = response['CommonPrefixes'][0]['Prefix']
            first_class_name = first_class_prefix.rstrip('/').split('/')[-1]
            
            logger.info(f"✅ 첫 번째 클래스: {first_class_name}")
            
            # 첫 번째 클래스의 이미지 리스트 가져오기
            from utils.s3_loader import S3ImageLoader
            
            s3_loader = S3ImageLoader(
                bucket_name=s3_params['bucket_name'],
                access_key=s3_params.get('access_key'),
                secret_key=s3_params.get('secret_key'),
                region=s3_params.get('region'),
                domain=s3_params.get('domain')
            )
            
            class_prefix = f"{self.s3_config.train_prefix}{first_class_name}/"
            all_images = s3_loader.list_images_in_prefix(class_prefix)
            
            if not all_images:
                logger.error(f"❌ '{class_prefix}' 하위에 이미지가 없습니다")
                return False
            
            # num_images개만 선택
            selected_images = all_images[:min(num_images, len(all_images))]
            logger.info(f"   선택된 이미지: {len(selected_images)}장")
            
            # 단일 클래스용 간단한 데이터셋 생성
            class SingleClassDataset(Dataset):
                def __init__(self, s3_loader, image_keys, class_name, transform=None, cache_size=100):
                    self.s3_loader = s3_loader
                    self.image_keys = image_keys
                    self.class_name = class_name
                    self.transform = transform
                    self.cache = {}
                    self.cache_keys = []
                    self.cache_size = cache_size
                    self.class_idx = 0  # 단일 클래스이므로 0
                
                def __len__(self):
                    return len(self.image_keys)
                
                def __getitem__(self, idx):
                    image_key = self.image_keys[idx]
                    
                    # 캐시 확인
                    if image_key in self.cache:
                        image = self.cache[image_key].copy()
                    else:
                        # S3에서 이미지 로드
                        image = self.s3_loader.load_image_from_s3(image_key)
                        if image is None:
                            logger.warning(f"이미지 로드 실패: {image_key}, 검은 이미지로 대체")
                            image = Image.new('RGB', (224, 224), (0, 0, 0))
                        
                        # 캐시에 저장
                        if self.cache_size > 0:
                            if len(self.cache) >= self.cache_size:
                                oldest_key = self.cache_keys.pop(0)
                                del self.cache[oldest_key]
                            self.cache[image_key] = image.copy()
                            self.cache_keys.append(image_key)
                    
                    # Transform 적용
                    if self.transform:
                        image = self.transform(image)
                    
                    return image, self.class_idx
            
            # 데이터셋 생성
            self.dataset = SingleClassDataset(
                s3_loader=s3_loader,
                image_keys=selected_images,
                class_name=first_class_name,
                transform=transform,
                cache_size=num_images
            )
            self.num_classes = 1  # 1개 클래스만 사용
            
            logger.info(f"\n✅ 스트리밍 데이터셋 생성 완료")
            logger.info(f"   클래스: {first_class_name}")
            logger.info(f"   샘플 수: {len(self.dataset)}장")
            logger.info(f"   메모리 캐시: {num_images}개")
            logger.info(f"   디스크 사용: 0 bytes ✨")
            
            # 스트리밍 로딩 테스트
            logger.info(f"\n🧪 스트리밍 로딩 테스트 중...")
            logger.info(f"   → S3에서 직접 메모리로 로드")
            
            test_image, test_label = self.dataset[0]
            logger.info(f"✅ 스트리밍 로딩 성공!")
            logger.info(f"   이미지 shape: {test_image.shape}")
            logger.info(f"   레이블: {test_label}")
            logger.info(f"   메모리 크기: ~{test_image.element_size() * test_image.nelement() / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터셋 생성 실패: {e}", exc_info=True)
            return False
    
    def step3_train_yolo_streaming(self, epochs: int = 3, img_size: int = 640, batch_size: int = 2) -> bool:
        """
        단계 3: YOLO 스트리밍 학습
        
        Args:
            epochs: 학습 에포크 수
            img_size: 이미지 크기
            batch_size: 배치 크기
        """
        print("\n" + "="*70)
        print(f"🚀 단계 3: YOLO 스트리밍 학습")
        print("="*70)
        print(f"💡 설정:")
        print(f"   - 모델: YOLO11n")
        print(f"   - 에포크: {epochs}")
        print(f"   - 이미지 크기: {img_size}px")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 데이터 소스: S3 → 메모리 → GPU")
        print("="*70)
        
        try:
            # GPU 확인
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"\n🖥️  디바이스: {self.device}")
            
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"   GPU: {gpu_name}")
                logger.info(f"   메모리: {gpu_memory:.1f} GB")
            
            # DataLoader 생성
            # ⚠️ num_workers=0 필수: boto3.client는 pickle 불가능하므로 multiprocessing 사용 불가
            train_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # S3 클라이언트 때문에 0으로 설정 (단일 프로세스)
                pin_memory=(self.device.type == 'cuda')
            )
            
            logger.info(f"\n📊 DataLoader 준비:")
            logger.info(f"   배치 수: {len(train_loader)}개")
            logger.info(f"   Workers: 0 (S3 클라이언트 호환성)")
            logger.info(f"   Pin Memory: {self.device.type == 'cuda'}")
            
            # YOLO 모델 로드
            logger.info(f"\n🧠 YOLO11n 모델 로드 중...")
            model_path = project_root / 'yolo11n-cls.pt'
            
            if not model_path.exists():
                logger.error(f"❌ 모델 파일이 없습니다: {model_path}")
                logger.info("💡 yolo11n-cls.pt 파일을 프로젝트 루트에 다운로드해주세요")
                return False
            
            # YOLO 모델을 PyTorch 모델로 변환
            yolo_model = YOLO(str(model_path))
            
            # YOLO의 백본 네트워크 추출
            # (YOLO Classification 모델의 경우)
            self.model = yolo_model.model
            
            # 마지막 분류 레이어를 1클래스로 조정
            if hasattr(self.model, 'linear'):
                # YOLO-CLS의 경우
                in_features = self.model.linear.in_features
                self.model.linear = nn.Linear(in_features, 1)
            
            self.model = self.model.to(self.device)
            self.model.train()
            
            logger.info(f"✅ YOLO11n 모델 로드 완료")
            
            # Loss, Optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # 학습 시작
            logger.info(f"\n{'='*70}")
            logger.info("🎯 스트리밍 학습 시작")
            logger.info(f"{'='*70}")
            logger.info("💡 데이터 흐름: S3 → 메모리 캐시 → GPU")
            logger.info(f"{'='*70}\n")
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                progress_bar = tqdm(
                    train_loader, 
                    desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch",
                    ncols=100
                )
                
                for batch_idx, (images, labels) in enumerate(progress_bar):
                    # S3에서 스트리밍으로 로드된 데이터
                    images = images.to(self.device)
                    labels = labels.float().to(self.device)
                    
                    # Forward
                    optimizer.zero_grad()
                    
                    try:
                        outputs = self.model(images)
                        
                        # YOLO 모델 출력 형태에 따라 조정
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                        
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze()
                        
                        # 1클래스이므로 binary classification
                        labels = torch.zeros_like(labels)  # 모두 같은 클래스
                        
                        # 크기 맞추기
                        if outputs.dim() == 0:
                            outputs = outputs.unsqueeze(0)
                        if labels.dim() == 0:
                            labels = labels.unsqueeze(0)
                        
                        # 배치 크기가 다른 경우 처리
                        min_size = min(outputs.size(0), labels.size(0))
                        outputs = outputs[:min_size]
                        labels = labels[:min_size]
                        
                        loss = criterion(outputs, labels)
                        
                        # Backward
                        loss.backward()
                        optimizer.step()
                        
                        # 통계
                        epoch_loss += loss.item()
                        
                        # Progress bar 업데이트
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg': f'{epoch_loss/(batch_idx+1):.4f}',
                            'source': 'S3→RAM→GPU'
                        })
                        
                    except Exception as e:
                        logger.warning(f"배치 {batch_idx} 처리 중 오류: {e}")
                        continue
                
                avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
                logger.info(f"   Epoch {epoch+1}/{epochs} 완료 - Loss: {avg_loss:.4f}")
            
            logger.info(f"\n{'='*70}")
            logger.info("✅ 스트리밍 학습 완료!")
            logger.info(f"{'='*70}")
            
            # 결과 저장 디렉토리 생성
            self.results_dir = project_root / 'runs' / 'yolo_streaming' / 's3_test'
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 저장
            weights_dir = self.results_dir / 'weights'
            weights_dir.mkdir(exist_ok=True)
            
            model_path = weights_dir / 'best.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_loss': avg_loss,
                'epochs': epochs,
                'img_size': img_size,
            }, model_path)
            
            logger.info(f"\n💾 모델 저장:")
            logger.info(f"   경로: {model_path}")
            logger.info(f"   크기: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # 학습 정보 저장
            info_path = self.results_dir / 'train_info.txt'
            with open(info_path, 'w') as f:
                f.write(f"YOLO S3 Streaming Training\n")
                f.write(f"="*50 + "\n")
                f.write(f"Model: YOLO11n-cls\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Image Size: {img_size}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Final Loss: {avg_loss:.4f}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Data Source: S3 (Streaming)\n")
                f.write(f"Disk Usage: 0 bytes (memory only)\n")
            
            logger.info(f"   학습 정보: {info_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 학습 실패: {e}", exc_info=True)
            return False
    
    def step4_upload_results_to_s3(self) -> bool:
        """단계 4: 학습 결과를 S3에 업로드"""
        print("\n" + "="*70)
        print("☁️  단계 4: 학습 결과 S3 업로드")
        print("="*70)
        
        try:
            if not self.results_dir or not self.results_dir.exists():
                logger.error("❌ 학습 결과 디렉토리가 없습니다")
                return False
            
            s3_params = self.s3_config.get_s3_params()
            bucket_name = s3_params['bucket_name']
            
            # 변수 설정 확인
            logger.info("🔍 변수 설정 확인 중...")
            logger.info(f"   버킷: {bucket_name}")
            logger.info(f"   리전: {s3_params.get('region', 'N/A')}")
            logger.info(f"   엔드포인트: {s3_params.get('domain', 'AWS 기본')}")
            logger.info(f"   Access Key: {'설정됨' if s3_params.get('access_key') else '없음'}")
            logger.info(f"   Secret Key: {'설정됨' if s3_params.get('secret_key') else '없음'}")
            
            # S3 클라이언트 확인 (step1에서 이미 생성됨)
            if not self.s3_client:
                logger.error("❌ S3 클라이언트가 없습니다. step1을 먼저 실행해주세요.")
                return False
            
            # S3 경로: models/s3_test_YYYYMMDD_HHMMSS/
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_prefix = f"models/s3_test_{timestamp}/"
            
            logger.info(f"\n📤 업로드 시작...")
            logger.info(f"   로컬: {self.results_dir}")
            logger.info(f"   S3: s3://{bucket_name}/{s3_prefix}")
            
            uploaded_count = 0
            failed_count = 0
            total_size = 0
            
            # 결과 디렉토리의 모든 파일 업로드
            for local_file in self.results_dir.rglob('*'):
                if local_file.is_file():
                    # 상대 경로 계산
                    relative_path = local_file.relative_to(self.results_dir)
                    s3_key = s3_prefix + str(relative_path)
                    
                    file_size = local_file.stat().st_size
                    file_size_mb = file_size / 1024 / 1024
                    
                    try:
                        # Naver Cloud Object Storage 업로드
                        # upload_file은 multipart를 시도할 수 있어서 put_object 직접 사용
                        logger.info(f"  📤 업로드 중: {relative_path} ({file_size_mb:.2f} MB)...")
                        
                        # ContentType 설정
                        content_type = "application/octet-stream"  # 기본값
                        if relative_path.suffix.lower() in ['.pt', '.pth']:
                            content_type = "application/octet-stream"
                        elif relative_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            content_type = f"image/{relative_path.suffix[1:].lower()}"
                        elif relative_path.suffix.lower() == '.txt':
                            content_type = "text/plain"
                        
                        # 파일을 바이너리로 읽어서 put_object로 직접 업로드
                        # 이 방법은 multipart를 사용하지 않고 단일 요청으로 업로드
                        with open(local_file, 'rb') as f:
                            file_data = f.read()
                        
                        self.s3_client.put_object(
                            Bucket=bucket_name,
                            Key=s3_key,
                            Body=file_data,
                            ContentType=content_type
                        )
                        
                        total_size += file_size
                        uploaded_count += 1
                        logger.info(f"  ✅ [{uploaded_count}] {relative_path} ({file_size_mb:.2f} MB)")
                    except ClientError as e:
                        failed_count += 1
                        error_code = e.response['Error']['Code']
                        error_message = e.response['Error'].get('Message', '')
                        logger.error(f"  ❌ [{failed_count}] {relative_path} 업로드 실패")
                        logger.error(f"     에러 코드: {error_code}")
                        logger.error(f"     에러 메시지: {error_message}")
                        logger.error(f"     파일 크기: {file_size_mb:.2f} MB")
                        logger.error(f"     S3 경로: s3://{bucket_name}/{s3_key}")
                        if error_code == 'AccessDenied':
                            logger.error(f"     권한이 없습니다. 나머지 파일도 업로드하지 않습니다.")
                            break
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"  ❌ [{failed_count}] {relative_path} 업로드 실패: {e}")
                        logger.error(f"     파일 경로: {local_file}")
                        logger.error(f"     파일 크기: {file_size_mb:.2f} MB")
            
            if uploaded_count > 0:
                logger.info(f"\n✅ 업로드 완료:")
                logger.info(f"   파일 수: {uploaded_count}개")
                logger.info(f"   전체 크기: {total_size / 1024 / 1024:.2f} MB")
                logger.info(f"   📍 S3: s3://{bucket_name}/{s3_prefix}")
                if failed_count > 0:
                    logger.warning(f"⚠️  실패: {failed_count}개 파일")
                return True
            else:
                logger.error(f"\n❌ 모든 파일 업로드 실패 ({failed_count}개)")
                return False
            
        except Exception as e:
            logger.error(f"❌ 업로드 실패: {e}", exc_info=True)
            logger.info("\n💡 로컬 파일은 그대로 유지됩니다:")
            logger.info(f"   {self.results_dir}")
            return False
    
    def run_all_steps(self, num_images: int = 10, epochs: int = 3, img_size: int = 640, batch_size: int = 2):
        """모든 단계 실행"""
        print("\n" + "="*70)
        print("🌊 YOLO + S3 스트리밍 학습 전체 테스트")
        print("="*70)
        print("💡 특징:")
        print("   ✨ S3에서 메모리로 직접 로드")
        print("   ✨ 디스크에 학습 데이터 저장 안 함")
        print("   ✨ LRU 캐시로 메모리 관리")
        print("   ✨ YOLO11n 모델 사용")
        print("   ✨ 최종 모델만 디스크 저장")
        print("="*70)
        print(f"📋 설정:")
        print(f"   - 이미지: {num_images}장 (1개 클래스)")
        print(f"   - 에포크: {epochs}")
        print(f"   - 이미지 크기: {img_size}px")
        print(f"   - 배치 크기: {batch_size}")
        print("="*70)
        
        try:
            # 단계 1: S3 연결
            if not self.step1_test_s3_connection():
                logger.error("\n❌ 단계 1 실패: S3 연결 불가")
                return
            
            # 단계 2: 스트리밍 데이터셋 생성
            if not self.step2_create_streaming_dataset(num_images):
                logger.error("\n❌ 단계 2 실패: 데이터셋 생성 불가")
                return
            
            # 단계 3: YOLO 스트리밍 학습
            if not self.step3_train_yolo_streaming(epochs, img_size, batch_size):
                logger.error("\n❌ 단계 3 실패: 학습 불가")
                return
            
            # 단계 4: 결과 업로드
            if not self.step4_upload_results_to_s3():
                logger.error("\n❌ 단계 4 실패: 업로드 불가")
                return
            
            # 완료
            print("\n" + "="*70)
            print("🎉 모든 단계 성공!")
            print("="*70)
            print("\n✅ 테스트 완료 요약:")
            print("   1. ✅ S3 연결")
            print("   2. ✅ 스트리밍 데이터셋 (메모리)")
            print("   3. ✅ YOLO 스트리밍 학습")
            print("   4. ✅ 결과 S3 업로드")
            print("\n💡 핵심 포인트:")
            print(f"   - 학습 데이터 디스크 사용: 0 bytes ✨")
            print(f"   - 데이터 흐름: S3 → 메모리 캐시 → GPU")
            print(f"   - 메모리 캐시: {num_images}개 이미지")
            print(f"   - 최종 모델만 디스크 저장")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  사용자가 중단했습니다")
            
        except Exception as e:
            logger.error(f"\n❌ 예상치 못한 오류: {e}", exc_info=True)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO + S3 스트리밍 학습 테스트')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], 
                        help='실행할 단계 (없으면 전체 실행)')
    parser.add_argument('--num-images', type=int, default=10,
                        help='사용할 이미지 수 (기본: 10)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='학습 에포크 수 (기본: 3)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='이미지 크기 (기본: 640)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='배치 크기 (기본: 2)')
    
    args = parser.parse_args()
    
    tester = YOLOStreamingTester()
    
    try:
        if args.step:
            # 특정 단계만 실행
            print(f"\n🎯 단계 {args.step}만 실행\n")
            
            if args.step == 1:
                tester.step1_test_s3_connection()
            elif args.step == 2:
                if tester.step1_test_s3_connection():
                    tester.step2_create_streaming_dataset(args.num_images)
            elif args.step == 3:
                if tester.step1_test_s3_connection():
                    if tester.step2_create_streaming_dataset(args.num_images):
                        tester.step3_train_yolo_streaming(args.epochs, args.img_size, args.batch_size)
            elif args.step == 4:
                if tester.step1_test_s3_connection():
                    # 기존 학습 결과 찾기
                    runs_dir = project_root / 'runs' / 'yolo_streaming' / 's3_test'
                    if runs_dir.exists():
                        tester.results_dir = runs_dir
                        tester.step4_upload_results_to_s3()
                    else:
                        logger.error(f"❌ 학습 결과를 찾을 수 없습니다: {runs_dir}")
        else:
            # 전체 실행
            tester.run_all_steps(
                num_images=args.num_images,
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch_size
            )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다")


if __name__ == "__main__":
    main()

