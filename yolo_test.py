from ultralytics import YOLO

# model = YOLO("yolo11n.yaml")
#
# # model = YOLO("yolo11n.pt")
#
# results = model.train(data="coco8.yaml", epochs=3)
#
# success = model.export(format="onnx")

# 학습된 모델 로드
model = YOLO("trains/runs/classify/korean_food/weights/best.pt")

# 테스트할 이미지 경로
img_path = "C:/Users/Quzz/Downloads/4QEyaXWxmg3mT9fR-resized.webp"

# 예측 수행
results = model(img_path)

# 결과 확인
predicted_class = results[0].names[results[0].probs.top1]
confidence = results[0].probs.top1conf.item()

print(f"예측: {predicted_class} (신뢰도: {confidence:.1%})")

# 상위 3개 예측
top3_indices = results[0].probs.top5[:3]
top3_names = [results[0].names[int(i)] for i in top3_indices]
top3_confs = [results[0].probs.data[int(i)].item() for i in top3_indices]

for i, (name, conf) in enumerate(zip(top3_names, top3_confs), 1):
    print(f"{i}순위: {name} ({conf:.1%})")

results[0].plot()  # 예측 클래스 표시된 이미지 생성
results[0].show()  # 창으로 보기