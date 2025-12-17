from ultralytics import YOLO



# Build a YOLOv9c model from pretrained weight
model = YOLO("/workspace3/weights/last.pt")


# # # Train the model on the COCO8 example dataset for 100 epochs
results_train = model.train(data="/workspace/dataset/data.yaml", epochs=200, imgsz=416,name = "/workspace/workspace",batch=64,device = [0,1],lr0=0.005)


# results_val = model.val(
#             data='/workspace/data.yaml',         # 데이터셋 YAML 파일 경로
#             split="val",           # 검증 데이터셋 분할 (e.g., "test")
#             save_json=True,         # 검증 결과를 JSON 파일로 저장 (True 또는 False)
#             project='/workspace',       # 검증 결과 저장 디렉토리
#             name="validation_results",      # 저장 폴더 이름
#             imgsz=640,              # 이미지 크기
#             batch=1,                # 배치 크기 (사용자의 컴퓨팅 자원에 맞게 설정)          # 검증에 사용할 디바이스 ('cpu': CPU, 0: 첫 번째 GPU, [0,1,2,3]: 여러 GPU)
#             plots=True,
#             exist_ok=True         )     # 플롯 생성 여부