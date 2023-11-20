from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8s.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='SK-Shielders-Module-Project-3_KoreanFOOD_Detecting-3\data.yaml', epochs=30)

# # Evaluate the model's performance on the validation set
# results = model.val()

# results = model.predict("datasets/OID/images/train/0a0a8ed8e0fc75cc_jpg.rf.910c6fdbe08b4b5668e7c769b25049ad.jpg")