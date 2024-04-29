from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
# results = model.predict(show = True, source = r"fd7e20787941f578c9952f44feccf10c.jpg")

# Real-time predict
results = model.predict(show = True, source = 0)