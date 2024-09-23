import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

area = [(30, 19), (16, 456), (1015, 451), (992, 40)]

input_folder = 'input_images/'
processed_numbers = set()

with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Unable to read {image_path}")
        continue
    
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    if len(results) > 0:
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d] if d < len(class_list) else 'Unknown'
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                text = pytesseract.image_to_string(gray).strip()
                text = re.sub(r'[^\w\s]', '', text)

                if text and text not in processed_numbers:
                    processed_numbers.add(text)
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open("car_plate_data.txt", "a") as file:
                        file.write(f"{text}\t{current_datetime}\n")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow('Cropped License Plate', crop)
                    print(f"Detected License Plate: {text}")

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

    cv2.imshow("Processed Image", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
