from flask import Flask, render_template, request, redirect
import cv2
import pandas as pd
import numpy as np
import pytesseract
from ultralytics import YOLO
from datetime import datetime
import os

app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\Dell\Downloads\license_plate_recognition1-master\license_plate_recognition1-master\uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO(r"C:\Users\Dell\Downloads\license_plate_recognition1-master\license_plate_recognition1-master\best.pt")

processed_numbers = set()

area = np.array([(10, 400), (10, 480), (1030, 480), (1030, 400)], np.int32)

def preprocess_image(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 20, 20)
    return gray

def extract_plate(frame):
    results = model.predict(frame)
    a = results[0].boxes.data
    if len(a) == 0:
        return [], pd.DataFrame()  
    
    px = pd.DataFrame(a).astype("float")
    extracted_texts = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        crop = frame[y1:y2, x1:x2]
        processed_crop = preprocess_image(crop)
        text = pytesseract.image_to_string(processed_crop).strip()
        cleaned_text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').strip()
        if cleaned_text and cleaned_text not in processed_numbers:
            processed_numbers.add(cleaned_text)
            extracted_texts.append(cleaned_text)
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("car_plate_data.txt", "a") as file:
                file.write(f"{cleaned_text}\t{current_datetime}\n")

    return extracted_texts, px

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    extracted_texts = []
    count = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break  

        if count % 3 != 0:  
            continue

        frame = cv2.resize(frame, (1020, 500))
        cv2.polylines(frame, [area], True, (255, 0, 0), 2)

        results = model.predict(frame)
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")
        
        if px.empty:
            print("No detections in this frame.")
            continue 

        for index, row in px.iterrows():
            x1, y1, x2, y2, conf, cls = row  
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        
            print(f"Detected box: {(x1, y1, x2, y2)} with confidence: {conf}")

            
            if (x1 >= area[0][0] and x2 <= area[2][0] and
                y1 >= area[0][1] and y2 <= area[2][1]):
                
               
                crop = frame[y1:y2, x1:x2]
                processed_crop = preprocess_image(crop)

                text = pytesseract.image_to_string(processed_crop).strip()
                cleaned_text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').strip()
                
                if cleaned_text and cleaned_text not in processed_numbers:
                    processed_numbers.add(cleaned_text) 
                    extracted_texts.append(cleaned_text)
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("car_plate_data.txt", "a") as file:
                        file.write(f"{cleaned_text}\t{current_datetime}\n")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print(f"Detected but already processed: {cleaned_text}")

        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()
   
    return "\n".join(set(extracted_texts))

def process_image(image_path):
    img = cv2.imread(image_path)
    frame = cv2.resize(img, (1020, 500))
    extracted_texts, _ = extract_plate(frame)
    return extracted_texts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):  
        extracted_texts = process_video(file_path)
        extracted_text_output = extracted_texts.split("\n") if extracted_texts else "No plates detected."
        return render_template('index.html', extracted_text=extracted_text_output)
    else: 
        extracted_texts = process_image(file_path)
        extracted_text_output = extracted_texts if extracted_texts else "No plates detected."
        return render_template('index.html', extracted_text=extracted_text_output)

if __name__ == '__main__':
    app.run(debug=True)

