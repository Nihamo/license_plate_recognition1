from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
import os
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\Dell\Downloads\license_plate_recognition1-master\license_plate_recognition1-master\uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO(r"C:\Users\Dell\Downloads\license_plate_recognition1-master\license_plate_recognition1-master\best.pt")

def process_image(image_path):
    img = cv2.imread(image_path)
    frame = cv2.resize(img, (1020, 500))
    
    results = model.predict(frame)
    boxes = results[0].boxes.data
    extracted_text = ""
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 10, 20, 20)
        text = pytesseract.image_to_string(gray).strip()
        extracted_text += text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
        
    return extracted_text

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

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        extracted_text = process_image(file_path)

        return render_template('index.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
