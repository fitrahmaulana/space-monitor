from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import json
from parking_management import ParkingManagement
from werkzeug.utils import secure_filename
import logging

# Konfigurasi aplikasi
class Config:
    UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Direktori untuk menyimpan file yang diunggah
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp4', 'avi', 'mov', 'json'}  # Ekstensi file yang diperbolehkan
    MODEL_PATH = "./best.pt"  # Jalur ke model yang digunakan untuk prediksi
    POLYGON_JSON_PATH = "bounding_boxes.json"  # Jalur ke file JSON yang berisi bounding boxes
    SECRET_KEY = 'supersecretkey'  # Kunci rahasia untuk aplikasi Flask

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config.from_object(Config)

# Membuat direktori upload jika belum ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Inisialisasi logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    """Cek apakah file diizinkan berdasarkan ekstensi."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_bounding_boxes(bounding_boxes):
    """Menyimpan bounding boxes ke file JSON."""
    try:
        with open(app.config['POLYGON_JSON_PATH'], 'w') as f:
            json.dump(bounding_boxes, f)
        logging.info("Bounding boxes saved successfully.")
    except Exception as e:
        logging.error(f"Error saving bounding boxes: {e}")
        raise

def process_image(file_path, conf_threshold, show_labels):
    """Memproses gambar untuk prediksi."""
    management = ParkingManagement(model_path=app.config['MODEL_PATH'])
    image = cv2.imread(file_path)
    json_data = management.parking_regions_extraction(app.config['POLYGON_JSON_PATH'])
    results = management.model.track(image, conf=conf_threshold, persist=True)

    if results[0] and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        conf = results[0].boxes.conf.cpu().tolist()
        management.process_data(json_data, image, boxes, clss, conf, show_labels)

    return image

def gen_frames(video_path, conf_threshold, show_labels):
    """Menghasilkan frame dari video untuk streaming."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error reading video file {video_path}")
        return

    management = ParkingManagement(model_path=app.config['MODEL_PATH'])
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        json_data = management.parking_regions_extraction(app.config['POLYGON_JSON_PATH'])
        results = management.model.track(frame, conf=conf_threshold, persist=True, classes=[3, 4, 5, 8, 9])

        if results[0] and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            conf = results[0].boxes.conf.cpu().tolist()
            management.process_data(json_data, frame, boxes, clss, conf, show_labels)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Halaman utama."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediksi dari file yang diunggah."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    conf_threshold = request.form.get('conf_threshold', type=float, default=0.5)
    show_labels = request.form.get('show_labels') == "on"
    bounding_box_file = request.files.get('bounding_box')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if bounding_box_file and allowed_file(bounding_box_file.filename):
            try:
                bounding_boxes = json.load(bounding_box_file)
                save_bounding_boxes(bounding_boxes)
            except Exception as e:
                return "Error reading bounding boxes.", 500

        if filename.split('.')[-1].lower() in {'mp4', 'avi', 'mov'}:
            logging.info(f"Video file uploaded: {file_path}")
            return jsonify({'message': 'Video file successfully uploaded', 'video_path': file_path, 'conf_threshold': conf_threshold, 'show_labels': show_labels})
        else:
            try:
                image = process_image(file_path, conf_threshold, show_labels)
                prediction_filename = "prediction_" + filename
                prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction_filename)
                cv2.imwrite(prediction_path, image)

                image_url = os.path.join('uploads', filename).replace('\\', '/')
                prediction_url = os.path.join('uploads', prediction_filename).replace('\\', '/')

                logging.info(f"Image processed and saved: {prediction_path}")
                return jsonify({'message': 'Image successfully processed', 'image_path': image_url, 'prediction_path': prediction_url})
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return jsonify({'error': 'Error processing image'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/save_boxes', methods=['POST'])
def save_boxes():
    """Menyimpan bounding boxes ke file JSON."""
    boxes = request.json
    try:
        save_bounding_boxes(boxes)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/video_feed')
def video_feed():
    """Streaming video dengan bounding boxes."""
    video_path = request.args.get('video_path')
    conf_threshold = request.args.get('conf_threshold', type=float, default=0.5)
    show_labels = request.args.get('show_labels') == 'true'
    return Response(gen_frames(video_path, conf_threshold=conf_threshold, show_labels=show_labels), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
