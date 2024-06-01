from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import json
from parking_management import ParkingManagement
from werkzeug.utils import secure_filename
import logging

# Configuration class
class Config:
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
    MODEL_PATH = "./best.pt"
    POLYGON_JSON_PATH = "bounding_boxes.json"
    SECRET_KEY = 'supersecretkey'  # Necessary for flashing messages

# Initialize the Flask app
app = Flask(__name__)
app.config.from_object(Config)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    """Check if the file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def gen_frames(video_path):
    """Generate frames from the video for streaming."""
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
        results = management.model.track(frame, persist=True, show=False, classes=[3, 4, 5, 8, 9])
        
        if results[0] and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            management.process_data(json_data, frame, boxes, clss)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.split('.')[-1].lower() in {'mp4', 'avi', 'mov'}:
            # Video prediction
            logging.info(f"Video file uploaded: {file_path}")
            return jsonify({'message': 'Video file successfully uploaded', 'video_path': file_path})
        else:
            # Image prediction
            management = ParkingManagement(model_path=app.config['MODEL_PATH'])
            image = cv2.imread(file_path)
            json_data = management.parking_regions_extraction(app.config['POLYGON_JSON_PATH'])
            results = management.model.track(image)
            
            if results[0] and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                management.process_data(json_data, image, boxes, clss)
            
            prediction_filename = "prediction_" + filename
            prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction_filename)
            cv2.imwrite(prediction_path, image)
            
            image_url = os.path.join('uploads', filename).replace('\\', '/')
            prediction_url = os.path.join('uploads', prediction_filename).replace('\\', '/')
            
            logging.info(f"Image processed and saved: {prediction_path}")
            return jsonify({'message': 'Image successfully processed', 'image_path': image_url, 'prediction_path': prediction_url})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/save_boxes', methods=['POST'])
def save_boxes():
    boxes = request.json
    try:
        with open(app.config['POLYGON_JSON_PATH'], 'w') as f:
            json.dump(boxes, f)
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error saving bounding boxes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
