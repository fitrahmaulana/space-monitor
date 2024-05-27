# Import necessary libraries
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import os
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
        flash('No file part')
        return redirect("/")
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect("/")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.split('.')[-1].lower() in {'mp4', 'avi', 'mov'}:
            # Video prediction
            logging.info(f"Video file uploaded: {file_path}")
            flash('Video file successfully uploaded')
            return redirect(url_for('video_prediction', video_path=file_path))
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
            flash('Image successfully processed')
            return render_template('prediction.html', image_path=image_url, prediction_path=prediction_url)
    
    flash('File type not allowed')
    return redirect("/")

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_prediction')
def video_prediction():
    video_path = request.args.get('video_path')
    return render_template('videoPrediction.html', video_path=video_path)

if __name__ == "__main__":
    app.run(debug=True)
