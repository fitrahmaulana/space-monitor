# Import necessary libraries
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from parking_management import ParkingManagement
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Path to json file, that created with above point selection app
polygon_json_path = "bounding_boxes.json"

# Initialize parking management object
management = ParkingManagement(model_path="./best.pt")


def gen_frames(video_path):  
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    while True:
        success, frame = cap.read()  # read the cap frame
        if not success:
            break
        else:
            json_data = management.parking_regions_extraction(polygon_json_path)

            results = management.model.track(frame, persist=True, show=False, classes=[3,4,5,8,9])

            if results[0] and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                management.process_data(json_data, frame, boxes, clss)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imagePrediction')
def imagePrediction():
    return render_template('imagePrediction.html')

@app.route('/videoPrediction')
def videoPrediction():
    video_path = request.args.get('video_path')
    return render_template('videoPrediction.html', video_path=video_path)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('videoPrediction', video_path=file_path))

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image and save the prediction result
        image = cv2.imread(file_path)
        json_data = management.parking_regions_extraction(polygon_json_path)
        results = management.model.track(image)

        if results[0] and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            management.process_data(json_data, image, boxes, clss)
        
        prediction_filename = "prediction_" + filename
        prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction_filename)
        cv2.imwrite(prediction_path, image)

        # Ensure the paths are in URL format
        image_url = os.path.join('uploads', filename).replace('\\', '/')
        prediction_url = os.path.join('uploads', prediction_filename).replace('\\', '/')

        return render_template('imagePrediction.html', 
                               image_path=image_url,
                               prediction_path=prediction_url)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
