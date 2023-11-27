from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import os
from web import perform_style_transfer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods=['POST'])
def upload_image():
    print("Received request: POST /uploader")
    if request.method == 'POST':
        style_file = request.files['style_image']
        content_file = request.files['content_image']
        if style_file and content_file:
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_file.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_file.filename)
            style_file.save(style_path)
            content_file.save(content_path)

            output_path = os.path.join(OUTPUT_FOLDER, 'output_image.jpg')
            perform_style_transfer(style_path, content_path, output_path)

            return send_file(output_path, mimetype='image/jpeg')
        else:
            return 'No selected file'

@app.route('/latest-image')
def latest_image():
    try:
        files = os.listdir(OUTPUT_FOLDER)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)), reverse=True)
        if files:
            print(files)
            newest_file = files[0]
            with open(os.path.join(OUTPUT_FOLDER, newest_file), 'rb') as image_file:
                return send_file(image_file, mimetype='image/jpeg')
        else:
            return 'No images found', 404
    except FileNotFoundError:
        return 'No images found', 404
    
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)