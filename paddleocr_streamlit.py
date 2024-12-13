from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings("ignore")
import os
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)
app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
def image_to_text(files):
    try:
        if not files:
            return jsonify({"error": "No files provided"}), 400
        detected_text = ""
        for idx, file in enumerate(files, start=1):
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return jsonify({"error": f"Invalid file type: {file.filename}"}), 400
            file_path = f"temp_image_{idx}.png"
            file.save(file_path)
            result = ocr.ocr(file_path)
            #detected_text += f"Image {idx} text:\n"
            for line in result[0]:
                detected_text += line[1][0] +" "
            detected_text += "\n\n"
            os.remove(file_path)
        return detected_text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/image_text', methods=['POST'])
def ocr():
    files = request.files.getlist("file")
    output_answer = image_to_text(files)
    return output_answer

if __name__ == "__main__":
    print("Starting the server on port 8080")
    #flask_app.run(debug=False, host="0.0.0.0", port=8080)
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    print('Server running on http://0.0.0.0:8080')
    http_server.serve_forever()

