import os
# import JSON
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image


import webDocParser

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 #16 MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/uploads')
def list_files():
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    return filenames

@app.route('/process/<name>')
def process_document(name):
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], name))
    results = webDocParser.runInference(img)
    print(type(results))
    return results

@app.route('/preview/<name>')
def preview_results(name):
    return f'''
<canvas id="Preview"/>
<script>
const id2label = {{0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}}
const id2color = ["violet", "orange", "orange", "blue", "blue", "green", "green"]

const canva = document.getElementById("Preview")
const ctx = canva.getContext("2d");
const img = new Image();

img.src = "/uploads/{name}"
img.addEventListener("load", () => {{
  canva.width = img.width;
  canva.height = img.height;
  ctx.drawImage(img, 0, 0);
}});
fetch("/process/{name}").then(res => res.json().then(j => {{
  for (var i = 0; i < j["boxes"].length; i++){{
    const bbox = j["boxes"][i]
    let box = bbox;
    box[3] = bbox[3] - bbox[1];
    box[2] = bbox[2] - bbox[0];
    console.log(j["pred"][i])
    ctx.strokeStyle = id2color[j["pred"][i]]
    ctx .strokeRect(...box);
  }}
}}))
</script>
'''