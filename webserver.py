import os
# import JSON
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from PIL import Image


import gptinf
import server.webDocParser

UPLOAD_FOLDER = './'
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
    results = gptinf.get_rel(img)
    # class_results = server.webDocParser.runInference(img)
    # print(type(results))
    return {"links": results["link_boxes"], "key_val": results["key_val"], "pred": results["classes"], "boxes": results["block_bboxes"], "ocr_boxes": results["ocr_boxes"]}

@app.route('/preview/<name2>')
def preview_results(name2):
    results = process_document(name2)
    return render_template("demo.html", key_val=results["key_val"], name=name2)

if __name__ == "__main__":
    app.run()