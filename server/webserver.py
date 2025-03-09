import os, sys
# import JSON
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from config import Config
from sqlalchemy.sql import text
from werkzeug.utils import secure_filename
from PIL import Image
from multiprocessing import freeze_support
import regex as re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gptinf


from werkzeug.security import safe_join


UPLOAD_FOLDER = './media/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 #16 MB
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
app.config.from_object(Config)

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def dasboard():
        return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
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
    print(safe_join(app.config["UPLOAD_FOLDER"], name))
    print(os.path.abspath(safe_join(app.config["UPLOAD_FOLDER"], name)))
    # return send_file(safe_join(app.config["UPLOAD_FOLDER"], name),  _root_path="./")
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/uploads')
def list_files():
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    return filenames

# Other pages
@app.route('/process/<name>')
def process_document(name):
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], name))
    results = gptinf.get_rel(img)
    # class_results = server.webDocParser.runInference(img)
    # print(type(results))
    return {"links": results["link_boxes"], "key_val": results["key_val"], "pred": results["classes"], "boxes": results["block_bboxes"], "ocr_boxes": results["ocr_boxes"]}

@app.route('/preview/<name2>', methods=['GET', 'POST'])
def preview_results(name2):
    if request.method == 'POST':
        table_name = request.form['table_name']
        field_names = request.form.getlist('field_name[]')  # Get field names
        field_values = request.form.getlist('field_value[]')  # Get field names
        field_types = request.form.getlist('field_type[]')  # Get field types
        print("field names contain:", field_names)
        for i in range(len(field_names)):
            field_names[i] = re.sub('[^A-Za-z0-9 ]+', '', field_names[i])
            field_names[i] = field_names[i].lstrip().rstrip()

        # Prepare SQL statement to create the table
        columns = []
        for field_name, field_type in zip(field_names, field_types):
            if field_type == 'String':
                columns.append(f'`{field_name}` VARCHAR(255)')
            elif field_type == 'Integer':
                columns.append(f'`{field_name}` INT')
            elif field_type == 'Boolean':
                columns.append(f'`{field_name}` BOOLEAN')
            # Add more types if necessary

        # Create the SQL query to create the new table
        columns_sql = ", ".join(columns)
        print(columns_sql)
        create_table_sql = text(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});")

        # Execute the raw SQL to create the table
        db.session.execute(create_table_sql)
        db.session.commit()

        #Change this for something useful later
        insert_value = []
        # for i in range(len(field_values)):
        for field_value, field_type in zip(field_values, field_types):
            if field_type == 'String':
                insert_value.append(f'"{field_value}"')
            elif field_type == 'Integer':
                insert_value.append(f'{field_value}')
            elif field_type == 'Boolean':
                insert_value.append(f'"{field_value}"')
        insert_sql = ", ".join(insert_value)
        insert_statement = text(f"INSERT INTO {table_name} VALUES({insert_sql})")
        db.session.execute(insert_statement)
        db.session.commit()

        return '''
    <!doctype html>
    <title>Table Created Successfully</title>
    <h1>Table Created Successfully!</h1>
    '''

    results = process_document(name2)
    return render_template("demo.html", key_val=results["key_val"], name=name2)

# Route to create dynamic table
@app.route('/create_table', methods=['GET', 'POST'])
def create_table():
    if request.method == 'POST':
        table_name = request.form['table_name']
        field_names = request.form.getlist('field_name[]')  # Get field names
        field_types = request.form.getlist('field_type[]')  # Get field types

        # Prepare SQL statement to create the table
        columns = []
        for field_name, field_type in zip(field_names, field_types):
            if field_type == 'String':
                columns.append(f"{field_name} VARCHAR(255)")
            elif field_type == 'Integer':
                columns.append(f"{field_name} INT")
            elif field_type == 'Boolean':
                columns.append(f"{field_name} BOOLEAN")
            # Add more types if necessary

        # Create the SQL query to create the new table
        print(columns)
        columns_sql = ", ".join(columns)
        print(columns_sql)
        create_table_sql = text(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});")

        # Execute the raw SQL to create the table
        db.session.execute(create_table_sql)
        db.session.commit()

        #Change this for something useful later
        return '''
    <!doctype html>
    <title>Table Created Successfully</title>
    <h1>Table Created Successfully!</h1>
    '''
        # return redirect(url_for('view_table', table_name=table_name))

    return render_template('create_table.html')

# Route to view the newly created table
@app.route('/view_table/<table_name>')
def view_table(table_name):
    # Query the table and get data (raw SQL query)
    result = db.session.execute(text(f"SELECT * FROM {table_name}"))
    rows = result.fetchall()

    return render_template('view_table.html', rows=rows, table_name=table_name)

# Route to add new file information to the table
@app.route('/table/<table_name>/process/<file_name>', methods=["GET", "POST"])
def add_form_to_table(table_name, file_name):
    if request.method == 'POST':
        # table_name = request.form['table_name']
        # field_names = request.form.getlist('field_name[]')  # Get field names
        field_values = request.form.getlist('field_value[]')  # Get field names
        field_types = request.form.getlist('field_type[]')  # Get field types

        #Change this for something useful later
        insert_value = []
        for field_value, field_type in zip(field_values, field_types):
            if field_type == 'String':
                insert_value.append(f'"{field_value}"')
            elif field_type == 'Integer':
                insert_value.append(f'{field_value}')
            elif field_type == 'Boolean':
                insert_value.append(f'"{field_value}"')
        insert_sql = ", ".join(insert_value)
        insert_statement = text(f"INSERT INTO {table_name} VALUES({insert_sql})")
        db.session.execute(insert_statement)
        db.session.commit()
        
        return redirect(url_for("view_table", table_name=table_name))

    # Query the table and get data (raw SQL query)
    # To try a better method of getting column names
    db_result = db.session.execute(text(f"SELECT * FROM {table_name}"))
    rows = db_result.fetchall()
    new_key_val = dict()
    # column_names = []
    for column in rows[0]._fields:
        # column_names.append(column)
        new_key_val[column] = ""

    results = process_document(file_name)
    for key, val in results["key_val"]:
        field_name = re.sub('[^A-Za-z0-9 ]+', '', key)
        field_name = field_name.lstrip().rstrip()
        if field_name in new_key_val.keys():
            new_key_val[field_name] = val

    print(new_key_val)
    print(type(new_key_val))
    for key, val in new_key_val.items():
        print(key, val)
    return render_template("ExistingFormResultsPreview.html", new_key_val=new_key_val, key_val=results["key_val"], name=file_name, table_name=table_name)

# Home route
@app.route('/')
def index():
    return 'Welcome to the dynamic table app!'

if __name__ == '__main__':
    freeze_support()
    app.run(port = 12494, debug=True)






