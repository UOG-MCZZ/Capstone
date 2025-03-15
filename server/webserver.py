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
from datetime import date

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

@app.route('/new_table', methods=['GET', 'POST'])
def new_table():
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

        return redirect(url_for('view_table', table_name=table_name))


    return render_template("new_table.html")

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

# Route to view Certification tables
@app.route('/view_cert_table/<table_name>')
def view_cert_table(table_name):
    # Query the table and get data (raw SQL query)
    frequencies = []
    results = []
    db_frequencies = db.session.execute(text(f"SELECT DISTINCT(SurveillanceFrequency) FROM {table_name};"))
    for frequency in db_frequencies.fetchall():
        frequency = frequency[0]
        frequencies.append(frequency)
        result = db.session.execute(text(f"SELECT * FROM {table_name} WHERE SurveillanceFrequency = '{frequency}';"))
        rows = result.fetchall()
        results.append(rows)

    return render_template('view_table2.html', data_tables=zip(frequencies, results), table_name=table_name)

# Route to view Certification MEC Table
@app.route('/view_cert_table/<table_name>/<cert_name>')
def view_cert_mec_table(table_name, cert_name):
    # Query the table and get data (raw SQL query)
    result = db.session.execute(text(f"SELECT * FROM {table_name}_MEC WHERE InitialCBWCertNumber = '{cert_name}';"))
    rows = result.fetchall()

    return render_template('view_mec_table.html', rows=rows, table_name=table_name, cert_name=cert_name)

# Route to view the newly created table
@app.route('/view_table/<table_name>')
def view_table(table_name):
    # Query the table and get data (raw SQL query)
    result = db.session.execute(text(f"SELECT * FROM {table_name};"))
    rows = result.fetchall()

    return render_template('view_table.html', rows=rows, table_name=table_name)

# Route to view the newly created table
@app.route('/dash')
def dashboard2():
    table_list = ["CLS1B"] # get this dynamically
    expiry_info_list = []

    # Query the table and get data (raw SQL query)
    for table in table_list:
        # can be faster with single query, need parse after tho, so? faster?
        # expiring_result = db.session.execute(text(f"SELECT SurveillanceStatusMonitoring, COUNT(1) FROM {table} GROUP BY SurveillanceStatusMonitoring;")).fetchall()
        expiring_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table} WHERE SurveillanceStatusMonitoring = 'Surveillance Expiring';")).fetchone()
        expired_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table} WHERE SurveillanceStatusMonitoring = 'Surveillance Expired';")).fetchone()
        completed_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table} WHERE SurveillanceStatusMonitoring = 'Surveillance Completed';")).fetchone()
        expiry_info_list.append((table, expiring_result[0], completed_result[0], expired_result[0]))

    return render_template('dashboard2.html', expiry_info_list=expiry_info_list)

#route to get information of each month
@app.route("/api/get/cert/<table_name>/monthly_summary")
def get_certificate_monthly_summary(table_name):
    d = date.today().replace(day=1)
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    expiring = []
    expired = []
    completed = []

    for month_idx, month in enumerate(months, 1):
        d = d.replace(month=month_idx)
        date_str = str(d)
        if d.month < date.today().month:
            expiring_result = [0]
            expired_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring = 'Surveillance Expired' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 1 month;")).fetchone()
            completed_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring = 'Surveillance Completed' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 1 month;")).fetchone()
        elif d.month == date.today().month:
            expired_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring = 'Surveillance Expired' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 1 month;")).fetchone()
            completed_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring = 'Surveillance Completed' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 1 month;")).fetchone()
            expiring_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring != 'Surveillance Completed'  AND SurveillanceStatusMonitoring != 'Surveillance Expired' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 3 month;;")).fetchone()
        else:
            expired_result = [0]
            expiring_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring != 'Surveillance Completed' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 3 month;")).fetchone()
            completed_result = db.session.execute(text(f"SELECT COUNT(1) FROM {table_name} WHERE SurveillanceStatusMonitoring = 'Surveillance Completed' AND SurveillanceYearEndDate between '{date_str}' and '{date_str}' + interval 1 month;")).fetchone()
        expired.append(expired_result[0])
        expiring.append(expiring_result[0])
        completed.append(completed_result[0])

    return {"months": months, "expiring": expiring, "completed": completed, "expired": expired}

@app.route("/api/get/cert/<table_name>/<cert_name>/<SurveillanceSN>")
def get_mec_info(table_name, cert_name, SurveillanceSN):
    # Query the table and get data (raw SQL query)
    result = db.session.execute(text(f"SELECT * FROM {table_name}_MEC WHERE SurveillanceSN = '{SurveillanceSN}';"))
    row = result.fetchone()
    print(list(zip(row._fields, row._t)))

    return list(zip(row._fields, row._t))

# Route to add view DB information with an image
@app.route('/view_cert_table/<table_name>/<cert_name>/<SurveillanceSN>', methods=["GET", "POST"])
def view_current_table_data(table_name, cert_name, SurveillanceSN):
    if request.method == 'POST':
        table_name = request.form['table_name']
        field_names = request.form.getlist('field_name[]')  # Get field names
        field_values = request.form.getlist('field_value[]')  # Get field names
        # field_types = request.form.getlist('field_type[]')  # Get field types

        sn_index = field_names.index("SurveillanceSN")
        #This one should be an UPdate
        update_value = []
        for field_name, field_value in zip(field_names, field_values):
            update_value.append(f"{field_name} = '{field_value}'")
        update_sql = ", ".join(update_value)
        update_statement = text(f"UPDATE {table_name}_MEC SET {update_sql} WHERE SurveillanceSN = '{field_values[sn_index]}'")
        db.session.execute(update_statement)
        db.session.commit()
        
        return redirect(url_for("view_cert_mec_table", table_name=table_name, cert_name=cert_name))

    return render_template("ExistingFormResultsPreview.html", name=SurveillanceSN, table_name=table_name, cert_name=cert_name, SurveillanceSN=SurveillanceSN)

# Home route
@app.route('/')
def index():
    return 'Welcome to the dynamic table app!'

if __name__ == '__main__':
    freeze_support()
    app.run(port = 12494, debug=True)






