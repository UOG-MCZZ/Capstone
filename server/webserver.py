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
from datetime import date, datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference


from werkzeug.security import safe_join


UPLOAD_FOLDER = './media/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xlsx'}

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

# Route to view the newly created table
@app.route('/')
def dasboard():
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

def insert_cert_row(row, table_name=""):
    insert_value = []
    for key, value in row.items():
        if pd.isnull(value) or pd.isna(value):
            insert_value.append("NULL")
        elif isinstance(value, datetime):
            insert_value.append(f"'{value.strftime('%Y-%m-%d')}'")
        else:
            insert_value.append(f"'{value}'")

    insert_sql = ", ".join(insert_value)
    insert_statement = text(f"INSERT INTO {table_name} VALUES({insert_sql})")

    try:
        db.session.execute(insert_statement)
        db.session.commit()
    except Exception as e:
        print(e)
        return "Failed"
    return "Passed"

@app.route('/upload_excel/<table_name>', methods=['GET', 'POST'])
def upload_excel_file(table_name):
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            cert_sheet = pd.read_excel(filepath, sheet_name="SurvSheet")
            results = cert_sheet.apply(insert_cert_row, axis=1, result_type="reduce", table_name=table_name)
            return render_template("upload_excel.html", table_name=table_name, results=zip(cert_sheet["Initial CBW Cert No."], results), fail_count=results.value_counts()["Failed"])
    return render_template("upload_excel.html", table_name=table_name)

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
    results = inference.get_rel(img)
    # class_results = server.webDocParser.runInference(img)
    # print(type(results))
    return {"links": results["link_boxes"], "key_val": results["key_val"], "pred": results["classes"], "boxes": results["block_bboxes"], "ocr_boxes": results["ocr_boxes"]}

@app.route('/new_table', methods=['GET', 'POST'])
def new_table():
    if request.method == 'POST':
        table_name = request.form['table_name']
        field_names = request.form.getlist('field_name[]')  # Get field names
        column_names = request.form.getlist('column_name[]')  # Get field names
        field_values = request.form.getlist('field_value[]')  # Get field names
        field_types = request.form.getlist('field_type[]')  # Get field types

        if len(field_names) != len(set(field_names)): return "duplicate field name", 400
        if len(column_names) != len(set(column_names)): return "duplicate column name", 400
        print("field names contain:", field_names)
        for i in range(len(field_names)):
            field_names[i] = re.sub('[^A-Za-z0-9 ]+', '', field_names[i])
            field_names[i] = field_names[i].lstrip().rstrip()

        # Prepare SQL statement to create the table
        columns = []
        for column_name, field_type in zip(column_names, field_types):
            if field_type == 'String':
                columns.append(f'`{column_name}` VARCHAR(255)')
            elif field_type == 'Integer':
                columns.append(f'`{column_name}` INT')
            elif field_type == 'Boolean':
                columns.append(f'`{column_name}` BOOLEAN')
            elif field_type == 'Date':
                columns.append(f'`{column_name}` DATE')
            # Add more types if necessary

        # Create the SQL query to create the new table
        columns_sql = ", ".join(columns)
        print(columns_sql)
        create_table_sql = text(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});")

        # Execute the raw SQL to create the table
        db.session.execute(create_table_sql)
        db.session.commit()

        # Prepare SQL statement to link form fields to column names
        for field_name, column_name in zip(field_names, column_names):
            values = (f"('{table_name}', '{field_name}', '{column_name}')")
            create_sql = text(f"INSERT INTO FormColumnConverter (TableName, FormFieldName, ColumnName) VALUES {values} ON DUPLICATE KEY UPDATE ColumnName = '{column_name}';")
            db.session.execute(create_sql)
        db.session.commit()

        #Change this for something useful later
        insert_value = []
        # for i in range(len(field_values)):
        for field_value, field_type in zip(field_values, field_types):
            if field_type == 'Integer':
                insert_value.append(f"{field_value}")
            else:
                insert_value.append(f"'{field_value}'")
        insert_sql = ", ".join(insert_value)
        insert_statement = text(f"INSERT INTO {table_name} VALUES({insert_sql})")
        db.session.execute(insert_statement)
        db.session.commit()

        return redirect(url_for('view_table', table_name=table_name))


    return render_template("new_table.html")

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

# Route to view the list of created tables
@app.route('/view_created_tables/')
def view_created_tables():
    # Query the table and get data (raw SQL query)
    result = db.session.execute(text(f"SHOW TABLES WHERE tables_in_dynamic_tables NOT LIKE 'CLS%' AND tables_in_dynamic_tables != 'FormColumnConverter';"))
    rows = result.fetchall()
    table_names = []
    for row in rows:
        table_names.append(row[0])

    return render_template('view_created_tables.html', table_names=table_names)

#Route to add data to created tables
@app.route('/add_table/<table_name>', methods=["GET", "POST"])
def add_table_data(table_name):
    if request.method == 'POST':
        table_name = request.form['table_name']
        # field_names = request.form.getlist('field_name[]')  # Get field names
        column_names = request.form.getlist('column_name[]')  # Get column names
        field_values = request.form.getlist('field_value[]')  # Get field values
        field_types = request.form.getlist('field_type[]')  # Get field types
        print(table_name)

        # prepare column names for insert
        column_values = []
        for column_name in column_names:
            column_values.append(f"`{column_name}`")
        column_sql = ", ".join(column_values)

        insert_value = []
        # for i in range(len(field_values)):
        for field_value, field_type in zip(field_values, field_types):
            if field_type == 'Integer':
                insert_value.append(f"{field_value}")
            else:
                insert_value.append(f"'{field_value}'")
        insert_sql = ", ".join(insert_value)
        insert_statement = text(f"INSERT INTO {table_name} ({column_sql}) VALUES({insert_sql})")
        db.session.execute(insert_statement)
        db.session.commit()
        
        if table_name.endswith("_MEC"): table_name = table_name[:-4] 
        return redirect(url_for("view_table", table_name=table_name))

    return render_template("add_to_existing_table.html", table_name=table_name)

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
        # field_names = request.form.getlist('field_name[]')  # Get field names
        column_names = request.form.getlist('column_name[]')  # Get column names
        field_values = request.form.getlist('field_value[]')  # Get field values
        # field_types = request.form.getlist('field_type[]')  # Get field types

        sn_index = column_names.index("SurveillanceSN")
        #This one should be an UPdate
        update_value = []
        for column_names_name, field_value in zip(column_names, field_values):
            update_value.append(f"{column_names_name} = '{field_value}'")
        update_sql = ", ".join(update_value)
        update_statement = text(f"UPDATE {table_name}_MEC SET {update_sql} WHERE SurveillanceSN = '{field_values[sn_index]}'")
        db.session.execute(update_statement)
        db.session.commit()
        
        return redirect(url_for("view_cert_mec_table", table_name=table_name, cert_name=cert_name))

    # table_name += "_MEC"
    return render_template("ExistingFormResultsPreview.html", name=SurveillanceSN, table_name=table_name, cert_name=cert_name, SurveillanceSN=SurveillanceSN)

# Route to add view DB information with an image
@app.route('/add_cert_table/<table_name>/<cert_name>', methods=["GET", "POST"])
def add_cert_table_data(table_name, cert_name):
    if request.method == 'POST':
        table_name = request.form['table_name']
        # field_names = request.form.getlist('field_name[]')  # Get field names
        column_names = request.form.getlist('column_name[]')  # Get column names
        field_values = request.form.getlist('field_value[]')  # Get field values
        field_types = request.form.getlist('field_type[]')  # Get field types
        print(table_name)

        # prepare column names for insert
        column_values = []
        for column_name in column_names:
            column_values.append(f"`{column_name}`")
        column_sql = ", ".join(column_values)

        insert_value = []
        # for i in range(len(field_values)):
        for field_value, field_type in zip(field_values, field_types):
            if field_type == 'Integer':
                insert_value.append(f"{field_value}")
            else:
                insert_value.append(f"'{field_value}'")
        insert_sql = ", ".join(insert_value)
        insert_statement = text(f"INSERT INTO {table_name} ({column_sql}) VALUES({insert_sql})")
        db.session.execute(insert_statement)
        db.session.commit()
        
        if table_name.endswith("_MEC"): table_name = table_name[:-4] 
        return redirect(url_for("view_cert_mec_table", table_name=table_name, cert_name=cert_name))

    # table_name += "_MEC"
    return render_template("add_to_existing_mec_table.html", table_name=table_name, cert_name=cert_name)

# Route to get the conversion info from form field to column name
@app.route('/api/get_table_conversion/<table_name>')
def get_table_converter(table_name):
    sql = text(f"SELECT * FROM  FormColumnConverter WHERE TableName = '{table_name}';")
    rows = db.session.execute(sql).fetchall()
    ret = {}
    for tableName, form_name, column_name in rows:
        ret[form_name] = column_name
    print(ret)
    return ret

if __name__ == '__main__':
    freeze_support()
    app.run(port = 12494, debug=True)