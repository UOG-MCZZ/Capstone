// let isnum = (val) => /^\d+$/.test(val);
let isDate = (val) => /\d{1,4}\D+(\d{1,2}|\w{3,9})\D+\d{1,4}/.test(val);

function addField(fieldName, columnName, fieldValue) {
    var fieldDiv = document.createElement('div');
    fieldDiv.classList.add('field-group');
    
    var fieldNameInput = document.createElement('input');
    fieldNameInput.setAttribute('type', 'text');
    fieldNameInput.setAttribute('name', 'field_name[]');
    fieldNameInput.setAttribute('placeholder', 'Field name');
    // fieldNameInput.setAttribute('pattern', '[a-zA-Z0-9]+');
    fieldNameInput.value = fieldName;
    fieldNameInput.readOnly = true;
    
    var columnNameInput = document.createElement('input');
    columnNameInput.setAttribute('type', 'text');
    columnNameInput.setAttribute('name', 'column_name[]');
    columnNameInput.setAttribute('placeholder', 'Column name');
    columnNameInput.setAttribute('pattern', '[a-zA-Z0-9]+');
    columnNameInput.value = columnName;
    columnNameInput.readOnly = true;
    columnNameInput.required = true;

    var fieldValueInput = document.createElement('input');
    fieldValueInput.setAttribute('type', 'text');
    fieldValueInput.setAttribute('name', 'field_value[]');
    fieldValueInput.setAttribute('placeholder', 'Field value');
    fieldValueInput.value = fieldValue;
    if (columnName) fieldValueInput.id = columnName  + "_valueInput";

    var fieldTypeSelect = document.createElement('select');
    fieldTypeSelect.setAttribute('name', 'field_type[]');
    var types = ['String', 'Integer', 'Boolean', 'Date'];
    fieldTypeSelect.value = 'String'
    if (columnName) fieldTypeSelect.id = columnName  + "_typeInput";
    types.forEach(function(type) {
        var option = document.createElement('option');
        option.setAttribute('value', type);
        option.textContent = type;
        fieldTypeSelect.appendChild(option);
    });
    fieldTypeSelect.onchange = (evt) => {
        if (evt.target.value == "Date"){
            const d = new Date(fieldValueInput.value)
            fieldValueInput.setAttribute('type', 'date')
            if (d == "Invalid Date"){
                fieldValueInput.valueAsNumber = Date.now()
            } else{
                fieldValueInput.valueAsDate = d
            }
        }else{
            fieldValueInput.setAttribute('type', 'text');
        }
    }

    const d = new Date(fieldValue)
    if (isISODate(fieldValue) && d != "Invalid Date"){
        fieldTypeSelect.value = 'Date'
        const d = new Date(fieldValueInput.value)
        fieldValueInput.setAttribute('type', 'date')
        fieldValueInput.valueAsDate = d
    }

    // var removeButton = document.createElement('button');
    // removeButton.textContent = 'X';
    // removeButton.type = 'button';
    // removeButton.onclick = function() {
    //     fieldDiv.remove();
    // };

    fieldDiv.appendChild(fieldNameInput);
    fieldDiv.appendChild(columnNameInput);
    fieldDiv.appendChild(fieldValueInput);
    fieldDiv.appendChild(fieldTypeSelect);
    // fieldDiv.appendChild(removeButton);
    
    document.getElementById('fields-container').appendChild(fieldDiv);
}

function addFieldsFromDB(table_name, cert_name, SurveillanceSN) {
    fetch("/api/get/cert/" + table_name + "/" + cert_name + "/" + SurveillanceSN).then(res => res.json().then(j => {
        let a = {}
        for (const [label, value] of j){
            a[label] = value
            // addField("", label, value)
        }
        
        getFieldsToColumnNames(table_name, cert_name).then((converter) => {
            console.log(converter)
            console.log(a)
            for (const fieldName in converter){
                addField(fieldName, converter[fieldName], a[converter[fieldName]])
            }
        })
    }))
}

async function getFieldsToColumnNames(table_name, cert_name){
    if (cert_name) table_name += "_MEC"
    resp = await fetch("/api/get_table_conversion/" + table_name)
    return await resp.json()
}