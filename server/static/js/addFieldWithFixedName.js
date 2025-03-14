let isnum = (val) => /^\d+$/.test(val);

function addField(fieldName, fieldValue) {
    var fieldDiv = document.createElement('div');
    fieldDiv.classList.add('field-group');
    
    var fieldNameInput = document.createElement('input');
    fieldNameInput.setAttribute('type', 'text');
    fieldNameInput.setAttribute('name', 'field_name[]');
    fieldNameInput.setAttribute('placeholder', 'Field name');
    // fieldNameInput.setAttribute('pattern', '[a-zA-Z0-9]+');
    fieldNameInput.value = fieldName;
    fieldNameInput.readOnly = true;
    fieldNameInput.required = true;

    var fieldValueInput = document.createElement('input');
    fieldValueInput.setAttribute('type', 'text');
    fieldValueInput.setAttribute('name', 'field_value[]');
    fieldValueInput.setAttribute('placeholder', 'Field value');
    fieldValueInput.value = fieldValue;

    var fieldTypeSelect = document.createElement('select');
    fieldTypeSelect.setAttribute('name', 'field_type[]');
    var types = ['String', 'Integer', 'Boolean', 'Date'];
    fieldTypeSelect.value = 'String'
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
                fieldValueInput.valueAsDate = Date.now()
            } else{
                fieldValueInput.valueAsDate = d
            }
        }else{
            fieldValueInput.setAttribute('type', 'text');
        }
    }

    const d = new Date(fieldValue)
    if (!isnum(fieldValue) && d != "Invalid Date"){
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
    fieldDiv.appendChild(fieldValueInput);
    fieldDiv.appendChild(fieldTypeSelect);
    // fieldDiv.appendChild(removeButton);
    
    document.getElementById('fields-container').appendChild(fieldDiv);
}

function addFieldsFromDB(table_name, cert_name, SurveillanceSN) {
    fetch("/api/get/cert/" + table_name + "/" + cert_name + "/" + SurveillanceSN).then(res => res.json().then(j => {
        console.log(j)
        for (const [label, value] of j){
            addField(label, value)
        }
    }))
}