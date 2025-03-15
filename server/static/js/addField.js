function addField(fieldName, fieldValue) {
    var fieldDiv = document.createElement('div');
    fieldDiv.classList.add('field-group');
    
    var fieldNameInput = document.createElement('input');
    fieldNameInput.setAttribute('type', 'text');
    fieldNameInput.setAttribute('name', 'field_name[]');
    fieldNameInput.setAttribute('placeholder', 'Field name');
    // fieldNameInput.setAttribute('pattern', '[a-zA-Z0-9]+');
    fieldNameInput.value = fieldName;
    fieldNameInput.required = true;

    var fieldValueInput = document.createElement('input');
    fieldValueInput.setAttribute('type', 'text');
    fieldValueInput.setAttribute('name', 'field_value[]');
    fieldValueInput.setAttribute('placeholder', 'Field value');
    fieldValueInput.value = fieldValue;

    var fieldTypeSelect = document.createElement('select');
    fieldTypeSelect.setAttribute('name', 'field_type[]');
    var types = ['String', 'Integer', 'Boolean', 'Date'];
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

    var removeButton = document.createElement('button');
    removeButton.textContent = 'X';
    removeButton.type = 'button';
    removeButton.onclick = function() {
        fieldDiv.remove();
    };

    fieldDiv.appendChild(fieldNameInput);
    fieldDiv.appendChild(fieldValueInput);
    fieldDiv.appendChild(fieldTypeSelect);
    fieldDiv.appendChild(removeButton);
    
    document.getElementById('fields-container').appendChild(fieldDiv);
}