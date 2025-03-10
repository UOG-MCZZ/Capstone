var MLResults = undefined

function getMLResults (name){
  if (MLResults) return new Promise(() => {return MLResults})

  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  // const img = new Image();
  const img = document.getElementById("Preview")
  const btns = document.getElementById("PreviewButtons")
  
  btns.style.top = img.height
  // img.src = "/uploads/" + name
  if (img.complete) {
    canva.width = img.width;
    canva.height = img.height;
    canva.style.width = img.style.width;
    canva.style.height = img.style.height;
  } else
    img.addEventListener("load", () => {
      canva.width = img.width;
      canva.height = img.height;
    });

  return fetch("/process/" + name).then(res => res.json().then(j => {
    MLResults = j;
    return MLResults
  }))
}

function drawLinkedBoxes (name) {
  const id2color = ["violet", "orange", "blue", "green"]
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");

  getMLResults(name).then((MLResults) => {
    for (var i = 0; i < MLResults["boxes"].length; i++){
      const bbox = MLResults["boxes"][i]
      let box = [...bbox[0], ...bbox[2]];
      box[3] = box[3] - box[1];
      box[2] = box[2] - box[0];
      console.log(MLResults["pred"][i])
      console.log(box)
      ctx.lineWidth = 10;
      ctx.strokeStyle = id2color[MLResults["pred"][i]]
      ctx .strokeRect(...box);
    }
  })
}

function drawLinkLines(name) {
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  
  getMLResults(name).then((MLResults) => {
    ctx.strokeStyle = "black"
    for (var i = 0; i < MLResults["links"].length; i++){
      const link = MLResults["links"][i]
      console.log(link);
      let box = link[0];
      ctx.moveTo(...box[1]);
      box = link[1]
      ctx.lineTo(...box[0]);
      ctx .stroke();
    }
  })

  return fetch("/process/" + name).then(res => res.json().then(j => {
    MLResults = j;
    return MLResults
  }))
}

function drawLinkedBoxes (name) {
  const id2color = ["violet", "orange", "blue", "green"]
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");

  getMLResults(name).then((MLResults) => {
    for (var i = 0; i < MLResults["boxes"].length; i++){
      const bbox = MLResults["boxes"][i]
      let box = [...bbox[0], ...bbox[2]];
      box[3] = box[3] - box[1];
      box[2] = box[2] - box[0];
      console.log(MLResults["pred"][i])
      console.log(box)
      ctx.lineWidth = 10;
      ctx.strokeStyle = id2color[MLResults["pred"][i]]
      ctx .strokeRect(...box);
    }
  })
}

function drawLinkLines(name) {
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  
  getMLResults(name).then((MLResults) => {
    ctx.strokeStyle = "black"
    for (var i = 0; i < MLResults["links"].length; i++){
      const link = MLResults["links"][i]
      console.log(link);
      let box = link[0];
      ctx.moveTo(...box[1]);
      box = link[1]
      ctx.lineTo(...box[0]);
      ctx .stroke();
    }
  })
}

function drawAllBoxes(name) {
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  
  getMLResults(name).then((MLResults) => {
    ctx.strokeStyle = "red"
    for (var i = 0; i < MLResults["ocr_boxes"].length; i++){
      const bbox = MLResults["ocr_boxes"][i]
      let box = [...bbox[0], ...bbox[2]];
      box[3] = box[3] - box[1];
      box[2] = box[2] - box[0];
      ctx .strokeRect(...box);
    }
  })
}

function clearlabel () {
function drawAllBoxes(name) {
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  
  getMLResults(name).then((MLResults) => {
    ctx.strokeStyle = "red"
    for (var i = 0; i < MLResults["ocr_boxes"].length; i++){
      const bbox = MLResults["ocr_boxes"][i]
      let box = [...bbox[0], ...bbox[2]];
      box[3] = box[3] - box[1];
      box[2] = box[2] - box[0];
      ctx .strokeRect(...box);
    }
  })
}

function clearlabel () {
  const canva = document.getElementById("Preview")
  const ctx = canva.getContext("2d");
  ctx.clearRect(0,0,canva.width, canva.height)
}