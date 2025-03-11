var MLResults = undefined

async function getMLResults (name){
  if (MLResults) return MLResults

  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  const img = new Image();
  const btns = document.getElementById("PreviewButtons")
  const imgContainer = document.getElementById("image-container")

  const styleSetUp = (e) => {
    canva.width = img.width;
    canva.height = img.height;
    // ctx.drawImage(img, 0, 0);
    imgContainer.style.height = img.height+ "px";
    imgContainer.style.width = img.width+ "px";
    canva.width = img.width;
    canva.height = img.height;
    img.style.position = "absolute"
    img.style.top = "0"
    img.style.left = "0"
    img.removeEventListener("load", styleSetUp)
    imgContainer.appendChild(img)
    canva.style.maxWidth = "100%";
    canva.style.maxHeight = "90%";
    img.style.maxHeight = "90%";
    img.style.maxWidth = "100%";
    btns.style.top = img.offsetHeight + "px";
  }
  img.src = "/uploads/" + name
  img.addEventListener("load", styleSetUp);

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
      ctx.lineWidth = Math.floor((canva.height + canva.width) / 200) + 1;
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
  const canva = document.getElementById("LabelPreview")
  const ctx = canva.getContext("2d");
  ctx.clearRect(0,0,canva.width, canva.height)
}