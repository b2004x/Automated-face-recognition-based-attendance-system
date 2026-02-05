from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Form
import io, tempfile, torch
import joblib
import numpy as np
import streamlit as st
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, ImageDraw, ImageFont
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import subprocess
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision.models import vit_b_16
from torchvision import transforms
import json
import io
import base64
from pathlib import Path
app = FastAPI(title="Face detection with RetinaFace and Vision Transformers")

# Load model
##############################################################################################################
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
model.eval()

base = Path(__file__).resolve().parent

##############################################################################################################
model_celeb_face = vit_b_16(weights="IMAGENET1K_V1")
weight_path = base.parent.parent / "model" / "Face_recognition" / "best_vit_face.pth"
num_classes = 17  # 👈 change this to your number of celebrity classes
model_celeb_face.heads.head = torch.nn.Linear(model_celeb_face.heads.head.in_features, num_classes)
state_dict = torch.load(weight_path, map_location="cpu")
model_celeb_face.load_state_dict(state_dict)
model_celeb_face.eval()



transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

with open("class_names.json", "r") as f:
    class_names = json.load(f)

############################################################################################################################################
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
##############################################################################################################################
# 2. Preprocess and get embedding with mtcnn and facenet_pytorch
@app.post("/predict")

def get_face_embedding(image):
    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected!")
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding.numpy()

# Function to get embedding from image path with vision_transformer
def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output  # [1, hidden_size]
    return embedding.numpy()

def face_detection(tmp_path):

    project_root = Path(__file__).resolve().parent 

    model_path = (project_root / ".." / ".." / "model" / "Face_detection" / "resnet34_final.pth").resolve()
    retinaface_path = (project_root / ".." / ".." / "retinaface-pytorch").resolve()
    
    
    
    process = subprocess.Popen(
        [
            "python", "detect.py",
            "-n", "resnet34",
            "-w", str(model_path),
            "-s",
            "--image-path", tmp_path
        ],
        cwd= retinaface_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
        )   

    output_area = st.empty()  # a placeholder for updating logs

    log = ""
    for line in process.stdout:
            log += line
            output_area.text(log)  # update Streamlit UI with new output

    process.wait()
    st.success("✅ RetinaFace finished!")

def get_crop_face_detection(ref_img):
    data_path = base.parent.parent  / "retinaface-pytorch" / "detections.npz"

    data = np.load(data_path)
    st.success(data.files)
    boxes = data['boxes']  # x1, y1, x2, y2
    scores = data['scores']  
    # 2️⃣ Load the same image used in inference
    image = Image.open(ref_img).convert("RGB")
    H, W = image.height, image.width

    # 3️⃣ Convert to NumPy array for cropping
    threshold = 0.5
    mask = scores > threshold
    boxes = boxes[mask]

    face_regions = []
    for x1, y1, x2, y2 in boxes:
        # clamp coordinates to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))
        
        # crop region using PIL
        face_region = image.crop((x1, y1, x2, y2))
        face_regions.append((face_region, (x1, y1, x2, y2)))
        
        # optional preview
        # face_region.show()
    return face_regions

def compare_embeddings(emb1, emb2):
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return sim

def draw_rectangle(face, celeb_image):
    data_path = base.parent.parent  / "retinaface-pytorch" / "detections.npz"
    data = np.load(data_path)
    boxes = data['boxes']




@app.post("/Save_Face_Embedding/")
async def Save_Face_Embedding(file: UploadFile = File(...), name: str = Form(...)):
    ## Load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
    face_detection(tmp_path)
    ## Get face regions
    face_regions = get_crop_face_detection(tmp_path)

    if not face_regions:
        return JSONResponse({"message": "❌ No face detected in image."}, status_code=400)
    
    ## Get embedding for the first detected face
    first_face, _ = face_regions[0]
    ref_embedding = get_face_embedding(first_face)

    if os.path.exists("face_embeddings.npy"):
        database = np.load("face_embeddings.npy", allow_pickle=True).item()
    else:
        database = {}
    
    if name in database:
        # Average new embedding with old one (optional for consistency)
        old_emb = database[name]
        database[name] = (old_emb + ref_embedding) / 2.0
    else:
        database[name] = ref_embedding

    np.save("face_embeddings.npy", database)

    ###draw bounding box on full image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for face, (x1, y1, x2, y2) in face_regions:
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, max(0, y1 - 25)), name, fill="green", font=font)
    
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "message": f"✅ Face embedding saved for '{name}'. Total entries: {len(database)}",
        "saved_names": list(database.keys()),
        "image_base64": img_str
    })


@app.post("/Recognize_Face/")
async def Recognize_Face(file: UploadFile = File(...)):
    ## Load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
    face_detection(tmp_path)
    face_regions = get_crop_face_detection(tmp_path)

    if not face_regions:
        return JSONResponse({"message": "No face detected in image."}, status_code=400)
    
    ## Get embedding for the first detected face
    first_face, _ = face_regions[0]
    query_embedding = get_face_embedding(first_face)
    database = np.load("face_embeddings.npy", allow_pickle=True).item()

    draw = ImageDraw.Draw(image)  
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    ## Compare with database
    matches = []
    for name, embedding in database.items():
        result = compare_embeddings(query_embedding, embedding)
        if result > 0.7:
           matches.append((name, result))
    ###draw bounding box on full image
    for face, (x1, y1, x2, y2) in face_regions:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if matches:
            text = ", ".join([f"{n} ({s:.2f})" for n, s in matches])
            draw.text((x1, max(0, y1 - 25)), text, fill="green", font=font)
    # Convert image to base64 string for JSON response 
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    if matches:
        msg = "Match found: " + ", ".join([f"{n} ({s:.2f})" for n, s in matches])
    else:
        msg = "No match found."

    return JSONResponse({
        "message": msg,
        "matches": [{"name": n, "similarity": float(s)} for n, s in matches],
        "image_base64": img_str
    })

@app.post("/Recognize_Celebrity/")
async def Recognize_Celebrity(file: UploadFile = File(...)):
    ## Load image
    image_bytes = await file.read()
    celeb_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            celeb_image.save(tmp.name)
            tmp_path = tmp.name
    face_detection(tmp_path)    
    face_regions = get_crop_face_detection(tmp_path)
    ## Recognize celebrity for each detected face
    draw = ImageDraw.Draw(celeb_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for face, (x1, y1, x2, y2) in face_regions:
        face_tensor = transforms(face).unsqueeze(0)
        with torch.no_grad():
            outputs = model_celeb_face(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()

            predicted_name = class_names[pred_class]

                # draw rectangle on face on FULL IMAGE
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 30), predicted_name, fill="Green", font=font)
    ## Return image with bounding boxes and names
    buf = io.BytesIO()
    celeb_image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({
        "message": "Celebrity recognition completed.",
        "image_base64": img_str
    })
        
    
    


