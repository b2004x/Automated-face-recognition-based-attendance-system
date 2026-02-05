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


# Load model
##############################################################################################################
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
model.eval()

##############################################################################################################
model_celeb_face = vit_b_16(weights="IMAGENET1K_V1")
weight_path = r"G:\hoc\private\Face_detection\model\Face_recognition\best_vit_face.pth"
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
# 2. Preprocess and get embedding
def get_face_embedding(image):
    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected!")
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding.numpy()

# Function to get embedding from image path
def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output  # [1, hidden_size]
    return embedding.numpy()



def face_detection(tmp_path):
    process = subprocess.Popen(
        [
            "python", "detect.py",
            "-n", "resnet34",
            "-w", r"G:\hoc\private\Face_detection\model\Face_detection\resnet34_final.pth",
            "-s",
            "--image-path", tmp_path
        ],
        cwd=r"G:\hoc\private\Face_detection\retinaface-pytorch",
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

    data = np.load(r"G:\hoc\private\Face_detection\retinaface-pytorch\detections.npz")
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
        face_region.show()
    return face_regions

def compare_embeddings(emb1, emb2):
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return sim

def draw_rectangle(face, celeb_image):
    data = np.load(r"G:\hoc\private\Face_detection\retinaface-pytorch\detections.npz")
    boxes = data['boxes']


    







############################################################################################################################################

st.set_page_config(page_title="Face Image Demo", layout="wide")

st.title("🧠 Face Image Upload Demo")

col1, col2, col3 = st.columns(3)

# 1️⃣ Reference Face (embedding)
with col1:
    st.header("Reference Face")
    ref_img = st.file_uploader("Upload reference face image", type=["jpg", "jpeg", "png"], key="ref")
    if ref_img:
        st.image(Image.open(ref_img), caption="Reference Face", use_column_width=True)
    else:
        st.info("Please upload a reference image.")

# 2️⃣ Image to Compare
with col2:
    st.header("Test Image")
    test_img = st.file_uploader("Upload test image", type=["jpg", "jpeg", "png"], key="test")
    if test_img:
        st.image(Image.open(test_img), caption="Test Image", use_column_width=True)
    else:
        st.info("Please upload a test image.")

# 3️⃣ Another General Image
with col3:
    st.header("Celebrity detection Image")
    other_img = st.file_uploader("Upload another image", type=["jpg", "jpeg", "png"], key="other")
    if other_img:
        st.image(Image.open(other_img), caption="Other Image", use_column_width=True)
    else:
        st.info("Please upload an additional image.")

############################################################################################################################################
# Optional: a save/check button section
st.divider()
st.subheader("Actions")
col_a, col_b , col_c = st.columns(3)



with col_a:
    if st.button("💾 Save Face Embedding"):
        with st.spinner("Saves faces... Please wait"):
            ref_image = Image.open(ref_img).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                ref_image.save(tmp.name)
                tmp_path = tmp.name

            st.write(f"Temporary path: `{tmp_path}`")
            face_detection(tmp_path)
            face_regions = get_crop_face_detection(tmp_path)
            first_face, _ = face_regions[0]
            ref_embedding = get_embedding(first_face)  # assuming first detected face
            st.success("Reference face embedding saved!")
            database = {
                "person": ref_embedding,
            }
            np.save("face_embeddings.npy", database)
            print("Saved embeddings to face_embeddings.npy")
        st.success("✅ Face embedding saved!")





with col_b:
    if st.button("🔍 Compare Faces"):
        with st.spinner("Compares faces... Please wait"):
            test_image = Image.open(test_img).convert("RGB")
            ref_image = Image.open(ref_img).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                test_image.save(tmp.name)
                tmp_path = tmp.name

            st.write(f"Temporary path: `{tmp_path}`")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                ref_image.save(tmp.name)
                tmp_path2 = tmp.name

            st.write(f"Temporary path: `{tmp_path2}`")
            # draw_test = ImageDraw.Draw(test_image)
            face_detection(tmp_path)
            face_regions = get_crop_face_detection(tmp_path)
            first_face, _ = face_regions[0]
            # draw_test.rectangle( _ , outline="red", width=3)

            # draw_ref = ImageDraw.Draw(ref_image)
            face_detection(tmp_path2)
            face_regions2 = get_crop_face_detection(tmp_path2)
            first_face_2, _ = face_regions2[0]
            # draw_ref.rectangle( _ , outline="red", width=3)
            
            test_embedding = get_face_embedding(first_face)# assuming first detected face  
            ref_embedding =  get_face_embedding(first_face_2)  # assuming first detected face     
            database = np.load("face_embeddings.npy", allow_pickle=True).item()
            
            # Access the embedding for the person
            ref_embedding_2 = database["person"]  # assuming first detected face
            st.write(test_embedding)
            st.write(ref_embedding)
            results = compare_embeddings(test_embedding, ref_embedding)
            st.write(f"Cosine Similarity: {results:.4f}")
            if results > 0.8:
                st.success(" Same person")
            else:
                st.error(" Different persons")
        st.success("Face comparison done!")

        


with col_c: 
    if st.button("🔍 Detect Celebrity Faces"):
        with st.spinner("Recognize Celebrity faces... Please wait"):
            celeb_image = Image.open(other_img).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                celeb_image.save(tmp.name)
                tmp_path = tmp.name

            st.write(f"Temporary path: `{tmp_path}`")
            face_detection(tmp_path)
            face_regions = get_crop_face_detection(tmp_path)
            st.success(f"Detected {len(face_regions)} faces.")
            draw = ImageDraw.Draw(celeb_image)
            font = ImageFont.truetype("arial.ttf", 20)

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
            st.image(celeb_image, caption="Detected Celebrity Faces", use_column_width=True)
        st.success("Celebrity face recognition done!")

            