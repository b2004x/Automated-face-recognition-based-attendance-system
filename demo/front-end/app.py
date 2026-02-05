import streamlit as st
import requests
import base64
from PIL import Image

# FastAPI backend base URL
BACKEND_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="Face Image Demo", layout="wide")

st.title("Face Image Upload Demo")


tab1, tab2, tab3 = st.tabs(["Save Face Embedding", "Recognize Face", "Recognize Celebrity"])

with tab1:
    st.header("Save Face Embedding")
    upload_file  = st.file_uploader("Upload reference face image", type=["jpg", "jpeg", "png"], key="ref")
    if upload_file:
        st.image(Image.open(upload_file ), caption="Reference Face", use_column_width=True)
    else:
        st.info("Please upload a reference image.")
    name = st.text_input("Enter name for this person:")

    if upload_file and name and st.button("Save Face Embedding"):
        with st.spinner("Saves faces... Please wait"):
            files = {"file": (upload_file.name, upload_file.getvalue(), "image/jpeg")}
            data = {"name": name}
            try:
                res = requests.post(f"{BACKEND_URL}/Save_Face_Embedding/", files=files, data=data)
                if res.status_code == 200:
                    result = res.json()
                    st.success(result["message"])
                    st.write("Saved names:", ", ".join(result.get("saved_names", [])))
                    img_data = base64.b64decode(result["image_base64"])
                    st.image(img_data, caption="Annotated Image", use_container_width=True)
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.header("Recognize a Face from Database")
    upload_file = st.file_uploader("Upload a face image to recognize", type=["jpg", "jpeg", "png"], key="recognize_upload")

    if upload_file:
        st.image(Image.open(upload_file ), caption="Recognize Face", use_column_width=True)
    else:
        st.info("Please upload a reference image.")

    if upload_file and st.button("Recognize Face"):
        with st.spinner("Compares faces... Please wait"):
            files = {"file": (upload_file.name, upload_file.getvalue(), "image/jpeg")}
            try:
                res = requests.post(f"{BACKEND_URL}/Recognize_Face/", files=files)
                if res.status_code == 200:
                    result = res.json()
                    st.success(result["message"])
                    if "matches" in result:
                        st.json(result["matches"])
                    if "image_base64" in result:
                        img_data = base64.b64decode(result["image_base64"])
                        st.image(img_data, caption="Recognition Result", use_container_width=True)
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab3:
    st.header("🌟 Recognize Celebrity")

    upload_file = st.file_uploader("Upload a celebrity image", type=["jpg", "jpeg", "png"], key="celeb_upload")

    if upload_file:
        st.image(Image.open(upload_file ), caption="Recognize Celebrity Face", use_column_width=True)
    else:
        st.info("Please upload a reference image.")

    if upload_file and st.button("Recognize Celebrity"):
        with st.spinner("Recognize Celebrity faces... Please wait"):
            files = {"file": (upload_file.name, upload_file.getvalue(), "image/jpeg")}
            try:
                res = requests.post(f"{BACKEND_URL}/Recognize_Celebrity/", files=files)
                if res.status_code == 200:
                    result = res.json()
                    st.success(result["message"])
                    img_data = base64.b64decode(result["image_base64"])
                    st.image(img_data, caption="Celebrity Recognition", use_container_width=True)
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"Connection error: {e}")




    