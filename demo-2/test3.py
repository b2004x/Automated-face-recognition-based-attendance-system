from deepface import DeepFace
image_path = "D:\Face_detection\demo-2\z7502212317107_4285c8e68b90e68d87a69a123762538c.jpg"
print(DeepFace.extract_faces(img_path = image_path, anti_spoofing=True)[0]['is_real'])  
