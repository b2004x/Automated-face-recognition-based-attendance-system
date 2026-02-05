from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
# Load pretrained ViT (you can replace this with a face-specific model)
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
base = Path(__file__).resolve().parent
# Function to get embedding from image path
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output  # [1, hidden_size]
    return embedding.numpy()


# Paths to face images
face1_path = base.parent.parent / "test" / "face2.jpg"
face2_path = base.parent.parent / "test" / "face3.jpg"

# Compute embeddings
emb1 = get_embedding(face1_path)
emb2 = get_embedding(face2_path)

# Compare embeddings
sim = cosine_similarity(emb1, emb2)[0][0]
print(f"Cosine Similarity: {sim:.4f}")

if sim > 0.8:
    print("✅ Same person")
else:
    print("❌ Different persons")
