from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data/lost_found_dataset_cleaned.csv")

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_embeddings = []

for path in df['image_path']:
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        # emb is now a BaseModelOutputWithPooling
        # extract the tensor from the 'pooler_output' attribute
        tensor_emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
        # move to CPU and convert to numpy
        image_embeddings.append(tensor_emb.detach().cpu().numpy()[0])

image_embeddings = np.array(image_embeddings)
np.save("embeddings/image_embeddings.npy", image_embeddings)
print("Image embeddings saved ✅")
