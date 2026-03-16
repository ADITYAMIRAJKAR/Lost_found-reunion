import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize
from PIL import Image

index = faiss.read_index("vector.index")
df = pickle.load(open("data.pkl", "rb"))

text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def search(query, k=3):

    query_embedding = text_model.encode([query], normalize_embeddings=True)

    padding = np.zeros((1, 512))

    combined_query = np.hstack((query_embedding, padding))
    combined_query = normalize(combined_query)

    distances, indices = index.search(combined_query, k)

    results = df.iloc[indices[0]].copy()
    results["confidence"] = distances[0]

    return results


def search_by_image(image_file, k=3):

    image = Image.open(image_file).convert("RGB")

    inputs = clip_processor(images=image, return_tensors="pt")

    outputs = clip_model.get_image_features(**inputs)

    image_embedding = outputs.detach().numpy()

    padding = np.zeros((1, 384))

    combined_query = np.hstack((padding, image_embedding))
    combined_query = normalize(combined_query)

    distances, indices = index.search(combined_query, k)

    results = df.iloc[indices[0]].copy()
    results["confidence"] = distances[0]

    return results


def explain_match(query, description):

    prompt = f"""
A student reported losing this item:
{query}

A possible matching item in the database:
{description}

Explain briefly why this could be the same item.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
