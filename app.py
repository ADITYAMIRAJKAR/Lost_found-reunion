import streamlit as st
import chromadb
import torch
from transformers import CLIPTokenizer, CLIPModel

st.title("🔍 Lost Item Semantic Search")

st.write("Type a description of a lost item to find similar reports.")

# Load model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return model, tokenizer

model, tokenizer = load_model()

# Load ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("lost_items")

st.write(f"📦 Total items in database: {collection.count()}")

query = st.text_input("Enter lost item description")

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a description.")
    else:

        inputs = tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.text_model(**inputs)
            embedding = outputs.pooler_output

        vector = embedding.cpu().numpy().flatten().tolist()

        results = collection.query(
            query_embeddings=[vector],
            n_results=5
        )

        st.subheader("Top Matches")

        docs = results["documents"][0]

        if len(docs) == 0:
            st.write("No matches found.")
        else:
            for i, doc in enumerate(docs):
                st.write(f"{i+1}. {doc}")
