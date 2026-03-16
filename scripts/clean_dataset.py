import pandas as pd
import os

df=pd.read_csv("data/scraped_products.csv")

df=df.drop_duplicates()
df=df.dropna()

df["image_path"]=df.index.map(lambda x:f"images/product_{x}_0.jpg")

def category(name):

    name=name.lower()

    if "laptop" in name:
        return "laptop"
    if "phone" in name:
        return "phone"
    if "tablet" in name:
        return "tablet"

    return "electronics"

df["category"]=df["product_name"].apply(category)

df=df[df["image_path"].apply(os.path.exists)]

df["searchable_text"]=(
    df["product_name"]+" "+
    df["description"]+" "+
    df["lost_description"]+" "+
    df["category"]
)

expanded=pd.concat([df]*4,ignore_index=True)

expanded=expanded.sample(550)

expanded.to_csv("data/lost_found_dataset_cleaned.csv",index=False)
