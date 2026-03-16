import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)

url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

products = []

for i, item in enumerate(soup.select(".thumbnail")):

    name = item.select_one(".title").text.strip()
    description = item.select_one(".description").text.strip()

    image_url = item.select_one("img")["src"]
    image_url = "https://webscraper.io" + image_url

    image_path = f"images/product_{i}.jpg"

    img_data = requests.get(image_url).content

    with open(image_path, "wb") as f:
        f.write(img_data)

    products.append({
        "name": name,
        "description": description,
        "image": image_path
    })

df = pd.DataFrame(products)

df.to_csv("data/scraped_products.csv", index=False)

print("Products and images scraped successfully")
