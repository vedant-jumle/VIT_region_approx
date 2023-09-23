import pandas as pd
import requests
import io
import threading
from PIL import Image
import numpy as np
from random import randint
from multiprocessing import Pool
from tqdm import tqdm


def download_image_buffer(item):
    global urls

    i, url = item
    # print(i)
    try:
        data = requests.get(url, headers={"user-agent": "I am a valid user, please give me image"}).content
        img = Image.open(io.BytesIO(data))
        filename = f"{i}.jpg"
        img.save(f"./datasets/conceptual-12m/images/{filename}")
    except Exception as e:
        # print("ERROR", i, e)
        filename = "unavailable"
    return filename


if __name__ == "__main__":

    _start = 0
    _end = 100000

    print("loading urls")
    with open("./datasets/conceptual-12m/cc12m.tsv", "r", encoding="utf-8") as dataset_file:
        urls, texts = [], []

        for i, item in enumerate(dataset_file):
            url, text = item.split("\t")

            urls.append(url)
            texts.append(text)

    print("\n\n")

    urls = pd.DataFrame({
        "url": urls,
        "caption": texts
    })

    if "filename" not in urls.columns:
        urls["filename"] = ["-1"] * len(urls)

    items = zip(urls.index[_start:_end], urls['url'][_start:_end])

    with Pool(40) as pool:
        filenames = pool.map(download_image_buffer, tqdm(items, total=_end-_start))

    urls.loc[_start:_end, "filename"] = filenames

    urls.to_csv("./datasets/conceptual-12m/download.csv", index=False)