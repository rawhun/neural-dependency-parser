import os
import requests
import zipfile

URL = "https://github.com/UniversalDependencies/UD_English-EWT/archive/refs/heads/master.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")

def download_and_extract():
    os.makedirs(OUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUT_DIR, "ud-ewt.zip")
    if not os.path.exists(zip_path):
        print("Downloading UD-EWT....")
        r = requests.get(URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)
    print("Extracting..")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    download_and_extract()