# Mindful Automations Table & Arabic OCR

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-main-blueviolet?logo=github)](https://github.com/facebookresearch/detectron2)
[![Google Cloud](https://img.shields.io/badge/Cloud-Google%20Cloud-blue?logo=googlecloud)](https://cloud.google.com/)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/keremberke/yolov8s-table-extraction)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/shyamgupta196/Arabic-ocr/blob/main/LICENSE)
[![Repo Stars](https://img.shields.io/github/stars/shyamgupta196/Arabic-ocr?style=social)](https://github.com/shyamgupta196/Arabic-ocr/stargazers)
[![Forks](https://img.shields.io/github/forks/shyamgupta196/Arabic-ocr?style=social)](https://github.com/shyamgupta196/Arabic-ocr/network/members)
[![Issues](https://img.shields.io/github/issues/shyamgupta196/Arabic-ocr?color=yellow)](https://github.com/shyamgupta196/Arabic-ocr/issues)
[![Views](https://komarev.com/ghpvc/?username=shyamgupta196&color=green)](https://github.com/shyamgupta196/arabic-ocr)

## Problem Statement

Processing invoices in Arabic was a tedious manual task—people relied on Google Camera to translate invoices, which was inefficient and time-consuming. I was tasked with automating this process: extracting and translating invoice data, then storing it in JSON format for streamlined workflows.

## Solution Approach

- **Initial Attempt:**  
    I started with [YOLOv8 Table Extraction](https://huggingface.co/keremberke/yolov8s-table-extraction), but it struggled with detecting boundaries and edges in complex invoices.

- **Improved Model:**  
    Switching to **Multi-Type-TD-TSR** significantly improved detection accuracy and saved time.

- **Optimal Choice:**  
    Ultimately, I discovered the **Google Cloud Vision API**, which provided robust OCR capabilities for Arabic text and proved to be the best fit for the job.

> [**Google Cloud Vision OCR Documentation**](https://cloud.google.com/vision/docs/ocr?_ga=2.134025102.-1000806481.1689720701&cloudshell=false&apix_params=%7B%22resource%22%3A%7B%22requests%22%3A%5B%7B%22features%22%3A%5B%7B%22type%22%3A%22TEXT_DETECTION%22%7D%5D%2C%22image%22%3A%7B%22source%22%3A%7B%22imageUri%22%3A%22gs%3A%2F%2Fcloud-samples-data%2Fvision%2Focr%2Fsign.jpg%22%7D%7D%7D%5D%7D%7D#try_it)

With this workflow, I automated the extraction, translation, and structured storage of invoice data, making the process efficient and scalable.

```bash
# install requirements.txt
pip install -r requirements.txt
```
# How It Works & Progress Notes

Upload your invoice images and the script extracts and returns the text from the invoices.

![Invoice Example](Mindful%20Automations%20Table%20&%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled.png)

---

## 1. Enable Google Cloud Vision API

```bash
# 1. Select your GCP project & enable the Vision API
# 2. Open Cloud Shell

mkdir google-cloud-vision-python && touch google-cloud-vision-python/app.py
cd google-cloud-vision-python
cloudshell open-workspace .
export PROJECT_ID=gcp-kubernetes-ml-app
```

---

## 2. Authenticate & Set Up Google CLI

```bash
# Create a service account for authentication
gcloud iam service-accounts create google-cloud-vision-quickstart --project gcp-kubernetes-ml-app

# Grant viewer role to the service account
gcloud projects add-iam-policy-binding gcp-kubernetes-ml-app \
    --member serviceAccount:google-cloud-vision-quickstart@gcp-kubernetes-ml-app.iam.gserviceaccount.com \
    --role roles/viewer

# Create a service account key
gcloud iam service-accounts keys create google-cloud-vision-key.json \
    --iam-account google-cloud-vision-quickstart@gcp-kubernetes-ml-app.iam.gserviceaccount.com

# Set the key as your default credentials
export GOOGLE_APPLICATION_CREDENTIALS=google-cloud-vision-key.json
```

---

## 3. Make API Calls with Google Cloud Vision

```bash
# Download a sample image
wget https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/vision/snippets/quickstart/resources/wakeupcat.jpg

# Open app.py in the Cloud Shell Editor
cloudshell open app.py

# Install the Cloud Vision client library
pip3 install --upgrade google-cloud-vision
```

---

## 4. Example: Detect Text in a Local File

> **Note:** The following code is adapted from Google Cloud documentation.

```python
from google.cloud import vision

def detect_text(path):
        """Detects text in the file."""
        client = vision.ImageAnnotatorClient()
        with open(path, "rb") as image_file:
                content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        print("Texts:")
        for text in texts:
                print(f'\n"{text.description}"')
                vertices = [f"({v.x},{v.y})" for v in text.bounding_poly.vertices]
                print("bounds: {}".format(",".join(vertices)))
        if response.error.message:
                raise Exception(
                        f"{response.error.message}\nFor more info, see: "
                        "https://cloud.google.com/apis/design/errors"
                )

detect_text('SHUKRAN.jpeg')
```

---

## 5. Run & Clean Up

```bash
python3 app.py

# Clean up: Delete your service account key file
rm google-cloud-vision-key.json
```

---

## 6. Example Service Account Key (for reference only)

```json
{
    "YOUR GCP CREDENTIALS"
}
```

---

## 7. Detect Text from a URL

```python
from google.cloud import vision

def detect_text_uri(uri):
        """Detects text in the file located in Google Cloud Storage or on the Web."""
        client = vision.ImageAnnotatorClient()
        image = vision.Image()
        image.source.image_uri = uri
        response = client.text_detection(image=image)
        texts = response.text_annotations
        print("Texts:")
        for text in texts:
                print(f'\n"{text.description}"')
                vertices = [f"({v.x},{v.y})" for v in text.bounding_poly.vertices]
                print("bounds: {}".format(",".join(vertices)))
        if response.error.message:
                raise Exception(
                        f"{response.error.message}\nFor more info, see: "
                        "https://cloud.google.com/apis/design/errors"
                )

detect_text_uri('URL')
```

---

## 8. Packaging & Running as a CLI Script

The project is packaged to run directly from the command line with arguments.  
You can also containerize it using Docker if needed.

---

## Progress & Implementation Notes

### ✅ Package Delivery & Setup

- **Task:** Delivered the required package as requested.
- **Prerequisites:**  
    - Installed `torch` and other dependencies.
    - Installed `detectron2`:

        ```bash
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        ```

- **Repository:**  
    - Cloned the repo locally.
    - Ran tests and fixed code issues:
        - Fixed device issue (missing `DEVICE` flag in `.yaml` for CPU support).
        - Fixed `cv2_imshow()` (Google Colab function) for local runs.

---

### 🥳 Local Table Detection: Successful

- Saved cropped images.
- Extracted all data locally.
- Set up Vision API locally.
- Stored extracted text in `.txt` files (named after images).
- Implemented reading files directly from a folder.
- Introduced valid CLI arguments.
- Removed unnecessary print statements.
- Integrated Vision API into the main function.
- Added arguments for showing cropped tables.
- Stored extracted text in `.txt` files.
- Created OCR file with bash commands using Google Vision API.
- Enabled full-page detection.
- Integrated translation functionality.

> **Translation Setup:**  
> Installed and enabled Google Translate API:

```bash
pip install google-cloud-translate==2.0.1
pip install --upgrade google-cloud-translate
```

---

### 📦 Data Storage

- Generated JSON files to store extracted data in the required format.

---

### 🛠️ Debugging & Validation

- Used `str(texts)` to debug encoding issues (resolved).

---

### 🚀 How to Run

```bash
python app.py --input_folder 'FILES/FILES' --detect_full_page True --translate True
```

---

### 🖼️ Example

**Input:**  
![Image](https://github.com/shyamgupta196/Arabic-ocr/blob/main/FILES/FILES/SHUKRAN.jpeg)

**Outputs:**  
![output](https://github.com/shyamgupta196/Arabic-ocr/blob/main/Mindful%20Automations%20Table%20%26%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled%201.png)  
![output](https://github.com/shyamgupta196/Arabic-ocr/blob/main/Mindful%20Automations%20Table%20%26%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled.png)

---

**Everything works! If any bugs arise, they will be fixed as needed.**

---

If you want to see more experiments I conducted, please check out my [notebook](https://github.com/shyamgupta196/Arabic-ocr/blob/main/main/Table_Recognition.ipynb).
