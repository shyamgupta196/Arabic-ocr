# Mindful Automations Table & Arabic OCR
I have been asked to apply OCR on invoices! Lets try to give our best shot at detecting tables and texts.

[https://huggingface.co/keremberke/yolov8s-table-extraction](https://huggingface.co/keremberke/yolov8s-table-extraction) 

The HF yolov8was total ❎!

****Multi-Type-TD-TSR - This worked.**** ✅

now looking for Arabic OCR. ❎

I think **GOOGLE VISION API** is best for the job.  

[**REFERENCE TO GCP](https://cloud.google.com/vision/docs/ocr?_ga=2.134025102.-1000806481.1689720701&cloudshell=false&apix_params=%7B%22resource%22%3A%7B%22requests%22%3A%5B%7B%22features%22%3A%5B%7B%22type%22%3A%22TEXT_DETECTION%22%7D%5D%2C%22image%22%3A%7B%22source%22%3A%7B%22imageUri%22%3A%22gs%3A%2F%2Fcloud-samples-data%2Fvision%2Focr%2Fsign.jpg%22%7D%7D%7D%5D%7D%7D#try_it) link to cloud page…..**

```bash
# install requirements.txt
pip install -r requirements.txt
```

### How it works + my progress + personal notes

![Untitled](Mindful%20Automations%20Table%20&%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled.png)

fresh results coming in !!

### Commands used in GCP vision API

```bash
# 1. select project & Enable vision API
# 2. open cloudshell

mkdir google-cloud-vision-python && touch google-cloud-vision-python/app.py
cd google-cloud-vision-python
cloudshell open-workspace .
export PROJECT_ID=gcp-kubernetes-ml-app
```

### Authentication

```bash
#Create a service account to authenticate your API requests:
gcloud iam service-accounts create google-cloud-vision-quickstart --project gcp-kubernetes-ml-app

#Grant your service account the roles/viewer role:
gcloud projects add-iam-policy-binding gcp-kubernetes-ml-app \
   --member serviceAccount:google-cloud-vision-quickstart@gcp-kubernetes-ml-app.iam.gserviceaccount.com \
   --role roles/viewer

#Create a service account key:
gcloud iam service-accounts keys create google-cloud-vision-key.json --iam-account  google-cloud-vision-quickstart@gcp-kubernetes-ml-app.iam.gserviceaccount.com

#Set the key as your default credentials:

  export GOOGLE_APPLICATION_CREDENTIALS=google-cloud-vision-key.json
```

### Call vision API

```bash
# Get the sample image used in the code sample
wget \
    https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/vision/snippets/quickstart/resources/wakeupcat.jpg

# Open app.py in the Cloud Shell Editor by running the following command in your terminal:
cloudshell open app.py

# Install the Cloud Vision client library:

pip3 install --upgrade \
    google-cloud-vision
```

### In app.py, add the following code (if local file)
**I have customised the codes for my needs**
```python
'''credits to google'''
# app.py
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
detect_text('SHUKRAN.jpeg')
```

# Execute

```bash
python3 app.py 

## Clean up
## Delete the file containing your service account key.

rm google-cloud-vision-key.json
```

```json
//google-cloud-vision-key.json

{
  "YOUR GCP CREDENTIALS"
}
```

### If URL file

```python
''' credits to google'''
def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

detect_text_uri('URL')
```

# Packaging it!

I know how to make a docker container, but I dont know how to package this whole thing as a project using [setup.py](http://setup.py) or .toml files ? ✅

************************************************they asked me to deliver a package and I dont know how to do that.************************************************ ✅(done)

- I install torch and other libs -yes
- installed detectron2 - yes

```python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

- Almost everything is ready I have to package stuff together
- ***Although I just fear for the code will not work, but let's just believe it to be fine***
**- clone the repo locally  -  yes**
**- test & fixed the code locally  !!!**
1. I fixed the Device issue, the .yaml was missing DEVICE flag, because of which the model was not running on CPU.
2. I fixed the cv2_imshow() google colab function which was not working locally !!!

# Local Runs For Table Detection- Successful 🥳

- change the code to save cropped images - done✅
- locally extract everything - done✅
- write commands to setup vision api locally - done✅
- STORE IN TEXT FILE SAME AS IMAGE NAME ! - ✅
- I have to read in files from FOLDER DIRECTLY. ✅
- INTRODUCE VALID ARGUMENTS ✅
- REMOVE UNECESSARY PRINT STATEMENTS ✅
- I WANT THE VISION API TO WORK IN THE MAIN FUNCTION AND NOT DIFFERENTLY ✅
- ADDING ARGUMENTS FOR SHOWING CROPPED TABLES ✅
- store extracted text in txt ✅
- make the OCR file with bash commands using google vision API. ✅
- detect full page ✅
- translate ✅

got translate results !! had to setup the api using pip and enable using console !

```bash

pip install google-cloud-translate==2.0.1
pip install --upgrade google-cloud-translate

```

![Untitled](Mindful%20Automations%20Table%20&%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled%201.png)

- make json files to store the data in format given ✅

## Done ! everything works , now If there is any bug i will fix it : )

CHECK using str(texts) on file and see why encoding not working ! ✅

```bash
# RUN it
python app.py --input_folder 'FILES/FILES' --detect_full_page True --translate True
```
