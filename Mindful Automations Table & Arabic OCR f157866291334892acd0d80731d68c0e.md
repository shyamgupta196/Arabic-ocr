# Mindful Automations Table & Arabic OCR

I have to apply OCR on invoices! Lets try to give our best shot at detecting tables and texts.

[https://huggingface.co/keremberke/yolov8s-table-extraction](https://huggingface.co/keremberke/yolov8s-table-extraction) 

The HF yolov8was total ##flopshow!

****Multi-Type-TD-TSR - This worked.****

now looking for Arabic OCR. ## flopshow

karan said **GOOGLE VISION API** can do the job.  

[**REFERENCE TO GCP](https://cloud.google.com/vision/docs/ocr?_ga=2.134025102.-1000806481.1689720701&cloudshell=false&apix_params=%7B%22resource%22%3A%7B%22requests%22%3A%5B%7B%22features%22%3A%5B%7B%22type%22%3A%22TEXT_DETECTION%22%7D%5D%2C%22image%22%3A%7B%22source%22%3A%7B%22imageUri%22%3A%22gs%3A%2F%2Fcloud-samples-data%2Fvision%2Focr%2Fsign.jpg%22%7D%7D%7D%5D%7D%7D#try_it) link to cloud page…..**

```bash
# install requirements.txt
pip install -r requirements.txt
```

# How it works + my progress + personal notes

![Untitled](Mindful%20Automations%20Table%20&%20Arabic%20OCR%20f157866291334892acd0d80731d68c0e/Untitled.png)

fresh results coming in !!

# Commands used in GCP vision API

```bash
# 1. select project & Enable vision API
# 2. open cloudshell

> mkdir google-cloud-vision-python && touch google-cloud-vision-python/app.py
> cd google-cloud-vision-python
> cloudshell open-workspace .
	> export PROJECT_ID=gcp-kubernetes-ml-app
```

# Authentication

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

# Call vision API

```bash
#Get the sample image used in the code sample
wget \
    https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/vision/snippets/quickstart/resources/wakeupcat.jpg

#Open app.py in the Cloud Shell Editor by running the following command in your terminal:
cloudshell open app.py

#Install the Cloud Vision client library:

pip3 install --upgrade \
    google-cloud-vision
```

## In app.py, add the following code (if local file)

```python
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
  "type": "service_account",
  "project_id": "gcp-kubernetes-ml-app",
  "private_key_id": "3edcd799c9ab9b6afb9291719e0fd9927c9fb0da",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCXgm0dQMIUwNRp\nQuA086UcNGMQNHHACcO29ersGB1AnfWpc5p0lcfAMn5JoB2ngdRpxncKOj8XHFQo\ndUM9KVmBx1IYaOPR21oRPMXN7h+Fl7x00/rMOFgdzHB5jyF08XJ5vMUsgtB5ocJZ\nACABzuMtgR4Op2kR5n5DEoHA9l8bVu0mNFjOs+Ba+91C9ySiIGhQ2T6ES8gBaaer\nOspi0Y4P3r/Qx+4FNIIas9spwjkJl8X26gVC1MHg0oqu889VdIoOw2h7JxXlsRZ1\nmLc+EKcPIV+uzhxkJcK1KmaxZvKv7DC4Jl431QYFwrRST2p9VnJh0wvDyc4IM4k1\nsQxLbz0nAgMBAAECggEAAx6RqziExnNHQaOO2jLqA1qmGin6QunwjaPNN8SHVafq\n7EI6ZowZlhYeuNqtwZ/QBR2A9e+dEkITDteF5Yx5u0z587wKdMJUvbtylyV/9frZ\ny838K+2SYuACsNNcQlF2T0CuY+w4Pp4wp4RzzusON7Gg3MxgQsNsIMAQy9hGvQP/\nsetqRiE/tHWcFGhdhyxj7naq7USagh97Kv2JwgkbN84/Ch4qLRSZ3IgtmuWYO7nU\nZ+BPiUfir6+cyllbB0rZGR5TXnBDIdgCe+LiqqOmNf2Ca664VI4PLX0kbtiPzxTM\nsm8cR/OF5g6XGH7taZHG1xqE0tE1VMAGyTX1SjYNgQKBgQDTgU81sdGXDz8jnAJ6\nm001RdyA2BdjjkbLF7N/p30nwjd1fwFSA3aHj9LtVu40bxpPsMa1Za9pzbPVTQVu\n4ABldO0wfUPT81DiTcq61YQSEJ4tKPZ1cdr3RF5VaPk6Ql8YMXVrWZsmOVMIgV+p\nbucf+GAvUhhD/JLSCvY2R9za5QKBgQC3YgdAKeAq3nXK+hi5SsLtMHBVt2yXykjQ\neJSOm2suemdBpkpZHUtVZQ2NHRk+B2Ea4hTAYlCbccxnmnDl5q2+qaFBtl5n/3CV\nkQfvuU+1kYmyS/M5C2NiCkksWjNJsn64CgQqMTxOTBcq+kq2Vm47emR0umos1f6n\nP5WuhPYbGwKBgAMZIc/niuprjsE2x9KD892T6Gb6w1zx+JeBAWAdU/gBIE7YTWym\nIFZcBPr1Cwg5mGkSbda6ZpdmK/wz5KB7J4ZU8CSFsTipl8W43f9eoVCiba67quCN\nimeU2MznfL8ducbg8pdf+KXWsSCuxHf25+vP95i52yEj0gLBplmlM9cBAoGAWqsy\nMIxD1I5HKUN8g+it9f4UHJ1jKK2QUNq9gMDhPoqwkOn6KpNsUt1y1MmFWIDnhxYu\n8mvptOGQEc1vcowabYGLRGU5yium65xYkzNJcNlzfl9E83ho++lgAnjakN6a/r2d\nD4tmaMQAVHSKChszx16dWoVsx5xKm0C57h65W+8CgYEAmi5l8lVDN5HJgEmD5Nbc\nnFm3lIbxw6pQW9X2W8+Ebyeeht6bq9wmdKS0v64VD8obUsYP/vIBX2Rkwvvbx1K4\nX779Jjc3/Phllh4sbii/rhtGZ6ur+mn00vmINgzBgmcgkrp6zD1f4mMW1YfqrZxE\nhkYnlfCPieCQEYz9avuAzK0=\n-----END PRIVATE KEY-----\n",
  "client_email": "google-cloud-vision-quickstart@gcp-kubernetes-ml-app.iam.gserviceaccount.com",
  "client_id": "117576378998333497632",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-cloud-vision-quickstart%40gcp-kubernetes-ml-app.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

# If URL file

```python
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

# Now I am packaging it, to run systematically line by line

I know how to make a docker container, but I dont know how to package this whole thing as a project using [setup.py](http://setup.py) or .toml files ? ✅

************************************************they asked me to deliver a package and I dont know how to do that.************************************************ ✅(done)

- I install torch and other libs -yes
- installed detectron2 - yes

```python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

- Almost everything is ready I have to package stuff together
- ***Although I just fear for the code will not work, but let's just believe it to be fine***
- clone the repo locally  -  yes
- test the code locally and debug !!!
1. I fixed the Device issue, the .yaml was missing DEVICE flag, because of which the model was not running on CPU.
2. I fixed the cv2_imshow() google colab function which was not working locally !!!

# Local Table Detection Runs Successful 🥳

- change the code to save cropped images - done
- locally extract everything - done
- write commands to setup vision api locally - done
- STORE IN TEXT FILE SAME AS IMAGE NAME ! - ✅
- I have to read in files from FOLDER DIRECTLY. ✅
- INTRODUCE VALID ARGUMENTS ✅
- REMOVE UNECESSARY PRINT STATEMENTS ❌
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

- make json files to store the data in format given by vaishnavi ✅

## Done ! everything works , now If there is any bug i will fix it : )

CHECK using str(texts) on file and see why encoding not working ! ✅

```bash
# RUN it
python app.py --input_folder 'FILES/FILES' --detect_full_page True --translate True
```