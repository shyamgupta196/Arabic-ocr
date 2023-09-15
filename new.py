import detectron2
from Multi_Type_TD_TSR.google_colab.deskew import deskewImage
import Multi_Type_TD_TSR.google_colab.table_detection as table_detection
import os
# from google_cloud_vision_python.vision import detect_text
from google_cloud_vision_python.translate import translate_text

import argparse
from detectron2.utils.logger import setup_logger
import json
# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
os.system("pip3 install --upgrade google-cloud-vision")

# from google.colab.patches import cv2_imshow
setup_logger()

parser = argparse.ArgumentParser(
    description="Detect table and do OCR using OpenCV + pytorch model + visionAPI."
)
# Define command-line arguments
parser.add_argument(
    "--input_folder", type=str, help="Path to the folder containing input images."
)
parser.add_argument(
    "--output_json",
    default="json_output",
    type=str,
    required=False,
    help="Path to the folder containing coordinates & text of outputs in JSON.",
)
parser.add_argument(
    "--deskew", type=bool, default=False, required=False, help="Straighten the images."
)
parser.add_argument(
    "--output_folder",
    default="outputs",
    type=str,
    required=False,
    help="Path to the folder for saving output images.",
)
parser.add_argument(
    "--show_results",
    default=False,
    type=str,
    required=False,
    help="Path to the folder for saving output images.",
)
parser.add_argument(
    "--detect_full_page",
    default=False,
    type= bool,
    required=False,
    help="instead of reading extracted tables it reads full page",
)

parser.add_argument(
    "--translate",
    default=False,
    type=str,
    required=False,
    help="translate the extracted text to english",
)
# Parse the command-line arguments
args = parser.parse_args()

# create detectron config
cfg = get_cfg()
# set yaml
cfg.merge_from_file("Multi_Type_TD_TSR\All_X152.yaml")
# set model weights
cfg.MODEL.WEIGHTS = "Multi_Type_TD_TSR\model_final.pth"  # Set path model .pth
predictor = DefaultPredictor(cfg)

# Ensure the output folder exists
os.makedirs(args.output_folder, exist_ok=True)
os.makedirs(args.output_json, exist_ok=True)

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print("Texts:")

    # for text in texts:
        # print(f'\n"{text.description}"')

        # vertices = [
            # f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        # ]

        # print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return texts

# Iterate through input images in the input folder
def main():
    try:
        for filename in os.listdir(args.input_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                # path to the image scan of the document
                path = os.path.join(args.input_folder, filename)
                jsonpath = open(os.path.join(args.output_json, f"{filename.split('.')[0]}.json"),'a')
                document_img = cv2.imread(r"{}".format(path))
                file = open(
                    f"{os.path.join(args.output_folder,filename.split('.')[0])}.txt",
                    "a",
                    encoding="utf-8",
                )
                # to straighten a image
                if args.deskew:
                    document_img = straighten_text_image(document_img)
                # if detecting full page worth of text and not just tables
                if args.detect_full_page:
                    texts = detect_text(path)
                    import IPython; IPython.embed(); exit(1)
                    file.write("\n\ntexts:")
                    file.write(texts[0].description)
                    if args.translate:
                        texts = translate_text(str.encode(texts))
                    texts = ' '.join([f'{i}' for i in texts])
                    json.dump(texts, jsonpath, indent=4)
                    jsonpath.close()
                    file.close()
                else:
                    table_list, table_coords, fnames = table_detection.make_prediction(
                        document_img, predictor, args.show_results
                    )
                    for num, coords in enumerate(table_coords):
                        file.write("tables: ")
                        file.write(f"table {num}: {coords}\n")

                    for fname in fnames:
                        texts = detect_text(fname)
                        file.write("\n\ntexts:")
                        file.write(texts[0].description)
                        if args.translate:
                            texts = translate_text(texts)
                        texts = ' '.join([f'{i}' for i in texts])
                        json.dump(texts, jsonpath, indent=4)
                        jsonpath.close()
                        file.close()
    except Exception as e:
        print("Error:", e)


def straighten_text_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find contours
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the text)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a bounding box to the contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Determine the angle of rotation
    angle = rect[2]

    if angle < -45:
        angle += 90

    # Rotate the image to straighten the text
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated_image


if __name__ == "__main__":
    main()
