# author - Shyam Gupta, github.com/shyamgupta196
# -*- coding: utf-8 -*-

import detectron2
from main.google_colab.deskew import deskewImage
import main.google_colab.table_detection as table_detection
import os
from google_cloud_vision_python.vision import detect_text
from google.cloud import translate
import json
import argparse
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import re

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog

# from google.colab.patches import cv2_imshow
setup_logger()
# create detectron config
cfg = get_cfg()
# set yaml
cfg.merge_from_file("main\All_X152.yaml")
# set model weights
cfg.MODEL.WEIGHTS = "main\model_final.pth"  # Set path model .pth

parser = argparse.ArgumentParser(
    description="Detect table and do OCR using OpenCV + pytorch model + visionAPI."
)
# Define command-line arguments
parser.add_argument(
    "--input_folder", type=str, help="Path to the folder containing input images."
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
    type=bool,
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

parser.add_argument(
    "--output_json",
    default="json_output",
    type=str,
    required=False,
    help="Path to the folder containing coordinates & text of outputs in JSON.",
)
# Parse the command-line arguments
args = parser.parse_args()

# Ensure the output folders exists
os.makedirs(args.output_folder, exist_ok=True)
os.makedirs(args.output_json, exist_ok=True)


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


def main():
    predictor = DefaultPredictor(cfg)
    try:
        # Iterate through input images in the input folder
        for filename in os.listdir(args.input_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                # path to the image scan of the document
                path = os.path.join(args.input_folder, filename)

                document_img = cv2.imread(r"{}".format(path))
                file = open(
                    f"{os.path.join(args.output_folder,filename.split('.')[0])}.txt",
                    "a",
                    encoding="utf-8",
                )

                if args.deskew:
                    document_img = straighten_text_image(document_img)

                if args.detect_full_page:
                    texts = detect_text(path)
                    for text in texts:
                        file.write("\n")
                        vertices = [
                            f"({vertex.x},{vertex.y})"
                            for vertex in text.bounding_poly.vertices
                        ]
                        file.write(f"text :{text.description} , ")
                        file.write("bbox: {}".format(",".join(vertices)))
                        # import IPython;IPython.embed();exit(1)
                    file.close()

                else:
                    print(" not detecting full")
                    table_list, table_coords, fnames = table_detection.make_prediction(
                        document_img, predictor, args.show_results
                    )
                    for num, coords in enumerate(table_coords):
                        file.write("tables: ")
                        file.write(f"table {num}: {coords}\n")
                    print("tables extracted")
                    for fname in fnames:
                        texts = detect_text(fname)
                        for text in texts:
                            file.write("\n")
                            vertices = [
                                f"({vertex.x},{vertex.y})"
                                for vertex in text.bounding_poly.vertices
                            ]
                            file.write(f"text :{text.description} , ")
                            file.write("bbox: {}".format(",".join(vertices)))
                    file.close()
    except Exception as e:
        print("Error:", e)


def translate_text(
    input: str, output: str, project_id: str = "gcp-kubernetes-ml-app"
) -> translate.TranslationServiceClient:

    """
    Translates all Arabic text files in the specified input directory to English using Google Cloud Translation API,
    extracts text and bounding box information from the translation response, and saves the results as JSON files
    in the output directory.
    Args:
        input (str): Path to the directory containing input .txt files with Arabic text.
        output (str): Path to the directory where output .json files will be saved.
        project_id (str, optional): Google Cloud project ID. Defaults to "gcp-kubernetes-ml-app".
    Returns:
        translate.TranslationServiceClient: An instance of the TranslationServiceClient used for translation.
    Notes:
        - Each .txt file in the input directory is processed and a corresponding .json file is created in the output directory.
        - The JSON output contains a list of dictionaries, each with 'text' (translated text) and 'bbox' (bounding box coordinates).
        - Bounding box extraction relies on specific formatting in the translation response.
        - Requires the Google Cloud Translation API and appropriate credentials.
    """

    # Initialize the TranslationServiceClient
    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"
    for filename in os.listdir(input):
        print("reading")
        print(filename)
        if filename.endswith(".txt"):
            file = open(
                os.path.join(input, filename),
                "r",
                encoding="utf-8",
            )
            translated_json = open(
                f"{os.path.join(output,filename.split('.')[0])}.json",
                "a",
                encoding="utf-8",
            )

            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [file.read()],
                    "mime_type": "text/plain",  # mime types: text/plain, text/html
                    "source_language_code": "ar",
                    "target_language_code": "en",
                }
            )

            formatted_data = []

            # import IPython;IPython.embed();exit(1)
            for count, item in enumerate(str(response).split("\\n")):
                if count == 0:
                    text = " ".join(
                        response.translations[0]
                        .translated_text.split("bbox")[0]
                        .split("\n")
                    )
                    bbox = (
                        "bbox"
                        + response.translations[0].translated_text.split("bbox")[1]
                    )
                    bbox_match = re.search(
                        "bbox: \((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\)",
                        bbox,
                    )
                    if bbox_match:
                        x1, y1, x2, y1, x2, y2, x1, y2 = map(int, bbox_match.groups())
                        item_data = {
                            "text": text,
                            "bbox": {
                                "x_low": min(x1, x2),
                                "y_low": min(y1, y2),
                                "x_high": max(x1, x2),
                                "y_high": max(y1, y2),
                            },
                        }
                        formatted_data.append(item_data)

                # Extract text and bounding box coordinates using regular expressions
                else:
                    text_match = re.search(r"text:(.*?),", item)
                    bbox_match = re.search(
                        r"bbox: \((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\)",
                        item,
                    )

                    if text_match and bbox_match:
                        text = text_match.group(1).strip()
                        x1, y1, x2, y1, x2, y2, x1, y2 = map(int, bbox_match.groups())
                        item_data = {
                            "text": text,
                            "bbox": {
                                "x_low": min(x1, x2),
                                "y_low": min(y1, y2),
                                "x_high": max(x1, x2),
                                "y_high": max(y1, y2),
                            },
                        }
                        formatted_data.append(item_data)

            json.dump(
                formatted_data, translated_json, ensure_ascii=False, indent=4
            )  # text description,\\nbbox,des,\\n,bbox\\n
            translated_json.close()
            file.close()
    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats

    # Display the translation for each input text provided
    # for translation in response.translations:
    #     print(f"Translated text: {translation.translated_text}")

    # return response


if __name__ == "__main__":
    main()
    if args.translate:
        print("translating")
        translate_text(args.output_folder, args.output_json)
    print("DONE !")
