import detectron2
from Multi_Type_TD_TSR.google_colab.deskew import deskewImage
import Multi_Type_TD_TSR.google_colab.table_detection as table_detection
import Multi_Type_TD_TSR.google_colab.table_structure_recognition_all as tsra
import Multi_Type_TD_TSR.google_colab.table_structure_recognition_lines as tsrl
import Multi_Type_TD_TSR.google_colab.table_structure_recognition_wol as tsrwol
import Multi_Type_TD_TSR.google_colab.table_structure_recognition_lines_wol as tsrlwol
import Multi_Type_TD_TSR.google_colab.table_xml as txml
import Multi_Type_TD_TSR.google_colab.table_ocr as tocr
import pandas as pd
import os
import json
import itertools
import random
from detectron2.utils.logger import setup_logger
# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
# from google.colab.patches import cv2_imshow



setup_logger()

#create detectron config
cfg = get_cfg()

#set yaml
cfg.merge_from_file('Multi_Type_TD_TSR\All_X152.yaml')

#set model weights
# cfg.MODEL.WEIGHTS = 'Multi_Type_TD_TSR\model_final.pth' # Set path model .pth

# predictor = DefaultPredictor(cfg) 

# path to the image scan of the document
# file = "Multi_Type_TD_TSR/images/rotated_example.jpeg" 
# original_image = cv2.imread(file)

def deskew(image):
    # load the image from disk
    deskewed_image = deskewImage(image)

    print("ORIGINAL IMAGE:")
    plt.imshow(image)
    plt.show()
    print("DESKEWED IMAGE:")
    plt.imshow(deskewed_image)
    plt.show()

# def table_detection():
document_img = cv2.imread(r"FILES\FILES\Screenshot 2023-08-22 135455.png")
# deskew(document_img)
# table_detection.plot_prediction(document_img, predictor)
# table_list, table_coords = table_detection.make_prediction(document_img, predictor)
# print(table_list, table_coords)
# # table_detection()

#





import cv2
import numpy as np

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
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

# Load the input image
input_image = cv2.imread('FILES\FILES\SHUKRAN.jpeg')

# Call the straighten_text_image function
straightened_image = straighten_text_image(input_image)

# Display the original and straightened images
cv2.imshow('Original Image', input_image)
cv2.imshow('Straightened Image', straightened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
