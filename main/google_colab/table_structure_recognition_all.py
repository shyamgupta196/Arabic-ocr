import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


# from google.colab.patches import cv2_imshow

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as tess
import pytesseract
def recognize_structure(img):
    """
    Recognizes the table structure in an image using OpenCV.
    Detects vertical and horizontal lines, finds contours, and sorts detected boxes into rows and columns.

    Args:
        img (numpy.ndarray): Input image (BGR).

    Returns:
        finalboxes (list): List of detected table cells organized by rows and columns.
        img_bin (numpy.ndarray): Binary image after thresholding.
    """
    def show(*images):
        """Utility to show multiple images for debugging."""
        for im in images:
            cv2.imshow(str(id(im)), im)

    def get_kernel(shape, size):
        """Utility to get structuring element."""
        return cv2.getStructuringElement(cv2.MORPH_RECT, size)

    def sort_contours(cnts, method="left-to-right"):
        """Sort contours according to the specified method."""
        reverse = method in ["right-to-left", "bottom-to-top"]
        i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts, boundingBoxes

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img_gray.shape
    show(img_gray)

    # Thresholding
    _, img_bin = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    show(img_bin)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    invert = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height and (w > max(10, img_width / 30) and h > max(10, img_height / 30))):
            invert = True
            img_bin[y:y+h, x:x+w] = 255 - img_bin[y:y+h, x:x+w]
    show(img_bin)

    img_bin = 255 - img_bin if invert else img_bin
    show(img_bin)

    img_bin_inv = 255 - img_bin
    show(img_bin_inv)

    # Kernel definitions
    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    ver_kernel = get_kernel(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = get_kernel(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = get_kernel(cv2.MORPH_RECT, (2, 2))

    # Vertical lines
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    show(image_1, vertical_lines)

    # Horizontal lines
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
    show(image_2, horizontal_lines)

    # Combine lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.dilate(img_vh, kernel, iterations=3)
    _, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
    show(img_vh)

    bitor = cv2.bitwise_or(img_bin, img_vh)
    img_median = bitor
    show(img_median)

    # More vertical/horizontal lines
    ver_kernel2 = get_kernel(cv2.MORPH_RECT, (5, img_height * 2))
    vertical_lines2 = cv2.erode(img_median, ver_kernel2, iterations=1)
    hor_kernel2 = get_kernel(cv2.MORPH_RECT, (img_width * 2, 3))
    horizontal_lines2 = cv2.erode(img_median, hor_kernel2, iterations=1)
    show(vertical_lines2, horizontal_lines2)

    # Combine again
    img_vh2 = cv2.addWeighted(vertical_lines2, 0.5, horizontal_lines2, 0.5, 0.0)
    show(img_vh2, ~img_vh2)

    img_vh2 = cv2.erode(~img_vh2, kernel, iterations=2)
    _, img_vh2 = cv2.threshold(img_vh2, 128, 255, cv2.THRESH_BINARY)
    show(img_vh2)

    bitxor = cv2.bitwise_xor(img_bin, img_vh2)
    bitnot = cv2.bitwise_not(bitxor)
    show(bitnot)

    # Contour detection
    contours, hierarchy = cv2.findContours(img_vh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            image = cv2.rectangle(img_gray.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    show(image)

    # Row/column sorting
    row, column = [], []
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    row.append(column)
            else:
                row.append(column)
                column = [box[i]]
                previous = box[i]

    countcol = max(len(r) for r in row)
    index = max(range(len(row)), key=lambda i: len(row[i]))
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    center = np.array(center)
    center.sort()

    finalboxes = []
    for i in range(len(row)):
        lis = [[] for _ in range(countcol)]
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin
