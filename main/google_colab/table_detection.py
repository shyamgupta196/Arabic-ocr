import os
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction(img, predictor):
    """
    Plots predicted bounding boxes on an input image using a given predictor.
    Args:
        img (numpy.ndarray or PIL.Image): The input image on which to plot predictions.
        predictor (callable): A function or model that takes an image and returns prediction outputs,
            expected to include bounding boxes under outputs["instances"].get_fields()["pred_boxes"].
    Returns:
        None. Displays the image with predicted bounding boxes using matplotlib.
    Note:
        - The function draws blue rectangles (bounding boxes) around detected objects (e.g., tables).
        - The image is displayed using matplotlib's imshow and show functions.
    """
    
    outputs = predictor(img)

    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2

    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        # Start coordinate 
        # represents the top left corner of rectangle 
        start_point = int(x1), int(y1) 
  
        # Ending coordinate
        # represents the bottom right corner of rectangle 
        end_point = int(x2), int(y2) 
  
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        img = cv2.rectangle(np.array(img, copy=True), start_point, end_point, color, thickness)
    # Displaying the image
    print("TABLE DETECTION:")  
    plt.imshow(img)
    plt.show()
    # save_image_incremental(img, 'plot_prediction_image', 'plotted_images')

def make_prediction(img, predictor, show_result=False):
    """
    Detects tables in an image using a given predictor, extracts the detected table regions,
    saves them as separate images, and optionally displays the results.
    Args:
        img (numpy.ndarray): The input image in which to detect tables.
        predictor (callable): A model or function that takes an image and returns detection outputs.
        show_result (bool, optional): If True, displays each detected table region. Defaults to False.
    Returns:
        tuple:
            table_list (list of numpy.ndarray): List of cropped table images extracted from the input image.
            table_coords (list of list of int): List of coordinates for each detected table in the format [x, y, width, height].
            filenames (list of str): List of filenames where each cropped table image has been saved.
    """

    
    #img = cv2.imread(img_path)
    outputs = predictor(img)

    table_list = []
    table_coords = []
    filenames = []

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()):
        x1, y1, x2, y2 = box
        table_list.append(np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True))
        table_coords.append([int(x1),int(y1),int(x2-x1),int(y2-y1)])
        filename = save_image_incremental(img[int(y1):int(y2), int(x1):int(x2)])
        filenames.append(filename)

        if show_result:
            plt.imshow(img[int(y1):int(y2), int(x1):int(x2)])
            plt.show()

    return table_list, table_coords, filenames



def save_image_incremental(image , base_filename='image', directory='extracted_tables'):
    """
    Saves an image to a specified directory with an incrementally numbered filename to avoid overwriting existing files.
    Args:
        image (numpy.ndarray): The image to be saved, represented as a NumPy array (typically from OpenCV).
        base_filename (str, optional): The base name for the saved image file. Defaults to 'image'.
        directory (str, optional): The directory where the image will be saved. Defaults to 'extracted_tables'.
    Returns:
        str: The full path of the saved image file.
    Notes:
        - The function creates the target directory if it does not exist.
        - The image is saved in PNG format.
        - The filename is automatically incremented (e.g., image_0.png, image_1.png, ...) to prevent overwriting.
    """

    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize an increment counter
    increment = 0

    while True:
        # Generate the filename with an increment
        filename = os.path.join(directory, f"{base_filename}_{increment}.png")

        # Check if the file already exists
        if not os.path.exists(filename):
            # Save the image using OpenCV
            cv2.imwrite(filename, image)
            print(f"Image saved as {filename}")
            break
        # Increment the counter to try the next filename
        increment += 1
    return filename