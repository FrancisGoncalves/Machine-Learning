## Imports
import glob
import cv2
import keyboard
import xml.etree.ElementTree as ET
import os
import time

## Paths
BASE_PATH = ''
# IMAGE_PATH = os.path.join(BASE_PATH, r"images_augmented/")
# ANNOTATION_PATH = os.path.join(BASE_PATH, r"annotations_yolo/")
IMAGE_PATH = "..\\obj_augmented\\"
ANNOTATION_PATH = "..\\obj_augmented\\"

print(IMAGE_PATH)

# Variables
directories = ['train/', 'val/', 'test/']
classes = {'white_boat': 0}

print('here')
## Draws bounding boxes on the provided image using its annotation file in YOLO format
def draw_bounding_boxes(image, annotation_yolo):
    # Initialize variables
    image_with_bounding_boxes = image
    
    top_left = None
    bottom_right = None
    

    # Read in all annotations
    f = open(annotation_yolo, "r")
    annotations = f.readlines()
    f.close()

    # Convert normalized box center position, width, and height to corners of bounding box
    bounding_boxes = []

    for annotation in annotations:
        
        # Split the line of text
        yolo_label = annotation.split()
        
        # Remove the new line character '\n' from the last value
        yolo_label[4].rstrip('\n')

        # Convert the classes from a dictionary to a list to access the object name
        class_list = list(classes)

        # Grab the object name
        object_name = class_list[int(yolo_label[0])]

        # Grab the bounding box center coordinates, width, and height
        bounding_box_center_x = float(yolo_label[1])
        bounding_box_center_y = float(yolo_label[2])
        bounding_box_width = float(yolo_label[3])
        bounding_box_height = float(yolo_label[4])

        # Convert the normalized coordinates and values into actual values
        bounding_box_center_x = int(bounding_box_center_x * image.shape[1])
        bounding_box_center_y = int(bounding_box_center_y * image.shape[0])
        bounding_box_width = int(bounding_box_width * image.shape[1])
        bounding_box_height = int(bounding_box_height * image.shape[0])

        # Use the center position, width, and height to construct pixel coordinates of the
        # top left and bottom right of the bounding box for the cv2.drawRectangle() function
        top_left = ((bounding_box_center_x - int(bounding_box_width / 2)),
                    (bounding_box_center_y - int(bounding_box_height / 2)))
        
        bottom_right = ((bounding_box_center_x + int(bounding_box_width / 2)),
                        (bounding_box_center_y + int(bounding_box_height / 2)))

        # print(top_left)
        # print(bottom_right)
        # Add the bounding box corners and the object label to a dictionary for later use
        bounding_boxes.append({'object': object_name, 'top_left': top_left, 'bottom_right': bottom_right})
        # print(bounding_boxes)
    print(image)
    for box in bounding_boxes:
        print(image)
        top_left = box['top_left']
        bottom_right = box['bottom_right']
        print(box)

        print("xmin: {0} ymin: {1} xmax: {2} ymax: {3}".format(str(top_left[0]), str(top_left[1]), str(bottom_right[0]), str(bottom_right[1])))
        
        image_with_bounding_boxes = cv2.rectangle(image_with_bounding_boxes, top_left, bottom_right, (255, 255, 255), 2)
        image_with_bounding_boxes = cv2.putText(image_with_bounding_boxes, str(box['object']), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    return image_with_bounding_boxes

## Main
def main():
    # Grab the images and YOLO annotations
    # NEW_IMAGE_PATH = IMAGE_PATH + "\\"
    image_paths = glob.glob(IMAGE_PATH + '/*.jpg')
    annotation_paths = glob.glob(ANNOTATION_PATH + '/*.txt')

    # Sort
    image_paths.sort()
    annotation_paths.sort()

    # Initialize index
    i = 0
    # print(i)
    # Run until window is closed
    # Read keyboard input from the user
    while(1):
        try:  # used try so that if user presdsed other than the given key error will not be shown
            # Read in the image
            # print(image_paths)
            image = cv2.imread(image_paths[i])
            print(image_paths[i])
            print(annotation_paths[i])
            # # Place bounding boxes on the image
            image = draw_bounding_boxes(image, annotation_paths[i])
            
            # Display the image
            window_name = image_paths[i].lstrip(IMAGE_PATH)
            # cv2.namedWindow(window_name)
            # cv2.moveWindow(window_name, 1000, 550)
            cv2.imshow(window_name, image)
            # Get user input
            k = cv2.waitKey(0) # 33

            # Esc key to stop
            if k == 27:
                cv2.destroyAllWindows()
                break

            # 'd' key is pressed
            elif k == 100:
                i = i + 1
                cv2.destroyAllWindows()

                if i > (len(image_paths) - 1):
                    i = len(image_paths) - 1

            # 'a' key is pressed
            elif k == 97:
                i = i - 1
                cv2.destroyAllWindows()

                if i < 0:
                    i = 0

        except:
            # print('error')
            pass

    # Debugging
    print("Complete")

##
if __name__ == "__main__":
    main()