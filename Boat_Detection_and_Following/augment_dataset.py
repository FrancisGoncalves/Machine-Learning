#!/usr/bin/env python3
##hello
## Imports
import cv2
import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.etree.ElementTree as ET
import numpy as np
import random

## Paths to source images and annotations
# BASE_PATH = r'C:\Users\Francisco Gonçalves\Desktop'
IMAGE_PATH = '.'
ANNOTATION_PATH = '.'
IMAGE_SAVE_PATH = "C:\\Users\\Francisco Gonçalves\\Desktop\\obj_augmented\\"
ANNOTATION_SAVE_PATH = "C:\\Users\\Francisco Gonçalves\\Desktop\\obj_augmented\\"
print(IMAGE_PATH)

## Seed random number generator
ia.seed(1)

## Functions
# Add salt and pepper noise to image, solution from https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
## Returns a list of modified images
def add_sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''

    modified_images = []

    for probability in prob:
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - probability 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < probability:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        modified_images.append(output)

    return modified_images

## Modifies the brightness of the image uniformly. Returns a list of modified images
## https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
def modify_brightness(image, brightness_modifier):
    modified_images = []

    # Iterate through list of brightness modifiers
    for brightness in brightness_modifier:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        #
        value = brightness

        # If the modifier is above 0, make the image brighter
        if value > 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            # Add the brightened image to the return image list
            modified_images.append(img)

        # If the modifier is below 0, make the image darker
        elif value <= 0:
            lim = 0 - value
            v[v < lim] = 0
            v[v >= lim] += np.uint8(value)

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            # Add the brightened image to the return image list
            modified_images.append(img)

    # Return the list of brightness modified images
    return modified_images

## Modifies the clarity of the source image. Returns a list of modified images
def blur_image(image, kernel_size):
    modified_images = []

    for kernel in kernel_size:
        image_aug = cv2.blur(image, ksize = kernel) # kernel is a tuple, e.g. (5,5)
        modified_images.append(image_aug)

    # Retutn the list of modified images
    return modified_images

## Save the image
def save_images(images, starting_number, destination, xml_path, bb_modified = False, updated_bb = None):
    number = starting_number

    i = 0

    # Iterate through the passed images
    for image in images:
        # Name the image
        # file_name = "Image" + str(number) + ".jpg"
        file_name = "..\\obj_augmented\\" + "Image" + str(number) + ".jpg"
        # Save the image
        # print(destination + file_name)
        # print(image)
        cv2.imwrite(file_name, image)

        # Update the xml file path and filename
        

        # Update bounding boxes
        if bb_modified == True:
            # Grab the root tag
            root = xml_file.getroot()

            # Grab all of the object tags
            objects = root.findall('object')

            # index
            j = 0

            # Iterate through the objects, updating bounding box coordinates
            for ob in objects:
                ob[4][0].text = str(updated_bb[i][j]['xmin'])
                ob[4][1].text = str(updated_bb[i][j]['ymin'])
                ob[4][2].text = str(updated_bb[i][j]['xmax'])
                ob[4][3].text = str(updated_bb[i][j]['ymax'])

                # Increment the index
                j = j + 1

        # Save the xml file
        # y = ''
        # xml_save_path = ANNOTATION_SAVE_PATH + file_name.rstrip('.jpg') + '.txt'
        # print(xml_path)
        # f = open(xml_path, "r")
        # annotations = f.readlines()
        # f.close()
        # for annotation in annotations:
        #   y = y + str(annotation) + ' '
        # # print(y)
        # ff = open(xml_save_path, 'w')
        # ff.write(y)
        # ff.close()

        # Increment number of saved images
        number = number + 1

        # Increment index
        i = i + 1

    # Return the number of saved images
    return number

# Return the value from an input tag parameter
def getTagContents(xml_file, tag):
    # Grab the root tag
    root = xml_file.getroot()

    # Check for object tag
    x = root.find(tag)
    if x is not None:
        #print(x.text)
        return x.text

    # Tag does not exist
    return None

# Update xml tag
def updateTag(xml_file, tag, new_value):
    # Grab the root tag
    root = xml_file.getroot()

    # Check for object tag
    x = root.find(tag)
    if x is not None:
        x.text = new_value

    return xml_file

# Check bounds of index. Should be 0-299
def validate_index(index):
    if index < 0:
        index = 0
    elif index > 299:
        index = 299

    return index

## Rotates the image. Returns an updated list of images and an updated list of bounding boxes
def rotate_image(image, xml_file, rotations, image_name):
    #
    e = [] # List of updated bounding box coordinates
    modified_images = []
    modified_bb = []

    # Grab the root tag
    root = xml_file.getroot()

    # Grab all object tags
    objects = root.findall('object')

    # If there are objects labeled
    if objects is not None:
        for rotation in rotations:
            a = [] # 

            # Grab all of the bounding boxes belonging to each individual object in the xml file
            for ob in objects:
                b = {'xmin': int(ob[4][0].text),
                        'ymin': int(ob[4][1].text),
                        'xmax': int(ob[4][2].text),
                        'ymax': int(ob[4][3].text)}

                # Pass the bounding box coordinates to the transformer
                bbs = BoundingBoxesOnImage([
                    BoundingBox(x1=b['xmin'],
                                y1=b['ymin'],
                                x2=b['xmax'],
                                y2=b['ymax'])
                ], shape=image.shape)

                # Rotate the image
                seq = iaa.Sequential([
                    iaa.Affine(rotate=rotation)
                ])

                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
                bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

                # Grab the updated bounding box coordinates from the list to use in updating the xml file
                try:
                    coordinates = {'xmin': int(bbs_aug.bounding_boxes[0].x1),
                                'ymin': int(bbs_aug.bounding_boxes[0].y1),
                                'xmax': int(bbs_aug.bounding_boxes[0].x2),
                                'ymax': int(bbs_aug.bounding_boxes[0].y2)
                                }

                    coordinates['xmin'] = validate_index(coordinates['xmin'])
                    coordinates['ymin'] = validate_index(coordinates['ymin'])
                    coordinates['xmax'] = validate_index(coordinates['xmax'])
                    coordinates['ymax'] = validate_index(coordinates['ymax'])

                except IndexError:
                    print('Index error when retrieving updated bounding boxes')
                    print(image_name)
                    print("Bounding box coordinates: {0} {1} {2} {3}".format(str(coordinates['xmin']), 
                                                                            str(coordinates['ymin']),
                                                                            str(coordinates['xmax']),
                                                                            str(coordinates['ymax'])))
                    cv2.imshow('window',image_aug)
                    cv2.waitKey(0)
                    
                    exit()

                # Populate a with the updated bounding box coordinates
                a.append(coordinates)

            # Add the new image and new bounding box coordinates to their respective lists
            modified_images.append(image_aug)
            modified_bb.append(a)

    # Return the list of images and list of bounding boxes
    return modified_images, modified_bb

## Main
def main():
    ## Image augmentation section
    images_processed = 0
    file_name = ''

    # Read in the paths
    image_paths = glob.glob(IMAGE_PATH + '/*.jpg')
    #xml_paths = glob.glob(ANNOTATION_PATH + '*.xml')

    # Sort the paths
    image_paths.sort()
    #xml_paths.sort()

    # Iterate through all images in the directory
    for i in range(len(image_paths)):
        # Grab the image name
        image_name = image_paths[i].lstrip(IMAGE_PATH)

        # Grab the image
        source_image = cv2.imread(image_paths[i])
        

        # Read in the xml file. This file will be updated and written multiple times
        xml_path = image_paths[i].replace('obj', 'obj')
        xml_path = xml_path.replace('jpg', 'txt')

        # source_xml_file = ET.parse(xml_path)

        # Save the original image w/ a new name
        images_processed = save_images([source_image], images_processed, IMAGE_SAVE_PATH, xml_path, bb_modified = False)

        # Add some noise to the image
        modified_images = add_sp_noise(source_image, [0.015, 0.02, 0.025])
        #cv2_imshow(image_aug)

        # Save the modified images
        images_processed = save_images(modified_images, images_processed, IMAGE_SAVE_PATH, xml_path, bb_modified = False)

        # Add some blur to the image
        modified_images = blur_image(source_image, kernel_size = [(3,3), (5,5), (7,7)])
        #cv2_imshow(image_aug)

        # Save the modified images
        images_processed = save_images(modified_images, images_processed, IMAGE_SAVE_PATH, xml_path, bb_modified = False)

        # Modify the brightness of the source image
        modified_images = modify_brightness(source_image, [25, 50, -10])

        # Save the modified images
        images_processed = save_images(modified_images, images_processed, IMAGE_SAVE_PATH, xml_path, bb_modified = False)

        # Rotate the image
        # modified_images, modified_bb = rotate_image(source_image, source_xml_file, [-40, -30, -20, -10, 10, 20, 30, 40], image_name)

        # Save the modified images
        # images_processed = save_images(modified_images, images_processed, IMAGE_SAVE_PATH, source_xml_file, bb_modified = True, updated_bb = modified_bb)

        # Execute code once
        #break

        # Debugging
        if i % 10 == 0:
           print("Images processed: {}".format(images_processed))
           #break

    print("Complete")

##
if __name__ == "__main__":
    main()