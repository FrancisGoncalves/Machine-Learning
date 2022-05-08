## Imports
import glob

## Paths
BASE_PATH = r"/content/gdrive/MyDrive"
IMAGE_PATH = "."
ANNOTATION_PATH = "."

## Compares the name of each image and xml annotation
def check_for_pairs(image_paths, xml_paths):
    for i in range(len(image_paths)):
        image_name = image_paths[i].lstrip(IMAGE_PATH).rstrip(".jpg")
        xml_name = xml_paths[i].lstrip(ANNOTATION_PATH).rstrip(".txt")

        if image_name != xml_name:
            print("Image {} does not have an annotation".format(image_paths[i]))

        else:
            print("ok")

## Main
def main():
    image_paths = glob.glob(IMAGE_PATH + '/*.jpg')
    xml_paths = glob.glob(ANNOTATION_PATH + '/*.txt')

    image_paths.sort()
    xml_paths.sort()

    check_for_pairs(image_paths, xml_paths)

##
if __name__ == "__main__":
    main()