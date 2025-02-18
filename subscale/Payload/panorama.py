import cv2
from stitching import Stitcher

image_paths = [
    "depth1.jpg",
    "depth2.jpg",
    "depth3.jpg",
    "depth4.jpg",
    # Add more images as required wrt to the pictures taken per revolution
]

images = [cv2.imread(img_path) for img_path in image_paths]

stitcher = Stitcher()

result = stitcher.stitch(images)

if result is not None:
    cv2.imwrite("panorama_360_depth.jpg", result)
    print("Panorama saved as 'panorama_360.jpg")

    cv2.imshow("360 Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Stitching failed. Ensure the images overlap sufficiently.")