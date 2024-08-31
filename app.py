import cv2
import pixellib
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation
segment_image.load_model("mask_rcnn_coco.h5")
camera = cv2.VideoCapture(0)