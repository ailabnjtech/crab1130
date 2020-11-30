import cv2
import numpy as np
import time
from PIL import Image
from yolo import YOLO

yolo = YOLO()

def find_rect(image_path):

    image = Image.open(image_path)
    img = cv2.imread(image_path)

    top, left, bottom, right = yolo.detect_image(image)
    w, h = right - left, bottom - top
    rect = img[top + h//6: bottom - h*10//23, left + w//10: right - w//10]

    w = 512
    rect = cv2.resize(rect, (w, rect.shape[0]*w//rect.shape[1]))
    return rect

def edge_sobel(src):
    """Sobel边缘检测"""
    kernelSize = (7, 7)
    gausBlurImg = cv2.GaussianBlur(src, kernelSize, 0)

    # 转换为灰度图
    channels = src.shape[2]
    if channels > 1:
        src_gray = cv2.cvtColor(gausBlurImg, cv2.COLOR_RGB2GRAY)
    else:
        src_gray = src.clone()

    depth = cv2.CV_16S

    # 求X方向梯度（创建grad_x, grad_y矩阵）
    grad_x = cv2.Sobel(src_gray, depth, 1, 0)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    # 求Y方向梯度
    grad_y = cv2.Sobel(src_gray, depth, 0, 1)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并梯度（近似）
    edgeImg = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edgeImg
    # return cv2.threshold(edgeImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

def find_sobel(img_path):
    return edge_sobel(find_rect(img_path))



sobel_img = find_sobel(img_path)
# cv2.imshow('sobel_img', sobel_img)
# cv2.waitKey(0)
