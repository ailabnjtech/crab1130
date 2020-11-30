import cv2
import numpy as np

def edge_sobel(src):
    """Sobel边缘检测"""
    kernelSize = (3, 3)
    gausBlurImg = cv2.GaussianBlur(src, kernelSize, 0)

    # 转换为灰度图
    channels = src.shape[2]
    if channels > 1:
        src_gray = cv2.cvtColor(gausBlurImg, cv2.COLOR_RGB2GRAY)
    else:
        src_gray = src.clone()

    depth = cv2.CV_16S

    # 求X方向梯度（创建grad_x, grad_y矩阵）
    grad_x = cv2.Sobel( src_gray, depth, 1, 0)
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    # 求Y方向梯度
    grad_y = cv2.Sobel( src_gray, depth, 0, 1)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并梯度（近似）
    edgeImg = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edgeImg

def test_sobel():
    for i in range(1,16):
        fp1 = r'E:\crabtest_all\0913test\cut/' + str(i) + '1.jpg'
        img1 = cv2.imread(fp1)

        # fp2 = r'E:\cut/' + str(k) + str(l) + '.jpg'
        # img2 = cv2.imread(fp2)

        sobelImg1 = edge_sobel(img1)

        # sobelImg2 = edge_sobel(img2)

        ret, thresh1 = cv2.threshold(sobelImg1, 20, 255, cv2.THRESH_BINARY)
        # ret, thresh2 = cv2.threshold(sobelImg2, 20, 255, cv2.THRESH_BINARY)

        cv2.imwrite(r'E:\crabtest_all\0913test\sobel/' + str(i) + '1.jpg', sobelImg1)
        # cv2.imshow('img2', img2)
        # cv2.imshow("sobelImg2", sobelImg2)
        # cv2.imwrite(r'D:\crabs/' + str(k) + str(l) + '.jpg', sobelImg2)





test_sobel()















