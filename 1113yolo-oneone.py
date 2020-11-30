import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from yolo import YOLO
yolo = YOLO()

def find_rect(image_path):

    image = Image.open(image_path)
    img = cv2.imread(image_path)

    top, left, bottom, right = yolo.detect_image(image)
    w, h = right - left, bottom - top
    rect = img[top + h//4: bottom - h*12//23, left + w//7: right - w//7]

    w = 512
    rect = cv2.resize(rect, (w, rect.shape[0]*w//rect.shape[1]))
    plt.title('Best Matching Points')
    plt.imshow(rect)
    plt.show()
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
    # cv2.imshow('1',edgeImg)
    # cv2.waitKey(0)
    return edgeImg
    # return cv2.threshold(edgeImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

def find_sobel(img_path):
    return edge_sobel(find_rect(img_path))


#fp1 = r'E:\pic_xinlixun\no1\sobel\a.jpg'
#img1 = cv2.imread(fp1)

#fp2 = r'E:\pic_xinlixun\no1\sobel\30.jpg'
#img2 = cv2.imread(fp2)

#training_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#query_image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
#query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
match_matrix = []
matrix = []


training_gray = find_sobel(r'E:\pictures-shidi\pictures\jiaozheng/30.bmp')
query_gray = find_sobel(r'E:\pictures-shidi\pictures\sj/31.jpg')
# cv2.imwrite(r'E:\pic_xinlixun\yolo\cut/' + str(k) + '.jpg', query_gray)

# plt.rcParams['figure.figsize'] = [14.0, 7.0]
# orb = cv2.ORB_create(400, 1.12, patchSize=55)
orb = cv2.ORB_create(400, 1.1)

keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key=lambda x: x.distance)
new_matches = []

for o in matches:
    train_idx = o.queryIdx
    query_idx = o.trainIdx
    train_point = keypoints_train[train_idx].pt
    query_point = keypoints_query[query_idx].pt
    x_gap = abs(train_point[0] - query_point[0])
    y_gap = abs(train_point[1] - query_point[1])
    if x_gap < 30 and y_gap < 30:
        new_matches.append(o)

new_matches = [p for p in new_matches if p.distance < 60]

# result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_gray, flags = 2)
# result = cv2.drawMatches(training_image, keypoints_train, query_image, keypoints_query, new_matches[:300],
# query_image, flags=2)

# plt.title('Best Matching Points')
# plt.imshow(result)
# plt.show()

# print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
# print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
# print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
matrix.append(len(new_matches))
# print(len(matches))
print(matrix)
match_matrix.append(matrix)
