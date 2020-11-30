import cv2
import matplotlib.pyplot as plt

match_matrix = []

for i in range(5):
    for j in range(2):
        matrix = []
        for k in range(5):
            for l in range(2):
                fp1 = r'E:\pictures-shidi\pictures\sobel2/' + str(i) + str(j) + '.jpg'
                img1 = cv2.imread(fp1)

                fp2 = r'E:\pictures-shidi\pictures\sobel2/' + str(k) + str(l) + '.jpg'
                img2 = cv2.imread(fp2)

                training_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                query_image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
                query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)


                orb = cv2.ORB_create(400,2)

                keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
                keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
                matches = bf.match(descriptors_train, descriptors_query)
                matches = sorted(matches, key = lambda x : x.distance)
                new_matches = []

                for o in matches:
                    train_idx = o.queryIdx
                    query_idx = o.trainIdx
                    train_point = keypoints_train[train_idx].pt
                    query_point = keypoints_query[query_idx].pt
                    x_gap = abs(train_point[0] - query_point[0])
                    y_gap = abs(train_point[1] - query_point[1])
                    if x_gap < 45 and y_gap < 30:
                        new_matches.append(o)

                new_matches = [p for p in new_matches if p.distance < 60]

                # result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_gray, flags = 2)
                result = cv2.drawMatches(training_image, keypoints_train, query_image, keypoints_query, new_matches[:300], query_image, flags = 2)

                # plt.title('Best Matching Points')
                # plt.imshow(result)
                # plt.show()

                #print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
                #print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
                #print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
                matrix.append(len(new_matches))
                #print(len(matches))

        print(matrix)
        match_matrix.append(matrix)
print(k)
z=(k+1)*2
# print(match_matrix)
c=[]
a=0
for j in range(z):
    for k in range(z):
        if k==j+1 and j%2==0:
            c.append(match_matrix[j][k])
        if j%2 ==0:
            if match_matrix[j][k]>=match_matrix[j][j+1]:
                a=a+1
            #(跟j,j+1比较)
        else:
            if match_matrix[j][k]>=match_matrix[j][j-1]:
                a=a+1
            #跟j，j-1比较
a=a-(j+1)*2
a=a*2
b=a/((j+1)*(j+1))
# print(a)
# print(j+1)
print(c)
print(b)