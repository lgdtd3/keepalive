import cv2#调用opencv的库
import time
start = time.time()
import numpy as np

def binarize_image(image_path, threshold):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 将大于阈值的像素设为0，小于阈值的像素设为255
    image = np.where(image < threshold, 0, 255)

    # 显示和保存二值化后的图像
    #cv2.imshow("Binarized Image", image)
    cv2.imwrite("binarized_image.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数进行图像二值化
image_path = r"C:/Documents_2\Image\2.jpg"
threshold = 128
binarize_image(image_path, threshold)
# img_name = r"C:\Documents_2\Image\2.jpg"
# img = cv2.imread(img_name,0)#灰度图
# # cv2.imshow("haha",img)
# # cv2.waitKey()
#
# thread = 128
# rows, cols = img.shape
# #####################################遍历图像中的所有像素点########
# for row in range(rows):
#     for col in range(cols):
#         value = img[row][col]
#         if value > thread:
#             img[row][col] = 255
#         else:
#             img[row][col] = 0
# # cv2.imshow("Binary_image",img)
# # cv2.waitKey()
# cv2.imwrite("Binary_image.png", img)

end = time.time()
print("time:",end-start)
# time: 0.69437227249145568


import cv2#调用opencv的库
import time
start = time.time()
img_name = r"C:\Documents_2\Image\2.jpg"
img = cv2.imread(img_name,0)#灰度图

Threshold = 128
rows, cols = img.shape

#####################################阈值法########
_,Threshold_img = cv2.threshold(img,Threshold,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold_img", Threshold_img)

#####################################自适应均值法########
adaptive_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("adaptive_mean", adaptive_mean)

cv2.waitKey()

cv2.imwrite("Threshold_img.png", Threshold_img)
cv2.imwrite("adaptive_mean.png", adaptive_mean)

end = time.time()
print("time:",end-start)