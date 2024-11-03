import cv2
import numpy as np

# 读取掩码图像（假设掩码图像为灰度图像）
mask_image = cv2.imread('sample_2.jpg', cv2.IMREAD_GRAYSCALE)
# 确保图像是二值图像（只有黑色和白色）
_, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
# 找到白色区域的轮廓
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建一个全白的图像（可以用作掩模）
white_image = np.ones_like(binary_mask) * 255
# 设置轮廓面积阈值
area_threshold = 70  # 根据需要调整阈值
# 创建一个全黑的图像，用于填充处理
result_image = np.copy(mask_image)
# 遍历每个轮廓
for contour in contours:
    # 计算轮廓的面积
    contour_area = cv2.contourArea(contour)
    if contour_area >= area_threshold:
        # 创建一个全黑的图像用于掩模
        contour_mask = np.zeros_like(binary_mask)
        # 填充轮廓区域
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        # 膨胀操作，扩展白色区域
        kernel = np.ones((35, 35), np.uint8)  # 根据需要调整膨胀核的大小
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)
        # 将膨胀后的区域内的黑色区域变为白色
        result_image[(dilated_mask == 255) & (binary_mask == 0)] = 255
# 保存处理后的图像
cv2.imwrite('0_2.jpg', result_image)