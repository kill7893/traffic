import cv2
import os
import numpy as np


def calculate_white_percentage_diff(img_a_path, img_b_path):
    # 读取图像
    img_a = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)

    # 计算总像素数量
    total_pixels = img_a.shape[0] * img_a.shape[1]

    # 计算白色区域的像素数量
    white_pixels_a = np.sum(img_a == 255)
    white_pixels_b = np.sum(img_b == 255)

    # 计算白色像素的百分比
    white_percentage_a = (white_pixels_a / total_pixels) * 100
    white_percentage_b = (white_pixels_b / total_pixels) * 100

    # 计算增加的白色像素百分比
    white_percentage_diff = white_percentage_b - white_percentage_a

    return total_pixels, white_percentage_a, white_percentage_b, white_percentage_diff


def compare_folders(folder_a, folder_b):
    # 获取文件夹中的文件列表
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))

    # 找出相同命名的文件
    common_files = files_a & files_b

    for file in common_files:
        img_a_path = os.path.join(folder_a, file)
        img_b_path = os.path.join(folder_b, file)

        # 计算白色像素百分比差异
        total_pixels, white_percentage_a, white_percentage_b, white_percentage_diff = calculate_white_percentage_diff(img_a_path, img_b_path)

        # 打印结果
        print(f"图像 {file}:")
        print(f"  总像素数量: {total_pixels}")
        print(f"  a 中白色像素百分比: {white_percentage_a:.2f}%")
        print(f"  b 中白色像素百分比: {white_percentage_b:.2f}%")
        print(f"  b 相较于 a 增加了 {white_percentage_diff:.2f}% 的白色像素\n")


# 文件夹路径
folder_a = 'a'
folder_b = 'b'

# 比较两个文件夹中的图像
compare_folders(folder_a, folder_b)