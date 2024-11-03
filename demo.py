import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
np.random.seed(3)

def save_first_mask(masks, scores, output_dir="masks"):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 取第一个掩码和分数
    first_mask = masks[0]
    first_score = scores[0]

    # 将掩码转换为图像格式并保存
    mask_image = Image.fromarray((first_mask * 255).astype(np.uint8))
    mask_path = os.path.join(output_dir, "sample_2.jpg")
    mask_image.save(mask_path)

    # 读取掩码图像（假设掩码图像为灰度图像）
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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
    cv2.imwrite('masks/0_2.jpg', result_image)


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

#只显示第一个分数最高的分割掩码
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    if len(masks) > 0:
        mask = masks[0]
        score = scores[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        plt.title(f"Mask 1, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
'''
#显示所有分割下的分割掩码
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
'''
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

image = Image.open("masks/test.jpg")
image = np.array(image.convert("RGB"))

a=503
b=104

c=178
d=472
input_point = np.array([[a, b],[c,d]])
input_label = np.array([1,1])

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    save_first_mask(masks,scores)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    masks.shape  # (number_of_masks) x H x W
    #show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    show_masks(image, [masks[0]], [scores[0]], point_coords=input_point, input_labels=input_label, borders=True)

