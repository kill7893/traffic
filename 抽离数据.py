import os
from pycocotools.coco import COCO
from PIL import Image

# 路径设置
coco_annotation_file = r'D:\丁政\数据集\annotations\instances_train2017.json'  # COCO 注释文件路径
image_folder = r'D:\丁政\数据集\train2017'  # COCO 图片所在的文件夹
output_images_folder = r'D:\丁政\数据集\output\images'  # 输出图片文件夹
output_labels_folder = r'D:\丁政\数据集\output\labels'  # 输出标注文件夹

# 创建输出文件夹
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# 读取 COCO 数据集
coco = COCO(coco_annotation_file)

# 类别名称和 ID 映射
categories_of_interest = ['bicycle', 'motorcycle', 'car','bus','truck']
category_ids = coco.getCatIds(catNms=categories_of_interest)
category_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(category_ids)}
category_name_to_id = {name: idx for idx, name in enumerate(categories_of_interest)}

# 为每个类别创建文件夹
category_folders = {name: os.path.join(output_images_folder, name) for name in categories_of_interest}
for folder in category_folders.values():
    os.makedirs(folder, exist_ok=True)

# 保存图片和标注
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_file_name = img_info['file_name']
    img_path = os.path.join(image_folder, img_file_name)

    # 检查图像是否有感兴趣的类别标注
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    filtered_anns = [ann for ann in anns if ann['category_id'] in category_ids]

    if filtered_anns:
        # 根据标注的类别创建文件夹
        for ann in filtered_anns:
            cat_id = ann['category_id']
            cat_name = category_id_to_name[cat_id]
            category_folder = category_folders[cat_name]
            output_img_path = os.path.join(category_folder, img_file_name)

            # 复制图像到对应的类别文件夹
            try:
                img = Image.open(img_path)
                img.save(output_img_path)
                print(f"图像 {img_file_name} 已保存到 {category_folder}")
            except Exception as e:
                print(f"无法处理图像 {img_file_name}: {e}")
                continue

            # 保存标注为 YOLO 格式
            yolo_file = os.path.join(output_labels_folder, os.path.splitext(img_file_name)[0] + '.txt')

            try:
                with open(yolo_file, 'w') as f:
                    for ann in filtered_anns:
                        bbox = ann['bbox']
                        x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
                        y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
                        width = bbox[2] / img_info['width']
                        height = bbox[3] / img_info['height']
                        f.write(
                            f"{category_name_to_id[category_id_to_name[ann['category_id']]]} {x_center} {y_center} {width} {height}\n")

                print(f"标注信息 {yolo_file} 已保存")
            except Exception as e:
                print(f"无法保存标注信息 {yolo_file}: {e}")

print("处理完成！")