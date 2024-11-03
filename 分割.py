import os

# 设置文件夹路径
images_folder = r'D:\丁政\数据集\output\images'
labels_folder = r'D:\丁政\数据集\output\labels'

# 获取 images 文件夹中的文件名（不包括扩展名）
images_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith('.jpg')}

# 获取 labels 文件夹中的文件名（不包括扩展名）
labels_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')}

# 找出 labels 文件夹中存在但 images 文件夹中不存在的文件名
files_to_delete = labels_files - images_files

# 删除 labels 文件夹中那些不在 images 文件夹中的标注文件
for file in files_to_delete:
    file_path = os.path.join(labels_folder, f'{file}.txt')
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'已删除: {file_path}')
    else:
        print(f'未找到文件: {file_path}')