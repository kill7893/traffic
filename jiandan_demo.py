import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import random
import sys
from ultralytics import YOLO  #检测到-world字样会自动调用YOLOWorld
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#用于第一次识别警告滞留车辆
def Warning_message():
    return



def process_masks_and_iou(result_image_path, information_id, output_dir):

    def create_mask_from_box(image_shape, box):
        """
        创建基于单个矩形框的掩码图像。
        :param image_shape: 图像的形状 (高度, 宽度)
        :param box: 单个矩形框 (x1, y1, x2, y2)
        :return: 掩码图像
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = map(int, box)  # 将坐标转换为整数
        mask[y1:y2, x1:x2] = 255
        return mask

    def calculate_iou(mask1, mask2):
        """
        计算两个二值掩码图像之间的IoU。
        :param mask1: 第一个掩码图像
        :param mask2: 第二个掩码图像
        :return: IoU 值
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union

    def draw_box_on_image(image, box, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制单个矩形框。
        :param image: 要绘制矩形框的图像
        :param box: 矩形框 (x1, y1, x2, y2)
        :param color: 矩形框的颜色 (B, G, R)
        :param thickness: 矩形框的厚度
        :return: 绘制了矩形框的图像
        """
        x1, y1, x2, y2 = map(int, box)  # 将坐标转换为整数
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    # 读取处理过的掩码图像
    result_image = cv2.imread(result_image_path, cv2.IMREAD_GRAYSCALE)
    # 获取图像形状
    image_shape = result_image.shape
    # 在结果图像上绘制所有矩形框
    result_image_color = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    # 计算每个矩形框与掩码的IoU
    iou_results = {}
    for id, info in information_id.items():
        box = info['last_coords']
        mask = create_mask_from_box(image_shape, box)
        iou = calculate_iou(result_image, mask)
        iou_results[id] = iou
        # 在图像上绘制矩形框
        result_image_color = draw_box_on_image(result_image_color, box)
    # 保存绘制了矩形框的图像
    output_path = os.path.join(output_dir, 'result_image_with_boxes.jpg')
    cv2.imwrite(output_path, result_image_color)
    # 保存 IoU 结果到文件
    iou_output_path = os.path.join(output_dir, 'iou_results.txt')
    with open(iou_output_path, 'w') as f:
        for id, iou in iou_results.items():
            if iou > 0:
                f.write(f'ID {id}: IoU = {iou}\n')
                print(f"当前滞留ID {id}: IoU = {iou}，占用道路！\n")
    print('处理完成，IoU 结果已保存到 iou_results.txt')


#获得分割掩码并修复掩码，并备份图片
def process_image(information_id,image_path, checkpoint_path, model_cfg_path, output_dir="masks", input_points=None, input_labels=None,
                  multimask_output=True):
    #分割得分最高的一个道路掩码，并对掩码进行修复
    def save_first_mask(masks, scores, output_dir,information_id):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        first_mask = masks[0]
        first_score = scores[0]
        # 将 first_mask 转换为 8-bit 格式 (0-255)
        first_mask = (first_mask * 255).astype(np.uint8)  # 假设掩码值在 [0, 1] 范围内
        # 保存掩码图像
        mask_file_path = os.path.join(output_dir, 'sample_2.jpg')
        cv2.imwrite(mask_file_path, first_mask)
        # 读取掩码图像（假设掩码图像为灰度图像）
        mask_image = cv2.imread('masks/sample_2.jpg', cv2.IMREAD_GRAYSCALE)
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
        cv2.imwrite('masks/optimize_sample_2.jpg', result_image)
        #利用修复后的掩码和车辆进行IOU计算
        process_masks_and_iou('masks/optimize_sample_2.jpg', information_id, 'masks')

    def show_mask(mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    predictor = SAM2ImagePredictor(build_sam2(model_cfg_path, checkpoint_path))
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Perform prediction
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output,
        )

        save_first_mask(masks, scores, output_dir,information_id)

        #下面这段代码可以注释掉，主要用于显示分割时候的显示
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        show_masks(image, [masks[0]], [scores[0]], point_coords=input_points, input_labels=input_labels, borders=True)

#分割指定类别
def sam_id(end_out,last_frame_path):
    information_id={}
    #用于道路训练识别之前 0
    number=0
    points_list = []
    for item in end_out:
        # 假设这里的tensor是PyTorch张量，使用.tolist()转换为列表
        first_coords_list = tuple(
            coord.tolist() if isinstance(coord, torch.Tensor) else coord for coord in item['first_coords'])
        last_coords_list = tuple(
            coord.tolist() if isinstance(coord, torch.Tensor) else coord for coord in item['last_coords'])
        '''
        print(f"ID: {item['id']}")
        print(f"Class: {item['class']}")
        print(f"First Coordinates: {first_coords_list}")
        print(f"Last Coordinates: {last_coords_list}")
        print(f"Speed: {item['speed'].item() if isinstance(item['speed'], torch.Tensor) else item['speed']}")
        '''
        #缺陷：只识别一个大的道路，如果面临多个道路如何处理------------------------------------------------------------------------------------>>>
        #
        # 获取类别,如果类别为0，则跳过该项
        cls = item['class']
        if cls == 0:
            number=number+1
            # print(f"ID: {item['id']}")
            # print(f"Class: {item['class']}")
            # 获取每个道路的对角线的坐标（左上角、右下角）并计算出矩形框的中心点坐标用于分割
            x1, y1, x2, y2 = last_coords_list
            center1_x = (x1 + x2) / 2
            center1_y = (y1 + y2) / 2
            new_point = [center1_x, center1_y]
            points_list.append(new_point)

            # 计算缩小后的范围
            width = (x2 - x1) / 2
            height = (y2 - y1) / 2
            # 随机生成一个点，范围在中心点的缩小区域内
            random_x = random.uniform(center1_x - width / 2, center1_x + width / 2)
            random_y = random.uniform(center1_y - height / 2, center1_y + height / 2)
            random_point = [random_x, random_y]
            points_list.append(random_point)
            continue
        # 将信息保存到information_id字典中
        information_id[item['id']] = {
            'class': cls,
            'last_coords': last_coords_list
        }
    print("提取的滞留物体矩形框信息：")
    print(information_id)
    # 定义参数
    image_path = last_frame_path  # 替换为你的图像文件路径
    checkpoint_path = "./weights/checkpoints/sam2_hiera_tiny.pt"  # 替换为你的模型检查点路径
    model_cfg_path = "sam2_hiera_t.yaml"  # 替换为你的模型配置文件路径
    output_dir = "./masks"  # 替换为你希望保存结果的目录路径


#------------------------------------------
    # 示例使用
   # new_point = [561, 1353]
   # points_list.append(new_point)
   # new_point = [89, 1703]
    #points_list.append(new_point)
    if len(points_list) != number*2:
        raise ValueError(f"points_list的长度 ({len(points_list)}) 不等于n ({number})")
    # 将列表转换为 NumPy 数组
    input_points = np.array(points_list)
#------------------------------------------
    input_labels = np.array([1]*number*2)  # 示例输入标签，1 表示正类，0 表示负类

    process_image(
            information_id=information_id,
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            model_cfg_path=model_cfg_path,
            output_dir=output_dir,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=True
        )
    return


#在视频中插入指定文本数据
def process_frame(frame,results):
    boxes_dict = {}
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy
                if len(xyxy) > 0 and len(xyxy[0]) == 4:
                    box_id = box.id.item() if isinstance(box.id, torch.Tensor) else box.id
                    box_cls = box.cls.item() if isinstance(box.cls, torch.Tensor) else box.cls
                    # 检查类别是否为 0（即 "道路" 类别），如果不是，则将其添加到字典中
                    if box_cls != 0:
                        if box_id not in boxes_dict:
                            boxes_dict[box_id] = []
                        boxes_dict[box_id].append({'class': int(box_cls)})
        else:
            print("结果对象没有 'boxes' 属性")
    # 初始化一个字典来统计每个 class 出现的次数
    class_count = {}
    for frame_key, boxes in boxes_dict.items():
        for box in boxes:
            class_value = box.get('class')
            if class_value is not None:
                if class_value in class_count:
                    class_count[class_value] += 1
                else:
                    class_count[class_value] = 1
    # 定义文本的属性
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_color = (0, 0, 255)  # 黑色
    line_type = 2
    # 初始化文本位置
    position = (10, 150)  # 起始位置
    N=0
    P=0
    for cls, count in class_count.items():
        #对class数值进行映射 "way","car","motorcycle","bicycle","bus","truck","person"
        if cls==1:
            name="car"
        elif cls==2:
            name="motorcycle"
        elif cls==3:
            name = "bicycle"
        elif cls ==4:
            name = "bus"
        elif cls==5:
            name = "truck"
        elif cls==6:
            name = "person"
        if cls != 6:
            N = N + count
        else:
            P=P+count
        K = float(N / 60)
        D = float(P / 600)
        text = f"Class {name}: {count} times"
        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)
        # 更新位置，使下一行文本显示在下一行
        position = (position[0], position[1] + 40)  # 每行间隔30像素
    # 写入固定文本信息，包含K的数值
    fixed_text_N = f"Traffic Density K: {K:.2f} Unit:[vehicle/M]"
    cv2.putText(frame, fixed_text_N, (10, 50), font, font_scale, font_color, line_type)
    fixed_text_P = f"Human flow density D: {D:.2f} Unit:[person/M*M]"
    cv2.putText(frame, fixed_text_P, (10, 90), font, font_scale, font_color, line_type)
    return frame

#分析第一帧和最后一帧结果，并计算指定ID的速度
def boxes_print_and_speed(first_results, last_results, frame_rate):
    # 计算速度的函数
    def calculate_speed(coords1, coords2, frame_rate):
        (x1, y1, x2, y2) = coords1
        (x1_next, y1_next, x2_next, y2_next) = coords2
        # 将张量移到 CPU 并转换为 NumPy 数组
        x2_next = x2_next.cpu().numpy()
        x2 = x2.cpu().numpy()
        y2_next = y2_next.cpu().numpy()
        y2 = y2.cpu().numpy()
        distance = np.sqrt((x2_next - x2) ** 2 + (y2_next - y2) ** 2)
        speed = distance * frame_rate
        return speed
    # 提取边界框信息的函数,提取的是xyxy，不是xywh。（每个yolov系列算法属性不一定相同。yolov8三个属性都有xyxy、xywh、xyxyn）
    def extract_boxes(results):
        boxes_dict = {}
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy
                    if len(xyxy) > 0 and len(xyxy[0]) == 4:
                        x1, y1, x2, y2 = xyxy[0]
                        box_id = box.id.item() if isinstance(box.id, torch.Tensor) else box.id
                        box_cls = box.cls.item() if isinstance(box.cls, torch.Tensor) else box.cls
                        # 检查类别是否为 0（即 "道路" 类别），如果不是，则将其添加到字典中
                        if box_cls != 0:
                            print("---------------------------------------------------------------------------------")
                        if box_id not in boxes_dict:
                            boxes_dict[box_id] = []
                        boxes_dict[box_id].append({'coordinates': (x1, y1, x2, y2), 'class': int(box_cls)})
            else:
                print("结果对象没有 'boxes' 属性")
        return boxes_dict
    # 提取边界框信息
    first_boxes = extract_boxes(first_results)
    last_boxes = extract_boxes(last_results)
    high_speed_boxes = []
    # 打印和计算速度
    for box_id in first_boxes:
        if box_id in last_boxes:
            first_coords = first_boxes[box_id][0]['coordinates']
            last_coords = last_boxes[box_id][0]['coordinates']
            speed = calculate_speed(first_coords, last_coords, frame_rate)
            print(f"ID: {int(box_id)}   类别: {first_boxes[box_id][0]['class']}")
            print(
                f"第一帧边界框坐标: ({first_coords[0]:.2f}, {first_coords[1]:.2f}), ({first_coords[2]:.2f}, {first_coords[3]:.2f})")
            print(
                f"最后一帧边界框坐标: ({last_coords[0]:.2f}, {last_coords[1]:.2f}), ({last_coords[2]:.2f}, {last_coords[3]:.2f})")
            print(f"计算的速度: {speed:.2f} 像素/秒")
            if speed < 4000:
                print("当前类别为滞留状态\n")
                high_speed_boxes.append({
                    'id': int(box_id),
                    'class': first_boxes[box_id][0]['class'],
                    'first_coords': first_coords,
                    'last_coords': last_coords,
                    'speed': speed
                })
        else:
            print("跟踪物体在最后一帧中没有找到")
    return high_speed_boxes

#用于计算车辆速度和行人速度
def boxes_vehicle_persong_speed(first_results, last_results, frame_rate):
    # 计算速度的函数
    def calculate_speed(coords1, coords2, frame_rate):
        (x1, y1, x2, y2) = coords1
        (x1_next, y1_next, x2_next, y2_next) = coords2
        # 将张量移到 CPU 并转换为 NumPy 数组
        x2_next = x2_next.cpu().numpy()
        x2 = x2.cpu().numpy()
        y2_next = y2_next.cpu().numpy()
        y2 = y2.cpu().numpy()
        # 计算欧几里得距离
        distance = np.sqrt(np.square(x2_next - x2) + np.square(y2_next - y2))
        # 根据帧率计算速度
        speed = distance * frame_rate
        return speed
    # 提取边界框信息的函数,提取的是xyxy，不是xywh。（每个yolov系列算法属性不一定相同。yolov8三个属性都有xyxy、xywh、xyxyn）
    def extract_boxes(results):
        boxes_dict = {}
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy
                    if len(xyxy) > 0 and len(xyxy[0]) == 4:
                        x1, y1, x2, y2 = xyxy[0]
                        box_id = box.id.item() if isinstance(box.id, torch.Tensor) else box.id
                        box_cls = box.cls.item() if isinstance(box.cls, torch.Tensor) else box.cls
                        # 检查类别是否为 0（即 "道路" 类别），如果不是，则将其添加到字典中
                        if box_cls != 0:
                            if box_id not in boxes_dict:
                                boxes_dict[box_id] = []
                            boxes_dict[box_id].append({'coordinates': (x1, y1, x2, y2), 'class': int(box_cls)})
            else:
                print("结果对象没有 'boxes' 属性")
        return boxes_dict
    # 提取边界框信息
    first_boxes = extract_boxes(first_results)
    last_boxes = extract_boxes(last_results)
    speed_boxes = []
    # 打印和计算速度
    for box_id in first_boxes:
        if box_id and box_id in last_boxes:
            first_coords = first_boxes[box_id][0]['coordinates']
            last_coords = last_boxes[box_id][0]['coordinates']
            speed = calculate_speed(first_coords, last_coords, frame_rate)
            # 计算像素与实际尺寸的转换因子（单位：米/像素）
            conversion_factor = 2.0 / 300
            # 计算实际位移（单位：米）
            actual_distance = speed * conversion_factor
            # 计算速度（单位：米/秒）
            velocity_meters_per_second = actual_distance * frame_rate

            speed_boxes.append({
                    'id': int(box_id),
                    'class': first_boxes[box_id][0]['class'],
                    'speed':  velocity_meters_per_second
                })
    return speed_boxes

if __name__ == '__main__':
    # 加载开集检测模型并给与提示
    model = YOLO("weights/yolov8x-worldv3.pt")
    #model.set_classes(["car", "bus", "motorcycle", "truck", "bicycle", "way"])
    #  model.set_classes(["person","car"])  0:person,1:car #第一个类别不进行绘制预测框
    model.set_classes(["way","car","motorcycle","bicycle","bus","truck","person"])
    # 检测视频路径
    video_path = "source/test.mp4"
    cap = cv2.VideoCapture(video_path)
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 定义输出视频文件
    output_path = "runs/test3.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 读取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数：{total_frames}")
    frame_count = 0
    # 循环处理视频帧
    old_r=[]
    speed=[]
    while cap.isOpened():
        # 读取一帧
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            new_r=results
            if old_r:
                speed=boxes_vehicle_persong_speed(old_r, new_r, fps)
                print(speed)
            if frame_count == 0:
                first_results = results
            #隔一段时间就检测一次，本实验是检测两次，第一次检测到先警告，第二次则判定处罚
            if ( frame_count == int(total_frames/2) ) or ( frame_count == int(total_frames-1)):
                last_results = results
                # 通过矩形框位置计算指定ID类别的运动速度，这里的矩形框位置是按照四个顶点坐标格式，不是中心点坐标、宽度高度
                end_out = boxes_print_and_speed(first_results, last_results, fps)
                first_results=last_results
                if end_out:
                    print("检测到疑似滞留物体：")
                    ids = [item['id'] for item in end_out]
                    annotated_frame = results[0].plot(appoint_ID=ids)
                    out.write(annotated_frame)
                    # 喊话警告一次
                    if frame_count == int(total_frames / 2):
                        frame_count = frame_count + 1
                        print("警告一次")
                        Warning_message()
                        continue
                    else:
                        if frame is not None:
                            last_frame_path = "last_frame.jpg"
                            cv2.imwrite(last_frame_path, frame)
                            print(f"最后一帧保存为: {last_frame_path}")
                        break
                else:
                    ids=[]
                    print("没有疑似滞留物体")
            old_r=new_r
            frame_count = frame_count + 1
            # 在帧上可视化跟踪结果
            annotated_frame = results[0].plot(speed=speed)
            # 处理当前帧，添加车辆数量（不同类型的车辆数量）、道路占比
            frame_with_text = process_frame(annotated_frame, results)
            out.write(frame_with_text)
            # 将注释后的帧写入输出视频
            out.write(annotated_frame)
        else:
            # 如果视频结束，则退出
            break

    # 释放视频捕捉和写入对象，关闭显示窗口
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # 增加判断机制
    if ids:
        print(f"当前滞留ID为{ids}，进一步判断是否占用车道：")
        # 对指定类别进行分割，并进行IOU检测
        sam_id(end_out, last_frame_path)
    else:
        print("没有检测到滞留物体！")






