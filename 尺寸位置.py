import cv2

# 全局变量来存储鼠标点击的位置
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将鼠标坐标转换为原始图像坐标
        scale_x = original_width / display_width
        scale_y = original_height / display_height
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        print(f"Mouse Position in Original Image: ({original_x}, {original_y})")

# 读取图像
image_path = 'masks/test.jpg'  # 请替换成你自己的图像路径
img = cv2.imread(image_path)
original_height, original_width = img.shape[:2]

# 设置显示窗口的初始大小
display_width, display_height = 800, 600

while True:
    # 调整图像大小以适应窗口
    resized_img = cv2.resize(img, (display_width, display_height))

    # 创建窗口并设置鼠标回调函数
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    # 显示图像
    cv2.imshow('Image', resized_img)

    # 等待按键事件
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按 'ESC' 键退出
        break
    elif key == ord('+'):
        # 增加显示窗口大小
        display_width = int(display_width * 1.1)
        display_height = int(display_height * 1.1)
    elif key == ord('-'):
        # 减少显示窗口大小
        display_width = int(display_width * 0.9)
        display_height = int(display_height * 0.9)

# 释放所有窗口
cv2.destroyAllWindows()