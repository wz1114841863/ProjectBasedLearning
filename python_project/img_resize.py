import cv2 as cv
import struct


def save_bin_file(img_path, save_bin_path):
    """ 将图像保存为二进制文件 """
    img = cv.imread(img_path)
    print(f"图片的原始形状: {img.shape}")

    target_shape = (1280, 720)  # H = 720, W = 1280
    img = cv.resize(img, target_shape)
    img_blue = img[:, :, 0]
    img_green = img[:, :, 1]
    img_red = img[:, :, 2]

    # 保存为二进制文件, 计算大小: W * H * C = 720 * 1280 * 1 = 921600 Byte = 900 KB
    # img_rgb = cv.merge((img_red, img_green, img_blue))
    with open(save_bin_path, "wb") as fp:
        # 逐行遍历图像,并依次写入 R, G, B 通道的值
        for row_red, row_green, row_blue in zip(img_red, img_green, img_blue):
            for r, g, b in zip(row_red, row_green, row_blue):
                # 按 RGB 的顺序写入三个通道的值,每个通道一个字节 (unsigned char)
                fp.write(struct.pack('B', r))  # 写入红色通道
                fp.write(struct.pack('B', g))  # 写入绿色通道
                fp.write(struct.pack('B', b))  # 写入蓝色通道


def scale_to_ascii(value, min_value=0, max_value=255):
    """ 将RGB的值缩放到ASCII字符范围内 (0-127) """
    return int((value / max_value) * 127)


def save_img_as_ascii(img, save_txt_path):
    """ 将图像的RGB值转为ASCII并保存到txt文件 """
    with open(save_txt_path, "w", encoding="ascii") as file:
        for row in img:
            for r, g, b in row:
                # 将每个通道的值缩放到ASCII范围内
                ascii_r = scale_to_ascii(r)
                ascii_g = scale_to_ascii(g)
                ascii_b = scale_to_ascii(b)

                # 将RGB值转换为ASCII字符
                file.write(chr(ascii_r))  # 红色通道
                file.write(chr(ascii_g))  # 绿色通道
                file.write(chr(ascii_b))  # 蓝色通道

            # 每行结束后加一个换行符
            file.write("\n")


def save_txt_file(img_path, save_txt_path):
    """ 将图像保存为ASCII文件 """
    img = cv.imread(img_path)

    target_shape = (1280, 720)
    img_resized = cv.resize(img, target_shape)

    save_img_as_ascii(img_resized, save_txt_path)
    print(f"图像的RGB值已转换为ASCII字符并保存到 {save_txt_path}")


if __name__ == "__main__":
    # 指定图片文件路径
    img_path = "D:\\Projects\\3895\\npu_srio\\user_space_yolov5\\img_cases\\img01.jpg"
    save_bin_path = ""
    save_txt_path = "./img01.txt"
    save_txt_file(img_path, save_txt_path)
