from PIL import Image


def process_and_save_image(input_path, output_path, target_size=(64, 64)):
    """将输入图片处理为指定分辨率，并保存为指定格式。

    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_size: 目标分辨率，默认为 (64, 64)
    """
    # 打开图片，并确保转换为24位RGB格式
    image = Image.open(input_path).convert('RGB')
    # 调整图片大小到目标分辨率
    image = image.resize(target_size, Image.LANCZOS)
    # 保存图片为PNG格式
    image.save(output_path, format='PNG')
    print(f"Image saved to {output_path}")


# 示例使用
input_path = '2.jpg'
output_path = '2-.png'
process_and_save_image(input_path, output_path)

# 验证图片尺寸
image = Image.open(output_path)
print(f"Processed image size: {image.size}")
