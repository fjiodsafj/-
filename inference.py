# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               ★★★请在空白处填写适当的语句，将模型推理应用流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose


def process_image(image_path, target_size=(64, 64)):
    """加载并处理图片，将图片调整为指定分辨率。

    :param image_path: 输入图片路径
    :param target_size: 目标分辨率，默认为 (64, 64)
    :return: 处理后的图片张量
    """
    image = Image.open(image_path).convert('RGB')  # 确保图片为24位（8位RGB通道）
    transform = Compose([Resize(target_size), ToTensor()])
    return transform(image)


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    # START----------------------------------------------------------
    # 处理图片
    image = process_image(image_path).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 返回预测结果
    predicted_class = predicted.item()
    print(f'Predicted class: {predicted_class}')
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 指定图片路径
    image_path = "1.jpg"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
