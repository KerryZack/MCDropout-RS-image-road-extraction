from PIL import Image
import os


def crop_images(input_folder_A, input_folder_B, output_folder_images, output_folder_labels, crop_size=512, stride=484):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder_images):
        os.makedirs(output_folder_images)
    if not os.path.exists(output_folder_labels):
        os.makedirs(output_folder_labels)

    # 获取文件夹 A 中的所有图片
    image_files = [f for f in os.listdir(input_folder_A) if f.endswith('.tiff')]

    for filename in image_files:
        # 构建图片和标签的完整路径
        image_path = os.path.join(input_folder_A, filename)
        label_path = os.path.join(input_folder_B, filename.replace('.tiff', '.tif'))

        # 读取图片和标签
        image = Image.open(image_path)
        label = Image.open(label_path)

        # 获取图片尺寸
        width, height = image.size

        # 计算裁剪后的图片数量
        num_crops = (height - crop_size) // stride + 1

        # 进行裁剪并保存裁剪后的图片和标签
        for i in range(num_crops):
            for j in range(num_crops):
                left = j * stride
                upper = i * stride
                right = left + crop_size
                lower = upper + crop_size

                # 裁剪图片和标签
                cropped_image = image.crop((left, upper, right, lower))
                cropped_label = label.crop((left, upper, right, lower))

                # 构建保存路径
                save_filename = f"{os.path.splitext(filename)[0]}_{i*num_crops + j + 1}.png"
                save_image_path = os.path.join(output_folder_images, save_filename)
                save_label_path = os.path.join(output_folder_labels, save_filename)

                # 保存裁剪后的图片和标签
                cropped_image.save(save_image_path)
                cropped_label.save(save_label_path)

                print(f"已保存裁剪后的图片和标签：{save_filename}")

# 定义输入和输出文件夹路径
input_folder_A = "../../dataset/Mass_g/val"
input_folder_B = "../../dataset/Mass_g/val_mask"
output_folder_images = "../../dataset/Mass/val"
output_folder_labels = "../../dataset/Mass/val_mask"

# 调用裁剪函数
crop_images(input_folder_A, input_folder_B, output_folder_images, output_folder_labels)
