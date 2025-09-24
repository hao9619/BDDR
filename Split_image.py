import os
from PIL import Image


def split_image_with_resize(input_folder, output_folder):
    """
    Split 4-in-1 images into individual images, resize them back to original dimensions,
    and save them to their original category folders.

    Args:
        input_folder (str): Path to the folder containing the 4-in-1 images.
        output_folder (str): Path to the folder where individual images will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    # Open the image
                    img = Image.open(file_path)
                    original_width, original_height = img.size

                    # Calculate dimensions for each sub-image
                    sub_width = original_width // 2
                    sub_height = original_height // 2

                    # Crop 4 sub-images
                    sub_images = [
                        img.crop((0, 0, sub_width, sub_height)),  # Top-left
                        img.crop((sub_width, 0, original_width, sub_height)),  # Top-right
                        img.crop((0, sub_height, sub_width, original_height)),  # Bottom-left
                        img.crop((sub_width, sub_height, original_width, original_height))  # Bottom-right
                    ]

                    # Create output sub-folder based on original folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    category_folder = os.path.join(output_folder, relative_path)

                    if not os.path.exists(category_folder):
                        os.makedirs(category_folder)

                    # Save each sub-image after resizing back to original dimensions
                    for idx, sub_img in enumerate(sub_images):
                        # Resize sub-image back to original dimensions
                        resized_sub_img = sub_img.resize((original_width, original_height), Image.LANCZOS)

                        sub_img_name = f"{os.path.splitext(file)[0]}_part{idx + 1}.png"
                        sub_img_path = os.path.join(category_folder, sub_img_name)
                        resized_sub_img.save(sub_img_path)

                    print(f"Processed: {file_path} -> 4 images resized to {original_width}x{original_height}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


def split_image_with_custom_resize(input_folder, output_folder, target_width=None, target_height=None):
    """
    Split 4-in-1 images into individual images and resize them to specified dimensions.

    Args:
        input_folder (str): Path to the folder containing the 4-in-1 images.
        output_folder (str): Path to the folder where individual images will be saved.
        target_width (int): Target width for resized images. If None, uses original width.
        target_height (int): Target height for resized images. If None, uses original height.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    # Open the image
                    img = Image.open(file_path)
                    original_width, original_height = img.size

                    # Set target dimensions
                    resize_width = target_width if target_width else original_width
                    resize_height = target_height if target_height else original_height

                    # Calculate dimensions for each sub-image
                    sub_width = original_width // 2
                    sub_height = original_height // 2

                    # Crop 4 sub-images
                    sub_images = [
                        img.crop((0, 0, sub_width, sub_height)),  # Top-left
                        img.crop((sub_width, 0, original_width, sub_height)),  # Top-right
                        img.crop((0, sub_height, sub_width, original_height)),  # Bottom-left
                        img.crop((sub_width, sub_height, original_width, original_height))  # Bottom-right
                    ]

                    # Create output sub-folder based on original folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    category_folder = os.path.join(output_folder, relative_path)

                    if not os.path.exists(category_folder):
                        os.makedirs(category_folder)

                    # Save each sub-image after resizing
                    for idx, sub_img in enumerate(sub_images):
                        # Resize sub-image to target dimensions
                        resized_sub_img = sub_img.resize((resize_width, resize_height), Image.LANCZOS)

                        sub_img_name = f"{os.path.splitext(file)[0]}_part{idx + 1}.png"
                        sub_img_path = os.path.join(category_folder, sub_img_name)
                        resized_sub_img.save(sub_img_path)

                    print(f"Processed: {file_path} -> 4 images resized to {resize_width}x{resize_height}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


def simain():
    # 使用示例1：还原到原图尺寸
    input_folder = "/root/autodl-tmp/DataAll/swift_data/20000_cifar10_poisoned_images_4"  # 替换为您的输入文件夹路径
    output_folder = "/root/autodl-tmp/DataAll/swift_data/20000_cifar10_poisoned_images"  # 替换为您的输出文件夹路径

    print("方法1：还原到原图尺寸")
    split_image_with_resize(input_folder, output_folder)

    # # 使用示例2：自定义尺寸
    # print("\n方法2：自定义尺寸 (例如: 512x512)")
    # split_image_with_custom_resize(input_folder, output_folder + "_custom", 512, 512)



