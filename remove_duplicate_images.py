import os

def manage_image_files(directory, extensions=[".jpg", ".png"]):
    """
    遍历指定目录，查找并处理同一图片不同分辨率的文件对。
    若存在xxx_0.<ext>和xxx_720.<ext>，则删除xxx_0.<ext>；
    若只有xxx_0.<ext>而没有对应xxx_720.<ext>，则不进行操作。
    
    :param directory: 要检查的目录路径
    :param extensions: 要处理的文件扩展名列表
    """
    directory_thumb = os.path.join(directory, "thumb")

    # 获取目录中所有的文件名
    files = os.listdir(directory_thumb)
    
    # 创建一个集合用于存放已经确认需要保留（即有高分辨率版本）的基名
    keep_set = set()
    
    # 先找出所有存在高分辨率版本（即xxx_720.<ext>）的基名，并添加到keep_set中
    for file in files:
        for ext in extensions:
            if file.endswith(f"_720{ext}"):
                base_name = file.rsplit('_', 1)[0]
                keep_set.add(base_name)
    
    # 再次遍历文件，这次根据我们的规则删除低分辨率版本
    for file in files:
        for ext in extensions:
            if file.endswith(f"_0{ext}"):
                base_name = file.rsplit('_', 1)[0]
                if base_name in keep_set:
                    os.remove(os.path.join(directory_thumb, file))
                    print(f"Deleted: {file}")

    # 下一步：ori目录中出现过的文件，删除掉thumb目录中的同名_0后缀文件
    directory_ori = os.path.join(directory, "ori")
    files_ori = os.listdir(directory_ori)
    for file in files_ori:
        for ext in extensions:
            if file.endswith(f"{ext}"):
                base_name = file.rsplit('.', 1)[0]
                thumb_file = os.path.join(directory_thumb, f"{base_name}_0{ext}")
                if os.path.exists(thumb_file):
                    os.remove(thumb_file)
                    print(f"Deleted from thumb: {thumb_file}")


# 使用示例：请将这里的目录替换为你想要检查的实际目录路径

# 对2024-01到2024-11目录进行处理

directory_path = "/mnt/e/qqpersonal/1623970771/nt_qq/nt_data/Pic/"

# 2024年1月到11月的目录
for month in range(1, 12):
    month_dir = os.path.join(directory_path, f"2024-{month:02d}")
    if os.path.exists(month_dir):
        manage_image_files(month_dir)
    else:
        print(f"Directory {month_dir} does not exist.")