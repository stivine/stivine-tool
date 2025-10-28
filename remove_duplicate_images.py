import os

# 输入QQ号
qq_number = input("请输入QQ号: ").strip()

# 使用例：处理2024年1月到11月的目录
# 目录名格式为"2024-01", "2024-02", ..., "2024-11"
begin_month = "2024-01"
num = 11

directory_path = f"/mnt/e/qqpersonal/{qq_number}/nt_qq/nt_data/Pic/"

# directory_path = f"/mnt/e/qqpersonal/{qq_number}/nt_qq/nt_data/Emoji/emoji-recv

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

for month in range(num):
    month_dir = os.path.join(directory_path, f"{begin_month[:5]}{str(month).zfill(2)}")
    if os.path.exists(month_dir):
        manage_image_files(month_dir)
    else:
        print(f"Directory {month_dir} does not exist.")