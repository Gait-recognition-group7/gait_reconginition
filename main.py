import background
import GEI
import os
path = "E:\\untitled\\video"
save_path = "E:\\untitled\\GEI"
for (root, dirs, files) in os.walk(path):
    for file in files:
        # 检测人形，提取剪影
        binary = background.extract_silhouette(os.path.join(path, file))
        # 制造GEI能量图
        GEI.get_gei_from_one_cycle(save_path, binary, file)
    pass
pass
