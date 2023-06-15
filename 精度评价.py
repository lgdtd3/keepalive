import numpy as np
import os
import cv2
from tqdm import trange


def para_cal(pre, gt, thread):
    pre_flag = pre > 255 * thread
    gt_flag = gt == 255
    pre_flag = pre_flag * 1.0
    gt_flag = gt_flag * 1.0

    TP = pre_flag + gt_flag == 2
    FP = pre_flag + (1 - gt_flag) == 2
    TN = (1 - pre_flag) + (1 - gt_flag) == 2
    FN = (1 - pre_flag) + gt_flag == 2

    image = np.zeros((pre.shape[0], pre.shape[1], 3))
    image[TP, :] = [255, 255, 255]  # 道路
    image[FP, :] = [255, 0, 0]  # 地面被预测为道路
    image[TN, :] = [0, 0, 0]  # 地面
    image[FN, :] = [0, 0, 255]  # 道路部分被预测为地面

    TP, FP, TN, FN = map(np.sum, [TP, FP, TN, FN])
    return TP, FP, TN, FN, image


# 更图像仪置
pre_root = r"./test_data\UNet_results/"  # 预测总
gt_root = r"./test_data\target maps/"
save_root = r"./test_data/eval"
os.makedirs(save_root, exist_ok=True)

if __name__ == "__main__":
    img_list = os.listdir(pre_root)
    img_list.sort(key=lambda x: int(x[1:-5]))
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in trange(len(img_list)):
        pre = cv2.imread(pre_root + img_list[i], 0)
        gt = cv2.imread(gt_root + img_list[i], 0)

        # 期候大小保持...数
        target_size = gt.shape[0]
        size = pre.shape[0]
        pre = cv2.resize(pre, dsize=None, fx=target_size / size, fy=target_size / size, interpolation=cv2.INTER_LINEAR)

        TP_temp, FP_temp, TN_temp, FN_temp, image = para_cal(pre, gt, 0.2)
        TP += TP_temp
        FP += FP_temp
        TN += TN_temp
        FN += FN_temp

        image_name = os.path.join(save_root, img_list[i])
        cv2.imwrite(image_name, image)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    Iou = TP / (TP + FN + FP)

    print("precision: {:.4f} \n recall: {:.4f} \n F1: {:.4f} \n IOU: {:.4f}".format(precision, recall, F1, Iou))
