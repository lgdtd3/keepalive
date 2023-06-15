import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import glob
from data_loader import RandomCrop, ToTensorLab, RescaleT, SalObjDataset
from unet import UNet
import time

# -------1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')


def main():
    # -------2. set the directory of training dataset --------
    model_name = 'UNet'
    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = "img2\\"
    tra_label_dir = "GT2\\"
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    epoch_num = 250
    batch_size_train = 4

    tra_img_name_list = glob.glob(os.path.join(data_dir + tra_image_dir + '*'))
    tra_lbl_name_list = glob.glob(os.path.join(data_dir + tra_label_dir + '*'))

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=2
    )

    # -------3. define model --------
    # define the net, whether to use GPU
    if torch.cuda.is_available():
        net = UNet().cuda()
    else:
        net = UNet()

    # -------4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(
        net.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0
    )

    # -------5. training process --------
    print("---start training...")
    ite_num, running_loss, ite_num4val, save_frq = 0, 0.0, 0, 10
    txt = open("training loss.txt", "a+")
    txt.write("epoch" + "\t\t" + "loss" + "\n")

    for epoch in range(epoch_num):
        net.train()
        start = time.time()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), \
                    Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), \
                    Variable(labels, requires_grad=False)

            optimizer.zero_grad()

            d0 = net(inputs_v)
            loss = bce_loss(d0, labels_v)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()

            # del temporary outputs and loss
            del d0, loss

            print("[epoch:%3d/%3d, batch:%5d/%5d, ite:%d] train loss:%3f" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

            log = str(epoch + 1) + "\t\t" + str(np.mean(running_loss / ite_num4val)) + "\n"
            txt.write(log)
            txt.flush()

        # ------- 6. Save weight --------
        if epoch > 0 and epoch % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name + "_%d.pth" % (epoch))

        running_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0

        end = time.time()
        print("Time:", end - start)


if __name__ == "__main__":
    main()