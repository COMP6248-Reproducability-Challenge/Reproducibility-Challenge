import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x1_0, resnet50
from my_dataset import MyDataSet,MyDataSet_REP
from multi_train_utils import train_one_epoch, evaluate, get_next

def color_distortion(s=1.0):
    """ A strong data transformation following SimCLR """
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    version = 6
    start_epoch = 0
    iter_epoch = 1

    namee= "new_train_hk_v"+str(version)+'.csv'
    namegt= "new_train_500_hk.csv"

    if os.path.exists("./weights_rep_v"+str(version)) is False:
        os.makedirs("./weights_rep_v"+str(version))

    data_transform = {
        "train_bs1": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     color_distortion(s=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                    #  transforms.RandomRotation(15),
                                     color_distortion(s=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    json_path = "./classes_name.json"
    # 实例化训练数据集
    # train_dataset = MyDataSet(root_dir=data_root,
    #                           csv_name="new_train.csv",
    #                           json_path=json_path,
    #                           transform=data_transform["train"])

    # # check num_classes
    # if args.num_classes != len(train_dataset.labels):
    #     raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),
    #                                                                     args.num_classes))

    # 实例化验证数据集
    # val_dataset = MyDataSet(root_dir=data_root,
    #                         csv_name="new_val.csv",
    #                         json_path=json_path,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)

    # create model
    # model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    model = resnet50(num_classes=args.num_classes).to(device)

    # 如果存在预训练权重则载入
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):

        # if epoch>start_epoch and epoch%iter_epoch==0:
        if epoch>0:
            checkpoint = torch.load("./weights_rep_v{}/model-{}.pth".format(version,epoch-1))
            model.load_state_dict(checkpoint)
            train_dataset = MyDataSet_REP(root_dir=data_root,
                                    csv_name=namegt,
                                    json_path=json_path,
                                    transform=data_transform["train_bs1"])

            # check num_classes
            # if args.num_classes != len(train_dataset.labels):
            #     raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),args.num_classes))
            train_loader_bs1 = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)
    
            get_next(model=model, data_loader=train_loader_bs1, device=device, namee=namee)

        # if  epoch<iter_epoch:
        if  epoch==0:
            train_dataset = MyDataSet_REP(root_dir=data_root,
                                csv_name=namegt,
                                json_path=json_path,
                                transform=data_transform["train"])
        else:
            train_dataset = MyDataSet_REP(root_dir=data_root,
                                    csv_name=namee,
                                    json_path=json_path,
                                    transform=data_transform["train"])

        # # check num_classes
        # if args.num_classes != len(train_dataset.labels):
        #     raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),args.num_classes))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)

        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    warmup=True)

        scheduler.step()

        # validate
        # acc = evaluate(model=model,
        #                data_loader=val_loader,
        #                device=device)

        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # tags = ["loss", "accuracy", "learning_rate"]
        tags = ["loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights_rep_v{}/model-{}.pth".format(version, epoch))        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.0001)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/home/yzbj10/Work/lxj/data/mini_imagenet/mini-imagenet")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
