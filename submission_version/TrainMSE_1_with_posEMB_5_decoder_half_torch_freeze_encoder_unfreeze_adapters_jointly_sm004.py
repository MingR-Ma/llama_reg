import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from DataLoad import DataGenerator
import argparse
import math
from datetime import datetime, timedelta, timezone
import numpy as np
import losses
from models.model import DualLLama, SecDecoder, OtherDecoder, SpatialTransform
import csv
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import json
from tqdm import tqdm
import sys
from tools import show, save_checkpoint, generate_grid, grid2contour
import time as t
from llama import LLaMATransformer
import matplotlib.pyplot as plt
from tools import jacobian_determinant
from tools import MidpointNormalize
from torch.cuda.amp import GradScaler, autocast

sys.path.append("/home/mamingrui/PycharmProjects/LightWeightModel/ext/neuron/")
sys.path.append("/home/mamingrui/PycharmProjects/LightWeightModel/ext/pynd-lib/")
sys.path.append("/home/mamingrui/PycharmProjects/LightWeightModel/ext/pytools-lib/")
sys.path.append("/home/mamingrui/PycharmProjects/LightWeightModel/ext/medipy-lib/")

import glob

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='param')
parser.add_argument('--iters', default=300001, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--id_param', default=1, type=float)
# parser.add_argument('--path', default='/home/mamingrui/data/slice/', type=str)
parser.add_argument('--atlas_names', default="./atlases.npy", type=str)
parser.add_argument('--valsets_names', default='./valsets_names.npy', type=str)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--checkpoint_path',
                    default="./Weights/",
                    type=str)
parser.add_argument('--checkpoint_file',
                    default="./Weights/checkpoint.pth.tar",
                    type=str)
parser.add_argument('--json_file',
                    default='./knee.json',
                    type=str)
parser.add_argument('--train_path', default='./train_path/', type=str)
parser.add_argument('--label_path', default='./labels/', type=str)

parser.add_argument('--log_folder',
                    default='./Log/', type=str)
parser.add_argument('--range_flow', default=1.0, type=float)
parser.add_argument('--reg_smooth', default=0.04, type=float)  # 1 for loss_smooth NCC
parser.add_argument('--VI', default=2000, type=int)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True


def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")


def validation1(range_flow, grid, atlases, atlases_label, valsets, valsets_label, atlas_show, val_show, model,
                labels, slice=None):
    start_time = t.time()
    vol_length = len(valsets)
    print(vol_length)
    atlas_length = len(atlases)
    print("Validation:")
    print(atlas_length, vol_length, len(labels))

    val_acc_sum = 0.0
    # val_acc_sum_=0.0
    jac_acc_sum = 0.0

    atlases = sorted(atlases)
    atlases_label = sorted(atlases_label)
    valsets = sorted(valsets)
    valsets_label = sorted(valsets_label)

    with torch.no_grad():

        STN = SpatialTransform([160, 160, 96]).cuda()
        model.float().eval()

        for atlas, atlas_label in zip(atlases, atlases_label):

            atlas_volume = np.load(atlas)
            # atlas_volume=imgnorm(Norm_Zscore(atlas_volume))
            # print(atlas_volume.max(),atlas_volume.mean(),atlas_volume.min())

            atlas_label = np.load(atlas_label)
            atlas_tensor = torch.Tensor(atlas_volume).unsqueeze(0).unsqueeze(0).cuda()

            acc_list = []
            jac_list = []

            # acc_list_=[]

            for val, val_label in zip(valsets, valsets_label):

                val_volume = np.load(val)
                # val_volume=imgnorm(Norm_Zscore(val_volume))
                # print(val_volume.max(),val_volume.mean(),val_volume.min())

                val_label = np.load(val_label)
                val_volume_tensor = torch.Tensor(val_volume).unsqueeze(0).unsqueeze(0).cuda()
                val_label_tensor = torch.Tensor(val_label).unsqueeze(0).unsqueeze(0).cuda()

                warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f, mid_fea, fea_1, fea_2, fea_3, fea_4 = model(
                    val_volume_tensor,
                    atlas_tensor)

                flow = f3
                pred = warped_x_1
                pred_label = STN(val_label_tensor, flow * range_flow, 'nearest')
                pred_label = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()

                flow = (flow * range_flow).permute(0, 2, 3, 4, 1)
                acc = np.sum(dice(atlas_label, pred_label, labels)) / len(labels)

                acc_list.append(acc)

                flow_per = flow.squeeze(0)
                flow_per = flow_per.detach().cpu()
                jac_det = jacobian_determinant(flow_per)

                jac_neg_per = np.sum([i <= 0 for i in jac_det]) / (
                        jac_det.shape[0] * jac_det.shape[1] * jac_det.shape[2])
                jac_list.append(jac_neg_per)

                if atlas == atlas_show and val == val_show:
                    atlas_slice = atlas_volume[:, slice, :]

                    volume_slice = val_volume[:, slice, :]
                    volume_label_slice = val_label[:, slice, :]

                    pred_slice = pred[0, 0, :, slice, :]
                    pred_slice = pred_slice.squeeze(0).squeeze(0).detach().cpu().numpy()

                    jac_det_slice = jac_det[:, slice, :]
                    atlas_label_slice = atlas_label[:, slice, :]
                    pred_label_slice = pred_label[:, slice, :]

            print(f"A atlas Accuracy dice : {acc_list}.")
            val_acc = np.sum(acc_list) / vol_length
            # val_acc_=np.sum(acc_list_)/vol_length
            print(f"Average predict for atlas dice is {val_acc}")
            jac_neg_per = np.sum(jac_list) / vol_length

            val_acc_sum = val_acc + val_acc_sum
            # val_acc_sum_=val_acc_+val_acc_sum_

            jac_acc_sum = jac_neg_per + jac_acc_sum

    jac_neg_per = jac_acc_sum / atlas_length
    val_acc = val_acc_sum / atlas_length
    # val_acc_ = val_acc_sum_ / atlas_length
    time_spend = t.time() - start_time
    return val_acc, time_spend, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_neg_per, atlas_label_slice, pred_label_slice


def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    :return a list as the label length
    """

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def freeze_params(model):
    model.conv_m_1.requires_grad_(False)
    model.conv_f_1.requires_grad_(False)

    model.d_conv_m_1.requires_grad_(False)
    model.d_conv_f_1.requires_grad_(False)
    model.conv_m_2.requires_grad_(False)
    model.conv_f_2.requires_grad_(False)

    model.d_conv_m_2.requires_grad_(False)
    model.d_conv_f_2.requires_grad_(False)
    model.conv_m_3.requires_grad_(False)
    model.conv_f_3.requires_grad_(False)

    model.d_conv_m_3.requires_grad_(False)
    model.d_conv_f_3.requires_grad_(False)
    model.conv_m_4.requires_grad_(False)
    model.conv_f_4.requires_grad_(False)

    model.d_conv_m_4.requires_grad_(False)
    model.d_conv_f_4.requires_grad_(False)

    model.llama_block[2].requires_grad_(False)
    model.llama_block[5].requires_grad_(False)

    return model


def train_val_data_load(train_path, label_path, atlases_names_file, valsets_names_file, json_file):
    train_path = train_path
    label_path = label_path

    atlases_names = np.load(atlases_names_file).tolist()

    valsets_names = np.load(valsets_names_file)
    atlases = [train_path + str(atlases_names) + '.npy' for test_name in atlases_names]
    atlases_label = [label_path + str(atlases_names) + '_label.npy' for test_name in atlases_names]
    print(f'Atlases :\n {atlases}')

    # valsets&label
    valsets = [train_path + str(valsets_names) + '.npy' for test_name in valsets_names]
    valsets_label = [label_path + str(valsets_names) + '_label.npy' for test_name in valsets_names]
    print(f'Validation :\n {valsets}')

    atlases = sorted(atlases)
    atlases_label = sorted(atlases_label)
    valsets = sorted(valsets)
    valsets_label = sorted(valsets_label)

    # trainset
    with open(json_file, 'r') as file:
        data_dict = json.load(file)
    train_names = data_dict[0:200]

    return atlases, atlases_label, valsets, valsets_label, train_names


def Train_1(iters, lr, train_path, label_path, atlases_names, valsets_names, labels, resume, json_file, checkpoint_path,
            checkpoint_file, range_flow,
            log_folder, reg_smooth, validation_iters):
    checkpoint_path = checkpoint_path + 'model_1/'
    log_folder = log_folder + 'model_1/'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    print(f"LOG_FLODER : {log_folder}")
    print(f"CHECKPOINT FOLDER : {checkpoint_path}")

    atlases, atlases_label, valsets, valsets_label, train_names = train_val_data_load(train_path, label_path,
                                                                                      atlases_names, valsets_names,
                                                                                      json_file)
    # tensorboardX
    writer = SummaryWriter(log_folder)

    # grid generator
    grid = generate_grid(list(image_size))
    print('grid.shape='.format(grid.shape))


    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()



    model = DualLLama(in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()

    transform = SpatialTransform([160, 160, 96]).cuda()  # grid sample in my code.
    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True

    # model=SYMNet().cuda()
    train_set = DataGenerator(train_names, train_path, 1)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)

    print("----------------Train phase----------------.")

    # checkpoint_file="/home/mamingrui/PycharmProjects/formal_version/proposed_64_5_cycle/att_checkpoint/lsNCClr0.0005/verdense/s1_1.0/range_8.0_id0.5/proposed_64_5_decon.Net/18000_checkpoint.pth.tar"
    # check_point = torch.load(checkpoint_file, map_location='cpu')
    # state_iter = check_point['epoch']

    # print(f'Training restart at : {state_iter}th epoch.', flush=True)
    # model.load_state_dict(check_point['state_dict'])

    loss_sim = torch.nn.MSELoss

    Grad_loss = losses.smoothloss

    opt = Adam(model.parameters(), lr=lr)

    if resume:
        counter = 0
        flag = 0.0
        check_point = torch.load(checkpoint_file, map_location='cpu')
        state_iter = check_point['epoch']

        best_acc1 = check_point['best_acc']

        print(f'Training restart at : {state_iter}th epoch.', flush=True)

        model.load_state_dict(check_point['state_dict'])
        opt.load_state_dict(check_point['optimizer'])
        opt.param_groups[0]['lr'] = lr
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        current_iter = state_iter + 1

    else:
        counter = 0
        flag = 0
        flag_jac = 0
        flag_iter = 0
        state_iter = 0
        current_iter = state_iter + 1

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    scaler = GradScaler()

    while current_iter <= iters:
        update = False

        loss_epoch = 0.0

        for X_cuda, Y_cuda in tqdm(trainset_loader, desc='Stage1:'):
            """
            X :fixed
            Y: atlas
            """

            with autocast():
                warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f, mid_fea, fea_1, fea_2, fea_3, fea_4 = model(
                    X_cuda.cuda(),
                    Y_cuda.cuda())

                loss_s_4 = loss_sim(warped_x_4, y4)
                loss_r_4 = Grad_loss(f1 * range_flow)
                loss_s_2 = loss_sim(warped_x_2, y2)
                loss_r_2 = Grad_loss(f2 * range_flow)
                loss_s_1 = loss_sim(warped_x_1, y_f)
                loss_r_1 = Grad_loss(f3 * range_flow)

                loss = 0.25 * (loss_s_4 + loss_r_4 * reg_smooth) + 0.5 * (loss_s_2 + loss_r_2 * reg_smooth) + (
                        loss_s_1 + loss_r_1 * reg_smooth)

            opt.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if current_iter % 50 == 0:
                print_memory_usage()

            if current_iter % validation_iters == 0 or current_iter - state_iter == 1:
                acc, val_time, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_det_neg_per, atlas_label_slice, pred_label_slice = \
                    validation1(range_flow, grid, atlases=atlases, atlases_label=atlases_label, valsets=valsets,
                                valsets_label=valsets_label, atlas_show=atlases[0], val_show=valsets[0],
                                model=model, labels=labels, slice=56)

                writer.add_scalars('dice score forward', {'dice_score': acc}, current_iter)

                fig = show(atlas_slice, volume_slice, pred_slice, volume_label_slice, atlas_label_slice,
                           pred_label_slice)

                flow_per = range_flow * f3.permute(0, 2, 3, 4, 1).squeeze(0)
                flow_per = flow_per.detach().cpu().numpy()
                flow_slice = np.stack([flow_per[:, 56, :, 0], flow_per[:, 56, :, 1]], axis=-1)
                print(flow_slice.shape)

                writer.add_figure('Validation', fig, current_iter)

                writer.add_scalars('jac_det negative percent', {'percent': jac_det_neg_per}, current_iter)
                writer.close()

                print(f"Iter:{current_iter}th. Present LR:{opt.state_dict()['param_groups'][0]['lr']}.")

                if flag < acc:
                    is_best = True
                    update = 'True'
                    save_checkpoint({'iter_th': current_iter, 'loss': loss_epoch,
                                     'state_dict': model.state_dict(), 'best_acc': acc,
                                     'optimizer': opt.state_dict(), },
                                    is_best, checkpoint_path)
                    flag = acc
                    flag_jac = jac_det_neg_per
                    flag_iter = current_iter

                print(''.center(80, '='), flush=True)
                print("\t\titers: {}".format(current_iter), flush=True)
                print("\t\tLoss: {}".format(current_iter), flush=True)
                print("\t\tAccuracy (Dice score): {}.".format(acc), flush=True)
                print("\t\tValidation time spend: {:.2f}s".format(val_time), flush=True)
                print(''.center(80, '='), flush=True)

                if not os.path.exists(checkpoint_path + f'{args.reg_smooth}' + '_log.csv'):
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = ['iter_th', 'LR', 'per_epoch_time', 'loss', 'validation', 'update']
                        csv_write.writerow(row)
                else:
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = [current_iter, opt.state_dict()['param_groups'][0]['lr'], loss_epoch,
                               acc, update]
                        csv_write.writerow(row)

            if current_iter % validation_iters == 0 and current_iter >= 1:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename=f'{current_iter}_checkpoint.pth.tar')
            current_iter += 1
            if current_iter > iters:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename='final_result.pth.tar')
                torch.save(model, checkpoint_path + '/model_with_params.pth')

                params_dict = {
                    'model name': model_name,
                    'iters': flag_iter,
                    'lr': lr, 'reg_smooth': reg_smooth,
                    'range_flow': range_flow, 'best dice': flag,
                    'nonpositive Jacobian determinant percent': flag_jac,

                }

                with open(f'{log_folder}/describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                with open(f'{checkpoint_path}describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                break


def Train_2(iters, lr, train_path, label_path, atlases_names, valsets_names, labels, resume, json_file, checkpoint_path,
            checkpoint_file, range_flow,
            log_folder, reg_smooth, validation_iters):
    checkpoint_path = checkpoint_path + 'model_2/'
    log_folder = log_folder + 'model_2/'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    print(f"LOG_FLODER : {log_folder}")
    print(f"CHECKPOINT FOLDER : {checkpoint_path}")
    with open(json_file, 'r') as file:
        data_dict = json.load(file)

    atlases, atlases_label, valsets, valsets_label, train_names = train_val_data_load(train_path, label_path,
                                                                                      atlases_names, valsets_names,
                                                                                      json_file)

    # atlas for validation
    # atlases = atlas_file
    # atlases_label = atlas_label
    print(f'Atlases :\n {atlases}')

    # valsets&label
    # valsets = valsets
    # valsets_label = valsets_label
    print(f'Validation :\n {valsets}')

    # trainset
    train_names = sorted(data_dict['train'])[0:200]

    # tensorboardX
    writer = SummaryWriter(log_folder)

    # grid generator
    grid = generate_grid(list(image_size))
    print('grid.shape='.format(grid.shape))

    # grid = torch.Tensor(grid).cuda()
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()
    # grid_patches,_=make_patches(grid)
    print(grid.shape)
    # init model
    # print("Train model : {}".format(Net))

    model1 = DualLLama(in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()

    model1.load_state_dict(torch.load(
        f"/home/mamingrui/PycharmProjects/RegLLaMA/MultiScaleLLaMA/Weights/Checkpoint/WithPosEmbed_cascade_more_new_mixed_freezepre_unfreeze_adas_jointly_change_smooth/Smooth_{reg_smooth}/range_1.0/model_1/model_best.pth.tar")[
                               'state_dict'])

    model1 = freeze_params(model1)

    model = SecDecoder(model1=model1, in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()

    transform = SpatialTransform([160, 160, 96]).cuda()  # grid sample in my code.
    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True

    # model=SYMNet().cuda()
    train_set = DataGenerator(train_names, train_path, 1)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)

    print("----------------Train phase----------------.")

    # checkpoint_file="/home/mamingrui/PycharmProjects/formal_version/proposed_64_5_cycle/att_checkpoint/lsNCClr0.0005/verdense/s1_1.0/range_8.0_id0.5/proposed_64_5_decon.Net/18000_checkpoint.pth.tar"
    # check_point = torch.load(checkpoint_file, map_location='cpu')
    # state_iter = check_point['epoch']

    # print(f'Training restart at : {state_iter}th epoch.', flush=True)
    # model.load_state_dict(check_point['state_dict'])

    loss_sim = torch.nn.MSELoss

    Grad_loss = losses.smoothloss

    opt = Adam(model.parameters(), lr=lr)

    if resume:
        counter = 0
        flag = 0.0
        check_point = torch.load(checkpoint_file, map_location='cpu')
        state_iter = check_point['epoch']

        best_acc1 = check_point['best_acc']

        print(f'Training restart at : {state_iter}th epoch.', flush=True)

        model.load_state_dict(check_point['state_dict'])
        opt.load_state_dict(check_point['optimizer'])
        opt.param_groups[0]['lr'] = lr
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        current_iter = state_iter + 1

    else:
        counter = 0
        flag = 0
        flag_jac = 0
        flag_iter = 0
        state_iter = 0
        current_iter = state_iter + 1

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    acc = 0.0

    # train
    loss_iters = 0.0
    tmp_a = 0.0
    tmp_b = 0.0
    tmp_c = 0.0
    tmp_d = 0.0

    scaler = GradScaler()

    while current_iter <= iters:
        update = False

        loss_epoch = 0.0

        for X_cuda, Y_cuda in tqdm(trainset_loader, desc='Stage2:'):
            """
            X :fixed
            Y: atlas
            """
            # X_cuda = X.unsqueeze(1).cuda()
            # Y_cuda = Y.unsqueeze(1).cuda()
            with autocast():
                warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f, mid_fea, fea_1, fea_2, fea_3, fea_4 = model(
                    X_cuda.cuda(),
                    Y_cuda.cuda())
                # pred = transform(X_cuda, flow* range_flow)

                loss_s_4 = loss_sim(warped_x_4, y4)
                loss_r_4 = Grad_loss(f1 * range_flow)
                loss_s_2 = loss_sim(warped_x_2, y2)
                loss_r_2 = Grad_loss(f2 * range_flow)
                loss_s_1 = loss_sim(warped_x_1, y_f)
                loss_r_1 = Grad_loss(f3 * range_flow)

                loss = 0.25 * (loss_s_4 + loss_r_4 * reg_smooth) + 0.5 * (loss_s_2 + loss_r_2 * reg_smooth) + (
                        loss_s_1 + loss_r_1 * reg_smooth)

            opt.zero_grad()
            # loss.backward()
            # opt.step()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if current_iter % validation_iters == 0 or current_iter - state_iter == 1:

                acc, val_time, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_det_neg_per, atlas_label_slice, pred_label_slice = \
                    validation1(range_flow, grid, atlases=atlases, atlases_label=atlases_label, valsets=valsets,
                                valsets_label=valsets_label, atlas_show=atlases[0], val_show=valsets[0],
                                model=model, labels=labels, slice=56)

                writer.add_scalars('dice score forward', {'dice_score': acc}, current_iter)

                fig = show(atlas_slice, volume_slice, pred_slice, volume_label_slice, atlas_label_slice,
                           pred_label_slice)

                flow_per = range_flow * f3.permute(0, 2, 3, 4, 1).squeeze(0)
                flow_per = flow_per.detach().cpu().numpy()
                # flow_slice = np.stack([flow_per[:, 56, :, 0], flow_per[:, 56, :, 1]], axis=-1)
                # print(flow_slice.shape)
                # flow_grid = grid2contour(flow_slice)
                #
                writer.add_figure('Validation', fig, current_iter)
                # writer.add_figure('grid', flow_grid, current_iter)

                writer.add_scalars('jac_det negative percent', {'percent': jac_det_neg_per}, current_iter)
                writer.close()

                # scheduler.step(acc)
                print(f"Iter:{current_iter}th. Present LR:{opt.state_dict()['param_groups'][0]['lr']}.")

                if flag < acc:
                    is_best = True
                    update = 'True'
                    save_checkpoint({'iter_th': current_iter, 'loss': loss_epoch,
                                     'state_dict': model.state_dict(), 'best_acc': acc,
                                     'optimizer': opt.state_dict(), },
                                    is_best, checkpoint_path)
                    flag = acc
                    flag_jac = jac_det_neg_per
                    flag_iter = current_iter

                print(''.center(80, '='), flush=True)
                print("\t\titers: {}".format(current_iter), flush=True)
                print("\t\tLoss: {}".format(current_iter), flush=True)
                print("\t\tAccuracy (Dice score): {}.".format(acc), flush=True)
                print("\t\tValidation time spend: {:.2f}s".format(val_time), flush=True)
                print(''.center(80, '='), flush=True)

                if not os.path.exists(checkpoint_path + f'{args.reg_smooth}' + '_log.csv'):
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = ['iter_th', 'LR', 'per_epoch_time', 'loss', 'validation', 'update']
                        csv_write.writerow(row)
                else:
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = [current_iter, opt.state_dict()['param_groups'][0]['lr'], loss_epoch,
                               acc, update]
                        csv_write.writerow(row)

                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename=f'{current_iter}_checkpoint.pth.tar')
            current_iter += 1
            if current_iter > iters:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename='final_result.pth.tar')
                torch.save(model, checkpoint_path + '/model_with_params.pth')

                params_dict = {
                    'model name': model_name,
                    'iters': flag_iter,
                    'lr': lr, 'reg_smooth': reg_smooth,
                    'range_flow': range_flow, 'best dice': flag,
                    'nonpositive Jacobian determinant percent': flag_jac,

                }

                with open(f'{log_folder}/describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                with open(f'{checkpoint_path}describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                break


def Train_3(iters, lr, train_path, label_path, atlases_names, valsets_names, labels, resume, json_file, checkpoint_path,
            checkpoint_file, range_flow,
            log_folder, reg_smooth, validation_iters):  # start here

    checkpoint_path = checkpoint_path + 'model_3/'
    log_folder = log_folder + 'model_3/'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    print(f"LOG_FLODER : {log_folder}")
    print(f"CHECKPOINT FOLDER : {checkpoint_path}")
    with open(json_file, 'r') as file:
        data_dict = json.load(file)

    atlases, atlases_label, valsets, valsets_label, train_names = train_val_data_load(train_path, label_path,
                                                                                      atlases_names, valsets_names,
                                                                                      json_file)
    # trainset

    # atlas for validation
    # atlases = atlas_file
    # atlases_label = atlas_label
    print(f'Atlases :\n {atlases}')

    # valsets&label
    # valsets = valsets
    # valsets_label = valsets_label
    print(f'Validation :\n {valsets}')

    # trainset
    train_names = sorted(data_dict['train'])[0:200]

    # tensorboardX
    writer = SummaryWriter(log_folder)

    # grid generator
    grid = generate_grid(list(image_size))
    print('grid.shape='.format(grid.shape))

    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    print(grid.shape)
    # init model
    # print("Train model : {}".format(Net))

    model1 = DualLLama(in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()
    model1.load_state_dict(torch.load(
        f"/home/mamingrui/PycharmProjects/RegLLaMA/MultiScaleLLaMA/Weights/Checkpoint/WithPosEmbed_cascade_more_new_mixed_freezepre_unfreeze_adas_jointly_change_smooth/Smooth_{reg_smooth}/range_1.0/model_1/model_best.pth.tar")[
                               'state_dict'])
    for param in model1.parameters():
        param.requires_grad = False

    model2 = SecDecoder(model1=model1, in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()
    model2.load_state_dict(torch.load(
        f"/home/mamingrui/PycharmProjects/RegLLaMA/MultiScaleLLaMA/Weights/Checkpoint/WithPosEmbed_cascade_more_new_mixed_freezepre_unfreeze_adas/Smooth_{reg_smooth}/range_1.0/model_2/132000_checkpoint.pth.tar")[
                               'state_dict'])
    for param in model2.parameters():
        param.requires_grad = False

    model = OtherDecoder(model=model2, in_channel=16, llama_model=llama, imgshape=(160, 160, 96)).cuda()

    transform = SpatialTransform([160, 160, 96]).cuda()  # grid sample in my code.

    train_set = DataGenerator(train_names, train_path, 1)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)

    print("----------------Train phase----------------.")

    # checkpoint_file="/home/mamingrui/PycharmProjects/formal_version/proposed_64_5_cycle/att_checkpoint/lsNCClr0.0005/verdense/s1_1.0/range_8.0_id0.5/proposed_64_5_decon.Net/18000_checkpoint.pth.tar"
    # check_point = torch.load(checkpoint_file, map_location='cpu')
    # state_iter = check_point['epoch']

    # print(f'Training restart at : {state_iter}th epoch.', flush=True)
    # model.load_state_dict(check_point['state_dict'])

    loss_sim = torch.nn.MSELoss

    Grad_loss = losses.smoothloss

    opt = Adam(model.parameters(), lr=lr)

    if resume:
        counter = 0
        flag = 0.0
        check_point = torch.load(checkpoint_file, map_location='cpu')
        state_iter = check_point['epoch']

        best_acc1 = check_point['best_acc']

        print(f'Training restart at : {state_iter}th epoch.', flush=True)

        model.load_state_dict(check_point['state_dict'])
        opt.load_state_dict(check_point['optimizer'])
        opt.param_groups[0]['lr'] = lr
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        current_iter = state_iter + 1

    else:
        counter = 0
        flag = 0
        flag_jac = 0
        flag_iter = 0
        state_iter = 0
        current_iter = state_iter + 1

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    scaler = GradScaler()

    while current_iter <= iters:
        update = False

        loss_epoch = 0.0

        for X_cuda, Y_cuda in tqdm(trainset_loader, desc='Stage3:'):
            """
            X :fixed
            Y: atlas
            """
            # X_cuda = X.unsqueeze(1).cuda()
            # Y_cuda = Y.unsqueeze(1).cuda()
            with autocast():
                warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f, mid_fea, fea_1, fea_2, fea_3, fea_4 = model(
                    X_cuda.cuda(),
                    Y_cuda.cuda())
                # pred = transform(X_cuda, flow* range_flow)

                loss_s_4 = loss_sim(warped_x_4, y4)
                loss_r_4 = Grad_loss(f1 * range_flow)
                loss_s_2 = loss_sim(warped_x_2, y2)
                loss_r_2 = Grad_loss(f2 * range_flow)
                loss_s_1 = loss_sim(warped_x_1, y_f)
                loss_r_1 = Grad_loss(f3 * range_flow)

                loss = 0.25 * (loss_s_4 + loss_r_4 * reg_smooth) + 0.5 * (loss_s_2 + loss_r_2 * reg_smooth) + (
                        loss_s_1 + loss_r_1 * reg_smooth)

            opt.zero_grad()
            # loss.backward()
            # opt.step()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if current_iter % validation_iters == 0 and current_iter >= 1:
                acc, val_time, atlas_slice, volume_slice, pred_slice, volume_label_slice, jac_det_neg_per, atlas_label_slice, pred_label_slice = \
                    validation1(range_flow, grid, atlases=atlases, atlases_label=atlases_label, valsets=valsets,
                                valsets_label=valsets_label, atlas_show=atlases[0], val_show=valsets[0],
                                model=model, labels=labels, slice=56)

                writer.add_scalars('dice score forward', {'dice_score': acc}, current_iter)

                fig = show(atlas_slice, volume_slice, pred_slice, volume_label_slice, atlas_label_slice,
                           pred_label_slice)

                flow_per = range_flow * f3.permute(0, 2, 3, 4, 1).squeeze(0)
                flow_per = flow_per.detach().cpu().numpy()
                flow_slice = np.stack([flow_per[:, 56, :, 0], flow_per[:, 56, :, 1]], axis=-1)
                print(flow_slice.shape)
                # flow_grid = grid2contour(flow_slice)

                writer.add_figure('Validation', fig, current_iter)
                # writer.add_figure('grid', flow_grid, current_iter)

                writer.add_scalars('jac_det negative percent', {'percent': jac_det_neg_per}, current_iter)
                writer.close()

                # scheduler.step(acc)
                print(f"Iter:{current_iter}th. Present LR:{opt.state_dict()['param_groups'][0]['lr']}.")

                if flag < acc:
                    is_best = True
                    update = 'True'
                    save_checkpoint({'iter_th': current_iter, 'loss': loss_epoch,
                                     'state_dict': model.state_dict(), 'best_acc': acc,
                                     'optimizer': opt.state_dict(), },
                                    is_best, checkpoint_path)
                    flag = acc
                    flag_jac = jac_det_neg_per
                    flag_iter = current_iter

                print(''.center(80, '='), flush=True)
                print("\t\titers: {}".format(current_iter), flush=True)
                print("\t\tLoss: {}".format(current_iter), flush=True)
                print("\t\tAccuracy (Dice score): {}.".format(acc), flush=True)
                print("\t\tValidation time spend: {:.2f}s".format(val_time), flush=True)
                print(''.center(80, '='), flush=True)

                if not os.path.exists(checkpoint_path + f'{args.reg_smooth}' + '_log.csv'):
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = ['iter_th', 'LR', 'per_epoch_time', 'loss', 'validation', 'update']
                        csv_write.writerow(row)
                else:
                    with open(checkpoint_path + f'{args.reg_smooth}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = [current_iter, opt.state_dict()['param_groups'][0]['lr'], loss_epoch,
                               acc, update]
                        csv_write.writerow(row)
            if current_iter % 100 == 0:
                print_memory_usage()
            # if current_iter % validation_iters == 0 and current_iter >= 1:
            #     save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
            #                      'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
            #                     checkpoint_path=checkpoint_path, filename=f'{current_iter}_checkpoint.pth.tar')
            current_iter += 1
            if current_iter > iters:
                save_checkpoint({'epoch': current_iter, 'loss': loss_epoch, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename='final_result.pth.tar')
                torch.save(model, checkpoint_path + '/model_with_params.pth')

                params_dict = {
                    'model name': model_name,
                    'iters': flag_iter,
                    'lr': lr, 'reg_smooth': reg_smooth,
                    'range_flow': range_flow, 'best dice': flag,
                    'nonpositive Jacobian determinant percent': flag_jac,
                }

                with open(f'{log_folder}/describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                with open(f'{checkpoint_path}describe.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in params_dict.items():
                        writer.writerow(row)

                break


if __name__ == '__main__':

    labels = [1.0, 2.0, 3.0, 4.0, 5.0]
    iters = args.iters
    lr = args.lr
    train_path = args.train_path
    label_path=args.lable_path
    atlases_names=args.atlas_names
    valsets_names=args.valsets_names
    resume = args.resume
    json_file = args.json_file
    checkpoint_file = args.checkpoint_file
    range_flow = args.range_flow
    log_folder = args.log_folder
    reg_smooth = args.reg_smooth
    validation_iters = args.VI

    model_name = str(DualLLama)
    model_name = model_name.split("'")
    model_name = model_name[1]

    image_size = (160, 160, 96)
    # Shanghai time zone.
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    time = beijing_now.strftime('%Y_%m_%d_%H_%M')

    checkpoint_path = args.checkpoint_path + f'Smooth_{reg_smooth}/' + f'range_{range_flow}/'
    log_folder = log_folder + f'{time}/' + f'Smooth_{reg_smooth}' + f'range_{range_flow}'

    llama_default_config = {"dim": 4096, "multiple_of": 256,
                            "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-6,
                            "vocab_size": -1, "first_layer": 31}
    llama = LLaMATransformer(llama_default_config)
    checkpoints = '/home/mamingrui/PycharmProjects/RegLLaMA/LLaMA/llama-2-7b/consolidated.00.pth'
    checkpoint = torch.load(checkpoints, map_location="cpu")
    llama.custom_load_state_dict(checkpoint)
    llama.cuda()

    for param in llama.parameters():
        param.requires_grad = False

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Now, this experiment's parameters [Learn Rate = {lr}]")

    with open(json_file, 'r') as file:
        data_dict = json.load(file)

    Train_1(iters=iters, lr=lr, train_path=train_path, label_path=label_path,
            atlases_names=atlases_names, valsets_names=valsets_names,
            labels=labels, resume=resume, json_file=json_file, checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file, range_flow=range_flow, log_folder=log_folder,
            reg_smooth=reg_smooth, validation_iters=validation_iters)

    Train_2(iters=iters, lr=lr, train_path=train_path, label_path=label_path,
            atlases_names=atlases_names, valsets_names=valsets_names,
            labels=labels, resume=resume, json_file=json_file, checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file, range_flow=range_flow, log_folder=log_folder,
            reg_smooth=reg_smooth, validation_iters=validation_iters)

    Train_3(iters=iters, lr=lr, train_path=train_path, label_path=label_path,
            atlases_names=atlases_names, valsets_names=valsets_names,
            labels=labels, resume=resume, json_file=json_file, checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file, range_flow=range_flow, log_folder=log_folder,
            reg_smooth=reg_smooth, validation_iters=validation_iters)
