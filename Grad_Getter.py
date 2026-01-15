import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch.utils.data import DataLoader,Dataset
from Utils import Network, Data, var_bins
import yaml
import pandas as pd
import skink as skplt
import re
import scipy.stats as ss


def get_best_epoch(net_filepath, early = True, given = None):

    ckpt_list = os.listdir(net_filepath)
    ckpt_list.sort()
    ckpt_list.pop(-1)

    if not early:    
        return ckpt_list[-1]

    if early and given is not None:
        return ckpt_list[given]

    ckpt_nums = []
    for ckpt in ckpt_list:
        for char in range(len(ckpt)):
            if ckpt[char].isdigit() and ckpt[char+1].isdigit():
                ckpt_nums.append(int(ckpt[char:char+2]))
                break
            elif ckpt[char].isdigit():
                ckpt_nums.append(int(ckpt[char]))
                break

    losses = []
    for num, ckpt in enumerate(ckpt_list):
        losses.append(re.findall(r'\d+',ckpt_list[num])[2])

    if early:
        return ckpt_list[np.argmin(losses)]

    return ckpt_list[-1]

def test_loop(dataloader, model):

    # Creates array for outputting
    out = []
    labels = []
    grads = []

    for vecs, weight, label in dataloader:

            vecs.retain_grad()

            # Calculates accuracy and avg loss for outputting
            pred=model(vecs)

            grad_output = torch.ones_like(pred)
            pred.backward(grad_output, retain_graph=True)

            grads.append(max(torch.norm(vecs.grad, dim = 1).numpy()))


    return grads

def grad_over_samples(model, sample_file, vars = None):

    # Grab Signal test and train files
    test_file = h5py.File(sample_file+"test.h5")

    # Converts Files to dataset
    test_tensors = []
    if vars is None:
        for var in variables:
            test_tensors.append(torch.from_numpy(test_file["Data"][var]))
    else:
        for var in vars:
            test_tensors.append(torch.from_numpy(test_file["Data"][var]))

    test_vecs = torch.stack((test_tensors),-1)
    test_weights = torch.from_numpy(test_file["Data"]["weight"]).unsqueeze(1)
    test_labels = torch.from_numpy(np.ones_like(test_file["Data"]["label"])).unsqueeze(1)

    # Closes h5 files
    test_file.close()

    test_vecs.requires_grad_(True)

    dataloader = DataLoader(Data(test_vecs, test_weights, test_labels),
                            batch_size = learning["batch_size"],
                            shuffle = True)

    grads = test_loop(dataloader,model)

    return max(grads)


def forward_pass(model, arr, boolarr = None, weights = None):

    torch_tensors = []
    for var in arr:
        torch_tensors.append(torch.from_numpy(var))
    
    torch_vecs = torch.stack((torch_tensors),-1)
    if weights is None:
        torch_weights = torch.from_numpy(np.ones_like(arr[0], dtype = np.float32)).unsqueeze(1)
    else:
        torch_weights = torch.from_numpy(weights).unsqueeze(1)
    torch_labels = torch.from_numpy(np.ones_like(arr[0], dtype = np.float32)).unsqueeze(1)

    dataloader = DataLoader(Data(torch_vecs, torch_weights, torch_labels),
                            shuffle = False)


    outputs, labels = test_loop(dataloader, model)

    if boolarr is None:
        boolarr = np.ones_like(outputs,bool)    

    return outputs[boolarr]


    
# Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_10.12.25_3b_Data_FULL_2b_Data_FULL_128_128_0.001_10000/ckpts/"
Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_9.11.25_3b_Data_CR_2b_Data_CR_128_64_0.001_2000/ckpts/"
early = True


Net_3b_2b_name = r"$\mathcal{D}^{3b1j}_{2b2j}$"
sample_3b_CR = "samples/bbb_bkg_run2_CR/"
sample_3b_SR = "samples/bbb_bkg_run2_SR/"


config_file = open("Config.yaml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

variables = config["variables"]
learning = config["training"]["learning"]

loss_function = nn.BCELoss()


Net_3b_2b_best_epoch = get_best_epoch(Net_filepath_3b_2b_ckpts, early)
print("3b 2b best epoch: " + str(Net_3b_2b_best_epoch))


Net_3b_2b_model = torch.load(Net_filepath_3b_2b_ckpts + Net_3b_2b_best_epoch, map_location=torch.device('cpu'))

print(grad_over_samples(Net_3b_2b_model, sample_3b_CR))
print(grad_over_samples(Net_3b_2b_model, sample_3b_SR))
