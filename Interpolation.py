import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch.utils.data import DataLoader,Dataset
from Utils import test_loop, Network, Data, var_bins
import yaml
import pandas as pd
import skink as skplt
import re
import scipy.stats as ss

def X_HH(mh1, mh2):

    R = ((mh1-124)/(0.1*mh1))**2 + ((mh2-117)/(0.1*mh2))**2

    return np.sqrt(R)

def R_CR(mh1, mh2):

    R = (mh1 - 1.05*124)**2 + (mh2 - 1.05*117)**2

    return np.sqrt(R)

def discriminant(arr: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return np.log((arr+eps)/(1-arr+eps))

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



def run_over_samples(model, sample_file, vars = None, boolarr = None, sigmoid = False):

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

    dataloader = DataLoader(Data(test_vecs, test_weights, test_labels),
                            batch_size = learning["batch_size"],
                            shuffle = False)

    outputs, labels = test_loop(dataloader,model)

    if boolarr is None:
        boolarr = np.ones_like(outputs,bool)    

    if sigmoid:
        outputs[boolarr] = 1/(1 + np.exp(-outputs[boolarr]))

    return outputs[boolarr], test_weights[boolarr]


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

# # 1st attempt
# Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_5.11.25_3b_Data_CR_2b_Data_CR_128_128_0.003_5000/ckpts/"
# Net_filepath_4b_3b_ckpts = "trained_networks/4b_3b_full_5.11.25_4b_Data_CR_3b_Data_CR_128_128_0.003_5000/ckpts/"

# Net_filepath_3b_2b_mass_ckpts = "trained_networks/3b_2b_mass_5.11.25_3b_Data_CR_2b_Data_CR_128_128_0.003_5000/ckpts/"
# Net_filepath_4b_3b_mass_ckpts = "trained_networks/4b_3b_mass_5.11.25_4b_Data_CR_3b_Data_CR_128_128_0.003_5000/ckpts/"

# 2nd attempt
# Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_7.11.25_3b_Data_CR_2b_Data_CR_128_128_0.001_5000/ckpts/"
Net_filepath_4b_3b_ckpts = "trained_networks/4b_3b_full_7.11.25_4b_Data_CR_3b_Data_CR_128_128_0.03_5000/ckpts/"

# Net_filepath_3b_2b_mass_ckpts = "trained_networks/3b_2b_mass_7.11.25_3b_Data_CR_2b_Data_CR_32_32_0.001_1000/ckpts/"
# Net_filepath_4b_3b_mass_ckpts = "trained_networks/4b_3b_mass_7.11.25_4b_Data_CR_3b_Data_CR_32_32_0.001_1000/ckpts/"

# # 3rd attempt
Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_9.11.25_3b_Data_CR_2b_Data_CR_64_64_0.001_2000/ckpts/"
Net_filepath_3b_2b_ckpts = "trained_networks/3b_2b_full_9.11.25_3b_Data_CR_2b_Data_CR_128_64_0.001_2000/ckpts/"
Net_filepath_3b_2b_ckpts = "trained_networks/Lipschitz_5_ord_1_01.12.25_3b_Data_CR_2b_Data_CR_128_128_128_0.003_8000/ckpts/"
# Net_filepath_4b_3b_ckpts = "trained_networks/4b_3b_full_9.11.25_4b_Data_CR_3b_Data_CR_128_128_0.003_10000/ckpts/"

Net_filepath_3b_2b_mass_ckpts = "trained_networks/3b_2b_mass_7.11.25_3b_Data_CR_2b_Data_CR_32_32_0.001_1000/ckpts/"
Net_filepath_4b_3b_mass_ckpts = "trained_networks/4b_3b_mass_7.11.25_4b_Data_CR_3b_Data_CR_32_32_0.001_1000/ckpts/"

# Lipschitz 3b_2b networks
# 1,5,10,100
lip_const = "5"
early = False
Net_filepath_3b_2b_ckpts = "trained_networks/Lipschitz_"+lip_const+"_ord_1_22.11.25_3b_Data_CR_2b_Data_CR_128_128_0.001_5000/ckpts/"


Net_3b_2b_name = r"$\mathcal{D}^{3b1j}_{2b2j}$"
Net_4b_3b_name = r"$\mathcal{D}^{4b}_{3b1j}$"

joint_3b_2b_disc = r"$\mathcal{D}^{3b1j}_{2b2j}(\vec{x},m_{H1},m_{H2})$"
mass_3b_2b_disc = r"$\mathcal{D}^{3b1j}_{2b2j}(m_{H1},m_{H2})$"
conditional_3b_2b_disc = r"$\mathcal{D}^{3b1j}_{2b2j}(\vec{x}\|m_{H1},m_{H2})$"

joint_4b_3b_disc = r"$\mathcal{D}^{4b}_{3b1j}(\vec{x},m_{H1},m_{H2})$"
mass_4b_3b_disc = r"$\mathcal{D}^{4b}_{3b1j}(m_{H1},m_{H2})$"
conditional_4b_3b_disc = r"$\mathcal{D}^{4b}_{3b1j}(\vec{x}\|m_{H1},m_{H2})$"

sample_3b_CR = "samples/bbb_bkg_run2_CR/"
sample_3b_SR = "samples/bbb_bkg_run2_SR/"

sample_2b_CR = "samples/bb_bkg_run2_CR/"
sample_2b_SR = "samples/bb_bkg_run2_SR/"

sample_4b_CR = "samples/bbbb_bkg_run2_CR/"

config_file = open("Config.yaml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

variables = config["variables"]
learning = config["training"]["learning"]

loss_function = nn.BCELoss()

#get best epoch
Net_3b_2b_best_epoch = get_best_epoch(Net_filepath_3b_2b_ckpts, early)
print("3b 2b best epoch: " + str(Net_3b_2b_best_epoch))
Net_3b_2b_mass_best_epoch = get_best_epoch(Net_filepath_3b_2b_mass_ckpts, True)
print("3b 2b (mass) best epoch: " + str(Net_3b_2b_mass_best_epoch))

Net_4b_3b_best_epoch = get_best_epoch(Net_filepath_4b_3b_ckpts, True)
print("4b 3b best epoch: " + str(Net_4b_3b_best_epoch))
Net_4b_3b_mass_best_epoch = get_best_epoch(Net_filepath_4b_3b_mass_ckpts, True)
print("4b 3b (mass) best epoch: " + str(Net_4b_3b_mass_best_epoch))

# Get mh1 and mh2
##################################################################################################################
CR_3b_mhs = []
CR_2b_mhs = []
CR_4b_mhs = []

with h5py.File(sample_3b_CR+"test.h5", "r") as file:
    CR_3b_mhs = {"mh1" : np.array(file["Data"]["m_h1"]),
                 "mh2" : np.array(file["Data"]["m_h2"])}

with h5py.File(sample_2b_CR+"test.h5", "r") as file:
    CR_2b_mhs = {"mh1" : np.array(file["Data"]["m_h1"]),
                 "mh2" : np.array(file["Data"]["m_h2"])}

with h5py.File(sample_4b_CR+"test.h5", "r") as file:
    CR_4b_mhs = {"mh1" : np.array(file["Data"]["m_h1"]),
                 "mh2" : np.array(file["Data"]["m_h2"])}


SR_3b_mhs = []
SR_2b_mhs = []

with h5py.File(sample_3b_SR+"test.h5", "r") as file:
    SR_3b_mhs = {"mh1" : np.array(file["Data"]["m_h1"]),
                 "mh2" : np.array(file["Data"]["m_h2"])}

with h5py.File(sample_2b_SR+"test.h5", "r") as file:
    SR_2b_mhs = {"mh1" : np.array(file["Data"]["m_h1"]),
                 "mh2" : np.array(file["Data"]["m_h2"])}

norm_2bCR = CR_4b_mhs["mh1"].size/(CR_2b_mhs["mh1"].size)
norm_3bCR = CR_4b_mhs["mh1"].size/(CR_3b_mhs["mh1"].size)
norm_4bCR = CR_4b_mhs["mh1"].size/(CR_4b_mhs["mh1"].size)

norm_2bSR = CR_4b_mhs["mh1"].size/(SR_2b_mhs["mh1"].size)
norm_3bSR = CR_4b_mhs["mh1"].size/(SR_3b_mhs["mh1"].size)


# Get RCR cut
##################################################################################################################

RCR = True
if RCR:
    CR_3b_RCR = R_CR(CR_3b_mhs["mh1"],CR_3b_mhs["mh2"]) < 45
    CR_2b_RCR = R_CR(CR_2b_mhs["mh1"],CR_2b_mhs["mh2"]) < 45
    CR_4b_RCR = R_CR(CR_4b_mhs["mh1"],CR_4b_mhs["mh2"]) < 45
else:
    CR_3b_RCR = np.ones_like(CR_3b_mhs["mh1"], bool)
    CR_2b_RCR = np.ones_like(CR_2b_mhs["mh1"], bool)
    CR_4b_RCR = np.ones_like(CR_4b_mhs["mh1"], bool)


# Load models
Net_3b_2b_model = torch.load(Net_filepath_3b_2b_ckpts + Net_3b_2b_best_epoch, map_location=torch.device('cpu'))
Net_4b_3b_model = torch.load(Net_filepath_4b_3b_ckpts + Net_4b_3b_best_epoch, map_location=torch.device('cpu'))

Net_3b_2b_mass_model = torch.load(Net_filepath_3b_2b_mass_ckpts + Net_3b_2b_mass_best_epoch, map_location = torch.device('cpu'))
Net_4b_3b_mass_model = torch.load(Net_filepath_4b_3b_mass_ckpts + Net_4b_3b_mass_best_epoch, map_location = torch.device('cpu'))

print(Net_3b_2b_model)

for layer in Net_3b_2b_model.operation:

    if type(layer) != type(nn.ReLU()):

        weight = layer.weight

        print("--------------")
        print(torch.linalg.matrix_norm(weight, ord = 1))
        print(torch.linalg.matrix_norm(weight, ord = 2))
        print(torch.linalg.matrix_norm(weight, ord = torch.inf))
        print("--------------")


raise("HI")
# Get network outputs
##################################################################################################################

sig = True

Net_3b_2b_CR_3b_outputs, CR_3b_weights = run_over_samples(Net_3b_2b_model, sample_3b_CR, boolarr = CR_3b_RCR, sigmoid = sig)
Net_3b_2b_CR_2b_outputs, CR_2b_weights = run_over_samples(Net_3b_2b_model, sample_2b_CR, boolarr = CR_2b_RCR, sigmoid = sig)

Net_3b_2b_SR_3b_outputs, SR_3b_weights = run_over_samples(Net_3b_2b_model, sample_3b_SR, sigmoid = sig)
Net_3b_2b_SR_2b_outputs, SR_2b_weights = run_over_samples(Net_3b_2b_model, sample_2b_SR, sigmoid = sig)

######

# Net_3b_2b_CR_3b_outputs = Sigmoid(Net_3b_2b_CR_3b_outputs)
# Net_3b_2b_CR_2b_outputs = Sigmoid(Net_3b_2b_CR_2b_outputs)

# Net_3b_2b_SR_3b_outputs = Sigmoid(Net_3b_2b_SR_3b_outputs)
# Net_3b_2b_SR_2b_outputs = Sigmoid(Net_3b_2b_SR_2b_outputs)


######

Net_4b_3b_CR_3b_outputs, CR_3b_weights = run_over_samples(Net_4b_3b_model, sample_3b_CR, boolarr = CR_3b_RCR)
Net_4b_3b_CR_4b_outputs, CR_4b_weights = run_over_samples(Net_4b_3b_model, sample_4b_CR, boolarr = CR_4b_RCR)

Net_4b_3b_SR_3b_outputs, SR_3b_weights = run_over_samples(Net_4b_3b_model, sample_3b_SR)

####################################################################################

Net_3b_2b_mass_CR_3b_outputs, CR_3b_weights = run_over_samples(Net_3b_2b_mass_model, sample_3b_CR, vars = ["m_h1","m_h2"], boolarr = CR_3b_RCR)
Net_3b_2b_mass_CR_2b_outputs, CR_2b_weights = run_over_samples(Net_3b_2b_mass_model, sample_2b_CR, vars = ["m_h1","m_h2"], boolarr = CR_2b_RCR)

Net_3b_2b_mass_SR_3b_outputs, SR_3b_weights = run_over_samples(Net_3b_2b_mass_model, sample_3b_SR, vars = ["m_h1","m_h2"])
Net_3b_2b_mass_SR_2b_outputs, SR_2b_weights = run_over_samples(Net_3b_2b_mass_model, sample_2b_SR, vars = ["m_h1","m_h2"])

Net_4b_3b_mass_CR_3b_outputs, CR_3b_weights = run_over_samples(Net_4b_3b_mass_model, sample_3b_CR, vars = ["m_h1","m_h2"], boolarr = CR_3b_RCR)
Net_4b_3b_mass_CR_4b_outputs, CR_4b_weights = run_over_samples(Net_4b_3b_mass_model, sample_4b_CR, vars = ["m_h1","m_h2"], boolarr = CR_4b_RCR)

Net_4b_3b_mass_SR_3b_outputs, SR_3b_weights = run_over_samples(Net_4b_3b_mass_model, sample_3b_SR, vars = ["m_h1","m_h2"])

# get discriminants
##################################################################################################################

Net_3b_2b_CR_3b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_CR_3b_outputs)
Net_3b_2b_CR_2b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_CR_2b_outputs)

Net_3b_2b_SR_3b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_SR_3b_outputs)
Net_3b_2b_SR_2b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_SR_2b_outputs)

Net_4b_3b_CR_3b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_CR_3b_outputs)
Net_4b_3b_CR_4b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_CR_4b_outputs)

Net_4b_3b_SR_3b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_SR_3b_outputs)

####################################################################################

Net_3b_2b_mass_CR_3b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_mass_CR_3b_outputs)
Net_3b_2b_mass_CR_2b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_mass_CR_2b_outputs)

Net_3b_2b_mass_SR_3b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_mass_SR_3b_outputs)
Net_3b_2b_mass_SR_2b_discs = np.apply_along_axis(discriminant,0,Net_3b_2b_mass_SR_2b_outputs)

Net_4b_3b_mass_CR_3b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_mass_CR_3b_outputs)
Net_4b_3b_mass_CR_4b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_mass_CR_4b_outputs)

Net_4b_3b_mass_SR_3b_discs = np.apply_along_axis(discriminant,0,Net_4b_3b_mass_SR_3b_outputs)

# get n_events in each sample
##################################################################################################################
N_3b_CR = Net_3b_2b_CR_3b_outputs.size
N_2b_CR = Net_3b_2b_CR_2b_outputs.size

N_3b_SR = Net_3b_2b_SR_3b_outputs.size
N_2b_SR = Net_3b_2b_SR_2b_outputs.size

N_4b_CR = Net_4b_3b_CR_4b_outputs.size

density = True
plot_dir_base = "plots/interpolation/"

###################################################################################################################
# MassPlane Ratio Plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "MassPlaneRatio"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

bins1 = skplt.get_bins(80,180,200)
bins2 = skplt.get_bins(75,170,200)

# bins1 = skplt.get_bins(105,150,45)
# bins2 = skplt.get_bins(100,140,40)


gridpoints = []
for m1 in bins1[1]:
    for m2 in bins2[1]:
        gridpoints.append([m1,m2])

gridpoints = np.array(gridpoints)

CR_bool = np.logical_and(R_CR(gridpoints[:,0], gridpoints[:,1]) < 45, X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.5)
SR_bool = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.5

m1m2_2bCR_hist2d = norm_2bCR * np.histogram2d(x = CR_2b_mhs["mh1"], y = CR_2b_mhs["mh2"], bins = [bins1[0], bins2[0]])[0].flatten()
m1m2_3bCR_hist2d = norm_3bCR * np.histogram2d(x = CR_3b_mhs["mh1"], y = CR_3b_mhs["mh2"], bins = [bins1[0], bins2[0]])[0].flatten()
m1m2_4bCR_hist2d = norm_4bCR * np.histogram2d(x = CR_4b_mhs["mh1"], y = CR_4b_mhs["mh2"], bins = [bins1[0], bins2[0]])[0].flatten()

m1m2_2bSR_hist2d = norm_2bSR * np.histogram2d(x = SR_2b_mhs["mh1"], y = SR_2b_mhs["mh2"], bins = [bins1[0], bins2[0]])[0].flatten()
m1m2_3bSR_hist2d = norm_3bSR * np.histogram2d(x = SR_3b_mhs["mh1"], y = SR_3b_mhs["mh2"], bins = [bins1[0], bins2[0]])[0].flatten()

m1m2_ratio_2b3bCR = skplt.CalcNProp("/", [m1m2_3bCR_hist2d, np.sqrt(norm_3bCR)*np.sqrt(m1m2_3bCR_hist2d)],
                                         [m1m2_2bCR_hist2d, np.sqrt(norm_2bCR)*np.sqrt(m1m2_2bCR_hist2d)])
m1m2_ratio_3b4bCR = skplt.CalcNProp("/", [m1m2_4bCR_hist2d, np.sqrt(norm_4bCR)*np.sqrt(m1m2_4bCR_hist2d)],
                                         [m1m2_3bCR_hist2d, np.sqrt(norm_3bCR)*np.sqrt(m1m2_3bCR_hist2d)])

m1m2_ratio_2b3bSR = skplt.CalcNProp("/", [m1m2_3bSR_hist2d, np.sqrt(m1m2_3bSR_hist2d)],
                                         [m1m2_2bSR_hist2d, np.sqrt(m1m2_2bSR_hist2d)])

#m1,m2 for all nbtag hists

plot_dir = plot_dir_temp + "/ALL"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

histplot = skplt.HistogramPlot(bins1, xlabel = r"$m_{H1}$", ylabel = "number of events", ratio = True, density = True)
histplot.Add(CR_2b_mhs["mh1"], label = "2b2j CR", reference = True)
histplot.Add(CR_3b_mhs["mh1"], label = "3b21j CR")
histplot.Add(CR_4b_mhs["mh1"], label = "4b CR")

histplot.Plot(plot_dir+"/mh1CR")

histplot = skplt.HistogramPlot(bins2, xlabel = r"$m_{H2}$", ylabel = "number of events", ratio = True, density = True)
histplot.Add(CR_2b_mhs["mh2"], label = "2b2j CR", reference = True)
histplot.Add(CR_3b_mhs["mh2"], label = "3b21j CR")
histplot.Add(CR_4b_mhs["mh2"], label = "4b CR")

histplot.Plot(plot_dir+"/mh2CR")

histplot = skplt.HistogramPlot(bins1, xlabel = r"$m_{H1}$", ylabel = "density of number of events", ratio = True, density = True)
histplot.Add(SR_2b_mhs["mh1"], label = "2b2j SR", reference = True)
histplot.Add(SR_3b_mhs["mh1"], label = "3b21j SR")

histplot.Plot(plot_dir+"/mh1SR")

histplot = skplt.HistogramPlot(bins2, xlabel = r"$m_{H2}$", ylabel = "density of number of events", ratio = True, density = True)
histplot.Add(SR_2b_mhs["mh2"], label = "2b2j SR", reference = True)
histplot.Add(SR_3b_mhs["mh2"], label = "3b21j SR")

histplot.Plot(plot_dir+"/mh2SR")


#CR 2b 3b
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0], gridpoints[:,1], weights = m1m2_ratio_2b3bCR[0], zlabel = "p(3b/2b)", cmin = 0.01, cmax = 2)
massplot2D.Plot(plot_dir + "/MassPlane")

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0], gridpoints[:,1], weights = (m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1], zlabel = "p(3b/2b) consistencey",
               cmin = None, cmax = None)
massplot2D.Plot(plot_dir + "/Consistency")

mean_2b_3b = np.mean(np.array((m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1])[np.isnan(((m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1]).flatten()) == False])
std_2b_3b = np.std(np.array((m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1])[np.isnan(((m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1]).flatten()) == False])

histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = "pull (3b/2b)", ylabel = "No. Bins", density = True)
histplot.Add((m1m2_ratio_2b3bCR[0]-1)/m1m2_ratio_2b3bCR[1], label = "2b/3b")
plt.plot(skplt.get_bins(-4,4, 30)[1], ss.norm(loc = mean_2b_3b, scale = std_2b_3b).pdf(skplt.get_bins(-4,4, 30)[1]))
histplot.Text(r"$\mu$ = "+str(round(mean_2b_3b, 2)), xpos = 0.73, ypos = 0.78)
histplot.Text(r"$\sigma$ = "+str(round(std_2b_3b, 2)), xpos = 0.73, ypos = 0.73)
histplot.Plot(plot_dir + "/Pulls")

#CR 3b 4b
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0], gridpoints[:,1], weights = m1m2_ratio_3b4bCR[0], zlabel = "p(4b/3b)", cmin = 0.01, cmax = 2)
massplot2D.Plot(plot_dir + "/MassPlane")

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0], gridpoints[:,1], weights = (m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1], zlabel = "p(4b/3b) consistencey",
               cmin = -10, cmax = None)
massplot2D.Plot(plot_dir + "/Consistency")

mean_3b_4b = np.mean(np.array((m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1])[np.isnan(((m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1]).flatten()) == False])
std_3b_4b = np.std(np.array((m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1])[np.isnan(((m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1]).flatten()) == False])

histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = "pull (4b/3b)", ylabel = "No. Bins", density = True)
histplot.Add((m1m2_ratio_3b4bCR[0]-1)/m1m2_ratio_3b4bCR[1], label = "3b/4b")
plt.plot(skplt.get_bins(-4,4, 30)[1], ss.norm(loc = mean_3b_4b, scale = std_3b_4b).pdf(skplt.get_bins(-4,4, 30)[1]))
histplot.Text(r"$\mu$ = "+str(round(mean_3b_4b, 2)), xpos = 0.73, ypos = 0.78)
histplot.Text(r"$\sigma$ = "+str(round(std_3b_4b, 2)), xpos = 0.73, ypos = 0.73)
histplot.Plot(plot_dir + "/Pulls")

###################################################################################################################
# MassPlane Interpolation Plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "MassPlane"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

forward_inputs = np.array([gridpoints[:,0], gridpoints[:,1]], dtype = np.float32)

# 3b2b 
gridscan_3b_2b = forward_pass(Net_3b_2b_mass_model, forward_inputs)

plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0][CR_bool], gridpoints[:,1][CR_bool], weights = gridscan_3b_2b[CR_bool], cmin = 0.01, cmax = 1, zlabel = "P(3b1j)")
massplot2D.Plot(plot_dir + "/MassPlane")


plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0][SR_bool], gridpoints[:,1][SR_bool], weights = gridscan_3b_2b[SR_bool], cmin = 0.01, cmax = 1, zlabel = "P(3b1j)")
massplot2D.Plot(plot_dir + "/MassPlane")


# 4b3b 
gridscan_4b_3b = forward_pass(Net_4b_3b_mass_model, forward_inputs)

plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

massplot2D = skplt.Hist2D(bins1,bins2, margins = False, xlabel = "m1", ylabel = "m2", cbar = True)
massplot2D.Set(gridpoints[:,0][CR_bool], gridpoints[:,1][CR_bool], weights = gridscan_4b_3b[CR_bool], cmin = 0.01, cmax = 1, zlabel = "P(4b)")
massplot2D.Plot(plot_dir + "/MassPlane")

###################################################################################################################
# Bin Selections
###################################################################################################################

output_bins = skplt.get_bins(0,1,40)
disc_bins = skplt.get_bins(-5,5,40)

###################################################################################################################
# Output plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "Outputs"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

##################################################################################################################

plot_dir_temp = plot_dir_base + "Outputs" + "/Joint"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_3b_2b_CR_3b_outputs, label = "3b1j_CR")
output_hist.Add(Net_3b_2b_CR_2b_outputs, label = "2b2j_CR")
output_hist.Plot(plot_dir + "/output")

# SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_3b_2b_SR_3b_outputs, label = "3b1j_SR")
output_hist.Add(Net_3b_2b_SR_2b_outputs, label = "2b2j_SR")
output_hist.Plot(plot_dir + "/output")

# CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_4b_3b_CR_4b_outputs, label = "4b_CR")
output_hist.Add(Net_4b_3b_CR_3b_outputs, label = "3b1j_CR")
output_hist.Plot(plot_dir + "/output")

##################################################################################################################

plot_dir_temp = plot_dir_base + "Outputs" + "/Mass"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

bins = skplt.get_bins(0,1,40)

# CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_3b_2b_mass_CR_3b_outputs, label = "3b1j_CR")
output_hist.Add(Net_3b_2b_mass_CR_2b_outputs, label = "2b2j_CR")
output_hist.Plot(plot_dir + "/output")

# SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_3b_2b_mass_SR_3b_outputs, label = "3b1j_SR")
output_hist.Add(Net_3b_2b_mass_SR_2b_outputs, label = "2b2j_SR")
output_hist.Plot(plot_dir + "/output")

# CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

output_hist = skplt.HistogramPlot(output_bins, xlabel = "Network Output", ylabel = "No. Events", density = density)
output_hist.Add(Net_4b_3b_mass_CR_4b_outputs, label = "4b_CR")
output_hist.Add(Net_4b_3b_mass_CR_3b_outputs, label = "3b1j_CR")
output_hist.Plot(plot_dir + "/output")



###################################################################################################################
# Disc Plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "Discs"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

bins = skplt.get_bins(-5,5,40)

##################################################################################################################

plot_dir_temp = plot_dir_base + "Discs" + "/Joint"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = joint_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_CR_3b_discs, label = "3b1j_CR")
disc_hist.Add(Net_3b_2b_CR_2b_discs, label = "2b2j_CR")
disc_hist.Plot(plot_dir + "/discs")

# SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = joint_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_SR_3b_discs, label = "3b1j_SR")
disc_hist.Add(Net_3b_2b_SR_2b_discs, label = "2b2j_SR")
disc_hist.Plot(plot_dir + "/discs")

# CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = joint_4b_3b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_4b_3b_CR_4b_discs, label = "4b_CR")
disc_hist.Add(Net_4b_3b_CR_3b_discs, label = "3b1j_CR")
disc_hist.Plot(plot_dir + "/discs")

##################################################################################################################

plot_dir_temp = plot_dir_base + "Discs" + "/Mass"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = mass_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_mass_CR_3b_discs, label = "3b1j_CR")
disc_hist.Add(Net_3b_2b_mass_CR_2b_discs, label = "2b2j_CR")
disc_hist.Plot(plot_dir + "/discs")

# SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = mass_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_mass_SR_3b_discs, label = "3b1j_SR")
disc_hist.Add(Net_3b_2b_mass_SR_2b_discs, label = "2b2j_SR")
disc_hist.Plot(plot_dir + "/discs")

# CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = mass_4b_3b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_4b_3b_mass_CR_4b_discs, label = "4b_CR")
disc_hist.Add(Net_4b_3b_mass_CR_3b_discs, label = "3b1j_CR")
disc_hist.Plot(plot_dir + "/discs")

##################################################################################################################

plot_dir_temp = plot_dir_base + "Discs" + "/Conditional"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = conditional_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_CR_3b_discs - Net_3b_2b_mass_CR_3b_discs, label = "3b1j_CR")
disc_hist.Add(Net_3b_2b_CR_2b_discs - Net_3b_2b_mass_CR_2b_discs, label = "2b2j_CR")
disc_hist.Plot(plot_dir + "/discs")

# SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = conditional_3b_2b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_3b_2b_SR_3b_discs - Net_3b_2b_mass_SR_3b_discs, label = "3b1j_SR")
disc_hist.Add(Net_3b_2b_SR_2b_discs - Net_3b_2b_mass_SR_2b_discs, label = "2b2j_SR")
disc_hist.Plot(plot_dir + "/discs")

# CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

disc_hist = skplt.HistogramPlot(disc_bins, xlabel = conditional_4b_3b_disc, ylabel = "No. Events", density = density)
disc_hist.Add(Net_4b_3b_CR_4b_discs - Net_4b_3b_mass_CR_4b_discs, label = "4b_CR")
disc_hist.Add(Net_4b_3b_CR_4b_discs - Net_4b_3b_mass_CR_4b_discs, label = "3b1j_CR")
disc_hist.Plot(plot_dir + "/discs")


###################################################################################################################
# Disc Ratio Plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "Ratio"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

##################################################################################################################

plot_dir_temp = plot_dir_base + "Ratio" + "/Joint"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_CR_3b_discs, bins =disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_CR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = joint_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_CR/N_3b_CR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_CR/N_3b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_SR_3b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_SR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = joint_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_SR/N_3b_SR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_SR/N_3b_SR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")


# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_4b_3b_CR_4b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_4b_3b_CR_3b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = joint_4b_3b_disc, ylabel = "Ratio of 4b/3b1j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_3b_CR/N_4b_CR), label = "Obtained from " + Net_4b_3b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_4b_CR/N_4b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

##################################################################################################################

plot_dir_temp = plot_dir_base + "Ratio" + "/Mass"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_mass_CR_3b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_mass_CR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = mass_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_CR/N_3b_CR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_CR/N_3b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_mass_SR_3b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_mass_SR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = mass_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_SR/N_3b_SR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_SR/N_3b_SR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")


# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_4b_3b_mass_CR_4b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_4b_3b_mass_CR_3b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = mass_4b_3b_disc, ylabel = "Ratio of 4b/3b1j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_3b_CR/N_4b_CR), label = "Obtained from " + Net_4b_3b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_4b_CR/N_4b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

##################################################################################################################

plot_dir_temp = plot_dir_base + "Ratio" + "/Conditional"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_CR_3b_discs - Net_3b_2b_mass_CR_3b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_CR_2b_discs - Net_3b_2b_mass_CR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = conditional_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_CR/N_3b_CR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_CR/N_3b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_3b_2b_SR_3b_discs - Net_3b_2b_mass_SR_3b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_3b_2b_SR_2b_discs - Net_3b_2b_mass_SR_2b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = conditional_3b_2b_disc, ylabel = "Ratio of 3b1j/2b2j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_2b_SR/N_3b_SR), label = "Obtained from " + Net_3b_2b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_2b_SR/N_3b_SR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")


# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

ratio_numerator = np.histogram(Net_4b_3b_CR_4b_discs - Net_4b_3b_mass_CR_4b_discs, bins = disc_bins[0])[0]
ratio_denominator = np.histogram(Net_4b_3b_CR_3b_discs - Net_4b_3b_mass_CR_3b_discs, bins = disc_bins[0])[0]

ratio_ratio, ratio_uncs = skplt.CalcNProp(operation = "/",
                                          xs = [ratio_numerator,
                                                np.sqrt(ratio_numerator)],
                                          ys = [ratio_denominator,
                                                np.sqrt(ratio_denominator)])

LinePlot = skplt.LinePlot(xs = disc_bins[1], xlabel = conditional_4b_3b_disc, ylabel = "Ratio of 4b/3b1j", ratio = True, logy = True, plot_unc = True)

LinePlot.Add(ys = np.exp(disc_bins[1]), label = "Ideal Ratio", linecolour = "grey", linestyle = "--", marker_size = 0, reference = True, uncs = np.zeros_like(np.exp(bins[1])))
LinePlot.Add(ys = ratio_ratio * (N_3b_CR/N_4b_CR), label = "Obtained from " + Net_4b_3b_name + " in CR", marker_size = 0, uncs = ratio_uncs * (N_4b_CR/N_4b_CR))

LinePlot.Plot(plot_dir + "/disc_ratio", legend_loc = "upper left")

###################################################################################################################
# Reweight Plotting
###################################################################################################################

plot_dir_temp = plot_dir_base + "Reweight"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

weightcut = 1000

##################################################################################################################

plot_dir_temp = plot_dir_base + "Reweight" + "/Joint"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    CR_2b_file = h5py.File(sample_2b_CR+"test.h5")
    CR_2b_var = CR_2b_file["Data"][reweight_var][CR_2b_RCR]
    CR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", reference = True)
    reweighthistplot.Add(data = CR_2b_var, label = "2b2j CR", addthis = False)
    reweighthistplot.Add(data = CR_2b_var, label = "reweighted 2b2j CR", weights = np.where(np.exp(Net_3b_2b_CR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_CR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    SR_3b_file = h5py.File(sample_3b_SR+"test.h5")
    SR_3b_var = SR_3b_file["Data"][reweight_var]
    SR_3b_file.close()

    SR_2b_file = h5py.File(sample_2b_SR+"test.h5")
    SR_2b_var = SR_2b_file["Data"][reweight_var]
    SR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = SR_3b_var, label = "3b1j SR", reference = True)
    reweighthistplot.Add(data = SR_2b_var, label = "2b2j SR", addthis = False)
    reweighthistplot.Add(data = SR_2b_var, label = "reweighted 2b2j SR", weights = np.where(np.exp(Net_3b_2b_SR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_SR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_4b_file = h5py.File(sample_4b_CR+"test.h5")
    CR_4b_var = CR_4b_file["Data"][reweight_var][CR_4b_RCR]
    CR_4b_file.close()

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_4b_var, label = "4b CR", reference = True)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", addthis = False)
    reweighthistplot.Add(data = CR_3b_var, label = "reweighted 3b1j CR", weights = np.where(np.exp(Net_4b_3b_CR_3b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_4b_3b_CR_3b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

##################################################################################################################

plot_dir_temp = plot_dir_base + "Reweight" + "/Mass"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    CR_2b_file = h5py.File(sample_2b_CR+"test.h5")
    CR_2b_var = CR_2b_file["Data"][reweight_var][CR_2b_RCR]
    CR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", reference = True)
    reweighthistplot.Add(data = CR_2b_var, label = "2b2j CR", addthis = False)
    reweighthistplot.Add(data = CR_2b_var, label = "reweighted 2b2j CR", weights = np.where(np.exp(Net_3b_2b_mass_CR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_mass_CR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    SR_3b_file = h5py.File(sample_3b_SR+"test.h5")
    SR_3b_var = SR_3b_file["Data"][reweight_var]
    SR_3b_file.close()

    SR_2b_file = h5py.File(sample_2b_SR+"test.h5")
    SR_2b_var = SR_2b_file["Data"][reweight_var]
    SR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = SR_3b_var, label = "3b1j SR", reference = True)
    reweighthistplot.Add(data = SR_2b_var, label = "2b2j SR", addthis = False)
    reweighthistplot.Add(data = SR_2b_var, label = "reweighted 2b2j SR", weights = np.where(np.exp(Net_3b_2b_mass_SR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_mass_SR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_4b_file = h5py.File(sample_4b_CR+"test.h5")
    CR_4b_var = CR_4b_file["Data"][reweight_var][CR_4b_RCR]
    CR_4b_file.close()

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_4b_var, label = "4b CR", reference = True)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", addthis = False)
    reweighthistplot.Add(data = CR_3b_var, label = "reweighted 3b1j CR", weights = np.where(np.exp(Net_4b_3b_mass_CR_3b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_4b_3b_mass_CR_3b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

##################################################################################################################

plot_dir_temp = plot_dir_base + "Reweight" + "/Conditional"

if not os.path.exists(plot_dir_temp):
	os.mkdir(plot_dir_temp)

# 3b 2b CR
plot_dir = plot_dir_temp + "/CR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    CR_2b_file = h5py.File(sample_2b_CR+"test.h5")
    CR_2b_var = CR_2b_file["Data"][reweight_var][CR_2b_RCR]
    CR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", reference = True)
    reweighthistplot.Add(data = CR_2b_var, label = "2b2j CR", addthis = False)
    reweighthistplot.Add(data = CR_2b_var, label = "reweighted 2b2j CR", weights = np.where(np.exp(Net_3b_2b_CR_2b_discs.flatten() - Net_3b_2b_mass_CR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_CR_2b_discs.flatten() - Net_3b_2b_mass_CR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 3b 2b SR
plot_dir = plot_dir_temp + "/SR_3b_2b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    SR_3b_file = h5py.File(sample_3b_SR+"test.h5")
    SR_3b_var = SR_3b_file["Data"][reweight_var]
    SR_3b_file.close()

    SR_2b_file = h5py.File(sample_2b_SR+"test.h5")
    SR_2b_var = SR_2b_file["Data"][reweight_var]
    SR_2b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = SR_3b_var, label = "3b1j SR", reference = True)
    reweighthistplot.Add(data = SR_2b_var, label = "2b2j SR", addthis = False)
    reweighthistplot.Add(data = SR_2b_var, label = "reweighted 2b2j SR", weights = np.where(np.exp(Net_3b_2b_SR_2b_discs.flatten() - Net_3b_2b_mass_SR_2b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_3b_2b_SR_2b_discs.flatten() - Net_3b_2b_mass_SR_2b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)

# 4b 3b CR
plot_dir = plot_dir_temp + "/CR_4b_3b"
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)

for count, reweight_var in enumerate(variables):

    CR_4b_file = h5py.File(sample_4b_CR+"test.h5")
    CR_4b_var = CR_4b_file["Data"][reweight_var][CR_4b_RCR]
    CR_4b_file.close()

    CR_3b_file = h5py.File(sample_3b_CR+"test.h5")
    CR_3b_var = CR_3b_file["Data"][reweight_var][CR_3b_RCR]
    CR_3b_file.close()

    bins = var_bins[reweight_var]

    reweighthistplot = skplt.HistogramPlot(bins = bins, xlabel = reweight_var, ylabel = "Number of Events", residual = True, ratio = True, density = density)
    reweighthistplot.Add(data = CR_4b_var, label = "4b CR", reference = True)
    reweighthistplot.Add(data = CR_3b_var, label = "3b1j CR", addthis = False)
    reweighthistplot.Add(data = CR_3b_var, label = "reweighted 3b1j CR", weights = np.where(np.exp(Net_4b_3b_CR_3b_discs.flatten() - Net_4b_3b_mass_CR_3b_discs.flatten()) > weightcut, 0,
                                                                                            np.exp(Net_4b_3b_CR_3b_discs.flatten() - Net_4b_3b_mass_CR_3b_discs.flatten())))

    reweighthistplot.Plot(plot_dir+"/"+reweight_var)
