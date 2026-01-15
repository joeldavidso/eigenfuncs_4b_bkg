import pyarrow.parquet as pa
import pyarrow.dataset as pd
import pyarrow as py
import numpy as np
import math
import skink as skplt
from tqdm import tqdm
import os
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", DeprecationWarning)

def X_HH(mh1, mh2):

    R = ((mh1-124)/(0.1*mh1))**2 + ((mh2-117)/(0.1*mh2))**2

    return np.sqrt(R)

def R_CR(mh1, mh2):

    R = (mh1 - 1.05*124)**2 + (mh2 - 1.05*117)**2

    return np.sqrt(R)

def Gridpoints(var1, var2, bins):
    # Gridpoints for 2d mass plot
    gridpoints = []
    for v1 in bins[var1][1]:
        for v2 in bins[var2][1]:
            gridpoints.append([v1,v2])

    return np.array(gridpoints)

def loss_per_bin(NSig, NBkg, NSig_tot, NBkg_tot):

    NBkg = NBkg * NSig_tot/NBkg_tot

    return (- NSig_tot * np.log(NSig/(NSig+NBkg)) - NBkg_tot * np.log(NBkg/(NSig+NBkg)))/(NSig_tot + NBkg_tot)

def polynomial_2D(xy, a, b, c, d, e, f):
    x = xy[:,0]
    y = xy[:,1]
    # 2D polynomial of degree 2
    return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

def poly2d_fit_func(XY, *coefficients):
    """
    A 2D polynomial function f(x, y) that takes a variable number of coefficients.
    
    The terms are: c_00 + c_10*x + c_01*y + c_20*x^2 + c_11*x*y + c_02*y^2 + ...
    The number of coefficients determines the max degree.
    """
    
    x, y = XY[:,0], XY[:,1]

    # c_00 is the first coefficient (constant term)
    result = coefficients[0] 

    # if const polynomial return this 
    if len(coefficients) == 1:
        return result * np.ones_like(x)

    # loop over 
    coeff_idx = 1
    degree = 1
    while coeff_idx < len(coefficients):
        # Iterate over all possible combinations of x^i * y^j where i+j = degree
        for i in range(degree + 1):
            j = degree - i
            if coeff_idx < len(coefficients):
                # Add the term: coefficient * x^i * y^j
                result += coefficients[coeff_idx] * (x**i) * (y**j)
                coeff_idx += 1
            else:
                break
        degree += 1

    return result

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def plot_uncs_ratio(bins, ratios, region, plotdir):

    plot_dir = plotdir + "/uncs_frac"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    mean, mean_SE = skplt.get_mean(ratios)
    std, std_SE = skplt.get_std(ratios)

    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$\sigma_r/\sigma_P$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
    histplot.Add(ratios, label = r"$\sigma_r/\sigma_P$")
    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.65, ypos = 0.78)
    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.65, ypos = 0.73)
    histplot.Plot(plot_dir + "/" + region)

def plot_pulls(pulls, region, plotdir):

    plot_dir = plotdir + "/pulls_2D"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
    histplot2d.Set(gridpoints[:,0],gridpoints[:,1],
                   weights = pulls,
                   zlabel = r"$(r_{3b}^{2b} - P)/(\sqrt{\sigma_r^2 - \sigma_P^2})$")
    histplot2d.Plot(plot_dir + "/" + region)

def plot_ratios(ratios, region, plotdir):

    plot_dir = plotdir + "/ratios_2D"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
    histplot2d.Set(gridpoints[:,0],gridpoints[:,1],
                   weights = ratios,
                   zlabel = r"$N_{3b}/N_{2b}$")
    histplot2d.Plot(plot_dir + "/" + region)


def plot_pulls_1D(bins, pulls, region, plotdir, Ord, chi, n_region_bins):

    plot_dir = plotdir + "/pulls_1D"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    mean, mean_SE = skplt.get_mean(pulls)
    std, std_SE = skplt.get_std(pulls)

    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$(r_{3b}^{2b} - P)/(\sqrt{\sigma_r^2 - \sigma_P^2})$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
    histplot.Add(pulls, label = "Order-"+str(Ord))
    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.52, ypos = 0.78)
    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.52, ypos = 0.73)
    histplot.Text(r"$\chi^2/n_{df} = $"+str(round(chi))+"/"+str(n_region_bins), xpos = 0.15, ypos = 0.82)
    histplot.Plot(plot_dir + "/" + region)

def plot_diff_mh1(bins, mh_value, poly_func, ratio_CR, ratio_SR, plotdir):

    plot_dir = plotdir + "/diff_mh1"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    lineplot = skplt.LinePlot(xs = bins[1], xlabel = r"$m_{h1}$", ylabel = r"$r^{3b}_{2b}$", ratio = True, residual = True)
    lineplot.Add(poly_func[0],
                    uncs = poly_func[1], label = "Order-"+str(poly_orders[Ord]),
                    plotunc = True, marker_size = 0)
    lineplot.Add(ratio_CR[0],
                    uncs = ratio_CR[1],
                    label = "CR", linewidth = 0, addthis = False, plotunc = True, marker_size = 5, unctype = "bars")
    lineplot.Add(ratio_SR[0],
                    uncs = ratio_SR[1],
                    label = "SR", linewidth = 0, addthis = False, plotunc = True, marker_size = 5, unctype = "bars")
    lineplot.Text(r"$m_{h2}$ =" + str(mh_value), xpos = 0.15, ypos = 0.82)
    temp_ref_line = [ratio_CR[0],
                        ratio_CR[1]]
    temp_ref_line[0][np.isnan(ratio_CR[0])] = ratio_SR[0][np.isnan(ratio_CR[0])]
    temp_ref_line[1][np.isnan(ratio_CR[0])] = ratio_SR[1][np.isnan(ratio_CR[0])]
    temp_ref_line[1][np.isnan(ratio_CR[1])] = ratio_SR[1][np.isnan(ratio_CR[1])]
    lineplot.Add(temp_ref_line[0], uncs = temp_ref_line[1],label = "",
                    linewidth = 0, marker_size = 0, reference = True)
    lineplot.Plot(plot_dir + "/" + str(mh_value))

def plot_diff_mh2(bins, mh_value, poly_func, ratio_CR, ratio_SR, plotdir):

    plot_dir = plotdir + "/diff_mh2"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    lineplot = skplt.LinePlot(xs = bins[1], xlabel = r"$m_{h2}$", ylabel = r"$r^{3b}_{2b}$", ratio = True, residual = True)
    lineplot.Add(poly_func[0],
                    uncs = poly_func[1], label = "Order-"+str(poly_orders[Ord]),
                    plotunc = True, marker_size = 0)
    lineplot.Add(ratio_CR[0],
                    uncs = ratio_CR[1],
                    label = "CR", linewidth = 0, addthis = True, plotunc = True, marker_size = 5, unctype = "bars")
    lineplot.Add(ratio_SR[0],
                    uncs = ratio_SR[1],
                    label = "SR", linewidth = 0, addthis = True, plotunc = True, marker_size = 5, unctype = "bars")
    lineplot.Text(r"$m_{h1}$ =" + str(mh_value), xpos = 0.15, ypos = 0.82)
    temp_ref_line = [ratio_CR[0],
                        ratio_CR[1]]
    temp_ref_line[0][np.isnan(ratio_CR[0])] = ratio_SR[0][np.isnan(ratio_CR[0])]
    temp_ref_line[1][np.isnan(ratio_CR[0])] = ratio_SR[1][np.isnan(ratio_CR[0])]
    temp_ref_line[1][np.isnan(ratio_CR[1])] = ratio_SR[1][np.isnan(ratio_CR[1])]
    lineplot.Add(temp_ref_line[0], uncs = temp_ref_line[1],label = "",
                    linewidth = 0, marker_size = 0, reference = True)
    lineplot.Plot(plot_dir + "/" + str(mh_value))

def plot_pull_uncs(bins, region, n_region_bins, plotdir, xpos, ypos, ydiff):

    plot_dir = plotdir + "/uncs_frac"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$\sigma_{r}/\sigma_P$", ylabel = "No. Bins", plot_unc = False, density = False)
    for Ord in range(len(poly_orders)):
        uncs_ratio = np.divide(m1m2_ratio_3b2b[region][1], np.array(poly_uncs[region][Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b[region][1], np.array(poly_uncs[region][Ord])))]
        histplot.Add(uncs_ratio, label = "Order-"+str(poly_orders[Ord]))
        histplot.Text(r"$\chi^2/n_{df} = $"+str(round(chi_squareds[region][Ord]))+"/"+str(n_region_bins), xpos = xpos, ypos = ypos - ydiff*Ord)
    histplot.Plot(plot_dir + "/" + region)

def plot_pull_1D_all(bins, region, n_region_bins, plotdir, xpos, ypos, ydiff):

    plot_dir = plotdir + "/pulls_1D"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$(r_{3b}^{2b} - P)/(\sqrt{\sigma_r^2 - \sigma_P^2})$", ylabel = "No. Bins", plot_unc = False, density = False)
    for Ord in range(len(poly_orders)):
        histplot.Add(pulls[region][Ord], label = "Order-"+str(poly_orders[Ord]))
        histplot.Text(r"$\chi^2/n_{df} = $"+str(round(chi_squareds[region][Ord]))+"/"+str(n_region_bins), xpos = xpos, ypos = ypos-ydiff*Ord)    
    histplot.Plot(plot_dir + "/" + region)


def param_plot(Order, params, mean_params, mean_cov, plotdir):

    plot_dir = plotdir + "/params"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    n_params = int(((Order+1)*(Order+2))/2)

    bins = skplt.get_bins(-4,4,30)    
    
    gauss = np.random.normal(0,1, 100_000)

    est_cov = np.cov(params.T)

    # plot param ratio
    histplot = skplt.HistogramPlot(bins, xlabel = "param", ylabel = "No. Fits", plot_unc = False, density = True)
    for param in range(n_params):
        standard = (params[:,param] - mean_params[param]) / np.sqrt(mean_cov[param,param])
        histplot.Add(standard, label = "a_"+str(param))
        histplot.Text(r"$\sigma$"+"_"+str(param)+ " = " + str(round(np.std(standard), 2)),  xpos = 0.15, ypos = 0.82 - 0.05*param)
    histplot.Add(gauss, label = r"$\mathcal{N}(0,1)$", drawstyle = "default")
    histplot.Plot(plot_dir+"/params")


def plot_cov_vs_uncs(bins, region, poly_uncs, cov_matrix, params, gridpoints, plotdir):

    plot_dir = plotdir + "/cov_frac"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    uncs_arr = []
    uncs_arr_J = []

    nparams = len(params)

    e, V = np.linalg.eigh(cov_matrix)
    D = np.diag(1/np.sqrt(e))


    for point in tqdm(range(len(gridpoints[:,0])), desc = "Calculating Cov Unc", leave = False):



        Point = gridpoints[point].reshape(1,2)

        if region == "SR" and X_HH(Point[:,0],Point[:,1]) > 1.6:
            uncs_arr.append(0)
            uncs_arr_J.append(0)
        elif region == "CR" and X_HH(Point[:,0],Point[:,1]) < 1.6 and R_CR(Point[:,0],Point[:,1]) > 45:
            uncs_arr.append(0)
            uncs_arr_J.append(0)
        elif region == "full" and R_CR(Point[:,0],Point[:,1]) > 45:
            uncs_arr.append(0)
            uncs_arr_J.append(0)
        else:
            # Get jacobian array of polynomial
            J = np.zeros(nparams)
            for param in range(nparams):
                temp_params = np.zeros(nparams)
                temp_params[param] = 1

                J[param] = poly2d_fit_func(Point, *temp_params)

            J = np.array(J)

            # perform atrix multiplication
            var_f = J @ cov_matrix @ np.transpose(J)

            uncs_arr_J.append(np.sqrt(var_f))

            f_point = poly2d_fit_func(Point, *params)
            sum_sqrd = 0
            for param in range(nparams):
                temp_params = np.zeros(nparams)
                temp_params[param] = 1

                temp_params = np.linalg.inv(D @ V.T) @ temp_params

                sum_sqrd += (f_point - poly2d_fit_func(Point, *(params + temp_params)))**2

            uncs_arr.append(np.sqrt(sum_sqrd)[0])                

    ratios = np.divide(np.array(uncs_arr), poly_uncs)

    ratios = ratios[ratios > 0]

    ratios_J = np.divide(np.array(uncs_arr_J), poly_uncs)

    ratios_J = ratios_J[ratios_J > 0]


    # Plot Ratio

    mean, mean_SE = skplt.get_mean(ratios_J)
    std, std_SE = skplt.get_std(ratios_J)
    meanJ, meanJ_SE = skplt.get_mean(ratios_J)
    stdJ, stdJ_SE = skplt.get_std(ratios_J)
    

    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$\sigma_{cov}/\sigma_P$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
    histplot.Add(ratios, label = r"$\sigma_{cov}/\sigma_P$")
    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.65, ypos = 0.78)
    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.65, ypos = 0.73)
    histplot.Add(ratios_J, label = r"$\sigma_{cov}/\sigma_P$" + "Jacobian")
    histplot.Text(r"$\mu$ = "+str(round(meanJ, 2))+r" $\pm$ "+str(round(meanJ_SE,4)), xpos = 0.35, ypos = 0.78)
    histplot.Text(r"$\sigma$ = "+str(round(stdJ, 2))+r" $\pm$ "+str(round(stdJ_SE,4)), xpos = 0.35, ypos = 0.73)
    histplot.Plot(plot_dir + "/" + region)


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

bins = {"m_h1": skplt.get_bins(80,180,60),
        "m_h2": skplt.get_bins(75,170,60),
        "dEta_hh": skplt.get_bins(0,1.5,30),
        "eta_h1": skplt.get_bins(-2.5,2.5,30),
        "eta_h2": skplt.get_bins(-2.5,2.5,30),
        "m_hh": skplt.get_bins(0,1000,30),
        "pt_h1": skplt.get_bins(0,600,30),
        "pt_h2": skplt.get_bins(0,600,30),
        "X_hh": skplt.get_bins(0,4,30)}

runs = ["run2", "run3"]
numerator = "3b1j"

for run in runs:

    plot_dir_base = "plots/parquet/"+run

    if not os.path.exists(plot_dir_base):
            os.mkdir(plot_dir_base)


    etacuts = ["", "revDEta"]


    for etacut in etacuts:
        for numerator in ["3b1j", "4b"]:

            m1_m2_histarrs_CR = {"3b1j": np.load("parquet_hists/"+run+"/CR_3b1j"+etacut+".npy"),
                                "2b2j": np.load("parquet_hists/"+run+"/CR_2b2j"+etacut+".npy"),
                                "4b": np.load("parquet_hists/"+run+"/CR_4b"+etacut+".npy")}

            m1_m2_histarrs_SR = {"3b1j": np.load("parquet_hists/"+run+"/SR_3b1j"+etacut+".npy"),
                                "2b2j": np.load("parquet_hists/"+run+"/SR_2b2j"+etacut+".npy"),
                                "4b": np.load("parquet_hists/"+run+"/SR_4b"+etacut+".npy")}

            m1_m2_histarrs_CR_temp = {numerator : []}
            m1_m2_histarrs_SR_temp = {numerator : []}

            cmin = -2
            cmax = 2

            down_ratio = np.sum(m1_m2_histarrs_CR[numerator])/np.sum(m1_m2_histarrs_CR["4b"])
            gridpoints = Gridpoints("m_h1", "m_h2", bins)

            CR_bool = np.logical_and(R_CR(gridpoints[:,0], gridpoints[:,1]) < 45, X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6)
            SR_bool = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6

            n_CR_bins = np.sum(CR_bool)
            n_SR_bins = np.sum(SR_bool)

            chis = {"full": [],
                    "CR": [],
                    "SR": []}

            ks =  {"full": [],
                    "CR": [],
                    "SR": []}

            opt_params = []
            opt_cov = []

            N_validate = 1 # N Loops
            get_poly_uncs = True
            ensemble = False
            N_trials = 1000 # N loops for poly uncs
            poly_orders = [1,2,3,4]

            poly_coeff_guesses = [np.ones(int((O+1)*(O+2)/2)) for O in poly_orders]

            poly_covs_boot = list(np.ones(len(poly_orders)))
            for i in range(len(poly_covs_boot)):
                poly_covs_boot[i] = []

            poly_means_save = list(np.ones(len(poly_orders)))
            for i in range(len(poly_means_save)):
                poly_means_save[i] = []


            poly_average = {"full": [],
                            "CR" : [],
                            "SR" : []}

            for N in range(len(poly_orders)):
                for key in poly_average.keys():
                    poly_average[key].append([])

            plot_dir_temp = plot_dir_base + "/Fitting"+etacut

            if not os.path.exists(plot_dir_temp):
                    os.mkdir(plot_dir_temp)


            cov_ratio_mean_boot = list(np.ones(len(poly_orders)))
            for i in range(len(cov_ratio_mean_boot)):
                cov_ratio_mean_boot[i] = []

            #Loop a load of times and to plot chisuared and k-s values
            for val in tqdm(range(N_validate), desc = "iterating-"+numerator+"-"+run+"-"+etacut, leave = True):

                # downsample 3b data to 4b-like stats
                # take poisson distributed counts based on new uncs
                # m1_m2_histarrs_CR_temp["3b1j"] = np.random.poisson(m1_m2_histarrs_CR["3b1j"]/down_ratio)
                # m1_m2_histarrs_SR_temp["3b1j"] = np.random.poisson(m1_m2_histarrs_SR["3b1j"]/down_ratio)

                if numerator == "3b1j":
                    m1_m2_histarrs_CR_temp["3b1j"] = m1_m2_histarrs_CR["3b1j"]/down_ratio
                    m1_m2_histarrs_SR_temp["3b1j"] = m1_m2_histarrs_SR["3b1j"]/down_ratio

                    m1m2_ratio_3b2b = {"full" : skplt.CalcNProp("/", [m1_m2_histarrs_CR_temp[numerator].flatten() + m1_m2_histarrs_SR_temp[numerator].flatten(),
                                                                    np.sqrt(m1_m2_histarrs_CR_temp[numerator].flatten() + m1_m2_histarrs_SR_temp[numerator].flatten())],
                                                                    [m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten(),
                                                                    np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten())]),
                                    "CR" : skplt.CalcNProp("/", [m1_m2_histarrs_CR_temp[numerator].flatten(), np.sqrt(m1_m2_histarrs_CR_temp[numerator].flatten())],
                                                                [m1_m2_histarrs_CR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten())]),
                                    "SR" : skplt.CalcNProp("/", [m1_m2_histarrs_SR_temp[numerator].flatten(), np.sqrt(m1_m2_histarrs_SR_temp[numerator].flatten())],
                                                                [m1_m2_histarrs_SR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_SR["2b2j"].flatten())])}

                else:

                    m1_m2_histarrs_CR_temp[numerator] = m1_m2_histarrs_CR[numerator]
                    m1_m2_histarrs_SR_temp[numerator] = m1_m2_histarrs_SR[numerator]


                    m1m2_ratio_3b2b = {"full" : skplt.CalcNProp("/", [m1_m2_histarrs_CR_temp[numerator].flatten(),
                                                                    np.sqrt(m1_m2_histarrs_CR_temp[numerator].flatten())],
                                                                    [m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten(),
                                                                    np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten())]),
                                    "CR" : skplt.CalcNProp("/", [m1_m2_histarrs_CR_temp[numerator].flatten(), np.sqrt(m1_m2_histarrs_CR_temp[numerator].flatten())],
                                                                [m1_m2_histarrs_CR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten())]),
                                    "SR" : skplt.CalcNProp("/", [np.zeros_like(m1_m2_histarrs_SR["2b2j"]).flatten(), np.sqrt(np.zeros_like(m1_m2_histarrs_SR["2b2j"]).flatten())],
                                                                [m1_m2_histarrs_SR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_SR["2b2j"].flatten())])}


                n_3b = {"full": np.sum(m1_m2_histarrs_CR[numerator]) + np.sum(m1_m2_histarrs_SR[numerator]), "CR": np.sum(m1_m2_histarrs_CR[numerator]), "SR" : np.sum(m1_m2_histarrs_SR[numerator])}
                n_2b = {"full": np.sum(m1_m2_histarrs_CR["2b2j"]) + np.sum(m1_m2_histarrs_SR["2b2j"]), "CR": np.sum(m1_m2_histarrs_CR["2b2j"]), "SR" : np.sum(m1_m2_histarrs_SR["2b2j"])}


                #########################################################################################
                #########################################################################################
                #########################################################################################
                #########################################################################################

                # function fitting

                validbool = np.logical_and(np.isfinite(m1m2_ratio_3b2b["CR"][0]),np.isfinite(m1m2_ratio_3b2b["CR"][1]))

                opt_coeffs = []
                opt_covs = []

                poly_values = {"full": [],
                            "CR": [],
                            "SR": []}

                poly_uncs = {"full": [],
                            "CR": [],
                            "SR": []}

                pulls = {"full": [],
                        "CR": [],
                        "SR": []}

                chi_squareds = {"full": [],
                                "CR": [],
                                "SR": []}

                poly_params = list(np.ones(len(poly_orders)))
                for i in range(len(poly_params)):
                    poly_params[i] = []

                # Fitting polynomial for diff orders
                for N in tqdm(range(len(poly_orders)), desc = "Order", leave = False):

                    temp_coeffs, temp_covs = so.curve_fit(poly2d_fit_func,
                                                        gridpoints[validbool],
                                                        np.array(m1m2_ratio_3b2b["CR"][0])[validbool],
                                                        p0 = poly_coeff_guesses[N],
                                                        sigma = np.array(m1m2_ratio_3b2b["CR"][1])[validbool])

                    opt_params.append(temp_coeffs)
                    opt_cov.append(temp_covs)

                    opt_coeffs.append(temp_coeffs)
                    opt_covs.append(temp_covs)

                    poly_means_save[N].append(temp_coeffs)

                    polynomial = poly2d_fit_func(gridpoints, *temp_coeffs)

                    poly_CR = polynomial
                    poly_CR[R_CR(gridpoints[:,0], gridpoints[:,1]) > 45] == 0
                    poly_CR[X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6] == 0

                    poly_SR = polynomial
                    poly_SR[X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6] == 0

                    poly_values["full"].append(polynomial)
                    poly_values["CR"].append(poly_CR)
                    poly_values["SR"].append(poly_SR)

                    #######################################################################################
                    # Get unc on fits
                    # We do this by varying the data according to being poisson distributed and getting per bin standard deviations



                    if get_poly_uncs:

                        trials_bin_values = []
                        param_temp_for_cov = []
                        for n in tqdm(range(N_trials), desc = "Calculating Uncertainties", leave = False):
                            # Vary the bin counts
                            # print(m1_m2_histarrs_CR_temp["3b1j"][np.logical_and(m1_m2_histarrs_CR_temp["3b1j"] < 10, m1_m2_histarrs_CR_temp["3b1j"] > 0)])
                            temp_m1m1_3b_counts = np.random.poisson(m1_m2_histarrs_CR_temp[numerator])
                            temp_m1m2_3b_2b_ratio = skplt.CalcNProp("/", [temp_m1m1_3b_counts.flatten(), np.sqrt(temp_m1m1_3b_counts.flatten())],
                                                                            [m1_m2_histarrs_CR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten())])

                            tempvalidbool = np.logical_and(np.isfinite(temp_m1m2_3b_2b_ratio[0]),np.isfinite(temp_m1m2_3b_2b_ratio[1]))

                            # Get a fit
                            temp_params, tem_covariance = so.curve_fit(poly2d_fit_func,
                                                                    gridpoints[tempvalidbool],
                                                                    np.array(temp_m1m2_3b_2b_ratio[0])[tempvalidbool],
                                                                    p0 = poly_coeff_guesses[N],
                                                                    sigma = np.array(temp_m1m2_3b_2b_ratio[1])[tempvalidbool])

                            # Apply polynomial to each bin 
                            trials_bin_values.append(poly2d_fit_func(gridpoints, *temp_params))

                            poly_params[N].append(temp_params)
                            param_temp_for_cov.append(temp_params)

                        # Calculate uncertainties)
                        temp_uncs = np.std(np.array(trials_bin_values), axis = 0)

                        tempcov = np.cov(np.array(param_temp_for_cov).T)
                        poly_covs_boot[N].append(tempcov)

                    else:
                        temp_uncs = np.zeros_like(gridpoints[:,0])

                    ########################################################################################

                    uncs_CR = temp_uncs
                    uncs_CR[R_CR(gridpoints[:,0], gridpoints[:,1]) > 45] == 0
                    uncs_CR[X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6] == 0

                    uncs_SR = temp_uncs
                    uncs_SR[X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6] == 0

                    poly_uncs["full"].append(temp_uncs)
                    poly_uncs["CR"].append(uncs_CR)
                    poly_uncs["SR"].append(uncs_SR)

                    # Pull calculations
                    pulls_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b["full"],
                                                [poly_values["full"][N], np.array(poly_uncs["full"][N])])
                    pulls_CR_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b["CR"],
                                                    [poly_values["CR"][N], np.array(poly_uncs["CR"][N])])
                    pulls_SR_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b["SR"],
                                                    [poly_values["SR"][N], np.array(poly_uncs["SR"][N])])

                    pulls["full"].append(pulls_temp[0]/pulls_temp[1])
                    pulls["CR"].append(pulls_CR_temp[0]/pulls_CR_temp[1])
                    pulls["SR"].append(pulls_SR_temp[0]/pulls_SR_temp[1])

                    # Chi Squared calculations
                    chi_squareds["full"].append(np.sum(np.square(pulls["full"][N][np.isfinite(pulls["full"][N])])))
                    chi_squareds["CR"].append(np.sum(np.square(pulls["CR"][N][np.isfinite(pulls["CR"][N])])))
                    chi_squareds["SR"].append(np.sum(np.square(pulls["SR"][N][np.isfinite(pulls["SR"][N])])))

                    chis["full"].append(np.sum(np.square(pulls["full"][N][np.isfinite(pulls["full"][N])])))
                    chis["CR"].append(np.sum(np.square(pulls["CR"][N][np.isfinite(pulls["CR"][N])])))
                    chis["SR"].append(np.sum(np.square(pulls["SR"][N][np.isfinite(pulls["SR"][N])])))

                    # Get avg polynomial values (ensembling)
                    if val == 0:
                        for key in poly_average.keys():
                            poly_average[key][N] = poly_values[key]
                    else: 
                        for key in poly_average.keys():
                            poly_average[key][N] += poly_values[key]

                cov_ratio_mean_boot[N].append(np.cov(np.array(poly_params[N]).T)/temp_covs)

            # average poly values
            for key in poly_average.keys():
                poly_average[key][N] = np.array(poly_average[key][N])
                poly_average[key][N] /= N_validate


            for N in range(len(poly_orders)):

                # save polynomial coefficiencts

                # np.save("opt_params/Order-" + str(poly_orders[N]) + ".npy", np.array(opt_params[N::len(poly_orders)]))

                plot_dir_temp = plot_dir_base + "/Fitting"+etacut+"/"+numerator+"_2b2j_order-"+str(poly_orders[N])

                if not os.path.exists(plot_dir_temp):
                    os.mkdir(plot_dir_temp)

                n_region_bins =  {"full" : n_CR_bins + n_SR_bins,
                                "CR" : n_CR_bins,
                                "SR" : n_SR_bins}
                region_names = ["full", "CR", "SR"]

                plot_dir = plot_dir_temp + "/chis"

                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)

                for region in region_names:

                    tempchis = chis[region][N::len(poly_orders)]/n_region_bins[region]

                    cushion = (1e-6 + np.max(tempchis)-np.min(tempchis))/10

                    mean, mean_SE = skplt.get_mean(tempchis)
                    std, std_SE = skplt.get_std(tempchis)

                    histplot = skplt.HistogramPlot(skplt.get_bins(np.min(tempchis)-cushion,np.max(tempchis)+cushion,30), xlabel = r"$\chi^2/n_{df}$" + " ["+region+"]", ylabel = "No. Iterations", plot_unc = False)
                    histplot.Add(tempchis, label = "Order-"+str(poly_orders[N]))
                    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.52, ypos = 0.78)
                    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.52, ypos = 0.73)
                    histplot.Plot(plot_dir + "/" + region)

                # Plot compatability of bootstrapped params with covariance from mean fit 

                if poly_orders[N] < 3 and get_poly_uncs:
                    param_plot(poly_orders[N], np.array(poly_params[N]), np.array(opt_params[N]), np.array(opt_covs[N]), plot_dir_temp)

                    # plot cov ratio mean
                    histplot = skplt.HistogramPlot(bins = skplt.get_bins(0,5,30), xlabel = r"$\bar{\Sigma_{mean}/\Sigma_{boot}}$", ylabel = "No. Fits", plot_unc = False)
                    histplot.Add(cov_ratio_mean_boot[N], label = "Order-"+str(poly_orders[N]))
                    histplot.Text(r"$\mu$ = " + str(round(np.mean(cov_ratio_mean_boot[N]), 2)), xpos = 0.52, ypos = 0.78)
                    histplot.Plot(plot_dir_temp + "/params/mean_cov")


            # Save fit means and covariance matricies
            for N in range(len(poly_orders)):

                save_location = "poly_means_covs/"+run+"_"+numerator+"_2b2j__Order-"+str(poly_orders[N])+etacut
                
                if not os.path.exists(save_location):
                        os.mkdir(save_location)

                np.savetxt(save_location+"/means", np.array(poly_means_save[N])[0], delimiter = ",")
                np.savetxt(save_location+"/covs", np.array(poly_covs_boot[N])[0], delimiter = ",")



            ##############################################################################################################################

            ## Single Order plots


            for Ord in range(len(poly_orders)):

                plot_dir_temp = plot_dir_base + "/Fitting"+etacut+"/"+numerator+"_2b2j_order-"+str(poly_orders[N])

                if not os.path.exists(plot_dir_temp):
                    os.mkdir(plot_dir_temp)



                region_names = ["full", "CR", "SR"]

                uncbins = skplt.get_bins(0,20,40)
                n_region_bins =  {"full" : n_CR_bins + n_SR_bins,
                                "CR" : n_CR_bins,
                                "SR" : n_SR_bins}

                for region in region_names:

                    #######################################################
                    ## pull uncertainty plot

                    uncs_ratio = np.divide(m1m2_ratio_3b2b[region][1], np.array(poly_uncs[region][Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b[region][1], np.array(poly_uncs[region][Ord])))]
                    plot_uncs_ratio(uncbins, uncs_ratio, region, plot_dir_temp)

                    #######################################################
                    # 2D massplane pulls

                    plot_pulls(pulls[region][Ord], region, plot_dir_temp)

                    # plot_ratios(m1m2_ratio_3b2b[region][Ord],region, plot_dir_temp)

                    #######################################################
                    # Pulls_1D Plotting

                    plot_pulls_1D(skplt.get_bins(-4,4,30), pulls[region][Ord][np.isfinite(pulls[region][Ord])], region, plot_dir_temp, Ord, chi_squareds[region][Ord], n_region_bins[region])


                    #######################################################
                    # covariance matrix uncertainty vs polynomial uncertainty

                    plot_cov_vs_uncs(skplt.get_bins(0,2,30), region, np.array(poly_uncs[region][Ord]), np.array(opt_covs[Ord]), opt_coeffs[Ord], gridpoints, plot_dir_temp)
                

                #######################################################
                # Differential mh1 mh2 plots

                for mh_bin in tqdm(range(40), leave = False, desc = "diferential plotting"):
                    mh_bin += 10

                    if ensemble:
                        plot_diff_mh1(bins["m_h1"], round(gridpoints[mh_bin*60:mh_bin*60 + 60][0,0]),
                                    [poly_values["full"][Ord][mh_bin*60:mh_bin*60 + 60], poly_uncs["full"][Ord][mh_bin*60:mh_bin*60 + 60]],
                                    [m1m2_ratio_3b2b["CR"][0][mh_bin*60:mh_bin*60 + 60], m1m2_ratio_3b2b["CR"][1][mh_bin*60:mh_bin*60 + 60]],
                                    [m1m2_ratio_3b2b["SR"][0][mh_bin*60:mh_bin*60 + 60], m1m2_ratio_3b2b["SR"][1][mh_bin*60:mh_bin*60 + 60]],
                                    plot_dir_temp)

                        plot_diff_mh2(bins["m_h2"], round(gridpoints[mh_bin::60][0,1]),
                                    [poly_values["full"][Ord][mh_bin::60], poly_uncs["full"][Ord][mh_bin::60]],
                                    [m1m2_ratio_3b2b["CR"][0][mh_bin::60], m1m2_ratio_3b2b["CR"][1][mh_bin::60]],
                                    [m1m2_ratio_3b2b["SR"][0][mh_bin::60], m1m2_ratio_3b2b["SR"][1][mh_bin::60]],
                                    plot_dir_temp)
                    else:
                        plot_diff_mh1(bins["m_h1"], round(gridpoints[mh_bin*60:mh_bin*60 + 60][0,0]),
                                    [poly2d_fit_func(gridpoints[mh_bin*60:mh_bin*60 + 60], *opt_coeffs[Ord]), poly_uncs["full"][Ord][mh_bin*60:mh_bin*60 + 60]],
                                    [m1m2_ratio_3b2b["CR"][0][mh_bin*60:mh_bin*60 + 60], m1m2_ratio_3b2b["CR"][1][mh_bin*60:mh_bin*60 + 60]],
                                    [m1m2_ratio_3b2b["SR"][0][mh_bin*60:mh_bin*60 + 60], m1m2_ratio_3b2b["SR"][1][mh_bin*60:mh_bin*60 + 60]],
                                    plot_dir_temp)


                        plot_diff_mh2(bins["m_h2"], round(gridpoints[mh_bin::60][0,1]),
                                    [poly2d_fit_func(gridpoints[mh_bin::60], *opt_coeffs[Ord]), poly_uncs["full"][Ord][mh_bin::60]],
                                    [m1m2_ratio_3b2b["CR"][0][mh_bin::60], m1m2_ratio_3b2b["CR"][1][mh_bin::60]],
                                    [m1m2_ratio_3b2b["SR"][0][mh_bin::60], m1m2_ratio_3b2b["SR"][1][mh_bin::60]],
                                    plot_dir_temp)


            ##############################################################################################################################

            ## All Order plots
            plot_dir_temp = plot_dir_base + "/Fitting"+etacut+"/All_Orders"

            if not os.path.exists(plot_dir_temp):
                os.mkdir(plot_dir_temp)

            #######################################################
            ## pull uncertainty plot


            region_names = ["full", "CR", "SR"]

            uncbins = skplt.get_bins(0,20,40)
            n_region_bins =  {"full" : n_CR_bins + n_SR_bins,
                                "CR" : n_CR_bins,
                                "SR" : n_SR_bins}

            xpos = 0.15
            ypos = 0.82
            ydiff = 0.05

            for region in region_names:

                #######################################################
                ## pull uncertainty plot

                plot_pull_uncs(uncbins, region, n_region_bins[region], plot_dir_temp, xpos, ypos, ydiff)

                #######################################################
                # Pulls_1D Plotting

                plot_pull_1D_all(skplt.get_bins(-4,4,30), region, n_region_bins[region], plot_dir_temp, xpos, ypos, ydiff)
