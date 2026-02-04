import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import skink as skplt
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import functools as ft
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", DeprecationWarning)


###############################################################################################
#Misc Functions


def X_HH(mh1, mh2):
    R = ((mh1 - 124) / (0.1 * mh1)) ** 2 + ((mh2 - 117) / (0.1 * mh2)) ** 2

    return np.sqrt(R)


def Gridpoints(m1_bins, m2_bins):
    # Gridpoints for 2d mass plot
    gridpoints = []
    for m1bin in m1_bins[1]:
        for m2bin in m2_bins[1]:
            gridpoints.append([m1bin,m2bin])

    return np.array(gridpoints)


def Check_Location(filepath):    
    if not os.path.exists(filepath):
        os.mkdir(filepath)


def Contour_Plot(function, xbins, ybins):
    X, Y = np.meshgrid(xbins,ybins)
    plt.contour(X, Y, function(X, Y), levels = [0], colors = "red", linewidths = 2, linestyles = "--")

###############################################################################################
# Polynomial Reading/Reweighting


def poly2d_fit_func(XY, params):
    """
    A 2D polynomial function f(x, y) that takes a variable number of coefficients.

    The terms are: c_00 + c_10*x + c_01*y + c_20*x^2 + c_11*x*y + c_02*y^2 + ...
    The number of coefficients determines the max degree.
    """

    x, y = XY[:, 0], XY[:, 1]

    # c_00 is the first coefficient (constant term)
    result = params[0]

    # if const polynomial return this
    if len(params) == 1:
        return result * np.ones_like(x)

    # loop over
    coeff_idx = 1
    degree = 1
    while coeff_idx < len(params):
        # Iterate over all possible combinations of x^i * y^j where i+j = degree
        for i in range(degree + 1):
            j = degree - i
            if coeff_idx < len(params):
                # Add the term: coefficient * x^i * y^j
                result += params[coeff_idx] * (x**i) * (y**j)
                coeff_idx += 1
            else:
                break
        degree += 1

    return result

def Read(means_file, covs_file):
    means = np.loadtxt(means_file, delimiter=",")
    cov = np.loadtxt(covs_file, delimiter=",")

    return means, cov


def poly_eigfs(means, cov):
    # This funtion takes the polynomial means parameters and covariance matrix
    # to return the nominal and variation polynomial function

    nparams = means.shape[0]

    nominal = ft.partial(poly2d_fit_func, params=means)

    e, V = np.linalg.eigh(cov)

    D = np.diag(1 / np.sqrt(e))

    U = D @ V.T

    variation = np.linalg.inv(U)

    return nominal, [
        ft.partial(poly2d_fit_func, params=means + variation[:, i])
        for i in range(nparams)
    ]


def reweight(funcs, m1m2s):
    # takes poly nominal and variations and feeds the m1m2 pairs into it
    nom_funcs, var_funcs = funcs
    return nom_funcs(m1m2s), [var_func(m1m2s) for var_func in var_funcs]


def get3b2b(year, order, path):

    filepath = path + "3b1j_2b2j__Order-"+str(order)+"/"+year

    means, cov = Read(filepath + "/means", filepath + "/covs")

    nom_func, var_funcs = poly_eigfs(means, cov)

    return nom_func, var_funcs


def get4b2b(year, order, path):

    filepath = path + "4b_2b2j__Order-"+str(order)+"/"+year

    means, cov = Read(filepath + "/means", filepath + "/covs")

    nom_func, var_funcs = poly_eigfs(means, cov)

    return nom_func, var_funcs


def Apply_Polynomial(m1s, m2s, numerator, order, poly_filepath, year):

    if numerator == "3b1j":
        nom_weights, var_weights = reweight(get3b2b(year, order, poly_filepath),np.array([m1s, m2s]).T)
    elif numerator == "4b":
        nom_weights, var_weights = reweight(get4b2b(year, order, poly_filepath),np.array([m1s, m2s]).T)

    return nom_weights, var_weights


###############################################################################################
# Edge Finding


def xhh_curve(x, y):
    term_x = (x - 124) / (0.1 * x)
    term_y = (y - 117) / (0.1 * y)

    return np.sqrt(term_x**2 + term_y**2) - 1.6


def calculate_cumulative_arc_length(points):
    segment_diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(segment_diffs**2, axis=1))

    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    return cumulative_lengths, np.sum(segment_lengths)


def edge_bool(mh1_bins, mh2_bins):
    x_min, x_max = mh1_bins[0], mh1_bins[-1]
    y_min, y_max = mh2_bins[0], mh2_bins[-1]
    N_grid = 300  # Resolution for the grid

    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, N_grid), np.linspace(y_min, y_max, N_grid)
    )

    # Calculate the value of F(x, y) for every point in the grid
    Z_grid = xhh_curve(X_grid, Y_grid)

    fig, ax = plt.subplots()
    contour_set = ax.contour(X_grid, Y_grid, Z_grid, levels=[0], linewidths=0)
    plt.close(fig)  # We close the figure since we only need the data, not the plot

    paths = contour_set.get_paths()
    if not paths:
        print("Error: Could not find the contour F(x, y) = 0 within the defined grid.")
        exit()

    # Get vertices from the first path
    initial_points = paths[0].vertices
    if not np.allclose(initial_points[0], initial_points[-1]):
        initial_points = np.vstack([initial_points, initial_points[0]])

    cumulative_s, total_L = calculate_cumulative_arc_length(initial_points)

    N_uniform = 10000  # Number of uniform points you want
    s_targets = np.linspace(
        0, total_L, N_uniform, endpoint=False
    )  # Exclude endpoint if loop

    # Interpolate X and Y coordinates as a function of the cumulative arc length 's'
    x_interp = interp1d(cumulative_s, initial_points[:, 0])
    y_interp = interp1d(cumulative_s, initial_points[:, 1])

    # Get the uniformly sampled coordinates
    x_uniform = x_interp(s_targets)
    y_uniform = y_interp(s_targets)

    uniform_points = np.array([x_uniform, y_uniform]).T

    return (
        np.histogram2d(
            uniform_points[:, 0], uniform_points[:, 1], bins=[mh1_bins, mh2_bins]
        )[0]
        > 0
    )


###############################################################################################
# General Tasks/ Plots

def Get_Data(nj_tags, sample_filedir, year_files):

    print("Reading Data")

    # Loop over the year files and save the m1,m2,SR and polynomial info for each event
    datasets = {}
    
    for NJ in nj_tags:
        datasets[NJ] = {}
        for year in year_files.keys():

            if type(year_files[year]) == str:   
                temp_arrs = Read_Files([sample_filedir+year_files[year]], NJ)
            else:
                temp_arrs = Read_Files([sample_filedir+year_file for year_file in year_files[year]], NJ)

            datasets[NJ][year] = {"m_h1": np.concatenate(temp_arrs["m_h1"]),
                                  "m_h2": np.concatenate(temp_arrs["m_h2"]),
                                  "SR": np.concatenate(temp_arrs["SR"])}

    return datasets

def Read_Files(files, NJ):

    arrays = {"m_h1": [],
              "m_h2": [],
              "SR": []}


    for file in files:
        dataset = pq.ParquetFile(file[:-8] + "_reduced_"+NJ+".parquet")

        for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size=10_000), desc=file, leave = False)):

            for key in arrays.keys():
                arrays[key].append(np.array(batch[key]))

    return arrays


def Get_MassPlane_Bins(m1_bins, m2_bins, datasets, plot = None):
    # Get The 2D MassPlane Binnings used for plotting/calculating everything
    # Also Plot MassPlane (Ratio + Counts) on 2D Histograms

    print("Getting Bin Counts")

    xhh_mask = edge_bool(m1_bins[0], m2_bins[0])

    hists_counts = {}
    hists_counts_uncs = {}

    # Get Hists
    for ntag in datasets.keys():
        hists_counts[ntag] = {}
        hists_counts_uncs[ntag] = {}
        for year in datasets[ntag].keys():
            hists_counts[ntag][year] = {}
            hists_counts_uncs[ntag][year] = {}

            SR_bool = datasets[ntag][year]["SR"]

            for region in ["CR", "SR"]:
                validbool = SR_bool if region == "SR" else np.logical_not(SR_bool)

                hists_counts[ntag][year][region] = np.histogram2d(datasets[ntag][year]["m_h1"][validbool],
                                                                datasets[ntag][year]["m_h2"][validbool],
                                                                bins = [m1_bins[0], m2_bins[0]])[0]
                hists_counts_uncs[ntag][year][region] = np.sqrt(hists_counts[ntag][year][region])

                # hists_counts[ntag][year][region][xhh_mask] = 0
                # hists_counts_uncs[ntag][year][region][xhh_mask] = 0


    #Plot Hists
    if type(plot) == str:
        
        #2D Plots
        Check_Location(plot)
        plotdir = plot + "MassPlane_2D/"
        Check_Location(plotdir)
    
        gridpoints = Gridpoints(m1_bins, m2_bins)

        for ntag in datasets.keys():
            for year in datasets[ntag].keys():

                temp_plotdir = plotdir + year + "/"
                Check_Location(temp_plotdir)                    
                # Plotting the bin counts for each ntag
                histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                          xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
                histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = (hists_counts[ntag][year]["CR"] + hists_counts[ntag][year]["SR"]).flatten(),
                               zlabel = ntag + " counts, " + year)
                Contour_Plot(xhh_curve,m1_bins[0], m2_bins[0])
                histplot2d.Plot(temp_plotdir + ntag)

        # 1D Plots
        Check_Location(plot)
        plotdir = plot + "MassPlane_1D/"
        Check_Location(plotdir)
        for region in ["CR", "SR"]:
            # Per Ntag
            for ntag in datasets.keys():
                temp_plotdir = plotdir + ntag + "/"
                Check_Location(temp_plotdir)

                histplot_m1 = skplt.HistogramPlot(bins = m1_bins, xlabel = r"$m_{H1}[GeV]$", ylabel = "Normalized No. "+ntag+" Events", plot_unc = True, density = True, ratio = True)
                for year in datasets[ntag].keys():            
                    weights = np.array([np.sum((hists_counts[ntag][year][region])[mh1,:]) for mh1 in range(len(m1_bins[1]))])    
                    histplot_m1.Add(data = m1_bins[1], weights = weights, uncs = np.sqrt(weights), label = year, reference = year == list(datasets[ntag].keys())[0])

                histplot_m1.Plot(temp_plotdir+"m1_"+region)

            for ntag in datasets.keys():
                temp_plotdir = plotdir + ntag + "/"
                Check_Location(temp_plotdir)

                histplot_m2 = skplt.HistogramPlot(bins = m2_bins, xlabel = r"$m_{H2}[GeV]$", ylabel = "Normalized No. "+ntag+" Events", plot_unc = True, density = True, ratio = True)
                for year in datasets[ntag].keys():
                    weights = np.array([np.sum((hists_counts[ntag][year][region])[:,mh2]) for mh2 in range(len(m2_bins[1]))])
                    histplot_m2.Add(data = m2_bins[1], weights = weights, uncs = np.sqrt(weights), label = year, reference = year == list(datasets[ntag].keys())[0])

                histplot_m2.Plot(temp_plotdir+"m2_"+region)

            # per Year
            for year in datasets["2b2j"].keys():
                temp_plotdir = plotdir + year + "/"
                Check_Location(temp_plotdir)

                histplot_m1 = skplt.HistogramPlot(bins = m1_bins, xlabel = r"$m_{H1}[GeV]$", ylabel = "Normalized No. "+year+" Events", plot_unc = True, density = True, ratio = True)
                for ntag in datasets.keys():
                    weights = np.array([np.sum((hists_counts[ntag][year][region])[mh1,:]) for mh1 in range(len(m1_bins[1]))])    
                    histplot_m1.Add(data = m1_bins[1], weights = weights, uncs = np.sqrt(weights), label = ntag, reference = ntag == list(datasets.keys())[0])
            
                histplot_m1.Plot(temp_plotdir+"m1_"+region)

            for year in datasets["2b2j"].keys():
                temp_plotdir = plotdir + year + "/"
                Check_Location(temp_plotdir)

                histplot_m2 = skplt.HistogramPlot(bins = m2_bins, xlabel = r"$m_{H2}[GeV]$", ylabel = "Normalized No. "+year+" Events", plot_unc = True, density = True, ratio = True)
                for ntag in datasets.keys():
                    weights = np.array([np.sum((hists_counts[ntag][year][region])[:,mh2]) for mh2 in range(len(m2_bins[1]))])    
                    histplot_m2.Add(data = m2_bins[1], weights = weights, uncs = np.sqrt(weights), label = ntag, reference = ntag == list(datasets.keys())[0])
            
                histplot_m2.Plot(temp_plotdir+"m2_"+region)


    return hists_counts, hists_counts_uncs


def Get_Ratio(numerator, hists_counts, m1_bins, m2_bins, plot = None):

    print("Getting "+numerator+"/2b2j")

    xhh_mask = edge_bool(m1_bins[0], m2_bins[0])
    gridpoints = Gridpoints(m1_bins,m2_bins)
    SR_mask = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6

    hists_ratio = {}
    hists_ratio_uncs = {}
    for year in hists_counts[numerator].keys():
        hists_ratio[year] = {}
        hists_ratio_uncs[year] = {}
        for region in hists_counts[numerator][year].keys():
            temp_ratio = skplt.CalcNProp("/", [hists_counts[numerator][year][region].flatten(), np.sqrt(hists_counts[numerator][year][region].flatten())],
                                              [hists_counts["2b2j"][year][region].flatten(), np.sqrt(hists_counts["2b2j"][year][region].flatten())])
            hists_ratio[year][region] = temp_ratio[0]
            hists_ratio_uncs[year][region] = temp_ratio[1]

            region_mask = SR_mask if region == "CR" else np.logical_not(SR_mask)

            # Check for full set of valid bins
            # if not all(np.isfinite(hists_ratio[year][region].flatten()[region_mask])):
            #     raise("Invalid ratio encountered in counts")
            # if not all(np.isfinite(hists_ratio_uncs[year][region].flatten()[region_mask])):
            #     if region == "CR" or numerator != "4b": 
            #         raise("Invalid ratio encountered in uncertainties")

            hists_ratio[year][region][xhh_mask.flatten()] = np.nan
            hists_ratio_uncs[year][region][xhh_mask.flatten()] = np.nan

            hists_ratio[year][region][region_mask] = np.nan
            hists_ratio_uncs[year][region][region_mask] = np.nan


    if type(plot) == str:

        # 2D Plots
        Check_Location(plot)
        plotdir = plot + "Ratios_2D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)

        gridpoints = Gridpoints(m1_bins, m2_bins)

        for year in hists_ratio.keys():               
            # Plotting the ratios for each year
            num = hists_counts[numerator][year]["CR"]+hists_counts[numerator][year]["SR"]

            den = hists_counts["2b2j"][year]["CR"]+hists_counts["2b2j"][year]["SR"]

            histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                      xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
            histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = (num/den).flatten(),
                           zlabel = numerator+"/2b2j, " + year)
            Contour_Plot(xhh_curve, m1_bins[0], m2_bins[0])
            histplot2d.Plot(plotdir + year)

        # 1D Plots

        plotdir = plot +"Ratios_1D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"2b2j/"
        Check_Location(plotdir)
        
        for ntag in hists_counts.keys():

            histplot_m1 = skplt.HistogramPlot(bins = m1_bins, xlabel = r"m_{H1}[GeV]", ylabel = "Ratio of "+numerator+" events to 2b2j events", plot_unc = True, density = False, ratio = True)
            for year in datasets[ntag].keys():          
                temp_numerator = np.array([np.sum((hists_counts[numerator][year]["CR"] + hists_counts[numerator][year]["SR"])[mh1,:]) for mh1 in range(len(m1_bins[1]))])
                temp_denominator = np.array([np.sum((hists_counts["2b2j"][year]["CR"] + hists_counts["2b2j"][year]["SR"])[mh1,:]) for mh1 in range(len(m1_bins[1]))])

                weights = skplt.CalcNProp("/",[temp_numerator, np.sqrt(temp_numerator)], [temp_denominator, np.sqrt(temp_denominator)])
                histplot_m1.Add(data = m1_bins[1], weights = weights[0], uncs = weights[1], label = year, reference = year == list(datasets[ntag].keys())[0])

            histplot_m1.Plot(plotdir+"m1")

        for ntag in datasets.keys():

            histplot_m2 = skplt.HistogramPlot(bins = m2_bins, xlabel = r"m_{H2}[GeV]", ylabel = "Ratio of "+numerator+" events to 2b2j events", plot_unc = True, density = False, ratio = True)
            for year in datasets[ntag].keys():
                temp_numerator = np.array([np.sum((hists_counts[numerator][year]["CR"] + hists_counts[numerator][year]["SR"])[:,mh2]) for mh2 in range(len(m2_bins[1]))])
                temp_denominator = np.array([np.sum((hists_counts["2b2j"][year]["CR"] + hists_counts["2b2j"][year]["SR"])[:,mh2]) for mh2 in range(len(m2_bins[1]))])

                weights = skplt.CalcNProp("/",[temp_numerator, np.sqrt(temp_numerator)], [temp_denominator, np.sqrt(temp_denominator)])
                histplot_m2.Add(data = m2_bins[1], weights = weights[0], uncs = weights[1], label = year, reference = year == list(datasets[ntag].keys())[0])

            histplot_m2.Plot(plotdir+"m2")


    return hists_ratio, hists_ratio_uncs


def Get_Polynomial_Weights(datasets, numerator, poly_orders, poly_filepath):

    print("Getting "+numerator+"/2b2j weights")

    polynomial_weights = {}
    for year in year_files.keys():
        polynomial_weights[year] = {}
        for order in poly_orders:
            polynomial_weights[year][str(order)] = {}
            nom_weights, var_weights = Apply_Polynomial(datasets["2b2j"][year]["m_h1"],datasets["2b2j"][year]["m_h2"], numerator, order, poly_filepath, year)
            polynomial_weights[year][str(order)]["nom"] = nom_weights
            polynomial_weights[year][str(order)]["vars"] = {}
            for var, count in enumerate(var_weights):
                polynomial_weights[year][str(order)]["vars"]["var"+str(var)] = var_weights[var]

    return polynomial_weights


def Get_Bin_Weights(numerator, m1_bins, m2_bins, poly_order, poly_filepath, year_files, plot = False):

    print("Getting "+numerator+" Bin weights")

    gridpoints = Gridpoints(m1_bins, m2_bins)
    SR_bool = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6

    binned_weights = {}
    binned_weights_uncs = {}
    for year in year_files.keys():
        binned_weights[year] = {}
        binned_weights_uncs[year] = {}
        for order in poly_orders:
            binned_weights[year][str(order)] = {}

            nom_weights, var_weights = Apply_Polynomial(gridpoints[:,0], gridpoints[:,1], numerator, order,
                                                        poly_filepath, year)

            binned_weights_uncs[year][str(order)] = np.zeros_like(nom_weights)

            binned_weights[year][str(order)]["nom"] = nom_weights
            binned_weights[year][str(order)]["vars"] = {}
            for count, var in enumerate(var_weights):
                binned_weights[year][str(order)]["vars"]["var"+str(count)] = var 
                binned_weights_uncs[year][str(order)] = np.sqrt(binned_weights_uncs[year][str(order)]**2 + (nom_weights - var)**2)


    if type(plot) == str:

        # 2D Plots
        Check_Location(plot)
        plotdir = plot + "Weights_Binned_2D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)

        for year in binned_weights.keys():
            plotdir_temp = plotdir + year + "/"
            Check_Location(plotdir_temp)

            for order in binned_weights[year].keys(): 
                        
                histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                        xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
                histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = binned_weights[year][order]["nom"].flatten(),
                            zlabel = numerator+"/2b2j, " + year)
                Contour_Plot(xhh_curve, m1_bins[0], m2_bins[0])
                histplot2d.Plot(plotdir_temp + "Order-"+order)

    return binned_weights, binned_weights_uncs


def Get_Polynomial_Prediction(numerator, hists_counts, polynomial_weights, m1_bins, m2_bins, plot = None):

    print("Getting "+numerator+" predictions")

    hists_predictions = {}
    hists_predictions_uncs = {}
    for year in polynomial_weights.keys():
        hists_predictions[year] = {}
        hists_predictions_uncs[year] = {}     
        for order in polynomial_weights[year].keys():
            hists_predictions[year][order] = {}     
            hists_predictions_uncs[year][order] = {}     
            for region in ["CR", "SR"]:

                # Get Prediction            
                hists_predictions[year][order][region] = hists_counts["2b2j"][year][region].flatten() * polynomial_weights[year][order]["nom"]

                # Calc Uncertainties
                hists_predictions_uncs[year][order][region] = polynomial_weights[year][order]["nom"] * np.sqrt(hists_counts["2b2j"][year][region].flatten())

                for var in polynomial_weights[year][order]["vars"].keys():
                    hists_predictions_uncs[year][order][region] = np.sqrt(hists_predictions_uncs[year][order][region]**2 + (hists_counts["2b2j"][year][region].flatten()*(polynomial_weights[year][order]["nom"] - polynomial_weights[year][order]["vars"][var]))**2)

    if type(plot) == str:
        
        Check_Location(plot)
        plotdir = plot + "Predictions/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)

        gridpoints = Gridpoints(m1_bins, m2_bins)

        for year in hists_predictions.keys():
            for order in hists_predictions[year].keys():

                plotdir_temp = plotdir + "Order-"+str(order)+"/"
                Check_Location(plotdir_temp)

                for region in ["CR", "SR"]:

                    # Plotting the ratios for each year
                    histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                            xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
                    histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = hists_predictions[year][order][region].flatten(),
                                   zlabel = numerator+" count predictions from order-"+str(order)+" polynomial , " + year)
                    Contour_Plot(xhh_curve, m1_bins[0], m2_bins[0])
                    histplot2d.Plot(plotdir_temp + year+"_"+region)

    return hists_predictions, hists_predictions_uncs


def Get_Pulls(numerator, observed, observed_uncs, predicted, predicted_uncs, m1_bins, m2_bins, plot = False):

    if type(plot) == str:            
        print("Getting "+numerator+" Pulls")

    gridpoints = Gridpoints(m1_bins, m2_bins)
    xhh_mask = edge_bool(m1_bins[0], m2_bins[0]).flatten()
    SR_mask = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6

    pulls = {}
    for year in observed[numerator].keys():
        pulls[year] = {}
        for order in predicted[year].keys():
            pulls[year][order] = {}

            for region in ["SR", "CR"]:
                # pulls_temp = skplt.CalcNProp("-", [observed[numerator][year][region].flatten(), observed_uncs[numerator][year][region].flatten()],
                #                                   [predicted[year][str(order)][region], predicted_uncs[year][str(order)][region]])
                
                num = (observed[numerator][year][region].flatten() - predicted[year][str(order)][region])
                den = np.sqrt((observed_uncs[numerator][year][region].flatten())**2 + (predicted_uncs[year][str(order)][region])**2)
                pulls[year][order][region] = num/den
                

    if type(plot) == str:
                
        # 2D plot
        Check_Location(plot)
        plotdir = plot + "Pulls_2D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)
        
        gridpoints = Gridpoints(m1_bins, m2_bins)

        for year in pulls.keys():
            for order in pulls[year].keys():
                for region in pulls[year][order].keys():

                    plotdir_temp = plotdir + "Order-"+str(order)+"/"
                    Check_Location(plotdir_temp)

                    region_mask = SR_mask if region == "CR" else np.logical_not(SR_mask)
                    # bounded_mask = np.logical_not(np.logical_or(region_mask, xhh_mask))
                    bounded_mask = np.logical_not(region_mask)


                    mean, mean_SE = skplt.get_mean(pulls[year][order][region].flatten()[bounded_mask])
                    std, std_SE = skplt.get_std(pulls[year][order][region].flatten()[bounded_mask])
                    chi = np.sum(np.square(pulls[year][order][region].flatten()[bounded_mask]))
                    n_region_bins = np.sum(bounded_mask)

                    # 2D plot
                    temp_pull_arr = pulls[year][order][region].flatten()
                    temp_pull_arr[np.logical_not(bounded_mask)] = np.nan 
                    histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                            xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
                    histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = temp_pull_arr,
                                zlabel = r"$(Obs_{3b} - Pred_{3b})/(\sigma_{obs - Pred})$")
                    Contour_Plot(xhh_curve, m1_bins[0], m2_bins[0])
                    histplot2d.Plot(plotdir_temp + year+"_"+region)

        # 1D plot
        Check_Location(plot)
        plotdir = plot + "Pulls_1D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)
        
        bins = skplt.get_bins(-4,4,30)

        for year in pulls.keys():
            for order in pulls[year].keys():
                for region in pulls[year][order].keys():

                    plotdir_temp = plotdir + "Order-"+str(order)+"/"
                    Check_Location(plotdir_temp)

                    region_mask = SR_mask if region == "CR" else np.logical_not(SR_mask)
                    # bounded_mask = np.logical_not(np.logical_or(region_mask, xhh_mask))
                    bounded_mask = np.logical_not(region_mask)

                    mean, mean_SE = skplt.get_mean(pulls[year][order][region].flatten()[bounded_mask])
                    std, std_SE = skplt.get_std(pulls[year][order][region].flatten()[bounded_mask])
                    chi = np.sum(np.square(pulls[year][order][region].flatten()[bounded_mask]))
                    n_dof = np.sum(bounded_mask) + (int(order)+1)*(int(order)+2)/2 -1

                    # 1D plot
                    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$(Obs_{3b} - Pred_{3b})/(\sigma_{obs - Pred})$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
                    histplot.Add(pulls[year][order][region].flatten()[bounded_mask], label = "Order-"+str(order))
                    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.52, ypos = 0.78)
                    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.52, ypos = 0.73)
                    histplot.Text(r"$\chi^2/n_{df} = $"+str(round(chi))+"/"+str(n_dof), xpos = 0.15, ypos = 0.82)
                    histplot.Plot(plotdir_temp + "/" + year+"_"+region)

    return pulls

def Get_Chi2s(numerator, observed, observed_uncs, predicted, predicted_uncs, m1_bins, m2_bins, N_samples, plot = False):

    print("Getting "+numerator+" Chi2s")

    gridpoints = Gridpoints(m1_bins, m2_bins)
    SR_mask = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6

    n_bins_region = {"SR": np.sum(SR_mask),
                     "CR": np.sum(np.logical_not(SR_mask))}

    #define chi2 dictionaries
    chi2s = {}
    for year in observed[numerator].keys():
        chi2s[year] = {}
        for order in predicted[year].keys():
            chi2s[year][order] = {}
            for region in observed[numerator][year].keys():
                chi2s[year][order][region] = []

    for sample in range(N_samples):
        # Poisson sample the observed dataset
        temp_observed, temp_observed_uncs = {numerator: {}}, {numerator: {}}
        for year in observed[numerator].keys():
            temp_observed[numerator][year] = {}
            temp_observed_uncs[numerator][year] = {}
            for region in observed[numerator][year].keys():
                    temp_observed[numerator][year][region] = np.random.poisson(observed[numerator][year][region])
                    temp_observed_uncs[numerator][year][region] = np.sqrt(temp_observed[numerator][year][region])

        pulls = Get_Pulls(numerator, temp_observed, temp_observed_uncs, predicted, predicted_uncs, m1_bins, m2_bins)

        for year in chi2s.keys():
            for order in chi2s[year].keys():
                chi2s[year][order]["CR"].append(np.sum(np.square(pulls[year][order][region].flatten()[np.logical_not(SR_mask)])))
                chi2s[year][order]["SR"].append(np.sum(np.square(pulls[year][order][region].flatten()[SR_mask])))

    if type(plot) == str:

        plotdir = plot + "Chi2s/"
        Check_Location(plotdir)

        for year in chi2s.keys():
            for order in chi2s[year].keys():
                plotdir_temp = plotdir + order + "/"
                Check_Location(plotdir_temp)

                for region in chi2s[year][order].keys():

                    ndof = n_bins_region[region] + (int(order)+1)*(int(order)+2)/2 - 1
                    chi2s[year][order][region] = np.array(chi2s[year][order][region])/ndof
                    mean, mean_SE = skplt.get_mean(chi2s[year][order][region])
                    std, std_SE = skplt.get_std(chi2s[year][order][region])

                    # 1D plot
                    histplot = skplt.HistogramPlot(bins = skplt.get_bins(0,4,30), xlabel = r"$\chi^2$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
                    histplot.Add(chi2s[year][order][region], label = "Order-"+str(order))
                    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.52, ypos = 0.78)
                    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.52, ypos = 0.73)
                    histplot.Plot(plotdir_temp + "/" + year+"_"+region)

    return chi2s

def Single_Bin(numerator, obs_counts, obs_counts_uncs, pred_counts, pred_counts_uncs):

    for year in obs_counts["2b2j"].keys():
        print("-------------------------------------------------------------------")
        print("Year: "+year)

        obs_total, obs_unc = 0,0
        for obs_bin in obs_counts[numerator][year]["SR"].flatten():
            if np.isfinite(obs_bin):
                obs_total += obs_bin
                obs_unc = np.sqrt(obs_unc**2 + np.sqrt(obs_bin)**2)

        print("Observed "+numerator+" SR events: "+str(round(obs_total))+"+-"+str(round(obs_unc)))
        pred_total, pred_unc = 0,0
        for order in pred_counts[year].keys():
            for pred_bin in pred_counts[year][order]["SR"].flatten():
                if np.isfinite(pred_bin):
                    pred_total += pred_bin
            for pred_unc_bin in pred_counts_uncs[year][order]["SR"].flatten():
                if np.isfinite(pred_unc_bin):
                    pred_unc = np.sqrt((pred_unc)**2+(pred_unc_bin)**2)

            print("Predicted "+numerator+" SR events from order-"+order+" polynomial : "+str(round(pred_total))+"+-"+str(round(pred_unc)))

#############

        obs_total, obs_unc = 0,0
        for obs_bin in obs_counts[numerator][year]["CR"].flatten():
            if np.isfinite(obs_bin):
                obs_total += obs_bin
                obs_unc = np.sqrt(obs_unc**2 + np.sqrt(obs_bin)**2)

        print("Observed "+numerator+" CR events: "+str(round(obs_total))+"+-"+str(round(obs_unc)))
        pred_total, pred_unc = 0,0
        for order in pred_counts[year].keys():
            for pred_bin in pred_counts[year][order]["CR"].flatten():
                if np.isfinite(pred_bin):
                    pred_total += pred_bin
                    pred_unc = np.sqrt(pred_unc**2 + np.sqrt(pred_bin)**2)

            print("Predicted "+numerator+" CR events from order-"+order+" polynomial : "+str(round(pred_total))+"+-"+str(round(pred_unc)))


def Get_GP_Predictions(gp_filebase, gp_files, m1_bins, m2_bins):

    hists_GPs = {"prediction": {},
                 "error_up": {},
                 "error_down": {}}

    for year in gp_files.keys():

        hists_GPs["prediction"][year] = np.histogram2d([], [], bins = [m1_bins[0], m2_bins[0]])[0]
        hists_GPs["error_up"][year] = np.histogram2d([], [], bins = [m1_bins[0], m2_bins[0]])[0]
        hists_GPs["error_down"][year] = np.histogram2d([], [], bins = [m1_bins[0], m2_bins[0]])[0]

        if type(gp_files[year]) == str:
            files_arr = [gp_files[year]]
        else:
            files_arr = gp_files[year]
        for year_file in files_arr:
            file = gp_filebase + year_file

            dataset = pq.ParquetFile(file)
            for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size = 10_000), desc = file, leave = False)):                

                    hists_GPs["prediction"][year] += np.histogram2d(np.array(batch["m_h1"]),
                                                                    np.array(batch["m_h2"]),
                                                                    bins = [m1_bins[0], m2_bins[0]],
                                                                    weights = np.array(batch["weight"]))[0]

                    hists_GPs["error_up"][year] += np.histogram2d(np.array(batch["m_h1"]),
                                                                  np.array(batch["m_h2"]),
                                                                  bins = [m1_bins[0], m2_bins[0]],
                                                                  weights = np.array(batch["weight"]) - np.array(batch["weight_GP_up"]))[0]

                    hists_GPs["error_down"][year] += np.histogram2d(np.array(batch["m_h1"]),
                                                                    np.array(batch["m_h2"]),
                                                                    bins = [m1_bins[0], m2_bins[0]],
                                                                    weights = - np.array(batch["weight"]) + np.array(batch["weight_GP_down"]))[0]

    return hists_GPs


def Single_Bin_GP(GP_pred, m1_bins, m2_bins):

    gridpoints = Gridpoints(m1_bins, m2_bins)
    SR_bool = X_HH(gridpoints[:,0],gridpoints[:,1]) < 1.6
    xhh_mask = edge_bool(m1_bins[0], m2_bins[0]).flatten()

    region_mask_CR = np.logical_not(np.logical_or(SR_bool, xhh_mask))
    region_mask_SR = np.logical_not(np.logical_or(np.logical_not(SR_bool), xhh_mask))

    for year in GP_pred["prediction"].keys():
        print("-------------------------------------------------------------------")
        print("Year: "+year)
        
        pred_arr = GP_pred["prediction"][year].flatten()[region_mask_SR]
        up_arr = GP_pred["error_up"][year].flatten()[region_mask_SR]
        down_arr = GP_pred["error_down"][year].flatten()[region_mask_SR]

        print("Predicted 4b SR events from GPs : "+str(round(np.sum(pred_arr)))+"+-"+str(round(np.sum(up_arr))))

        pred_arr = GP_pred["prediction"][year].flatten()[region_mask_CR]
        up_arr = GP_pred["error_up"][year].flatten()[region_mask_CR]
        down_arr = GP_pred["error_down"][year].flatten()[region_mask_CR]

        print("Predicted 4b CR events from GPs : "+str(round(np.sum(pred_arr)))+"+-"+str(round(np.sum(up_arr))))


def Get_Slices_Ratio(numerator, mh_bin, m1_bins, m2_bins, ratio_hists, ratio_hists_uncs, polynomial_weights, polynomial_weights_uncs, plot):

    # m1 diff plot
    plotdir = plot+"MassPlane_Diff/"
    Check_Location(plotdir)
    plotdir = plot+"MassPlane_Diff/m1/"
    Check_Location(plotdir)

    n_m1 = m1_bins[1].shape[0]
    gridpoints = Gridpoints(m1_bins, m2_bins)
    mh_value = round(gridpoints[mh_bin*n_m1:mh_bin*n_m1 + n_m1][0,0])

    for year in ratio_hists.keys():
        plotdir_temp = plotdir + year+"/"
        Check_Location(plotdir_temp)

            
        lineplot = skplt.LinePlot(xs = m1_bins[1], xlabel = r"$m_{H1}", ylabel = numerator+"/2b2j", ratio = True, residual = True)

        for region in ratio_hists[year].keys():
            lineplot.Add(ratio_hists[year][region][mh_bin*n_m1:mh_bin*n_m1 + n_m1], uncs = ratio_hists_uncs[year][region][mh_bin*n_m1:mh_bin*n_m1 + n_m1],
                        label = region, linewidth = 0, addthis = False, plotunc = True, marker_size = 5, unctype = "bars")

        for order in polynomial_weights[year].keys():
            lineplot.Add(polynomial_weights[year][order]["nom"][mh_bin*n_m1:mh_bin*n_m1 + n_m1], uncs = polynomial_weights_uncs[year][order][mh_bin*n_m1:mh_bin*n_m1 + n_m1],
                         label = "Order-"+order, plotunc = False, marker_size = 0)

        ref_line = ratio_hists[year]["CR"][mh_bin*n_m1:mh_bin*n_m1 + n_m1]
        ref_line[np.logical_not(np.isfinite(ref_line))] = ratio_hists[year]["SR"][mh_bin*n_m1:mh_bin*n_m1 + n_m1][np.logical_not(np.isfinite(ref_line))]

        ref_line_uncs = ratio_hists_uncs[year]["CR"][mh_bin*n_m1:mh_bin*n_m1 + n_m1]
        ref_line_uncs[np.logical_not(np.isfinite(ref_line_uncs))] = ratio_hists_uncs[year]["SR"][mh_bin*n_m1:mh_bin*n_m1 + n_m1][np.logical_not(np.isfinite(ref_line_uncs))]

        lineplot.Text(r"$m_{h2}$ =" + str(mh_value), xpos = 0.15, ypos = 0.82)
        lineplot.Add(ref_line,  uncs = ref_line_uncs, label = "", linewidth = 0, marker_size = 00, reference = True)
        lineplot.Plot(plotdir_temp + "/" + str(mh_value))

    # m2 diff plot
    plotdir = plot+"MassPlane_Diff/m2/"
    Check_Location(plotdir)

    n_m2 = m2_bins[1].shape[0]
    mh_value = round(gridpoints[mh_bin::n_m2][0,0])

    for year in ratio_hists.keys():
        plotdir_temp = plotdir + year+"/"
        Check_Location(plotdir_temp)

        lineplot = skplt.LinePlot(xs = m1_bins[1], xlabel = r"$m_{H1}", ylabel = numerator+"/2b2j", ratio = True, residual = True)

        for region in ratio_hists[year].keys():
            lineplot.Add(ratio_hists[year][region][mh_bin::n_m2], uncs = ratio_hists_uncs[year][region][mh_bin::n_m2],
                        label = region, linewidth = 0, addthis = False, plotunc = True, marker_size = 5, unctype = "bars")

        for order in polynomial_weights[year].keys():
            lineplot.Add(polynomial_weights[year][order]["nom"][mh_bin::n_m2], uncs = polynomial_weights_uncs[year][order][mh_bin::n_m2],
                         label = "Order-"+order, plotunc = False, marker_size = 0)

        ref_line = ratio_hists[year]["CR"][mh_bin::n_m2]
        ref_line[np.logical_not(np.isfinite(ref_line))] = ratio_hists[year]["SR"][mh_bin::n_m2][np.logical_not(np.isfinite(ref_line))]

        ref_line_uncs = ratio_hists_uncs[year]["CR"][mh_bin::n_m2]
        ref_line_uncs[np.logical_not(np.isfinite(ref_line_uncs))] = ratio_hists_uncs[year]["SR"][mh_bin::n_m2][np.logical_not(np.isfinite(ref_line_uncs))]

        lineplot.Text(r"$m_{h2}$ =" + str(mh_value), xpos = 0.15, ypos = 0.82)
        lineplot.Add(ref_line,  uncs = ref_line_uncs, label = "", linewidth = 0, marker_size = 00, reference = True)
        lineplot.Plot(plotdir_temp + "/" + str(mh_value))



#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


# This code will read the reduced data files from Parquet_Maker
# and plot relevant info wrt polynomial background estimate using 
# the skink plotting code
if __name__ == "__main__":


    polyfit_filepath = "/home/epp/phubsg/Postgrad/HH_BBBB/background_estimate/background_estimate/polyfits_info/"

    sample_filedir = "2b_v7/"

    year_files = {
                    "16": "combined_skim_data16__Nominal.parquet",
                    "17": "combined_skim_data17__Nominal.parquet",
                    "18": "combined_skim_data18__Nominal.parquet",
                    "22": "combined_skim_data22__Nominal.parquet",
                    "23": "combined_skim_data23__Nominal.parquet",        
                    "run2": ["combined_skim_data16__Nominal.parquet",
                             "combined_skim_data17__Nominal.parquet",
                             "combined_skim_data18__Nominal.parquet"],
                    "run3": ["combined_skim_data22__Nominal.parquet",
                             "combined_skim_data23__Nominal.parquet"]
                    }

    gp_filebase = "/home/epp/phubsg/Postgrad/HH_BBBB/eigenfuncs_4b_bkg/GP_samples/"

    gp_files = {
                    "16": "df_gp_pipe_16_target_16_20xGP.parquet",
                    "17": "df_gp_pipe_17_target_17_20xGP.parquet",
                    "18": "df_gp_pipe_18_target_18_20xGP.parquet",
                    "22": "df_gp_pipe_22_target_22_20xGP.parquet",
                    "23": "df_gp_pipe_23_target_23_20xGP.parquet",        
                    "run2": ["df_gp_pipe_16_target_16_20xGP.parquet",
                             "df_gp_pipe_17_target_17_20xGP.parquet",
                             "df_gp_pipe_18_target_18_20xGP.parquet"],
                    "run3": ["df_gp_pipe_22_target_22_20xGP.parquet",
                             "df_gp_pipe_23_target_23_20xGP.parquet"]
                    }

    nj_tags = ["2b2j", "3b1j", "4b"]

    poly_orders = [2]

    plot_dir = "plots/Massplane_Estimate_Validation/"

    m1_bins, m2_bins = skplt.get_bins(80,180,40),skplt.get_bins(70,170,40)

    # Get datasets
    datasets = Get_Data(nj_tags, sample_filedir, year_files)

    hists_counts, hists_counts_uncs = Get_MassPlane_Bins(m1_bins, m2_bins, datasets, plot = plot_dir)


    # Get the ratios for 3b/2b & 4b/2b
    # ratios_3b2b, ratios_3b2b_uncs = Get_Ratio("3b1j", hists_counts, m1_bins, m2_bins, plot = plot_dir)
    # ratios_4b2b, ratios_4b2b_uncs = Get_Ratio("4b", hists_counts, m1_bins, m2_bins, plot = plot_dir)

    # Get the polynomial weights for each bin centre (For plotting) and event (for prediction calculations)
    # and get the variations with the uncertainties
    bin_weights_3b2b, bin_weights_3b2b_uncs = Get_Bin_Weights("3b1j", m1_bins, m2_bins, poly_orders, polyfit_filepath, year_files, plot = plot_dir)
    # bin_weights_4b2b, bin_weights_4b_2b_uncs = Get_Bin_Weights("4b", m1_bins, m2_bins, poly_orders, polyfit_filepath, year_files, plot = plot_dir)

    # Get The Polynomial Nominal Predictions for all Years + Orders and save
    # as histogram of predictions + Uncertainties (from variation)
    # Also Plot The Prediction & Polynomial values in the MassPlane (Counts + Ratio)
    predictions_3b2b, predictions_3b2b_uncs = Get_Polynomial_Prediction("3b1j", hists_counts, bin_weights_3b2b, m1_bins, m2_bins, plot = plot_dir)
    # predictions_4b2b, predictions_4b2b_uncs = Get_Polynomial_Prediction("4b", hists_counts, bin_weights_4b2b, m1_bins, m2_bins, plot = plot_dir)

    GP_4b_prediction = Get_GP_Predictions(gp_filebase, gp_files, m1_bins, m2_bins)

    # Get The pulls for each bin between observed counts 
    # Plot Pulls 
    # pulls_counts_3b2b = Get_Pulls("3b1j", hists_counts, hists_counts_uncs, predictions_3b2b, predictions_3b2b_uncs, m1_bins, m2_bins, plot = plot_dir)
    # pulls_counts_4b2b = Get_Pulls("4b", hists_counts, hists_counts_uncs, predictions_4b2b, predictions_4b2b_uncs, m1_bins, m2_bins,  plot = plot_dir)

    # Get Chi2s
    chi2s_3b2b = Get_Chi2s("3b1j", hists_counts, hists_counts_uncs, predictions_3b2b, predictions_3b2b_uncs, m1_bins, m2_bins, 1000, plot = plot_dir)

    # Do The Unbinned non-closure test
    Single_Bin("3b1j", hists_counts, hists_counts_uncs, predictions_3b2b, predictions_3b2b_uncs)
    # Single_Bin("4b", hists_counts, hists_counts_uncs, predictions_4b2b, predictions_4b2b_uncs)
    Single_Bin_GP(GP_4b_prediction, m1_bins, m2_bins)

    # Do The Differential Plots with slices in m1 & m2
    # Slowest Plots To Make So last one
    for mh_value in range(m1_bins[1].shape[0]):
        Get_Slices_Ratio("3b1j", mh_value, m1_bins, m2_bins, ratios_3b2b, ratios_3b2b_uncs, bin_weights_3b2b, bin_weights_3b2b_uncs, plot_dir)
    # Get_Slices_Counts()

    # def Get_Slices_Ratio(numerator, mh_bin, m1_bins, m2_bins, ratio_hists, ratio_hists_uncs, polynomial_weights, plot):

    print("DONE!!!")