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

            hists_counts[ntag][year]["SR"] = np.histogram2d(datasets[ntag][year]["m_h1"][SR_bool],
                                                            datasets[ntag][year]["m_h2"][SR_bool],
                                                            bins = [m1_bins[0], m2_bins[0]])[0]
            hists_counts_uncs[ntag][year]["SR"] = np.sqrt(hists_counts[ntag][year]["SR"])

            hists_counts[ntag][year]["CR"] = np.histogram2d(datasets[ntag][year]["m_h1"][np.logical_not(SR_bool)],
                                                            datasets[ntag][year]["m_h2"][np.logical_not(SR_bool)],
                                                            bins = [m1_bins[0], m2_bins[0]])[0]
            hists_counts_uncs[ntag][year]["CR"] = np.sqrt(hists_counts[ntag][year]["CR"])

    #Plot Hists
    if type(plot) == str:
        Check_Location(plot)
        plotdir = plot + "MassPlane/"
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

    return hists_counts, hists_counts_uncs


def Get_Ratio(numerator, hists_counts, m1_bins, m2_bins, plot = None):

    print("Getting "+numerator+"/2b2j")

    hists_ratio = {}
    hists_ratio_uncs = {}
    validbools = {}
    for year in hists_counts[numerator].keys():
        hists_ratio[year] = {}
        hists_ratio_uncs[year] = {}
        validbools[year] = {}
        for region in hists_counts[numerator][year].keys():
            temp_ratio = skplt.CalcNProp("/", [hists_counts[numerator][year][region].flatten(), np.sqrt(hists_counts[numerator][year][region].flatten())],
                                              [hists_counts["2b2j"][year][region].flatten(), np.sqrt(hists_counts["2b2j"][year][region].flatten())])
            hists_ratio[year][region] = temp_ratio[0]
            hists_ratio_uncs[year][region] = temp_ratio[1]
            validbools[year][region] = np.logical_and(hists_ratio[year][region] > 0, np.isfinite(hists_ratio[year][region]))
            validbools[year][region] = np.logical_and(np.logical_and(hists_ratio_uncs[year][region] > 0, np.isfinite(hists_ratio_uncs[year][region])),
                                                      validbools[year][region])


    if type(plot) == str:
        Check_Location(plot)
        plotdir = plot + "Ratios/"
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

    return hists_ratio, hists_ratio_uncs, validbools


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
            for var in range(int((order+1)*(order+2)*(1/2))):
                polynomial_weights[year][str(order)]["vars"]["var"+str(var)] = var_weights[var]

    return polynomial_weights


def Get_Polynomial_Prediction(datasets, numerator, hists_counts, polynomial_weights, m1_bins, m2_bins, plot = None):

    print("Getting "+numerator+" predictions")

    hists_predictions = {}
    hists_predictions_uncs = {}
    for year in polynomial_weights.keys():
        hists_predictions[year] = {}
        hists_predictions_uncs[year] = {}     
        for order in polynomial_weights[year].keys():
            hists_predictions[year][order] = {}     
            hists_predictions_uncs[year][order] = {}     
                
            SR_bool = datasets["2b2j"][year]["SR"]

            hists_predictions[year][order]["SR"] = np.histogram2d(datasets["2b2j"][year]["m_h1"][SR_bool],
                                                                  datasets["2b2j"][year]["m_h2"][SR_bool],
                                                                  bins = [m1_bins[0], m2_bins[0]],
                                                                  weights = polynomial_weights[year][order]["nom"][SR_bool])[0]

            hists_predictions[year][order]["CR"] = np.histogram2d(datasets["2b2j"][year]["m_h1"][np.logical_not(SR_bool)],
                                                                  datasets["2b2j"][year]["m_h2"][np.logical_not(SR_bool)],
                                                                  bins = [m1_bins[0], m2_bins[0]],
                                                                  weights = polynomial_weights[year][order]["nom"][np.logical_not(SR_bool)])[0]

            # Calc uncs
            temp_unc_hist2d = np.sqrt(hists_counts["2b2j"][year]["SR"])
            for var in polynomial_weights[year][order]["vars"].keys():

                temp_var_hist = np.histogram2d(datasets["2b2j"][year]["m_h1"][SR_bool],
                                               datasets["2b2j"][year]["m_h2"][SR_bool],
                                               bins = [m1_bins[0], m2_bins[0]],
                                               weights = polynomial_weights[year][order]["vars"][var][SR_bool])[0]

                temp_unc_hist2d = np.sqrt(temp_unc_hist2d**2 + (hists_predictions[year][order]["SR"] - temp_var_hist)**2)

            hists_predictions_uncs[year][order]["SR"] = temp_unc_hist2d


            temp_unc_hist2d = np.sqrt(hists_counts["2b2j"][year]["CR"])
            for var in polynomial_weights[year][order]["vars"].keys():

                temp_var_hist = np.histogram2d(datasets["2b2j"][year]["m_h1"][np.logical_not(SR_bool)],
                                               datasets["2b2j"][year]["m_h2"][np.logical_not(SR_bool)],
                                               bins = [m1_bins[0], m2_bins[0]],
                                               weights = polynomial_weights[year][order]["vars"][var][np.logical_not(SR_bool)])[0]

                temp_unc_hist2d = np.sqrt(temp_unc_hist2d**2 + (hists_predictions[year][order]["CR"] - temp_var_hist)**2)

            hists_predictions_uncs[year][order]["CR"] = temp_unc_hist2d

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


def Get_Pulls(numerator, observed, observed_uncs, predicted, predicted_uncs, plot = False):

    print("Getting "+numerator+" Pulls")

    pulls = {}
    for year in observed[numerator].keys():
        pulls[year] = {}
        for order in predicted[year].keys():
            pulls[year][str(order)] = {}

            for region in ["SR", "CR"]:
                pulls_temp = skplt.CalcNProp("-", [observed[numerator][year][region], observed_uncs[numerator][year][region]],
                                                  [predicted[year][str(order)][region], predicted_uncs[year][str(order)][region]])
                pulls[year][str(order)][region] = pulls_temp[0]/pulls_temp[1]

    if type(plot) == str:
                
        # 2D plot
        Check_Location(plot)
        plotdir = plot + "Pulls_2D/"
        Check_Location(plotdir)
        plotdir = plotdir + "_"+numerator+"_2b2j/"
        Check_Location(plotdir)
        
        for year in pulls.keys():
            for order in pulls[year].keys():
                for region in pulls[year][order].keys():

                    plotdir = plotdir + "Order-"+str(order)+"/"
                    Check_Location(plotdir)

                    # 2D plot
                    histplot2d = skplt.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                            xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
                    histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = pulls[year][order][region].flatten(),
                                zlabel = numerator+" count predictions from order-"+str(order)+" polynomial , " + year)
                    Contour_Plot(xhh_curve, m1_bins[0], m2_bins[0])
                    histplot2d.Plot(plotdir + year)

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

                    plotdir = plotdir + "Order-"+str(order)+"/"
                    Check_Location(plotdir)

                    mean, mean_SE = skplt.get_mean(pulls[year][order][region].flatten())
                    std, std_SE = skplt.get_std(pulls[year][order][region].flatten())
                    chi = np.sum(np.square(pulls[year][order][region].flatten()))
                    n_region_bins = shape(pulls[year][order][region].flatten())[0]

                    # 1D plot
                    histplot = skplt.HistogramPlot(bins = bins, xlabel = r"$(Obs_{3b} - Pred_{3b})/(\sigma_{obs - Pred})$", ylabel = "No. " + region + " Bins", plot_unc = False, density = False)
                    histplt.Add(pulls[year][order][region].flatten(), label = "Order-"+str(Ord))
                    histplot.Text(r"$\mu$ = "+str(round(mean, 2))+r" $\pm$ "+str(round(mean_SE,4)), xpos = 0.52, ypos = 0.78)
                    histplot.Text(r"$\sigma$ = "+str(round(std, 2))+r" $\pm$ "+str(round(std_SE,4)), xpos = 0.52, ypos = 0.73)
                    histplot.Text(r"$\chi^2/n_{df} = $"+str(round(chi))+"/"+str(n_region_bins), xpos = 0.15, ypos = 0.82)
                    histplot.Plot(plot_dir + "/" + region)

    return pulls

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
                    "23": "combined_skim_data23__Nominal.parquet"
                    }

    nj_tags = ["2b2j", "3b1j", "4b"]

    poly_orders = [2]

    plot_dir = "plots/Massplane_Estimate_Validation/"

    # Get datasets
    datasets = Get_Data(nj_tags, sample_filedir, year_files)

    m1_bins, m2_bins = skplt.get_bins(80,180,30),skplt.get_bins(70,170,30)
    hists_counts, hists_counts_uncs = Get_MassPlane_Bins(m1_bins, m2_bins, datasets, plot = plot_dir)

    # Get the ratios for 3b/2b & 4b/2b
    ratios_3b2b, ratios_3b2b_uncs, ratios_3b2b_valid = Get_Ratio("3b1j", hists_counts, m1_bins, m2_bins, plot = plot_dir)
    ratios_4b2b, ratios_4b2b_uncs, ratios_4b2b_valid = Get_Ratio("4b", hists_counts, m1_bins, m2_bins, plot = plot_dir)

    # Get the polynomial weights for each bin centre (For plotting) and event (for prediction calculations)
    # and get the variations with the uncertainties
    event_weights_3b2b = Get_Polynomial_Weights(datasets, "3b1j", poly_orders, polyfit_filepath)
    event_weights_4b2b = Get_Polynomial_Weights(datasets, "4b", poly_orders, polyfit_filepath)

    # Get The Polynomial Nominal Predictions for all Years + Orders and save
    # as histogram of predictions + Uncertainties (from variation)
    # Also Plot The Prediction & Polynomial values in the MassPlane (Counts + Ratio)
    predictions_3b2b, predictions_3b2b_uncs = Get_Polynomial_Prediction(datasets, "3b1j", hists_counts, event_weights_3b2b, m1_bins, m2_bins, plot = plot_dir)
    predictions_4b2b, predictions_4b2b_uncs = Get_Polynomial_Prediction(datasets, "4b", hists_counts, event_weights_4b2b, m1_bins, m2_bins, plot = plot_dir)

    # Get The pulls for each bin between observed counts 
    # Plot Pulls 
    pulls_counts_3b2b = Get_Pulls("3b1j", hists_counts, hists_counts_uncs, predictions_3b2b, predictions_3b2b_uncs, plot = plot_dir)
    pulls_counts_4b2b = Get_Pulls("4b", hists_counts, hists_counts_uncs, predictions_4b2b, predictions_4b2b_uncs, plot = plot_dir)

    # Do The Differential Plots with slices in m1 & m2
    # Slowest Plots To Make So last one
    Get_Slices_Ratio()
    Get_Slices_Counts()

    # Do The Unbinned non-closure test
    Single_Bin()



    print("DONE!!!")