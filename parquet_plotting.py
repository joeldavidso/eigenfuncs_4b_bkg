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
from scipy.interpolate import interp1d

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

    # XY is a 2xN array where XY[0] is x and XY[1] is y
    x, y = XY[:,0], XY[:,1]

    # c_00 is the first coefficient (constant term)
    result = coefficients[0] 

    if len(coefficients) == 1:
        return result * np.ones_like(x)


    # Current coefficient index, starting after c_00
    coeff_idx = 1
    
    # The max degree is inferred from the number of coefficients
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

def Gridpoints(var1_bins, var2_bins):
    # Gridpoints for 2d mass plot
    gridpoints = []
    for v1 in var1_bins[1]:
        for v2 in var2_bins[1]:
            gridpoints.append([v1,v2])

    return np.array(gridpoints)

def xhh_curve(x, y):
        
    term_x = (x - 124) / (0.1 * x)
    term_y = (y - 117) / (0.1 * y)
    
    return np.sqrt(term_x**2 + term_y**2) - 1.6

def rcr_curve(x, y):

    term_x = x - 124*1.05
    term_y = y - 117*1.05

    return np.sqrt(term_x**2 + term_y**2) - 45

def calculate_cumulative_arc_length(points):
    segment_diffs = np.diff(points, axis=0) 
    segment_lengths = np.sqrt(np.sum(segment_diffs**2, axis=1))
    
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    return cumulative_lengths, np.sum(segment_lengths)

def edge_bool(m1_bins, m2_bins, edge):

    x_min, x_max = 80, 180
    y_min, y_max = 75, 170
    N_grid = 300 # Resolution for the grid

    X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, N_grid),
                                np.linspace(y_min, y_max, N_grid))

    # Calculate the value of F(x, y) for every point in the grid
    if edge == "xhh":
        Z_grid = xhh_curve(X_grid, Y_grid)
    elif edge == "rcr":
        Z_grid = rcr_curve(X_grid, Y_grid)

    fig, ax = plt.subplots()
    contour_set = ax.contour(X_grid, Y_grid, Z_grid, levels=[0], linewidths=0)
    plt.close(fig) # We close the figure since we only need the data, not the plot yet

    if not contour_set.collections:
        print("Error: Could not find the contour F(x, y) = 0 within the defined grid.")
        exit()

    initial_points = contour_set.collections[0].get_paths()[0].vertices
    if not np.allclose(initial_points[0], initial_points[-1]):
        initial_points = np.vstack([initial_points, initial_points[0]])
        
    cumulative_s, total_L = calculate_cumulative_arc_length(initial_points)

    N_uniform = 10000 # Number of uniform points you want
    s_targets = np.linspace(0, total_L, N_uniform, endpoint=False) # Exclude endpoint if loop

    # Interpolate X and Y coordinates as a function of the cumulative arc length 's'
    x_interp = interp1d(cumulative_s, initial_points[:, 0])
    y_interp = interp1d(cumulative_s, initial_points[:, 1])

    # Get the uniformly sampled coordinates
    x_uniform = x_interp(s_targets)
    y_uniform = y_interp(s_targets)

    uniform_points = np.array([x_uniform, y_uniform]).T

    return np.histogram2d(uniform_points[:, 0], uniform_points[:, 1], bins = [m1_bins[0], m2_bins[0]])[0] > 0


sample_filedir = "2b_v7/"

run = "run3"
if run == "run2":
    sample_filenames = ["combined_skim_data16__Nominal.parquet"
                        ,"combined_skim_data17__Nominal.parquet"
                        ,"combined_skim_data18__Nominal.parquet"
                        ]
elif run == "run3":
    sample_filenames = ["combined_skim_data22__Nominal.parquet"
                        ,"combined_skim_data23__Nominal.parquet"
                        ,"combined_skim_data24__Nominal.parquet"
                        ]


plot_dir_base = "plots/parquet/"+run

if not os.path.exists(plot_dir_base):
        os.mkdir(plot_dir_base)

etacut = "revDEta" # either "revDEta" or ""
etacut = ""

cmin = -2
cmax = 2

etacuts = ["", "revDEta"]

for etacut in etacuts:

    # get total number of events in samples

    parquet_files = [sample_filedir+sample_filenames[i] for i in range(len(sample_filenames))]
    datasets = [pa.ParquetFile(file) for file in parquet_files]

    exp = 0
    CRcounts = {"2b2j": 0,
                "3b1j": 0,
                "4b": 0}
    SRcounts = {"2b2j": 0,
                "3b1j": 0,
                "4b": 0}

    ntag_keys = list(CRcounts.keys())

    #Initialize Histgram arrays for filling
    eta_edges = [0,1.5] if etacut != "revDEta" else [1.5,7]

    bins = {"m_h1": skplt.get_bins(80,180,60),
            "m_h2": skplt.get_bins(75,170,60),
            "dEta_hh": skplt.get_bins(eta_edges[0],eta_edges[1],30),
            "eta_h1": skplt.get_bins(-2.5,2.5,30),
            "eta_h2": skplt.get_bins(-2.5,2.5,30),
            "m_hh": skplt.get_bins(0,1000,30),
            "pt_h1": skplt.get_bins(0,600,30),
            "pt_h2": skplt.get_bins(0,600,30),
            "X_hh": skplt.get_bins(0,4,30)}

    m1_m2_histarrs_CR = {"2b2j": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0],
                        "3b1j": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0],
                        "4b": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0]}

    m1_m2_histarrs_SR = {"2b2j": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0],
                        "3b1j": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0],
                        "4b": np.histogram2d([],[],bins = [bins["m_h1"][0], bins["m_h2"][0]])[0]}

    var_histarrs_CR = {}
    for var in bins.keys():
        tempdict = {}
        for ntag in CRcounts.keys():
            tempdict[ntag] = np.histogram([], bins = bins[var][0])[0]
        var_histarrs_CR[var] = tempdict

    var_histarrs_SR = {}
    for var in bins.keys():
        tempdict = {}
        for ntag in SRcounts.keys():
            tempdict[ntag] = np.histogram([], bins = bins[var][0])[0]
        var_histarrs_SR[var] = tempdict

    for dataset in datasets:
        for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size = 5000))):

            exp += np.array(batch["m_h1"]).size

            # Get Boolean Arrs
            if etacut == "revDEta":
                resolved_bool = np.logical_and(np.array(batch["dEta_hh"]) > 1.5, np.array(batch["pass_resolved"]))
            else:
                resolved_bool = np.logical_and(np.array(batch["dEta_hh"]) < 1.5, np.array(batch["pass_resolved"]))


            SR_bool = np.array(batch["X_hh"]) < 1.6
            CR_bool = np.logical_and(R_CR(np.array((batch["m_h1"])),np.array(batch["m_h2"])) < 45, np.array(batch["X_hh"]) > 1.6)

            resolved_bool_CR = np.logical_and(resolved_bool, CR_bool)
            resolved_bool_SR = np.logical_and(resolved_bool, SR_bool)

            Nj_bools = {"2b2j": np.array(batch["ntag"]) == 2,
                        "3b1j": np.array(batch["ntag"]) == 3,
                        "4b": np.array(batch["ntag"]) >= 4}

            # Fill Histogram arrays and get numbers
            for ntag in ntag_keys:

                CRcounts[ntag] += np.sum(np.logical_and(resolved_bool_CR, Nj_bools[ntag]))
                SRcounts[ntag] += np.sum(np.logical_and(resolved_bool_SR, Nj_bools[ntag]))
                m1_m2_histarrs_CR[ntag] += np.histogram2d(np.array(batch["m_h1"])[np.logical_and(Nj_bools[ntag], resolved_bool_CR)],
                                                          np.array(batch["m_h2"])[np.logical_and(Nj_bools[ntag], resolved_bool_CR)],
                                                          bins = [bins["m_h1"][0], bins["m_h2"][0]])[0]
                m1_m2_histarrs_SR[ntag] += np.histogram2d(np.array(batch["m_h1"])[np.logical_and(Nj_bools[ntag], resolved_bool_SR)],
                                                          np.array(batch["m_h2"])[np.logical_and(Nj_bools[ntag], resolved_bool_SR)],
                                                          bins = [bins["m_h1"][0], bins["m_h2"][0]])[0]

                for var in bins.keys():
                    var_histarrs_CR[var][ntag] += np.histogram(np.array(batch[var])[np.logical_and(Nj_bools[ntag], resolved_bool_CR)], bins = bins[var][0])[0]

                    var_histarrs_SR[var][ntag] += np.histogram(np.array(batch[var])[np.logical_and(Nj_bools[ntag], resolved_bool_SR)], bins = bins[var][0])[0]

    # remove edge bins

    # get edge bool masks
    rcr_bool_mask = edge_bool(bins["m_h1"], bins["m_h2"], "rcr")
    xhh_bool_mask = edge_bool(bins["m_h1"], bins["m_h2"], "xhh")

    for ntag in ntag_keys:

        m1_m2_histarrs_CR[ntag][rcr_bool_mask] = 0
        m1_m2_histarrs_CR[ntag][xhh_bool_mask] = 0

        m1_m2_histarrs_SR[ntag][xhh_bool_mask] = 0


    # Plotting Time
    # Var Plotting
    plot_dir_temp = plot_dir_base + "/VarsCR"+etacut

    if not os.path.exists(plot_dir_temp):
            os.mkdir(plot_dir_temp)

    for var in bins.keys():
        histplot = skplt.HistogramPlot(bins = bins[var], xlabel = var, ylabel = "No. Events", density = True, ratio = True)
        histplot.Add(bins[var][1], weights = var_histarrs_CR[var]["2b2j"], 
                    uncs = np.sqrt(var_histarrs_CR[var]["2b2j"]),
                    label = "2b2j", reference = True)
        histplot.Add(bins[var][1], weights = var_histarrs_CR[var]["3b1j"], 
                    uncs = np.sqrt(var_histarrs_CR[var]["3b1j"]),
                    label = "3b1j")
        histplot.Add(bins[var][1], weights = var_histarrs_CR[var]["4b"], 
                    uncs = np.sqrt(var_histarrs_CR[var]["4b"]),
                    label = "4b")
        histplot.Plot(plot_dir_temp + "/"+var)

    #######################################################

    plot_dir_temp = plot_dir_base + "/VarsSR"+etacut

    if not os.path.exists(plot_dir_temp):
            os.mkdir(plot_dir_temp)

    for var in bins.keys():
        histplot = skplt.HistogramPlot(bins = bins[var], xlabel = var, ylabel = "No. Events", density = True, ratio = True)
        histplot.Add(bins[var][1], weights = var_histarrs_SR[var]["2b2j"], 
                    uncs = np.sqrt(var_histarrs_SR[var]["2b2j"]),
                    label = "2b2j",reference = True)
        histplot.Add(bins[var][1], weights = var_histarrs_SR[var]["3b1j"], 
                    uncs = np.sqrt(var_histarrs_SR[var]["3b1j"]),
                    label = "3b1j")
        histplot.Add(bins[var][1], weights = var_histarrs_SR[var]["4b"], 
                    uncs = np.sqrt(var_histarrs_SR[var]["4b"]),
                    label = "4b")
        histplot.Plot(plot_dir_temp + "/"+var)

    # Saving numpy arrays 
    np.save("parquet_hists/" +run+ "/CR_3b1j"+etacut+".npy", m1_m2_histarrs_CR["3b1j"])
    np.save("parquet_hists/" +run+ "/CR_2b2j"+etacut+".npy", m1_m2_histarrs_CR["2b2j"])
    np.save("parquet_hists/" +run+ "/CR_4b"+etacut+".npy", m1_m2_histarrs_CR["4b"])

    np.save("parquet_hists/" +run+ "/SR_3b1j"+etacut+".npy", m1_m2_histarrs_SR["3b1j"])
    np.save("parquet_hists/" +run+ "/SR_2b2j"+etacut+".npy", m1_m2_histarrs_SR["2b2j"])
    np.save("parquet_hists/" +run+ "/SR_4b"+etacut+".npy", m1_m2_histarrs_SR["4b"])

# #     #################################################################################################

#     # Massplane plotting
#     plot_dir_temp = plot_dir_base + "/MassPlaneRatio2b3bCR"+etacut

#     if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)

#     down_ratio = 10

#     # downsample 3b data to 4b-like stats
#     # take poisson distributed counts based on new uncs
#     # m1_m2_histarrs_CR["3b1j"] = np.random.poisson(m1_m2_histarrs_CR["3b1j"]/down_ratio)
#     # m1_m2_histarrs_SR["3b1j"] = np.random.poisson(m1_m2_histarrs_SR["3b1j"]/down_ratio)

#     m1m2_ratio_3b2b_CR = skplt.CalcNProp("/", [m1_m2_histarrs_CR["3b1j"].flatten(), np.sqrt(m1_m2_histarrs_CR["3b1j"].flatten())],
#                                             [m1_m2_histarrs_CR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten())])
#     m1m2_ratio_3b2b_SR = skplt.CalcNProp("/", [m1_m2_histarrs_SR["3b1j"].flatten(), np.sqrt(m1_m2_histarrs_SR["3b1j"].flatten())],
#                                             [m1_m2_histarrs_SR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_SR["2b2j"].flatten())])
#     m1m2_ratio_4b3b_CR = skplt.CalcNProp("/", [m1_m2_histarrs_CR["4b"].flatten(), np.sqrt(m1_m2_histarrs_CR["4b"].flatten())],
#                                         [m1_m2_histarrs_CR["3b1j"].flatten(), np.sqrt(m1_m2_histarrs_CR["3b1j"].flatten())])
#     gridpoints = Gridpoints("m_h1", "m_h2", bins)

#     m1m2_ratio_3b2b_full = skplt.CalcNProp("/", [m1_m2_histarrs_CR["3b1j"].flatten() + m1_m2_histarrs_SR["3b1j"].flatten(),
#                                                  np.sqrt(m1_m2_histarrs_CR["3b1j"].flatten() + m1_m2_histarrs_SR["3b1j"].flatten())],
#                                                 [m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten(),
#                                                  np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten() + m1_m2_histarrs_SR["2b2j"].flatten())])


#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = m1m2_ratio_3b2b_CR[0], zlabel = "n(3b)/n(2b)", cmin = 0.05, cmax = 0.15)
#     histplot2d.Text(r"$\frac{N_{3b}}{N_{2b}}$ = "+str(round(CRcounts["3b1j"]/CRcounts["2b2j"], 2)), xpos = 0.33, ypos = 0.45)
#     histplot2d.Plot(plot_dir_temp + "/Ratio")

#     raise("HI")

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1],
#                 weights = (m1m2_ratio_3b2b_CR[0]- (CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1],
#                 zlabel = "n(3b)/n(2b) pull",
#                 cmin = cmin, cmax = cmax)
#     histplot2d.Plot(plot_dir_temp + "/Pulls_2D")

#     mean = np.mean(((m1m2_ratio_3b2b_CR[0]-(CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1])[np.isfinite((m1m2_ratio_3b2b_CR[0]-(CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1])])
#     std = np.std(((m1m2_ratio_3b2b_CR[0]-(CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1])[np.isfinite((m1m2_ratio_3b2b_CR[0]-(CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1])])

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = "pull (3b/2b)", ylabel = "No. Bins", plot_unc = False, density = True)
#     histplot.Add((m1m2_ratio_3b2b_CR[0]-(CRcounts["3b1j"]/CRcounts["2b2j"]))/m1m2_ratio_3b2b_CR[1], label = "3b/2b")
#     histplot.Text(r"$\mu$ = "+str(round(mean, 2)), xpos = 0.7, ypos = 0.78)
#     histplot.Text(r"$\sigma$ = "+str(round(std, 2)), xpos = 0.7, ypos = 0.73)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D")

#     ##############################################

#     plot_dir_temp = plot_dir_base + "/MassPlaneRatio2b3bSR"+etacut

#     if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = m1m2_ratio_3b2b_SR[0], zlabel = "n(3b)/n(2b)", cmin = 0.05, cmax = 0.15)
#     histplot2d.Text(r"$\frac{N_{3b}}{N_{2b}}$ = "+str(round(SRcounts["3b1j"]/SRcounts["2b2j"], 2)), xpos = 0.33, ypos = 0.7)
#     histplot2d.Plot(plot_dir_temp + "/Ratio")

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1],
#                 weights = (m1m2_ratio_3b2b_SR[0]- (SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1],
#                 zlabel = "n(3b)/n(2b) pulls",
#                 cmin = cmin, cmax = cmax)
#     histplot2d.Plot(plot_dir_temp + "/Pulls_2D")

#     mean = np.mean(((m1m2_ratio_3b2b_SR[0]-(SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1])[np.isfinite((m1m2_ratio_3b2b_SR[0]-(SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1])])
#     std = np.std(((m1m2_ratio_3b2b_SR[0]-(SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1])[np.isfinite((m1m2_ratio_3b2b_SR[0]-(SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1])])

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = "pull (3b/2b)", ylabel = "No. Bins", plot_unc = False, density = True)
#     histplot.Add((m1m2_ratio_3b2b_SR[0]-(SRcounts["3b1j"]/SRcounts["2b2j"]))/m1m2_ratio_3b2b_SR[1], label = "3b/2b")
#     histplot.Text(r"$\mu$ = "+str(round(mean, 2)), xpos = 0.7, ypos = 0.78)
#     histplot.Text(r"$\sigma$ = "+str(round(std, 2)), xpos = 0.7, ypos = 0.73)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D")

#     ##############################################

#     plot_dir_temp = plot_dir_base + "/MassPlaneRatio3b4bCR"+etacut

#     if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = m1m2_ratio_4b3b_CR[0], zlabel = "n(4b)/n(3b)", cmin = 0.01, cmax = 0.2)
#     histplot2d.Text(r"$\frac{N_{4b}}{N_{3b}}$ = "+str(round(CRcounts["4b"]/CRcounts["3b1j"], 2)), xpos = 0.33, ypos = 0.45)
#     histplot2d.Plot(plot_dir_temp + "/Ratio")

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1],
#                 weights = (m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1],
#                 zlabel = "n(4b)/n(3b) pulls",
#                 cmin = cmin, cmax = cmax)
#     histplot2d.Plot(plot_dir_temp + "/Pulls_2D")

#     mean = np.mean(((m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1])[np.isfinite((m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1])])
#     std = np.std(((m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1])[np.isfinite((m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1])])

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = "pull (4b/3b)", ylabel = "No. Bins", plot_unc = False, density = True)
#     histplot.Add((m1m2_ratio_4b3b_CR[0]-(CRcounts["4b"]/CRcounts["3b1j"]))/m1m2_ratio_4b3b_CR[1], label = "4b/3b")
#     histplot.Text(r"$\mu$ = "+str(round(mean, 2)), xpos = 0.7, ypos = 0.78)
#     histplot.Text(r"$\sigma$ = "+str(round(std, 2)), xpos = 0.7, ypos = 0.73)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D")

#     # Lossplane plotting
#     plot_dir_temp = plot_dir_base + "/Losses"+etacut

#     if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)


#     losses_3b_2b = loss_per_bin(m1_m2_histarrs_CR["3b1j"].flatten(), m1_m2_histarrs_CR["2b2j"].flatten(),
#                                 CRcounts["3b1j"], CRcounts["2b2j"]).flatten()

#     histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#     histplot2d.Set(gridpoints[:,0], gridpoints[:,1], weights = losses_3b_2b, zlabel = "loss", cmin = 0.4, cmax = 0.8)
#     histplot2d.Plot(plot_dir_temp + "/3b_2b")

#     validbool = np.isfinite(losses_3b_2b) == False
#     avgloss = np.sum(losses_3b_2b[validbool])/np.sum(validbool)

#     print("Avg. Loss" + str(avgloss))


# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################
# #########################################################################################

#     # function fitting

#     poly_orders = [0,1,2,3]

#     poly_coeff_guesses = [np.ones(int((O+1)*(O+2)/2)) for O in poly_orders]

#     opt_coeffs = []
#     opt_covs = []
#     polys, polys_CR, polys_SR = [],[],[]
#     polys_uncs, polys_uncs_CR, polys_uncs_SR = [],[],[]

#     pulls, pulls_CR, pulls_SR = [],[],[]
#     chi_squareds, chi_squareds_CR, chi_squareds_SR = [],[],[]

#     validbool = np.logical_and(np.isfinite(m1m2_ratio_3b2b_CR[0]),np.isfinite(m1m2_ratio_3b2b_CR[1]))

#     # Fitting polynomial for diff orders
#     for N in tqdm(range(len(poly_orders)), desc = "Order"):

#         temp_coeffs, temp_covs = so.curve_fit(poly2d_fit_func,
#                                               gridpoints[validbool],
#                                               np.array(m1m2_ratio_3b2b_CR[0])[validbool],
#                                               p0 = poly_coeff_guesses[N],
#                                               sigma = np.array(m1m2_ratio_3b2b_CR[1])[validbool])


#         opt_coeffs.append(temp_coeffs)
#         opt_covs.append(temp_covs)

#         polynomial = poly2d_fit_func(gridpoints, *temp_coeffs)

#         poly_CR = polynomial
#         poly_CR[R_CR(gridpoints[:,0], gridpoints[:,1]) > 45] == 0
#         poly_CR[X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6] == 0

#         poly_SR = polynomial
#         poly_SR[X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6] == 0

#         polys.append(polynomial)
#         polys_CR.append(poly_CR)
#         polys_SR.append(poly_SR)

#         #######################################################################################
#         # Get unc on fits
#         # We do this by varying the data according to being poisson distributed and getting per bin standard deviations

#         N_trials = 1000

#         trials_bin_values = []
#         for n in tqdm(range(N_trials), desc = "Calculating Uncertainties"):
#             # Vary the bin counts
#             temp_m1m1_3b_counts = np.random.poisson(m1_m2_histarrs_CR["3b1j"])
#             temp_m1m2_3b_2b_ratio = skplt.CalcNProp("/", [temp_m1m1_3b_counts.flatten(), np.sqrt(temp_m1m1_3b_counts.flatten())],
#                                                          [m1_m2_histarrs_CR["2b2j"].flatten(), np.sqrt(m1_m2_histarrs_CR["2b2j"].flatten())])

#             tempvalidbool = np.logical_and(np.isfinite(temp_m1m2_3b_2b_ratio[0]),np.isfinite(temp_m1m2_3b_2b_ratio[1]))

#             # Get a fit
#             temp_params, tem_covariance = so.curve_fit(poly2d_fit_func,
#                                                        gridpoints[tempvalidbool],
#                                                        np.array(temp_m1m2_3b_2b_ratio[0])[tempvalidbool],
#                                                        p0 = poly_coeff_guesses[N],
#                                                        sigma = np.array(temp_m1m2_3b_2b_ratio[1])[tempvalidbool])

#             # Apply polynomial to each bin 
#             trials_bin_values.append(poly2d_fit_func(gridpoints, *temp_params))

#         # Calculate uncertainties
#         temp_uncs = np.std(np.array(trials_bin_values), axis = 0)

#         ########################################################################################

#         polys_uncs.append(temp_uncs)

#         uncs_CR = temp_uncs
#         uncs_CR[R_CR(gridpoints[:,0], gridpoints[:,1]) > 45] == 0
#         uncs_CR[X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6] == 0
#         polys_uncs_CR.append(uncs_CR)

#         uncs_SR = temp_uncs
#         uncs_SR[X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6] == 0
#         polys_uncs_SR.append(uncs_SR)

#         # Pull calculations
#         pulls_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b_full,
#                                     [polys[N], np.array(polys_uncs[N])])
#         pulls_CR_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b_CR,
#                                         [polys_CR[N], np.array(polys_uncs_CR[N])])
#         pulls_SR_temp = skplt.CalcNProp("-", m1m2_ratio_3b2b_SR,
#                                         [polys_SR[N], np.array(polys_uncs_SR[N])])

#         pulls.append(pulls_temp[0]/pulls_temp[1])
#         pulls_CR.append(pulls_CR_temp[0]/pulls_CR_temp[1])
#         pulls_SR.append(pulls_SR_temp[0]/pulls_SR_temp[1])

#         # Chi Squared calculations
#         chi_squareds.append(np.sum(np.square(pulls[N][np.isfinite(pulls[N])])))
#         chi_squareds_CR.append(np.sum(np.square(pulls_CR[N][np.isfinite(pulls_CR[N])])))
#         chi_squareds_SR.append(np.sum(np.square(pulls_SR[N][np.isfinite(pulls_SR[N])])))

#     CR_bool = np.logical_and(R_CR(gridpoints[:,0], gridpoints[:,1]) < 45, X_HH(gridpoints[:,0], gridpoints[:,1]) > 1.6)
#     SR_bool = X_HH(gridpoints[:,0], gridpoints[:,1]) < 1.6


# ##############################################################################################################################

#     ## Single Order plots

#     plot_dir_temp = plot_dir_base + "/Fitting"+etacut

#     if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)

#     for Ord in range(len(poly_orders)):

#         plot_dir_temp = plot_dir_base + "/Fitting"+etacut+"/order-"+str(poly_orders[Ord])

#         if not os.path.exists(plot_dir_temp):
#             os.mkdir(plot_dir_temp)

#         #######################################################
#         ## pull uncertainty plot

#         uncbins = skplt.get_bins(0,60,40)

#         uncs_ratio = np.divide(m1m2_ratio_3b2b_full[1], np.array(polys_uncs[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_full[1], np.array(polys_uncs[Ord])))]

#         histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_r}{\sigma_P}$", ylabel = "No. Bins", plot_unc = False, density = True)
#         histplot.Add(uncs_ratio, label = r"$\frac{\sigma_r}{\sigma_P}$")
#         histplot.Text(r"$\mu$ = "+str(round(np.mean(uncs_ratio), 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(np.std(uncs_ratio), 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Plot(plot_dir_temp + "/uncs_frac")

#         uncs_ratio = np.divide(m1m2_ratio_3b2b_CR[1], np.array(polys_uncs_CR[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_CR[1], np.array(polys_uncs_CR[Ord])))]

#         histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_r}{\sigma_P}$", ylabel = "No. CR Bins", plot_unc = False, density = True)
#         histplot.Add(uncs_ratio, label = r"$\frac{\sigma_r}{\sigma_P}$")
#         histplot.Text(r"$\mu$ = "+str(round(np.mean(uncs_ratio), 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(np.std(uncs_ratio), 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Plot(plot_dir_temp + "/uncs_frac_CR")

#         uncs_ratio = np.divide(m1m2_ratio_3b2b_SR[1], np.array(polys_uncs_SR[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_SR[1], np.array(polys_uncs_SR[Ord])))]

#         histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_r}{\sigma_P}$", ylabel = "No. SR Bins", plot_unc = False, density = True)
#         histplot.Add(uncs_ratio, label = r"$\frac{\sigma_r}{\sigma_P}$")
#         histplot.Text(r"$\mu$ = "+str(round(np.mean(uncs_ratio), 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(np.std(uncs_ratio), 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Plot(plot_dir_temp + "/uncs_frac_SR")

#         #######################################################
#         # 2D massplane pulls
#         histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#         histplot2d.Set(gridpoints[:,0],gridpoints[:,1],
#                     weights = pulls[Ord],
#                     zlabel = r"$\frac{r_{3b}^{2b} - P}{\sigma_r}$")
#         histplot2d.Plot(plot_dir_temp + "/Pulls_2D")

#         histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#         histplot2d.Set(gridpoints[:,0],gridpoints[:,1],
#                     weights = pulls_SR[Ord],
#                     zlabel = r"$\frac{r_{3b}^{2b} - P}{\sigma_r}$"
#                     )
#         histplot2d.Plot(plot_dir_temp + "/Pulls_2D_SR")

#         histplot2d = skplt.Hist2D(bins["m_h1"], bins["m_h2"], margins = False, xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$", cbar = True)
#         histplot2d.Set(gridpoints[:,0],gridpoints[:,1],
#                     weights = pulls_CR[Ord],
#                     zlabel = r"$\frac{r_{3b}^{2b} - P}{\sigma_r}$",
#                     cmin = cmin, cmax = cmax)
#         histplot2d.Plot(plot_dir_temp + "/Pulls_2D_CR")

#         #######################################################
#         # Pulls_1D Plotting
#         mean = np.mean(pulls[Ord][np.isfinite(pulls[Ord])])
#         std = np.std(pulls[Ord][np.isfinite(pulls[Ord])])
        
#         mean_SR = np.mean(pulls_SR[Ord][np.isfinite(pulls_SR[Ord])])
#         std_SR = np.std(pulls_SR[Ord][np.isfinite(pulls_SR[Ord])])
        
#         mean_CR = np.mean(pulls_CR[Ord][np.isfinite(pulls_CR[Ord])])
#         std_CR = np.std(pulls_CR[Ord][np.isfinite(pulls_CR[Ord])])

#         histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P }{\sigma_{(r - P)}}$", ylabel = "No. Bins", plot_unc = False, density = True)
#         histplot.Add(pulls[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\mu$ = "+str(round(mean, 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(std, 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds[Ord],2)), xpos = 0.7, ypos = 0.68)
#         histplot.Plot(plot_dir_temp + "/Pulls_1D")

#         histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P }{\sigma_{(r - P)}}$", ylabel = "No. SR Bins", plot_unc = False, density = True)
#         histplot.Add(pulls_SR[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\mu$ = "+str(round(mean_SR, 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(std_SR, 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds_SR[Ord],2)), xpos = 0.7, ypos = 0.68)
#         histplot.Plot(plot_dir_temp + "/Pulls_1D_SR")

#         histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P }{\sigma_{(r - P)}}$", ylabel = "No. CR Bins", plot_unc = False, density = True)
#         histplot.Add(pulls_CR[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\mu$ = "+str(round(mean_CR, 2)), xpos = 0.7, ypos = 0.78)
#         histplot.Text(r"$\sigma$ = "+str(round(std_CR, 2)), xpos = 0.7, ypos = 0.73)
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds_CR[Ord],2)), xpos = 0.7, ypos = 0.68)
#         histplot.Plot(plot_dir_temp + "/Pulls_1D_CR")

#         # Differential mh1 mh2 plots

#         mh1_bin = 30
#         mh2_bin = 30

#         lineplot = skplt.LinePlot(xs = bins["m_h1"], xlabel = r"$m_{h1}$", ylabel = r"$r^{3b}_{2b}$", residual = True)
#         lineplot.Add(poly2d_fit_func([bins["m_h1"],bins["m_h2"][mh2_bin]*np.ones_like(bins["m_h2"])], *opt_coeffs[Ord]),
#                      uncs = polys_uncs[mh1_bin*60:mh1_bin*60 + 60], reference = True)
#         lineplot.Add(m1m2_ratio_3b2b_CR[mh1_bin*60:mh1_bin*60 + 60], label = "CR")
#         lineplot.Plot(plot_dir_temp + "/diff_mh2")

# ##############################################################################################################################

#     ## All Order plots
#     plot_dir_temp = plot_dir_base + "/Fitting"+etacut+"/All_Orders"

#     if not os.path.exists(plot_dir_temp):
#         os.mkdir(plot_dir_temp)

#     #######################################################
#     ## pull uncertainty plot

#     xpos = 0.15
#     ypos = 0.82
#     ydiff = 0.05

#     histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_{r}}{\sigma_P}$", ylabel = "No. Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         uncs_ratio = np.divide(m1m2_ratio_3b2b_full[1], np.array(polys_uncs[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_full[1], np.array(polys_uncs[Ord])))]
#         histplot.Add(uncs_ratio, label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2$ = "+str(round(chi_squareds[Ord], 2)), xpos = xpos, ypos = ypos - ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/uncs_frac")

#     histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_{r}}{\sigma_P}$", ylabel = "No. CR Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         uncs_ratio = np.divide(m1m2_ratio_3b2b_CR[1], np.array(polys_uncs_CR[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_CR[1], np.array(polys_uncs_CR[Ord])))]
#         histplot.Add(uncs_ratio, label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2$ = "+str(round(chi_squareds_CR[Ord], 2)), xpos = xpos, ypos = ypos - ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/uncs_frac_CR")

#     histplot = skplt.HistogramPlot(bins = uncbins, xlabel = r"$\frac{\sigma_{r}}{\sigma_P}$", ylabel = "No. SR Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         uncs_ratio = np.divide(m1m2_ratio_3b2b_SR[1], np.array(polys_uncs_SR[Ord]))[np.isfinite(np.divide(m1m2_ratio_3b2b_SR[1], np.array(polys_uncs_SR[Ord])))]
#         histplot.Add(uncs_ratio, label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2$ = "+str(round(chi_squareds_SR[Ord], 2)), xpos = xpos, ypos = ypos - ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/uncs_frac_SR")

#     #######################################################
#     # Pulls_1D Plotting

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P}{\sigma_{(r-P)}}$", ylabel = "No. Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         histplot.Add(pulls[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds[Ord],2)), xpos = xpos, ypos = ypos-ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D")

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P}{\sigma_{(r-P)}}$", ylabel = "No. CR Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         histplot.Add(pulls_CR[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds_CR[Ord],2)), xpos = xpos, ypos = ypos-ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D_CR")

#     histplot = skplt.HistogramPlot(bins = skplt.get_bins(-4,4, 30), xlabel = r"$\frac{r^{3b}_{2b} - P}{\sigma_{(r-P)}}$", ylabel = "No. SR Bins", plot_unc = False, density = True)
#     for Ord in range(len(poly_orders)):
#         histplot.Add(pulls_SR[Ord], label = "Order-"+str(poly_orders[Ord]))
#         histplot.Text(r"$\chi^2 = $"+str(round(chi_squareds_SR[Ord],2)), xpos = xpos, ypos = ypos-ydiff*Ord)
#     histplot.Plot(plot_dir_temp + "/Pulls_1D_SR")


#     # Print relevant numbers

#     print("No. Events in files: " + str(exp))
#     print("No. Events in resolved CR: " + str(CRcounts["2b2j"]+CRcounts["3b1j"]+CRcounts["4b"]))
#     print("No. Events in resolved SR: " + str(SRcounts["2b2j"]+SRcounts["3b1j"]+SRcounts["4b"]))

