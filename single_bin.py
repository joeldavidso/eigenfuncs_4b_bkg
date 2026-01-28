import pyarrow.parquet as pa
import pyarrow.dataset as pd
import pyarrow as py
import numpy as np
import math
import skink as skplt
from tqdm import tqdm
import os
import functools as ft

def X_HH(mh1, mh2):

    R = ((mh1-124)/(0.1*mh1))**2 + ((mh2-117)/(0.1*mh2))**2

    return np.sqrt(R)

def R_CR(mh1, mh2):

    R = (mh1 - 1.05*124)**2 + (mh2 - 1.05*117)**2

    return np.sqrt(R)

def polynomial_2D(xy, a, b, c, d, e, f):
    x = xy[:,0]
    y = xy[:,1]
    # 2D polynomial of degree 2
    return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

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


def get3b2b(run, path):
    # run can be "run2" or "run3"
    filepath = path + run + "_3b1j_2b2j__Order-2"

    means, cov = Read(filepath + "/means", filepath + "/covs")

    nom_func, var_funcs = poly_eigfs(means, cov)

    return nom_func, var_funcs


def get4b2b(run, path):
    # run can be "run2" or "run3"
    filepath = path + run + "_4b_2b2j__Order-2"

    means, cov = Read(filepath + "/means", filepath + "/covs")

    nom_func, var_funcs = poly_eigfs(means, cov)

    return nom_func, var_funcs



sample_filedir = "2b_v7/"

run = "run2"
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

################################
# Get polynomial fits parameters
# Get functions as weel for mean + avariation functions

polyfit_filepath = "/home/epp/phubsg/Postgrad/HH_BBBB/background_estimate/background_estimate/polyfits_info/WIP"

# nom_func, var_funcs = get3b2b(run, polyfit_filepath)

nom_func, var_funcs = get4b2b(run, polyfit_filepath)

nom_weights = []
var_weights = []
for var in range(len(var_funcs)):
    var_weights.append([])

################################

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

bins = {"m_h1": skplt.get_bins(80,180,60),
        "m_h2": skplt.get_bins(75,170,60),
        "dEta_hh": skplt.get_bins(0,1.5,30),
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
    for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size = 10000))):

        exp += np.array(batch["m_h1"]).size

        # Get Boolean Arrs
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

            # Get polynomial weights for 2b SR massplane events
            
            if ntag == "2b2j":

                mhs = np.array([np.array(batch["m_h1"])[np.logical_and(Nj_bools[ntag], resolved_bool_SR)], np.array(batch["m_h2"])[np.logical_and(Nj_bools[ntag], resolved_bool_SR)]]).T

                temp_nom_weights, temp_var_weights = reweight((nom_func, var_funcs), mhs)

                # nom_weights = np.append(nom_weights, temp_nom_weights)
                nom_weights.append(temp_nom_weights)
                for var in range(len(var_weights)):
                    # var_weights[var] = np.append(var_weights[var], temp_var_weights[var])    
                    var_weights[var].append(temp_var_weights[var])


nom_weights = np.concatenate(nom_weights)
for var in range(len(var_weights)):
    var_weights[var] = np.concatenate(var_weights[var])

#############################
# Pritning counts and uncertainties

# sample_rw = (CRcounts["2b2j"])/(CRcounts["3b1j"])
sample_rw = 1

print("Observed 3b Counts in the SR:  " + str(round(SRcounts["3b1j"])) + " +- "+ str(round(np.sqrt(SRcounts["3b1j"]))))
print("Observed 2b Counts in the SR:  " + str(round(SRcounts["2b2j"])) + " +- "+ str(round(np.sqrt(SRcounts["2b2j"]))))
print("---------------------------------------------------------------")

print("Predicted 3b Counts in the SR (mean):  " + str(round(sample_rw*np.sum(nom_weights))) + " +- "+ str(round(sample_rw*np.sqrt(np.sum(nom_weights**2)))))
for var in range(len(var_weights)):
    print("Predicted 3b Counts in the SR (var"+str(var)+"):  " + str(round(sample_rw*np.sum(var_weights[var]))) + " +- "+ str(round(sample_rw*np.sqrt(np.sum(var_weights[var]**2)))))

uncs = (sample_rw*np.sqrt(np.sum(nom_weights**2)))**2
for var in range(len(var_weights)):
    uncs += (sample_rw*np.sum(nom_weights) - sample_rw*np.sum(var_weights[var]))**2
uncs = np.sqrt(uncs)

print("Overall predicted 3b Counts in the SR (sum_quad):  "+ str(round(sample_rw*np.sum(nom_weights))) + " +- " + str(round(uncs)))
