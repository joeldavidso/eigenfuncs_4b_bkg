import numpy as np
import pyarrow.parquet as pq


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

    # Loop over the year files and save the m1,m2,SR and polynomial info for each event
    datasets = {}
    
    for NJ in nj_tags:
        datasets[NJ] = {}
        for year in year_files.keys():

            temp_arrs = Read_File(sample_filedir+year_files[year])
            datasets[NJ][year] = {"m_h1": temp_arrs["m_h1"],
                                  "m_h2": temp_arrs["m_h2"],
                                  "SR": temp_arrs["SR"]}

    return datasets

def Read_File(filepaths, NJ):

    arrays = {"m_h1": [],
              "m_h2": [],
              "SR": []}

    dataset = pq.ParquetFile(file[:-8] + "_reduced_"+NJ+".parquet")

    for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size=10_000), desc=filepath, leave = False)):

        for key in arrays.keys():
            arrays[key].append(list(batch[key].to_numpy))

    return arrays


def Get_MassPlane_Bins(m1_bins, m2_bins, datasets):
    # Get The 2D MassPlane Binnings used for plotting/calculating everything
    # Also Plot MassPlane (Ratio + Counts) on 2D Histograms
    # Include SR Edge Bin Removal

    #TO DO:
    # - validbool
    #  - SR (XHH)
    # - plot hists
    # - return all hists2d

    gridpoints = Gridpoints(m1_bins, m2_bins)

    hists_counts = {}

    # Get Hists
    for ntag in datasets.keys():
        hists_counts[ntag] = {}
        for year in datasets[ntag].keys():
            hists_counts[ntag][year] = {}

            SR_bool = datasets[ntag][year]["SR"]

            hists_counts[ntag][year]["SR"] = np.histogram2d(datasets[ntag][year]["m_h1"][SR_bool],
                                                              datasets[ntag][year]["m_h2"][SR_bool],
                                                              bins = [m1_bins[0], m2_bins[0]])[0]

            hists_counts[ntag][year]["CR"] = np.histogram2d(datasets[ntag][year]["m_h1"][np.logical_not(SR_bool)],
                                                              datasets[ntag][year]["m_h2"][np.logical_not(SR_bool)],
                                                              bins = [m1_bins[0], m2_bins[0]])[0]

    #Plot Hists




    return hists_counts


def Get_Ratio(numerator, denominator, hists_counts):

    hists_ratio = {}
    for year in hists_counts[numerator].keys():
        hists_ratio[year] = {}
        for region in hists_counts[numerator][year].keys():

            hists_ratio[year][region] = skplt.CalcNProp("/", [hists_counts[numerator][year][region].flatten(), np.sqrt(hists_counts[numerator][year][region].flatten())],
                                                             [hists_counts[denominator][year][region].flatten(), np.sqrt(hists_counts[denominator][year][region].flatten())])

    return hists_ratio, hists_ratio_uncs

def Single_Bin():
    # Implement single bin code HERE
    # No need to return anything, just print and plot


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

    # Get datasets
    datasets = Get_Data(nj_tags, sample_filedir, year_files)

    m1_bins, m2_bins = skplt.get_bins(80,180,30),skplt.get_bins(70,170,30)
    hists_2b = Get_MassPlane_Bins(m1_bins, m2_bins, datasets)

    # Get The Polynomial Nominal Predictions for all Years + Orders and save
    # as histogram of predictions + Uncertainties (from variation)
    # Also Plot The Prediction & Polynomial values in the MassPlane (Counts + Ratio)
    Get_Polynomial_Prediction()

    # Get The pulls for each bin between observed counts 
    # Plot Pulls 
    pulls = Get_Pulls()

    # Get The Chi2 from all bins
    # Plot Chi2s
    Get_Chi2(pulls)

    # Do The Differential Plots with slices in m1 & m2
    # Slowest Plots To Make So last one
    Get_Slices()

    # Do The Unbinned non-closure test
    Single_Bin()