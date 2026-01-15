from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import numpy as np

# Dictionary of colours

C_Pallets = {"Pastel": {"Orange" : "#E78AA1",
                        "Blue" : "#11B7E0",
                        "Green" : "#00B689",
                        "Red" : "#DE0C62",
                        "Purple" : "#9213E0",},

             "Bold": {"Orange" : "#ff6600",
                      "Blue" : "#0033cc",
                      "Green" : "#009933",
                      "Red" : "#b80053",
                      "Purple" : "#670178",
                      "Yellow" : "#FFD700",
                      "Cyan" : "#00FFFF",
                      "Lime Green" : "#32CD32"}
            }

                                   ######################################################
##############################################       Misc Plot Functions      ###########################################
                                   ######################################################


# Simple function for creating bin edges and bin centres for use in plotting    
def get_bins(xmin, xmax, nbins):
    
    bin_edges = np.linspace(xmin,xmax,nbins+1)
    bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
    
    return [bin_edges,bin_centres]

# Function for propogating uncertainties when performing opertations
# Inputs in form of lists with 1st element values and second element the uncertainties
def CalcNProp(operation, xs, ys):
    # Available operations:
    # - Additon/Subtraction
    # - Multiplication/Division
    # - Square and Square root

    if xs[0].shape != ys[0].shape:
        raise("xs and ys not same shape!")
    if xs[1].shape != ys[1].shape:
        raise("xs and ys uncs not same shape!")
    if xs[0].shape != xs[1].shape:
        raise("xs and xs uncs not same shape!")


    zs = [0,0]

    if operation == "+" or operation == "-":
        if operation == "+":
            zs[0] = xs[0] + ys[0]
        elif operation == "-":
            zs[0] = xs[0] - ys[0]

        # Z = X + Y -> DZ = sqrt(DX^2 + DY^2)
        zs[1] = np.sqrt(xs[1]**2 + ys[1]**2)


    elif operation == "*" or operation == "/":
        if operation == "*":
            zs[0] = xs[0] * ys[0]
        elif operation == "/":
            zs[0] = xs[0] / ys[0]

        # Z = X * Y -> DZ/Z = sqrt((DX/X)^2 + (DY/Y)^2)
        zs[1] = zs[0]*np.sqrt((xs[1]/xs[0])**2 + (ys[1]/ys[0])**2)

    elif operation == "**2":
        if ys is not None:
            raise("ys are not None!!!")
        
        zs[0] = xs[0]**2
        
        # Z = X**2 -> DZ = 2X*DX
        zs[1] = 2*xs[0]*xs[1]

    elif operation == "sqrt":
        if ys is not None:
            raise("ys are not None!!!")
        
        zs[0] = np.sqrt(xs[0])

        # Z = sqrt(X) -> DZ = DX/(2sqrt(X))
        zs[1] = xs[1]/(2*np.sqrt(xs[0]))

    else:
        raise("Unkown Operation!!!!")

    return np.array(zs)

# returns the mean and the standard error on the mean
def get_mean(array):

    mean = np.mean(array)
    std = np.std(array)

    n = np.array(array.shape)[0]

    return mean, std/np.sqrt(n)

# Returns the mean and the standard error on the mean
# Assumes Gaussian!!!!!
def get_std(array):

    std = np.std(array)

    n = np.array(array.shape)[0]

    return std, std/np.sqrt(2*n - 2)

                                   ######################################################
##############################################         Plot Base Class        ###########################################
                                   ######################################################

class PlotBase:

    def __init__(self, xlabel, ylabel, ratio = False, residual = False, density = False, sizex = 4, sizey = 4, logy = False, logx = False):
        
        # Create figure and axes 
        n_add = 0
        height_ratios = [3]
        if ratio or residual:
            n_add = 2 if ratio and residual else 1
            height_ratios.append(1) if ratio else None
            height_ratios.append(1) if residual else None


        fig = plt.figure(2, figsize = (sizex, sizey + 1.2 * n_add))

        gs = mpl.gridspec.GridSpec(1+n_add, 1, height_ratios = height_ratios, wspace = 0 if ratio or residual else None, hspace = 0.07 if ratio or residual else None)

        ax_primary = plt.subplot(gs[0])

        self.figure = fig
        self.ax_primary = ax_primary
        self.density = density
        self.logy = logy
        self.logx = logx
        self.ratio = ratio
        self.residual = residual

        ax_primary.set_xlabel(xlabel)
        ax_primary.set_ylabel("Density from " + ylabel if density else ylabel)

        ax_primary.grid(linestyle = "--")
        ax_primary.set_axisbelow(True)


        ax_primary.tick_params(direction = "inout", which = "both", bottom = True, left = True)
        ax_primary.tick_params(direction = "in", which = "both", right = True, top = True)

        if ratio or residual:
            # Move x axis labels to below additional axis plot
            ax_primary.tick_params(labelbottom = False)            
            self.ax_primary.set_xlabel("")

        # Creates and configure additional axis
        self.ax_ratio = None
        if ratio:
            # Append additional axis under primary plot
            ax_ratio = plt.subplot(gs[1])

            self.ax_ratio = ax_ratio

            # Axes label and tick stuff and things
            ax_ratio.tick_params(labelleft = True, labelright = False)        
            ax_ratio.set_xlabel(xlabel)

            ax_ratio.yaxis.set_label_position("left")
            ax_ratio.yaxis.set_ticks_position("left")
            ax_ratio.set_ylabel("Ratio")
            
            ax_ratio.grid(linestyle = "--")
            ax_ratio.set_axisbelow(True)

            ax_ratio.tick_params(direction = "inout", which = "both", bottom = True, left = True)
            ax_ratio.tick_params(direction = "in", which = "both", right = True, top = True)



        self.ax_residual = None
        if residual:

            # Append additional axis under primary plot

            ax_residual = plt.subplot(gs[2]) if ratio else plt.subplot(gs[1])
            self.ax_residual = ax_residual

            # Axes label and tick stuff and things
            ax_residual.tick_params(labelleft = True, labelright = False)        
            ax_residual.set_xlabel(xlabel)

            label_side = "left"

            ax_residual.yaxis.set_label_position(label_side)
            ax_residual.yaxis.set_ticks_position(label_side)
            ax_residual.set_ylabel("Ratio")
            
            ax_ratio.set_xlabel("") if ratio else None
            ax_ratio.tick_params(labelbottom = False) if ratio else None

            ax_residual.grid(linestyle = "--")
            ax_residual.set_axisbelow(True)

            ax_residual.tick_params(direction = "inout", which = "both", bottom = True, left = True)
            ax_residual.tick_params(direction = "in", which = "both", right = True, top = True)


    def Text(self, text, xpos = 0, ypos = 1, size = 14):
        self.figure.text(x = xpos, y = ypos ,s = text, size = size)



                                   ######################################################
############################################           Hist Plot Class          ###########################################
                                   ######################################################


class HistogramPlot(PlotBase):

    def __init__(self, bins, xlabel = "X", ylabel = "Y", ratio = False, residual = False, density = False,
                 plot_unc = True, cpallet = "Bold", sizex = 4, sizey = 4, logy = False, logx = False):
        
        # Initialize relevant callable values
        self.histograms = []
        self.bin_edges = bins[0]
        self.bin_centres = bins[1]
        self.colours = C_Pallets[cpallet]
        self.plot_unc = plot_unc
        self.residual = residual
        self.ratio = ratio


        # if ratio and residual:
        #     raise("Can only plot one of ratio or residual, not both !!!")

        super().__init__(xlabel, ylabel, ratio, residual, density = density, sizex = 5, sizey = 5, logy = logy, logx = logx)

        if residual:
            
            brazil_alpha = 0.3

            self.ax_residual.fill_between(x=[self.bin_edges[0],self.bin_edges[-1]], y1=[1,1],y2=[-1,-1], color = "yellow", alpha = brazil_alpha,edgecolor = None)
            self.ax_residual.fill_between(x=[self.bin_edges[0],self.bin_edges[-1]], y1=[2,2],y2=[1,1], color = "green", alpha = brazil_alpha,edgecolor=None)
            self.ax_residual.fill_between(x=[self.bin_edges[0],self.bin_edges[-1]], y1=[-1,-1],y2=[-2,-2], color = "green", alpha = brazil_alpha,edgecolor=None)

    def Add(self, data, label = "Add Label !!!!!!!!!!", colour = None, uncs = None, drawstyle = "steps-mid",
                 fill = "00", shrink = None, linewidth = 2, linecolour = "black", linestyle = "-",
                 orientation = "vertical", reference = False, addthis = True, weights = None):

        # Set weights to one if None given
        weights = np.ones_like(data) if weights is None else weights

        # Bin Data into numpy histogram and get uncertainties and density factors
        n_entries = np.sum(weights[np.logical_and(data > self.bin_edges[0],data < self.bin_edges[-1])])

        uncs = np.sqrt(np.histogram(data, weights = weights**2, bins = self.bin_edges)[0]) if uncs is None else uncs
        data = np.histogram(data, weights = weights, bins = self.bin_edges)[0]

        density_factors = 1/(n_entries*(self.bin_edges[1:] - self.bin_edges[:-1])) if self.density else np.array([1 for b in self.bin_centres])

        # TODO: 
        # - Determine what exactly this should be (should this just be on the bin range (above) or the whole dataset (below)?)
        n_entries_2 = np.sum(weights)
        density_factors_2 = 1/(n_entries_2*(self.bin_edges[1:] - self.bin_edges[:-1])) if self.density else np.array([1 for b in self.bin_centres])
        # density_factors_2 = np.array([1/n_entries_2 for b in self.bin_centres]) if self.density else np.array([1 for b in self.bin_centres])



        # If no uncertatinty given then use sqrt(bin value)
        # / by 1/sqrt(n_entries * bin_width) if density is True
        # uncs = density_factors * uncs 

        self.histograms.append({"data":  data,
                                "label": label,
                                "colour": colour,
                                "uncs": uncs,
                                "density_factors": density_factors_2,
                                "fill": "ff" if fill == "full" else fill,
                                "shrink": shrink,
                                "linewidth": linewidth,
                                "linecolour": linecolour,
                                "linestyle": linestyle,
                                "orientation": orientation,
                                "reference": reference,
                                "add": addthis,
                                "drawstyle": drawstyle})

    def Plot_Unc(self, axis, hist):

        # Do errors for each bin rather than whole plot so that gaps can occur when shrink != None
        for bin_N in range(len(self.bin_centres)):

            x1 = self.bin_edges[bin_N]
            x2 = self.bin_edges[bin_N+1]

            if hist["shrink"] is not None:
                x1 = (1+hist["shrink"])*self.bin_centres[bin_N] - hist["shrink"]*x1
                x2 = (1+hist["shrink"])*self.bin_centres[bin_N] - hist["shrink"]*x2

            # split into upper and lower error for alpha matching whith overlap with transparent hist

            # Adding alphas is non-linear
            # For alpha_a on top of alpha_b alpha_both = alpha_a +(1-alpha_a)*alpha_b
            # In this case, alpha_a = fill_alpha/255 and alha_b = 0.3 
            alpha_unc = 0.3

            # Lower            
            axis.fill_between(x = [x1,x2],
                              y1 = np.array([hist["data"][bin_N] - hist["uncs"][bin_N], hist["data"][bin_N] - hist["uncs"][bin_N]])*hist["density_factors"][bin_N],
                              y2 = np.array([hist["data"][bin_N], hist["data"][bin_N]])*hist["density_factors"][bin_N],
                              color = hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = alpha_unc)

            # Upper 
            axis.fill_between(x = [x1,x2],
                              y1 = np.array([hist["data"][bin_N], hist["data"][bin_N]])*hist["density_factors"][bin_N],
                              y2 = np.array([hist["data"][bin_N] + hist["uncs"][bin_N], hist["data"][bin_N] + hist["uncs"][bin_N]])*hist["density_factors"][bin_N],
                              color = hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = min(alpha_unc + (1-alpha_unc)*(int(hist["fill"],16)/255), 1))


    def Plot_Ratio(self, hist):
        
        ref_bool = [hists["reference"] for hists in self.histograms]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Hist Specified !!!")

        ref_hist = self.histograms[np.nonzero(ref_bool)[0][0]]

        self.ax_ratio.set_ylabel("Ratio")

        ratio_calcs = CalcNProp("/", [hist["data"]*hist["density_factors"], hist["uncs"]*hist["density_factors"]],
                                     [ref_hist["data"]*ref_hist["density_factors"],ref_hist["uncs"]*ref_hist["density_factors"]])

        temp_hist = {"data": ratio_calcs[0],
                     "uncs": ratio_calcs[1],
                     "density_factors": [1 for b in self.bin_centres],
                     "shrink": None,
                     "fill": "00",
                     "colour": hist["colour"],
                     "label": hist["label"]}

        self.Plot_Unc(self.ax_ratio, temp_hist) if self.plot_unc else None

        self.ax_ratio.plot(self.bin_centres, temp_hist["data"], color = hist["colour"], label = temp_hist["label"], drawstyle = "steps-mid")


    def Plot_Residual(self, hist):
        
        ref_bool = [hists["reference"] for hists in self.histograms]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Hist Specified !!!")

        ref_hist = self.histograms[np.nonzero(ref_bool)[0][0]]

        self.ax_residual.set_ylabel("Residual")


        division_calc = CalcNProp("/", [hist["data"]*hist["density_factors"], hist["uncs"]*hist["density_factors"]],
                                       [ref_hist["data"]*ref_hist["density_factors"], ref_hist["uncs"]*ref_hist["density_factors"]])


        temp_hist = {"data":  (division_calc[0]-1)/division_calc[1],
                     "uncs": None,
                     "density_factors": [1 for b in self.bin_centres],
                     "shrink": None,
                     "fill": "00",
                     "colour": hist["colour"],
                     "label": hist["label"]}

        self.ax_residual.plot(self.bin_centres, temp_hist["data"], color = hist["colour"], label = temp_hist["label"], drawstyle = "steps-mid")


    def Plot(self, plot_path, legend_loc = "upper right", frame = False):

        # Plot each histogram on primary axis
        for count, hist in enumerate(self.histograms):

            hist["colour"] = hist["colour"] if hist["colour"] is not None else self.colours[list(self.colours.keys())[count]]
        
            # Uncertainty Plotting
            self.Plot_Unc(self.ax_primary, hist) if self.plot_unc else None

            self.ax_primary.plot(self.bin_centres, hist["data"]*hist["density_factors"], color = hist["colour"], linewidth = hist["linewidth"],
                                 label = hist["label"], drawstyle = hist["drawstyle"])

            # Ratio Plotting
            self.Plot_Ratio(hist) if self.ratio and hist["add"] else None

            # Ratio Plotting
            self.Plot_Residual(hist) if self.residual and hist["add"] else None

        self.ax_primary.margins(y = 0.1)
        self.ax_primary.set_ylim(bottom=0, top = self.ax_primary.get_ylim()[1]*1.2)
        self.ax_primary.set_xlim(left = self.bin_centres[0], right = self.bin_centres[-1])

        self.ax_ratio.margins(y = 0.05) if self.ratio else None
        self.ax_ratio.set_ylim(bottom = max(0,self.ax_ratio.get_ylim()[0]),
                               top = min(2,self.ax_ratio.get_ylim()[1])) if self.ratio else None
        self.ax_ratio.set_xlim(left = self.bin_centres[0], right = self.bin_centres[-1]) if self.ratio else None

        self.ax_residual.margins(y = 0.05) if self.residual else None
        self.ax_residual.set_ylim(top = max(abs(self.ax_residual.get_ylim()[0]),abs(self.ax_residual.get_ylim()[1])), 
                                  bottom = -max(abs(self.ax_residual.get_ylim()[0]),abs(self.ax_residual.get_ylim()[1]))) if self.residual else None
        self.ax_residual.set_xlim(left = self.bin_centres[0], right = self.bin_centres[-1]) if self.residual else None
        # self.ax_residual.set_ylim(bottom = max(0,self.ax_residual.get_ylim()[0])) if self.residual else None

        self.ax_primary.set_yscale("log") if self.logy else None
        self.ax_primary.set_xscale("log") if self.logx else None

        self.ax_primary.legend(loc = legend_loc, frameon = frame)

        if self.ax_ratio is not None and self.logx:
            self.ax_ratio.set_xscale("log")

        if self.ax_residual is not None and self.logx:
            self.ax_residual.set_xscale("log")

            # self.add_ax.legend(loc = legend_loc[1], frameon = frame)
            # self.add_ax.set_xscale("log") if self.logx else None

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)

                                   ######################################################
############################################           Line Plot Class          ###########################################
                                   ######################################################


class LinePlot(PlotBase):
    
    def __init__(self, xs, xlabel = "X", ylabel = "Y", ratio = False, residual = False, plot_unc = False, logy = False, logx = False, cpallet = "Bold", sizex = 4, sizey = 4):
        
        # Initialize relevant callable values
        self.lines = []
        self.xs = xs
        self.colours = C_Pallets[cpallet]
        self.plot_unc = plot_unc
        self.residual = residual
        self.ratio = ratio

        # if ratio and residual:
        #     raise("Can Only Plot Ratio or Residual Individually Not Together!!!!")

        super().__init__(xlabel, ylabel, ratio, residual, density = False, sizex = 5, sizey = 5, logy = logy, logx = logx)

        if residual:
            
            brazil_alpha = 0.3

            self.ax_residual.fill_between(x=[xs[0],xs[-1]], y1=[1,1],y2=[-1,-1], color = "yellow", alpha = brazil_alpha,edgecolor = None)
            self.ax_residual.fill_between(x=[xs[0],xs[-1]], y1=[2,2],y2=[1,1], color = "green", alpha = brazil_alpha,edgecolor=None)
            self.ax_residual.fill_between(x=[xs[0],xs[-1]], y1=[-1,-1],y2=[-2,-2], color = "green", alpha = brazil_alpha,edgecolor=None)

    
    def Add(self, ys, label = "Add Label !!!!!!!!!!", linecolour = None, uncs = None, addthis = True,
            linestyle = "-", linewidth = 2, marker = ".", marker_size = 10, reference = False, plotunc = False, unctype = "fill"):

        # Force data to be in binned form
        if len(ys) != len(self.xs):
            raise("xs & ys Not Same Shape !!!!")

        # If no uncertatinty given and plot_unc = True require uncs
        if self.plot_unc and uncs is None:
            raise("Plot Uncertainty True but No Uncertainty Given!!!!")         

        self.lines.append({"ys":  np.array(ys),
                           "label": label,
                           "uncs": np.array(uncs),
                           "unc": plotunc,
                           "unctype": unctype,
                           "linewidth": linewidth,
                           "linecolour": linecolour,
                           "linestyle": linestyle,
                           "marker": marker,
                           "marker_size": marker_size,
                           "reference": reference,
                           "add": addthis})


    def Plot_Unc(self, axis, line):

        if line["unctype"] == "fill":            
            axis.fill_between(x = self.xs,
                            y1 = line["ys"] - line["uncs"],
                            y2 = line["ys"] + line["uncs"],
                            color = line["linecolour"], zorder = 1, edgecolor = None, alpha = 0.3)
                            
        elif line["unctype"] == "bars":        
            axis.errorbar(self.xs, 
                        line["ys"],
                        line["uncs"],
                        ls = "none",
                        ecolor = line["linecolour"])

    def Plot_Ratio(self, line):
        
        ref_bool = [line["reference"] for line in self.lines]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Lines Specified !!!")

        ref_line = self.lines[np.nonzero(ref_bool)[0][0]]

        self.ax_ratio.set_ylabel("Ratio")

        temp_line = {"ys":  ref_line["ys"]/line["ys"],
                     "uncs": (line["ys"]/ref_line["ys"]) * np.sqrt((ref_line["uncs"]/ref_line["ys"])**2 + (line["uncs"]/line["ys"])**2) if self.plot_unc else None,
                     "label": line["label"],
                     "unctype": line["unctype"],
                     "linecolour": line["linecolour"],
                     "linestyle": line["linestyle"]}

        self.Plot_Unc(self.ax_ratio, temp_line) if self.plot_unc else None

        self.ax_ratio.plot(self.xs, temp_line["ys"], color = line["linecolour"], label = temp_line["label"], linestyle = temp_line["linestyle"])


    def Plot_Residual(self,line):

        ref_bool = [line["reference"] for line in self.lines]

        if np.sum(ref_bool) != 1:
            raise("Wrong Number of Refernece Lines Specified !!!")

        ref_line = self.lines[np.nonzero(ref_bool)[0][0]]

        self.ax_residual.set_ylabel("Pulls")

        diff = CalcNProp("-", [ref_line["ys"], ref_line["uncs"]], [line["ys"], line["uncs"]])

        temp_line = {"ys":  (diff[0])/(diff[1]),
                     "uncs": np.sqrt(ref_line["uncs"]**2 + line["uncs"]**2) if self.plot_unc else None,
                     "label": line["label"],
                     "unctype": line["unctype"],
                     "linecolour": line["linecolour"],
                     "linestyle": line["linestyle"]}


        self.ax_residual.plot(self.xs, temp_line["ys"], color = line["linecolour"], label = temp_line["label"], linestyle = temp_line["linestyle"])

        # divider = make_axes_locatable(self.ax_residual)

        # margin_hist = divider.append_axes("right", 0.6, pad=0, sharey = self.ax_residual)

        # margin_hist.axis("off")

        # margin_bine_edges = np.linspace(-2 , 2, 1 + 10)

        # margin_hist.hist(temp_line["ys"], margin_bine_edges, fc = line["linecolour"] + "90", ec = "none", linewidth = 0.5, orientation = "horizontal")



    def Plot(self, plot_path, legend_loc = "upper right", frame = False, ymax = None, ymin = None):

        # Plot each line on primary axis
        for count, line in enumerate(self.lines):

            line["linecolour"] = line["linecolour"] if line["linecolour"] is not None else self.colours[list(self.colours.keys())[count]]
    
            # Uncertainty Plotting
            self.Plot_Unc(self.ax_primary, line) if self.plot_unc else None

            self.Plot_Unc(self.ax_primary, line) if line["unc"] else None

            self.ax_primary.plot(self.xs, line["ys"], color = line["linecolour"], linestyle = line["linestyle"], marker = line["marker"], linewidth = line["linewidth"],
                                 ms = line["marker_size"], label = line["label"])

            # Ratio Plotting
            self.Plot_Ratio(line) if self.ratio and line["add"] else None

            # Residual Plotting
            self.Plot_Residual(line) if self.residual and line["add"] else None


        self.ax_primary.margins(y = 0.1)
        self.ax_primary.set_xlim(left = self.xs[0], right = self.xs[-1])

        self.ax_ratio.margins(y = 0.05) if self.ratio else None
        self.ax_ratio.set_ylim(bottom = max(0,self.ax_ratio.get_ylim()[0]),
                               top = min(2, self.ax_ratio.get_ylim()[1])) if self.ratio else None
        self.ax_ratio.set_xlim(left = self.xs[0], right = self.xs[-1]) if self.ratio else None

        self.ax_residual.margins(y = 0.05) if self.residual else None
        self.ax_residual.set_xlim(left = self.xs[0], right = self.xs[-1]) if self.residual else None
        # self.ax_residual.set_ylim(bottom = max(0,self.ax_residual.get_ylim()[0])) if self.residual else None

        self.ax_primary.set_yscale("log") if self.logy else None
        self.ax_primary.set_xscale("log") if self.logx else None

        self.ax_primary.legend(loc = legend_loc, frameon = frame)

        if self.ax_ratio is not None and self.logx:
            self.ax_ratio.set_xscale("log")

        if self.ax_residual is not None and self.logx:
            self.ax_residual.set_xscale("log")

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)


                                   ######################################################
############################################          Hist2D Plot Class         ###########################################
                                   ######################################################

class Hist2D(PlotBase):

    def __init__(self, xbins, ybins, xlabel = "X", ylabel = "Y", cmap = "cividis", sizex = 5, sizey = 5, margins = True, plot_unc = True, cbar = False):
        
        # Initialize relevant callable values
        self.xbin_edges = xbins[0]
        self.xbin_centres = xbins[1]
        self.ybin_edges = ybins[0]
        self.ybin_centres = ybins[1]
        self.cmap = cmap
        self.margins = margins
        self.plot_unc = plot_unc
        self.cbar = cbar

        super().__init__(xlabel, ylabel, sizex = 5, sizey = 5)


    def Set(self, xdata, ydata, weights = None, zlabel = "Add Label!!!", colour = "#0000FF", fill = "90", linecolour = "black", linewidth = 1.5, cmin = None, cmax = None):

        # Set weights to one if None given
        weights = np.ones_like(xdata) if weights is None else weights

        self.hist = {"xdata":  xdata,
                     "ydata": ydata,
                     "weights": weights,
                     "cmin": cmin,
                     "cmax": cmax,
                     "zlabel": zlabel,
                     ## The rest is for margins
                     "colour": colour,
                     "fill": fill,
                     "linecolour": linecolour,
                     "linewidth": linewidth
                     }
        
    def PlotMargins(self):
        
        divider = make_axes_locatable(self.ax_primary)

        ax_histx = divider.append_axes("top", 0.6, pad=0, sharex=self.ax_primary)
        ax_histy = divider.append_axes("right", 0.6, pad= -0.03 if self.plot_unc else 0, sharey=self.ax_primary)

        ax_histx.axis("off")
        ax_histy.axis("off")

        self.PlotUnc(ax_histx, self.hist["xdata"], self.xbin_edges, self.xbin_centres)
        self.PlotUnc(ax_histy, self.hist["ydata"], self.ybin_edges, self.ybin_centres)

        ax_histx.hist(self.hist["xdata"], self.xbin_edges, fc = self.hist["colour"] + "90", ec = "black", linewidth = 1.5)
        ax_histy.hist(self.hist["ydata"], self.ybin_edges, fc = self.hist["colour"] + "90", ec = "black", linewidth = 1.5, orientation = "horizontal")

    def PlotUnc(self, axis, data, bin_edges, bin_centres):

        data_hist = np.histogram(data, bin_edges)[0]

        hist = {"data": data_hist,
                "uncs": np.sqrt(data_hist)}

        # Do errors for each bin rather than whole plot so that gaps can occur when shrink != None
        for bin_N in range(len(bin_centres)):

            x1 = bin_edges[bin_N]
            x2 = bin_edges[bin_N+1]

            # split into upper and lower error for alpha matching whith overlap with transparent hist

            # Adding alphas is non-linear
            # For alpha_a on top of alpha_b alpha_both = alpha_a +(1-alpha_a)*alpha_b
            # In this case, alpha_a = fill_alpha/255 and alha_b = 0.3 
            alpha_unc = 0.3
            
            # Lower            
            axis.fill_between(x = [x1,x2],
                              y1 = [hist["data"][bin_N] - hist["uncs"][bin_N], hist["data"][bin_N] - hist["uncs"][bin_N]],
                              y2 = [hist["data"][bin_N], hist["data"][bin_N]],
                              color = self.hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = alpha_unc)

            # Upper 
            axis.fill_between(x = [x1,x2],
                              y1 = [hist["data"][bin_N], hist["data"][bin_N]],
                              y2 = [hist["data"][bin_N] + hist["uncs"][bin_N], hist["data"][bin_N] + hist["uncs"][bin_N]],
                              color = self.hist["colour"], zorder = 1, step = "pre", edgecolor = None, alpha = min(alpha_unc + (1-alpha_unc)*(int(self.hist["fill"],16)/255), 1))

    def Plot(self, plot_path):

    
        h = self.ax_primary.hist2d(self.hist["xdata"],self.hist["ydata"], bins = [self.xbin_edges, self.ybin_edges],
                                   weights = self.hist["weights"],
                                   cmap = "cividis",
                                   cmin = self.hist["cmin"],
                                   cmax = self.hist["cmax"])

        self.PlotMargins() if self.margins else None

        if self.cbar:
            cbar = plt.colorbar(h[3])
            cbar.set_label(self.hist["zlabel"])

        plt.draw()

        plt.savefig(plot_path+".pdf", bbox_inches = "tight")
        plt.savefig(plot_path+".png", bbox_inches = "tight")

        plt.close(self.figure)