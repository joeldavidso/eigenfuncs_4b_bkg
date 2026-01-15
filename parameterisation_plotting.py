import numpy as np
import skink as skplt

def y_p(t,a,r,b):
    return b/(1 + np.sqrt(r**2 - 100* (t**2-2*a*t + a**2)/(t**2)))

def y_m(t,a,r,b):
    return b/(1 - np.sqrt(r**2 - 100* (t**2 - 2*a*t + a**2)/(t**2)))

def y_p_circ(t,a,r,b):
    return b + np.sqrt(r**2 - t**2 + 2*a*t - a**2)


def y_m_circ(t,a,r,b):
    return b - np.sqrt(r**2 - t**2 + 2*a*t - a**2)

ts = np.linspace(0,200,1000)
a = 124
b = 117
r = 45

lineplot = skplt.LinePlot(xs = ts, xlabel = "X", ylabel = "Y")
# lineplot.Add(y_p_circ(ts,a,r,b), label = "plus", marker_size = 0)
# lineplot.Add(y_m_circ(ts,a,r,b), label = "minus", marker_size = 0)
lineplot.Add(y_p(ts,a,r,b), label = "plus", marker_size = 0)
lineplot.Add(y_m(ts,a,r,b), label = "minus", marker_size = 0)
lineplot.Plot("TEMP")
