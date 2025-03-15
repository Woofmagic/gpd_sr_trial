from utilities.plot_customizer import PlotCustomizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from networks.local_fit.cross_section_layers import F1F2, F_calc

import matplotlib as mpl

def genf(x,t,a,b,c,d,e,f):
    temp = (a*(x**2) + b*x)*np.exp(c*(t**2)+d*t+e)+f
    return temp

def ReHps(x,t):
    temp_a = -4.41
    temp_b = 1.68
    temp_c = -9.14
    temp_d = -3.57
    temp_e = 1.54
    temp_f = 2.07
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def ReEps(x,t):
    temp_a = -1.04
    temp_b = 0.46
    temp_c = 0.6
    temp_d = 1.95
    temp_e = 2.72
    temp_f = -0.95
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def ReHtps(x,t):
    temp_a = -1.86
    temp_b = 1.50
    temp_c = -0.29
    temp_d = -1.33
    temp_e = 0.46
    temp_f = -0.98
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def DVCSps(x,t):
    temp_a = 0.52
    temp_b = -0.41
    temp_c = 0.05
    temp_d = -0.25
    temp_e = 0.55
    temp_f = 0.173
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def generate_pseudodata(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize the dictionary to store the pseudodata
    pseudodata = {
        'set': [], 'k': [], 'QQ': [], 'x_b': [], 't': [], 'phi_x': [],
        'F': [], 'sigmaF': [], 'ReH': [], 'ReE': [], 'ReHt': [], 'dvcs': []
    }

    for i in range(len(df)):
        row = df.loc[i]
        
        # Extract values from the row
        set_id, QQ, x_b, t, k, phi_x, varF = row['set'], row['QQ'], row['x_b'], row['t'], row['k'], row['phi_x'], row['varF']
        
        # Compute physics-related values
        ReH = ReHps(x_b, t)
        ReE = ReEps(x_b, t)
        ReHt = ReHtps(x_b, t)
        dvcs = DVCSps(x_b, t)
        
        # Compute structure functions
        F1, F2 = fns.f1_f21(t)
        F_nominal = calc.fn_1([phi_x, QQ, x_b, t, k, F1, F2], [ReH, ReE, ReHt, dvcs])
        
        # Calculate the uncertainty and ensure it's positive
        sigmaF = np.abs(F_nominal * varF)
        
        # Generate a positive sampled F value
        while True:
            sampled_F = np.random.normal(loc=F_nominal, scale=sigmaF)
            if sampled_F > 0:
                break

        # Store values in the pseudodata dictionary
        pseudodata['set'].append(set_id)
        pseudodata['k'].append(k)
        pseudodata['QQ'].append(QQ)
        pseudodata['x_b'].append(x_b)
        pseudodata['t'].append(t)
        pseudodata['phi_x'].append(phi_x)
        pseudodata['ReH'].append(ReH)
        pseudodata['ReE'].append(ReE)
        pseudodata['ReHt'].append(ReHt)
        pseudodata['dvcs'].append(dvcs)
        pseudodata['F'].append(sampled_F)
        pseudodata['sigmaF'].append(sigmaF)

    return pd.DataFrame(pseudodata)


fns = F1F2()
calc = F_calc()

# (1): Load experimental data:
file_name = 'raw_experimental_data.csv'

# (2): Read the data file and drop all NaNs:
full_dataframe = pd.read_csv(file_name, dtype = np.float32).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')

tempPseudoDatadf = generate_pseudodata(full_dataframe)
NAME_OF_PSEUDODATA_FILE = 'psueodata_with_sampling.csv'
tempPseudoDatadf.to_csv(NAME_OF_PSEUDODATA_FILE, index=False)

df = pd.read_csv(NAME_OF_PSEUDODATA_FILE)

figure_1 = plt.figure(figsize=(8, 6))
axis_1 = figure_1.add_subplot(1, 1, 1, projection='3d')
plot_1 = PlotCustomizer(axis_1,
    title = r"[Pseudodata] Re[H] Values Across Phase Space",
    xlabel = r"$x_{{B}}$",
    ylabel = r"$Q^{{2}}$",
    zlabel = r"$-t$",
    grid = True)
plot_1.add_3d_scatter_plot(
    x_data = df['x_b'],
    y_data = df['QQ'],
    z_data = -df['t'],
    color = df['ReH'],
    marker = 'o',
    alpha = 0.30,
    colorbar_label = 'Re[H]')

figure_1.savefig("cff_h_vs_xb_and_t.png")
plt.close(figure_1)

figure_2 = plt.figure(figsize=(8, 6))
axis_2 = figure_2.add_subplot(1, 1, 1, projection='3d')
plot_2 = PlotCustomizer(axis_2,
    title = r"[Pseudodata] Re[E] Values Across Phase Space",
    xlabel = r"$x_{{B}}$",
    ylabel = r"$Q^{{2}}$",
    zlabel = r"$-t$",
    grid = True)
plot_2.add_3d_scatter_plot(
    x_data = df['x_b'],
    y_data = df['QQ'],
    z_data = -df['t'],
    color = df['ReE'],
    marker = 'o',
    alpha = 0.30,
    colorbar_label = 'Re[E]')

figure_2.savefig("cff_e_vs_xb_and_t.png")
plt.close(figure_2)

figure_3 = plt.figure(figsize=(8, 6))
axis_3 = figure_3.add_subplot(1, 1, 1, projection='3d')
plot_3 = PlotCustomizer(axis_3,
    title = r"[Pseudodata] Re[$\tilde{{H}}$] Values Across Phase Space",
    xlabel = r"$x_{{B}}$",
    ylabel = r"$Q^{{2}}$",
    zlabel = r"$-t$",
    grid = True)
plot_3.add_3d_scatter_plot(
    x_data = df['x_b'],
    y_data = df['QQ'],
    z_data = -df['t'],
    color = df['ReHt'],
    marker = 'o',
    alpha = 0.30,
    colorbar_label = r'Re[$\tilde{{H}}$]')

figure_3.savefig("cff_ht_vs_xb_and_t.png")
plt.close(figure_3)

figure_4 = plt.figure(figsize=(8, 6))
axis_4 = figure_4.add_subplot(1, 1, 1, projection='3d')
plot_4 = PlotCustomizer(axis_4,
    title = r"[Pseudodata] DVCS Values Across Phase Space",
    xlabel = r"$x_{{B}}$",
    ylabel = r"$Q^{{2}}$",
    zlabel = r"$-t$",
    grid = True)
plot_4.add_3d_scatter_plot(
    x_data = df['x_b'],
    y_data = df['QQ'],
    z_data = -df['t'],
    color = df['dvcs'],
    marker = 'o',
    alpha = 0.30,
    colorbar_label = r'DVCS')

figure_4.savefig("cff_dvcs_vs_xb_and_t.png")
plt.close(figure_4)