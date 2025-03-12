import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from networks.local_fit.local_fit_model import *
from mpl_toolkits.mplot3d import Axes3D

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
        set_id, QQ, x_b, t, k, phi_x, varF = row['#Set'], row['QQ'], row['x_b'], row['t'], row['k'], row['phi_x'], row['varF']
        
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
file_name = 'data.csv'

# (2): Read the data file and drop all NaNs:
full_dataframe = pd.read_csv(file_name, dtype = np.float32).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')

tempPseudoDatadf = generate_pseudodata(full_dataframe)
tempPseudoDatadf.to_csv('Pseudo_data_with_sampling.csv', index=False)

###### Making 3d scatter plots of CFFs ########

file_name = "Pseudo_data_with_sampling.csv"
df = pd.read_csv(file_name)

# Extract relevant columns
x_b_values = df['x_b'].values
t_values = df['t'].values

# Compute CFF values
ReH_values = df['ReH'].values
ReE_values = df['ReE'].values
ReHt_values = df['ReHt'].values
dvcs_values = df['dvcs'].values

# Create a single figure for all 4 scatter plots
fig = plt.figure(figsize=(16, 12))

# ReH scatter plot
ax1 = fig.add_subplot(221, projection='3d')
sc1 = ax1.scatter(x_b_values, t_values, ReH_values, c=ReH_values, marker='o')
ax1.set_xlabel('x_b')
ax1.set_ylabel('t')
ax1.set_zlabel('ReH')
ax1.set_title('ReH vs x_b and t')
fig.colorbar(sc1, ax=ax1, shrink=0.5)

# ReE scatter plot
ax2 = fig.add_subplot(222, projection='3d')
sc2 = ax2.scatter(x_b_values, t_values, ReE_values, c=ReE_values, marker='o')
ax2.set_xlabel('x_b')
ax2.set_ylabel('t')
ax2.set_zlabel('ReE')
ax2.set_title('ReE vs x_b and t')
fig.colorbar(sc2, ax=ax2, shrink=0.5)

# ReHt scatter plot
ax3 = fig.add_subplot(223, projection='3d')
sc3 = ax3.scatter(x_b_values, t_values, ReHt_values, c=ReHt_values, marker='o')
ax3.set_xlabel('x_b')
ax3.set_ylabel('t')
ax3.set_zlabel('ReHt')
ax3.set_title('ReHt vs x_b and t')
fig.colorbar(sc3, ax=ax3, shrink=0.5)

# DVCS scatter plot
ax4 = fig.add_subplot(224, projection='3d')
sc4 = ax4.scatter(x_b_values, t_values, dvcs_values, c=dvcs_values, marker='o')
ax4.set_xlabel('x_b')
ax4.set_ylabel('t')
ax4.set_zlabel('DVCS')
ax4.set_title('DVCS vs x_b and t')
fig.colorbar(sc4, ax=ax4, shrink=0.5)

# Save all scatter plots into a single PNG file
plt.tight_layout()
plt.savefig("CFF_Scatter_Plots.png", dpi=300)

# Close the plots to free memory
plt.close(fig)

print("Combined 3D scatter plots saved as 'CFF_Scatter_Plots.png'.")