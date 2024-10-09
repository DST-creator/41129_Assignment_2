#%%Imports
#General imports
import scipy
import numpy as np
import pandas as pd
import os

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%Global plot settings

#Figure size:
mpl.rcParams['figure.figsize'] = (16, 8)  

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['scatter.marker'] = "+"
mpl.rcParams['lines.color'] = "k"
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])
#Cycle through linestyles with color black instead of different colors
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])\
                                + mpl.cycler('linestyle', ['-', '--', '-.', ':'])\
                                + mpl.cycler('linewidth', [1.2, 1.2, 1.3, 1.8])
plt_marker = "d"


#Text sizes
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 20

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                 r'\usepackage{siunitx}'])
mpl.rcParams.update({"pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% Data Import 
struct_meas = scipy.io.loadmat("Exercise4.mat", 
                           struct_as_record=False, 
                           squeeze_me=True)["WBL"]
fld_path_matrans = "./MatRANS/TurbulenceBook/Exercise4/Case"
if os.path.exists(fld_path_matrans + "1" + "/out_MatRANS.mat"):
    struct_mdl1 = scipy.io.loadmat(fld_path_matrans + "1" + "/out_MatRANS.mat", 
                               struct_as_record=False, 
                               squeeze_me=True)["MatRANS"]
    struct_meas1 = struct_meas[0]
if os.path.exists(fld_path_matrans + "2" + "/out_MatRANS.mat"):
    struct_mdl2 = scipy.io.loadmat(fld_path_matrans + "2" + "/out_MatRANS.mat", 
                               struct_as_record=False, 
                               squeeze_me=True)["MatRANS"]
    struct_meas2 = struct_meas[1]
if os.path.exists(fld_path_matrans + "3" + "/out_MatRANS.mat"):
    struct_mdl3 = scipy.io.loadmat(fld_path_matrans + "3" + "/out_MatRANS.mat", 
                               struct_as_record=False, 
                               squeeze_me=True)["MatRANS"]
    struct_meas3 = struct_meas[2]
if os.path.exists(fld_path_matrans + "4" + "/out_MatRANS.mat"):
    struct_mdl4 = scipy.io.loadmat(fld_path_matrans + "4" + "/out_MatRANS.mat", 
                               struct_as_record=False, 
                               squeeze_me=True)["MatRANS"]
    struct_meas4 = struct_meas[3]

#%% Preparatory calculations

ang = np.arange(0, 181, 15)
n = np.arange(0, len(ang))
ang_dict = dict(zip(ang, n))

T = struct_meas[0].T        # [s] Time period
nu = struct_meas[0].nu      # [m^2/s] - kinematic viscosity
h_m = 0.145                 # [m] - model height

# Create the DataFrame (case_tbl)
data = {
    'U_0m': [struct_i.U0m for struct_i in struct_meas],
    'k_s': [0 if struct_i.k_s=="smooth" else struct_i.k_s 
            for struct_i in struct_meas],
    'Re': [7.2e4, 6.3e5, 5.4e6, 5.4e6],
    'a_ks': [np.inf, np.inf, np.inf, 3700],
    'f_w': np.zeros(4)
}
case_tbl = pd.DataFrame(data, index=['1', '2', '3', '4'])

# Task 1
for i in range(len(case_tbl)):
    Re_i = case_tbl.at[case_tbl.index[i], 'Re']
    k_s_i = case_tbl.at[case_tbl.index[i], 'k_s']
    
    if Re_i <= 1.5e5:  # Laminar flow
        case_tbl.at[case_tbl.index[i], 'f_w'] = 2 / np.sqrt(Re_i)  # Eq. 5.59
    elif Re_i >= 1.5e5:  # Turbulent flow
        if k_s_i == 0:  # Smooth wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = 0.035 / (Re_i ** 0.16)  # Eq. 5.60
        else:  # Rough wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = np.exp(5.5 * (case_tbl.at[case_tbl.index[i], 'a_ks']) ** (-0.16) - 6.7)  # Eq. 5.69
    else:  # Transitional flow
        case_tbl.at[case_tbl.index[i], 'f_w'] = 0.005  # Eq. 5.61

#Calculate a
case_tbl['a'] = [row.a_ks*row.k_s if not row.a_ks==np.inf 
                 else row.Re/row.U_0m*nu                        #Eq. 5.1
                 for _,row in case_tbl.iterrows()]

# Calculate U_fm
case_tbl['U_fm'] = np.sqrt(case_tbl['f_w'] / 2) * case_tbl['U_0m']

# Task 2
case_tbl['dy_max'] = nu / case_tbl['U_fm']  # Eq. 9.49

# Initialize k_s_plus column
case_tbl['k_s_plus'] = np.zeros(len(case_tbl)) + 0.1

# Identify rough cases and apply equation
i_rough = case_tbl['k_s'] != 0
case_tbl.loc[i_rough, 'k_s_plus'] = case_tbl.loc[i_rough, 'k_s'] \
                                    * case_tbl.loc[i_rough, 'U_fm'] / nu  # Eq. from Assignment

if (case_tbl.loc[i_rough, 'k_s_plus'] > 70).any():
    print("Rough case is hydraulically rough")

# Handle smooth cases
i_smooth = case_tbl['k_s'] == 0
case_tbl.loc[i_smooth, 'k_s'] = 0.1 * nu / case_tbl.loc[i_smooth, 'U_fm']  # Eq. from Assignment

del ang, n, i, Re_i, k_s_i, i_smooth, i_rough

#%% User input

exp_fld = "./00_export/"
if not os.path.isdir(exp_fld): os.mkdir(exp_fld)
replot_tasks = dict(C1=True, 
                    C2=True,
                    C3=True,
                    C4=True)
    
#%% Case 1: Laminar wave boundary layer 
if replot_tasks["C1"]:
    case="1"
    
    #Preparation
    phase_angles = np.array([0, 45, 90, 135])       #[°] - Phase angles to plot
    omega = 2*np.pi/T
    omegat = omega*struct_mdl1.t
    i_phases = [np.absolute(np.rad2deg(omegat)-phase_i).argmin() 
                for phase_i in phase_angles]
    rho = struct_mdl1.rho
    mu = rho*nu
    U_0m = struct_mdl1.U0m
    
    #Theoretical velocity
    delta_1 = np.sqrt(2*nu/omega)   #Eq. 5.13
    u_th = U_0m*np.sin(omegat.reshape(-1,1))\
           - U_0m*np.exp(-struct_mdl1.y.reshape(1,-1)/delta_1) \
                 *np.sin(omegat.reshape(-1,1)
                         -struct_mdl1.y.reshape(1,-1)/delta_1)      #Eq. 5.12
    
    #theoretical bed shear stress
    tau_0_th = (np.sqrt(2)*mu*U_0m)/delta_1 * np.sin(omegat+np.pi+4)    #Eq. 5.20
    
    #Plot u/U_0m over y/a
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl1.y/case_tbl.loc[case, 'a'],
                 struct_mdl1.u[i_p,:]/U_0m, 
                 label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- model",
                 zorder=3)
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl1.y/case_tbl.loc[case, 'a'],
                 u_th[i_p,:]/U_0m, 
                 label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- theoretical",
                 alpha=.5, zorder=2)
    
    #Formatting
    ax.set_xlabel(r'$\frac{y}{a}$',
                   fontsize =  1.5*mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(r'$\frac{u}{U_{0m}}$',
                   fontsize =  1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.legend(loc="center right", ncols=2)
    
    fname = exp_fld+"Case_1_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*U_om^2) over omega*t
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat),
             struct_mdl1.tau0/(rho*U_0m**2), 
             label="model",
             zorder=2)
    ax.plot(np.rad2deg(omegat),
            tau_0_th/(rho*U_0m**2), 
            label="theoretical",
            zorder=2)
    
    #Formatting
    ax.set_xlabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(r'$\omega t\:\unit{[\degree]}$')
    ax.grid(zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    fname = exp_fld+"Case_1_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
else:
    print("Plots for Case 1 not replotted")


#%% Case 2:
if replot_tasks["C2"]:
    case="2"  
else:
    print("Plots for Case 2 not replotted")
#%% Case 3:
if replot_tasks["C3"]:
    case="3"  
else:
    print("Plots for Case 3 not replotted")    
#%% Case 4:
if replot_tasks["C4"]:
    case="4"  
else:
    print("Plots for Case 4 not replotted")    