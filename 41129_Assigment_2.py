#%%Imports
#General imports
import re
import os
import scipy
import numpy as np
import xarray as xr
import pandas as pd

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

#Custom markers
import custom_markers

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
#Settings for specific markers
mss = {"+":dict(marker = "+", s=150, c="k"), 
       "d":dict(marker = "d", s=70, c="k"), 
       "1":dict(marker = "1", s=150, c="k"), 
       "v":dict(marker = "v", s=70, c="k"),
       "v_cm": dict(marker = custom_markers.triangle_down_center_mark(), 
                    s=200, lw=.8, facecolor="none", edgecolor="k"),
       "d_cm": dict(marker = custom_markers.diamond_center_mark(), 
                    s=200, lw=.8, facecolor="none", edgecolor="k"),
       "default": dict(marker = "+", s=100, c="k")
       }


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

T = struct_meas[0].T        # [s] Time period of one cycle
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
    elif Re_i >= 5e5:  # Turbulent flow
        if k_s_i == 0:  # Smooth wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = 0.035 / (Re_i ** 0.16)  # Eq. 5.60
        else:  # Rough wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = np.exp(5.5 * (case_tbl.at[case_tbl.index[i], 'a_ks']) ** (-0.16) - 6.7)  # Eq. 5.69
    else:  # Transitional flow
        f_w_turb = case_tbl.at[case_tbl.index[i], 'f_w'] = 0.035 / (Re_i ** 0.16)  # Eq. 5.60
        f_w_trans = 0.005 #Conservative choice from Eq. 5.61 (results in a larger U_fm and therefore smaller grid size)
        
        case_tbl.at[case_tbl.index[i], 'f_w'] = max(f_w_turb, f_w_trans)

#Calculate a
case_tbl['a'] = [row.a_ks*row.k_s if not row.a_ks==np.inf 
                 else row.Re/row.U_0m*nu                        #Eq. 5.1
                 for _,row in case_tbl.iterrows()]

# Calculate U_fm
case_tbl['U_fm'] = np.sqrt(case_tbl['f_w'] / 2) * case_tbl['U_0m']

# Task 2
case_tbl['dy_max'] = nu / case_tbl['U_fm']  # Eq. 9.49

# Initialize k_s_plus column
case_tbl['k_s_plus'] = np.full(len(case_tbl), .1)

# Identify rough cases and apply equation
i_rough = case_tbl['k_s'] != 0
case_tbl.loc[i_rough, 'k_s_plus'] = case_tbl.loc[i_rough, 'k_s'] \
                                    * case_tbl.loc[i_rough, 'U_fm'] / nu  # Eq. from Assignment

if (case_tbl.loc[i_rough, 'k_s_plus'] > 70).any():
    print("Rough case is hydraulically rough")

# Handle smooth cases
i_smooth = case_tbl['k_s'] == 0
case_tbl.loc[i_smooth, 'k_s'] = 0.1 * nu / case_tbl.loc[i_smooth, 'U_fm']  # Eq. from Assignment

del ang, n, i, Re_i, k_s_i, i_smooth, i_rough, data

#%% User input

exp_fld = "./00_export/"
if not os.path.isdir(exp_fld): os.mkdir(exp_fld)
replot_tasks = dict(C1=False, 
                    C2=False,
                    C3=False,
                    C4=True)

#%% Utility functions

def ensemble_averaging(var, omegat, T=9.72, N=5, sc=1):
    """
    Perform ensemble averaging for a sinusoidal time signal of velocities.
    
    Parameters:
        var (array-like): 
            m x n array of measurements which should be ensemble averaged, 
            where m is the number of time measurements (omega * t),
            and n is the number of spatial positions (y).
        omegat (array-like): 
            array of length m with the phase angles omega * t for each time 
            measurement
        N (int): 
            Total number of cycles sampled. Default: 5
        T (float): 
            Time period of one cycle (s). Default: 9.72
        sc (int):
            Cycle to start the calcuation at (in case that the model needed to 
            tune in for the first sc-1 cycles). Default 1 (i.e. use all cycles)
    
    Returns:
        var_mean (numpy array): 
            m_ps x n array of the mean of the variable where m_ps is the 
            number of time measurements per cycle and n is the number of 
            spatial positions
        var_fluc (numpy array): 
            N x m_ps x n array of the fluctuation of the variable around its 
            mean where N is the number of cycles (minus the skipped cycles at 
            the start - cf. parameter sc) m_ps is the number of time 
            measurements per cycle and n is the number of spatial positions.
        var_fluc_mean (numpy array): 
            m_ps x n array of the mean of the fluctuation of the variable 
            around its mean calculated over all cycles where m_ps is the 
            number of time measurements per cycle and n is the number of 
            spatial positions.
        omegat (numpy array): 
            array of length m_ps with the mean of the phase angles omega * t 
            where m_ps is the  number of time measurements per cycle
    """
    #Copy the input arrays so that they don't get altered outside the function
    var = var.copy()
    omegat=omegat.copy()
    
    
    m, n = var.shape
    
    # Number of time samples per cycle
    samples_per_cycle = m // N
    
    # Reshape u into an (N, n, samples_per_cycle) array and remove the sc-1 
    # first cycles 
    var_reshaped = var[:N*samples_per_cycle, :]\
                    .reshape(N, samples_per_cycle, n)[sc-1:,:,:]
    
    #Calculate corresponding omega*t values
    omegat = omegat[:N*samples_per_cycle].reshape(N, samples_per_cycle)[sc-1:,:]
    omegat -= (np.arange((sc-1),N)*2*np.pi).reshape(N-(sc-1),1)
    omegat = np.mean(omegat, axis=0)
    
    # Calculate the mean velocity for each spatial position by averaging over cycles
    var_mean = np.mean(var_reshaped, axis=0)
    
    # Calculate the fluctuating component of the velocity
    var_fluc = var_reshaped - var_mean[np.newaxis, :, :]
    
    # Calculate the RMS of the fluctuating velocity component
    var_fluc_mean = np.mean(var_fluc, axis=0)
    
    return var_mean, var_fluc, var_fluc_mean, omegat


#%% Case-independent variables
N_cycles = 5                #Number of cycles that were simulated
start_ang = 4*2*np.pi       #[rad]: Start angle for the plots (5th cycle)
omega = 2*np.pi/T           #[rad/s]: angular frequency 

#%% Case 1: Laminar wave boundary layer 
if replot_tasks["C1"]:
    case="1"
    
    #Preparation
    phase_angles = np.array([0, 45, 90, 135])       #[°] - Phase angles to plot
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
             label="Model",
             zorder=2)
    ax.plot(np.rad2deg(omegat),
            tau_0_th/(rho*U_0m**2), 
            label="Theoretical",
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
    
    #Preparation
    phase_angles = np.array([0, 45, 90, 135])       #[°] - Phase angles to plot
    omegat = omega*struct_mdl2.t
    i_phases = [np.absolute(np.rad2deg(omegat)-phase_i).argmin() 
                for phase_i in phase_angles]
    rho = struct_mdl2.rho
    mu = rho*nu
    U_0m = struct_mdl2.U0m
    
    #theoretical bed shear stress
    delta_1 = np.sqrt(2*nu/omega)   #Eq. 5.13
    tau_0_th = (np.sqrt(2)*mu*U_0m)/delta_1 * np.sin(omegat+np.pi+4)    #Eq. 5.20
    
    #Friction coefficient
    f_w_ast_meas = 2*struct_meas2.tau0/struct_meas2.rho \
                   / (struct_meas2.U0m**2 
                      * np.sin(np.deg2rad(struct_meas2.omegat_tau0) + np.pi/4))
    f_w_ast_mdl = 2*struct_mdl2.tau0/struct_mdl2.rho \
                   / (struct_mdl2.U0m**2 
                      * np.sin(omegat + np.pi/4))
    
    i_c2 = np.argwhere((omegat>=start_ang) 
                       & (omegat<=start_ang+np.deg2rad(135))).flatten()
    
    #Plot tau_0/(rho*U_om^2) over omega*t
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat[i_c2]-start_ang),
            (struct_mdl2.tau0[i_c2]/(struct_mdl2.rho*struct_mdl2.U0m**2)), 
            label="Model",
            zorder=2)
    ax.plot(np.rad2deg(omegat[i_c2]-start_ang),
            (tau_0_th[i_c2]/(rho*U_0m**2)), 
            label="Theoretical",
            zorder=2)
    ax.scatter(struct_meas2.omegat_tau0,
               struct_meas2.tau0/(rho*U_0m**2),
               label="Measurements",
               marker= "+", s=150, c="k", zorder=2)
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.grid(zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    fname = exp_fld+"Case_2_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot friction coefficient over omega*t
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat[i_c2]-start_ang),
            f_w_ast_mdl [i_c2], 
            label="Model",
            zorder=2)
    ax.scatter(struct_meas2.omegat_tau0,
               f_w_ast_meas, 
               label="Measurements",
               marker= "+", s=150, c="k", zorder=2)
    ax.axhline(case_tbl.loc[case, "f_w"], ls="--", lw=1.8, c="k")
    ax.text(.05, case_tbl.loc[case, "f_w"]*1.2,  
             r"$f_{w,theoretical}^*=" + f"{case_tbl.loc[case, 'f_w']:.4f}" + r"$", 
            color='k', va="bottom", ha="left", 
            fontsize = mpl.rcParams['ytick.labelsize'], 
            bbox=dict(facecolor='w', alpha=0.4, ls="none"),
            transform=ax.get_yaxis_transform())
    
    #Formatting
    ax.set_ylabel(r'$f_w^*$')
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.grid(zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    fname = exp_fld+"Case_2_fw_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
else:
    print("Plots for Case 2 not replotted")
#%% Case 3:
if replot_tasks["C3"]:
    case="3"  
    
    #Preparation
    phase_angles = np.array([0, 45, 90, 135])       #[°] - Phase angles to plot
    omegat = omega*struct_mdl3.t
    i_phases = [np.absolute(np.rad2deg(omegat)-phase_i).argmin() 
                for phase_i in phase_angles]
    rho = struct_mdl3.rho
    mu = rho*nu
    U_0m = struct_mdl3.U0m
    
    #Detemine indices of datapoints for specified phase angles
    i_c3_mdl = [np.argwhere(omegat>=start_ang+np.deg2rad(pa-.5))[0][0] 
                       for pa in phase_angles]
    i_c3_meas = [np.argwhere(struct_meas3.omegat>=pa-.5)[0][0] 
                 for pa in phase_angles]
    
    #Calculate turbulent kinetic energy
    k_meas = .65 * (struct_meas3.uu + struct_meas3.vv)  #Eq. 10.21 (cf. Assignment)
    
    #Calculate Reynolds stresses
    rs_meas = -struct_meas3.uv/struct_meas3.U0m**2
    
    #Calculate the boundary layer thickness
    delta = case_tbl.loc[case, "a"]*3*np.pi/4\
            * np.sqrt(2/case_tbl.loc[case, "Re"])
    

    #Plot u/U_om over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl3.y/case_tbl.loc[case, "a"],
                struct_mdl3.u[i_c3_mdl[i],:]/U_0m,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas3.y_u/case_tbl.loc[case, "a"],
                   struct_meas3.u[:,i_c3_meas[i]]/struct_meas3.U0m,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)

    #Formatting
    ax.set_ylabel(r'$\frac{\overline{u}}{U_{0m}}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_3_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot k/U_om^2 over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl3.y/case_tbl.loc[case, "a"],
                struct_mdl3.k[i_c3_mdl[i],:]/U_0m**2,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas3.y_uuvv/case_tbl.loc[case, "a"],
                   k_meas[:,i_c3_meas[i]]/struct_meas3.U0m**2,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)
    
    ax.axvline (delta/case_tbl.loc[case, "a"], c="k", ls="--", lw=1.2)
    
    #Formatting
    ax.set_ylabel(r'$\frac{k}{U_{0m}^2}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.set_xscale("log")
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_3_k_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot u over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl3.y/case_tbl.loc[case, "a"],
                struct_mdl3.u[i_c3_mdl[i],:]/U_0m,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas3.y_u/case_tbl.loc[case, "a"],
                   struct_meas3.u[:,i_c3_meas[i]]/struct_meas3.U0m,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)

    #Formatting
    ax.set_ylabel(r'$\frac{\overline{u}}{U_{0m}}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_3_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*U_0m^2) over omega*t
    # Get indices for the first half of the fourth cylcle
    i_c3_mdl = np.argwhere((omegat>=start_ang) 
                           & (omegat<=start_ang + np.pi)).flatten()
    i_c3_meas = np.argwhere(struct_meas3.omegat_tau0<=180).flatten()
    
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat[i_c3_mdl]-start_ang),
            struct_mdl3.tau0[i_c3_mdl]/U_0m**2,
            label = r"Model",
            zorder=2)
    ax.scatter(struct_meas3.omegat_tau0[i_c3_meas],
               struct_meas3.tau0[i_c3_meas]/struct_meas3.U0m**2,
               label = r"Measurements", 
               zorder=2, **mss["+"])
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim ([-5,185])
    ax.set_xticks(np.arange(0,181,10))
    ax.grid(zorder=1)
    ax.legend(loc="upper right")
    
    fname = exp_fld+"Case_3_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)

else:
    print("Plots for Case 3 not replotted")    
#%% Case 4:
if replot_tasks["C4"]:
    case="4"  
    
    #Preparation
    phase_angles = np.array([0, 45, 90, 135])       #[°] - Phase angles to plot
    omegat = omega*struct_mdl4.t
    i_phases = [np.absolute(np.rad2deg(omegat)-phase_i).argmin() 
                for phase_i in phase_angles]
    rho = struct_mdl4.rho
    mu = rho*nu
    U_0m = struct_mdl4.U0m
    
    #Detemine indices of datapoints for specified phase angles
    i_c4_mdl = [np.argwhere(omegat>=start_ang+np.deg2rad(pa-.5))[0][0] 
                       for pa in phase_angles]
    i_c4_meas = [np.argwhere(struct_meas4.omegat>=pa-.5)[0][0] 
                 for pa in phase_angles]
    
    #Calculate turbulent kinetic energy
    k_meas = .65 * (struct_meas4.uu + struct_meas4.vv)  #Eq. 10.21 (cf. Assignment)
    
    #Calculate Reynolds stresses
    rs_meas = -struct_meas4.uv/struct_meas4.U0m**2
    
    #Calculate the boundary layer thickness
    delta = case_tbl.loc[case, "a"]*3*np.pi/4\
            * np.sqrt(2/case_tbl.loc[case, "Re"])
    

    #Plot u/U_om over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl4.y/case_tbl.loc[case, "a"],
                struct_mdl4.u[i_c4_mdl[i],:]/U_0m,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas4.y_u/case_tbl.loc[case, "a"],
                   struct_meas4.u[:,i_c4_meas[i]]/struct_meas4.U0m,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)

    #Formatting
    ax.set_ylabel(r'$\frac{\overline{u}}{U_{0m}}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_4_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot k/U_om^2 over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl4.y/case_tbl.loc[case, "a"],
                struct_mdl4.k[i_c4_mdl[i],:]/U_0m**2,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas4.y_uuvv/case_tbl.loc[case, "a"],
                   k_meas[:,i_c4_meas[i]]/struct_meas4.U0m**2,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)
    
    ax.axvline (delta/case_tbl.loc[case, "a"], c="k", ls="--", lw=1.2)
    
    #Formatting
    ax.set_ylabel(r'$\frac{k}{U_{0m}^2}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.set_xscale("log")
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_4_k_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot u over y/a
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    for i,i_p in enumerate(i_phases):
        ax.plot(struct_mdl4.y/case_tbl.loc[case, "a"],
                struct_mdl4.u[i_c4_mdl[i],:]/U_0m,
                label = r"$\omega t = " + f"{phase_angles[i]}" 
                        + r"\:\unit{\degree}$ -- Model",
                zorder=2)
    for i,i_p in enumerate(i_phases):
        ms = mss[markers[i]] if mss.get(markers[i]) else mss["default"]
        ax.scatter(struct_meas4.y_u/case_tbl.loc[case, "a"],
                   struct_meas4.u[:,i_c4_meas[i]]/struct_meas4.U0m,
                   label = r"$\omega t = " + f"{phase_angles[i]}" 
                          + r"\:\unit{\degree}$ -- Measurements", 
                   zorder=2, **ms)

    #Formatting
    ax.set_ylabel(r'$\frac{\overline{u}}{U_{0m}}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\frac{y}{a}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.grid(zorder=1)
    ax.legend(loc="center right", ncols=2,
              bbox_to_anchor=(1, .4))
    
    fname = exp_fld+"Case_4_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*U_0m^2) over omega*t
    # Get indices for the first half of the fourth cylcle
    i_c4_mdl = np.argwhere((omegat>=start_ang) 
                           & (omegat<=start_ang + np.pi)).flatten()
    i_c4_meas = np.argwhere(struct_meas4.omegat_tau0<=180).flatten()
    
    markers = ["+", "1", "d_cm", "v_cm"]
    
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat[i_c4_mdl]-start_ang),
            struct_mdl4.tau0[i_c4_mdl]/U_0m**2,
            label = r"Model",
            zorder=2)
    ax.scatter(struct_meas4.omegat_tau0[i_c4_meas],
               struct_meas4.tau0[i_c4_meas]/struct_meas4.U0m**2,
               label = r"Measurements", 
               zorder=2, **mss["+"])
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim ([-5,185])
    ax.set_xticks(np.arange(0,181,10))
    ax.grid(zorder=1)
    ax.legend(loc="upper right")
    
    fname = exp_fld+"Case_4_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
else:
    print("Plots for Case 4 not replotted")    