#%% Imports
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

#%% User input

exp_fld = "./00_export/"
if not os.path.isdir(exp_fld): os.mkdir(exp_fld)
replot_tasks = dict(C1=True, 
                    C2=True,
                    C3=True,
                    C4=True,
                    T7=True)

#%% Global plot settings

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
                                + mpl.cycler('linestyle', ['-','--','-.',':'])\
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
mpl.rcParams['figure.subplot.top'] = .94 #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  
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
    elif Re_i >= 1e6:  # Turbulent flow
        if k_s_i == 0:  # Smooth wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = 0.035 / (Re_i ** 0.16)  # Eq. 5.60
        else:  # Rough wall
            case_tbl.at[case_tbl.index[i], 'f_w'] = \
                np.exp(5.5 * (case_tbl.at[case_tbl.index[i], 'a_ks'])**(-0.16)
                       - 6.7)  # Eq. 5.69
    else:  # Transitional flow
        f_w_turb = case_tbl.at[case_tbl.index[i], 'f_w'] = \
            0.035/(Re_i**0.16)  # Eq. 5.60
        f_w_trans = 0.005 #Conservative choice from Eq. 5.61 (results in a 
                          #larger U_fm and therefore smaller grid size)
        
        case_tbl.at[case_tbl.index[i], 'f_w'] = max(f_w_turb, f_w_trans)

#Calculate a
case_tbl['a'] = [row.Re/row.U_0m*nu                        #Eq. 5.1
                 for _,row in case_tbl.iterrows()]

# Calculate U_fm
case_tbl['U_fm'] = np.sqrt(case_tbl['f_w'] / 2) * case_tbl['U_0m']

# Task 2
case_tbl['dy_max'] = nu / case_tbl['U_fm']  # Eq. 9.49

# Initialize k_s_plus column
case_tbl['k_s_plus'] = np.full(len(case_tbl), .1)

# Identify rough cases and apply equation (with Eq. from Assignment)
i_rough = case_tbl['k_s'] != 0
case_tbl.loc[i_rough, 'k_s_plus'] = case_tbl.loc[i_rough, 'k_s'] \
                                    * case_tbl.loc[i_rough, 'U_fm'] / nu

if (case_tbl.loc[i_rough, 'k_s_plus'] > 70).any():
    print("Rough case is hydraulically rough")

# Handle smooth cases (with Eq. from Assignment)
i_smooth = case_tbl['k_s'] == 0
case_tbl.loc[i_smooth, 'k_s'] = 0.1 * nu / case_tbl.loc[i_smooth, 'U_fm']

del ang, n, i, Re_i, k_s_i, i_smooth, i_rough, data



#%% Utility functions

def ensemble_averaging(var, omegat, T=9.72, N=5, sc=1):
    """
    Perform ensemble averaging for a sinusoidal time signal of velocities.
    
    Parameters:
        var (array-like): 
            m x n array of Data from Jensen et. al. which should be ensemble 
            averaged, where m is the number of time Data from Jensen et. al. 
            (omega * t), and n is the number of spatial positions (y).
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
            number of time Data from Jensen et. al. per cycle and n is the 
            number of spatial positions
        var_fluc (numpy array): 
            N x m_ps x n array of the fluctuation of the variable around its 
            mean where N is the number of cycles (minus the skipped cycles at 
            the start - cf. parameter sc) m_ps is the number of time 
            Data from Jensen et. al. per cycle and n is the number of spatial 
            positions.
        var_fluc_mean (numpy array): 
            m_ps x n array of the mean of the fluctuation of the variable 
            around its mean calculated over all cycles where m_ps is the 
            number of time Data from Jensen et. al. per cycle and n is the 
            number of spatial positions.
        omegat (numpy array): 
            array of length m_ps with the mean of the phase angles omega * t 
            where m_ps is the  number of time Data from Jensen et. al. per 
            cycle
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
    omegat = omegat[:N*samples_per_cycle].reshape(N,samples_per_cycle)[sc-1:,:]
    omegat -= (np.arange((sc-1),N)*2*np.pi).reshape(N-(sc-1),1)
    omegat = np.mean(omegat, axis=0)
    
    # Calculate the mean velocity for each spatial position by averaging over 
    # cycles
    var_mean = np.mean(var_reshaped, axis=0)
    
    # Calculate the fluctuating component of the velocity
    var_fluc = var_reshaped - var_mean[np.newaxis, :, :]
    
    # Calculate the RMS of the fluctuating velocity component
    var_fluc_mean = np.mean(var_fluc, axis=0)
    
    return var_mean, var_fluc, var_fluc_mean, omegat


#%% Case-independent variables
N_cycles = 5                #Number of cycles that were simulated
start_ang = 4*2*np.pi       #[rad] - Start angle for the plots (5th cycle)

omega = 2*np.pi/T           #[rad/s] - Angular frequency 
rho = 1000                  #[kg/m^3] - Fluid density
mu = rho*nu                 #[kg/(m*s)] - Dynamic viscosity

phase_angles = np.array([0, 45, 90, 135])   #[°] - Phase angles to plot
markers = ["+", "1", "d_cm", "v_cm"]        #Markers for the four phase angles

#%% Case 1: Laminar wave boundary layer 
if 'struct_mdl1' in globals():
    case="1"
    
    #Preparation
    omegat_c1 = omega*struct_mdl1.t
    i_ph_c1_mdl = [np.argwhere(omegat_c1>=start_ang+np.deg2rad(pa-.5))[0][0] 
                       for pa in phase_angles]
    i_ph_c1_mdl_cont = np.argwhere((omegat_c1>=start_ang-np.deg2rad(1)) 
                                   & (omegat_c1<=start_ang+np.deg2rad(360))
                                   ).flatten() 
    
    #Theoretical velocity
    delta_1_c1 = np.sqrt(2*nu/omega)   #Eq. 5.13
    u_th_c1 = struct_mdl1.U0m*np.sin(omegat_c1.reshape(-1,1))\
           - struct_mdl1.U0m*np.exp(-struct_mdl1.y.reshape(1,-1)/delta_1_c1) \
                 *np.sin(omegat_c1.reshape(-1,1)
                         -struct_mdl1.y.reshape(1,-1)/delta_1_c1)      #Eq. 5.12
    
    #theoretical bed shear stress
    tau_0_th_c1 = (np.sqrt(2)*mu*struct_mdl1.U0m)\
                    /delta_1_c1 * np.sin(omegat_c1+np.pi/4)    #Eq. 5.20

if replot_tasks["C1"]:
    if not 'struct_mdl1' in globals():
        print("Data missing for Case 1 - Plots not replotted")
    
    #Plot u/U_om over y/a
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    ya_max = 4.5e-2
    for i,phase_i in enumerate(phase_angles):
        i_p = i_ph_c1_mdl[i]
        ax[i].plot(struct_mdl1.u[i_p,:]/struct_mdl1.U0m, 
                struct_mdl1.y/case_tbl.loc[case, 'a'],
                label=f"MatRANS Model", ls="-", zorder=3)
        ax[i].plot(u_th_c1[i_p,:]/struct_mdl1.U0m, 
                struct_mdl1.y/case_tbl.loc[case, 'a'], 
                label=f"Laminar Theory", ls="--", zorder=2)

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        fig.supxlabel(r'$\frac{\overline{u}}{U_{0m}}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_ylim([0, ya_max])
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
    
    fname = exp_fld+"Case_1_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*U_om^2) over omega*t
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat_c1[i_ph_c1_mdl_cont]-start_ang),
             struct_mdl1.tau0[i_ph_c1_mdl_cont]/(rho*struct_mdl1.U0m**2), 
             label="MatRANS Model",
             zorder=2)
    ax.plot(np.rad2deg(omegat_c1[i_ph_c1_mdl_cont]-start_ang),
            tau_0_th_c1[i_ph_c1_mdl_cont]/(rho*struct_mdl1.U0m**2), 
            label="Laminar theory",
            zorder=2)
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim([-5,365])
    ax.set_xticks(np.arange(0,361,20))
    ax.grid(zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    fname = exp_fld+"Case_1_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    print("Case 1 successfully plotted")
else:
    print("Plots for Case 1 not replotted")
#%% Case 2: Transitional wave boundary layer
if 'struct_mdl2' in globals():
    case="2"  
    
    #Preparation
    omegat_c2 = omega*struct_mdl2.t
    i_ph_c2_mdl = np.argwhere((omegat_c2>=start_ang-np.deg2rad(1)) 
                       & (omegat_c2<=start_ang+np.deg2rad(360))).flatten() 
    
    #theoretical bed shear stress
    delta_1_c2 = np.sqrt(2*nu/omega)   #Eq. 5.13
    tau_0_th_c2 = (np.sqrt(2)*mu*struct_mdl2.U0m)/delta_1_c2 \
                    * np.sin(omegat_c2+np.pi+4)    #Eq. 5.20
    
    #Friction coefficient
    f_w_ast_meas_c2 = 2*struct_meas2.tau0/struct_meas2.rho \
                   / (struct_meas2.U0m**2                   #Eq.5.22
                      * np.sin(np.deg2rad(struct_meas2.omegat_tau0) + np.pi/4))
    f_w_ast_mdl_c2 = 2*struct_mdl2.tau0/struct_mdl2.rho \
                   / (struct_mdl2.U0m**2 
                      * np.sin(omegat_c2 + np.pi/4))        #Eq.5.22
    f_w_lam_c2 = 2/np.sqrt(case_tbl.loc[case, "Re"])         #Eq.5.59
       
if replot_tasks["C2"]:
    if not 'struct_mdl2' in globals():
        print("Data missing for Case 2 - Plots not replotted")
    
    #Plot tau_0/(rho*U_om^2) over omega*t
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(omegat_c2[i_ph_c2_mdl]-start_ang),
            (struct_mdl2.tau0[i_ph_c2_mdl]
             /(struct_mdl2.rho*struct_mdl2.U0m**2)), 
            label="MatRANS Model",
            zorder=2)
    ax.plot(np.rad2deg(omegat_c2[i_ph_c2_mdl]-start_ang),
            (tau_0_th_c2[i_ph_c2_mdl]/(rho*struct_mdl2.U0m**2)), 
            label="Theoretical",
            zorder=2)
    ax.scatter(struct_meas2.omegat_tau0,
               struct_meas2.tau0/(struct_meas2.rho*struct_meas2.U0m**2),
               label="Data from Jensen et. al.",
               marker= "+", s=150, c="k", zorder=2)
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim([-5,365])
    ax.set_xticks(np.arange(0,361,20))
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
    ax.plot(np.rad2deg(omegat_c2[i_ph_c2_mdl]-start_ang),
            f_w_ast_mdl_c2 [i_ph_c2_mdl], 
            label="MatRANS Model",
            zorder=2)
    ax.scatter(struct_meas2.omegat_tau0,
               f_w_ast_meas_c2, 
               label="Data from Jensen et. al.",
               marker= "+", s=150, c="k", zorder=2)
    ax.axhline(f_w_lam_c2, ls="--", lw=1.8, c="k")
    ax.text(.05, f_w_lam_c2*1.2,  
             r"$f_{w,theoretical}^*=" + f"{f_w_lam_c2:.4f}" + r"$", 
            color='k', va="bottom", ha="left", 
            fontsize = mpl.rcParams['ytick.labelsize'], 
            bbox=dict(facecolor='w', alpha=0.4, ls="none"),
            transform=ax.get_yaxis_transform())
    
    #Formatting
    ax.set_ylabel(r'$f_w^*$')
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_ylim([0, (np.ceil(max(f_w_ast_meas_c2)*1000))/1000])
    ax.set_xlim([-5, 140])
    ax.set_xticks(np.arange(0,140,10))
    ax.grid(zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    fname = exp_fld+"Case_2_fw_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    print("Case 2 successfully plotted")
else:
    print("Plots for Case 2 not replotted")
#%% Case 3: Turbulent wave boundary layer, smooth wall
if 'struct_mdl3' in globals():
    case="3"  
    
    #Preparation
    omegat_c3 = omega*struct_mdl3.t
    
    #Detemine indices of datapoints for specified phase angles
    i_ph_c3_mdl = [np.argwhere(omegat_c3>=start_ang+np.deg2rad(pa-.5))[0][0] 
                       for pa in phase_angles]
    i_ph_c3_meas = [np.argwhere(struct_meas3.omegat>=pa-.5)[0][0] 
                 for pa in phase_angles]
    
    #Determine indices for the first half of the fifth cylcle
    i_ph_c3_mdl_cont = np.argwhere((omegat_c3>=start_ang-np.deg2rad(1)) 
                           & (omegat_c3<=start_ang+np.deg2rad(360))).flatten()
    i_ph_c3_meas_cont = np.argwhere(struct_meas3.omegat_tau0<=360).flatten()
    
    #Calculate turbulent kinetic energy
    k_meas_c3 = .65 *(struct_meas3.uu + struct_meas3.vv)   #Eq. 10.21
    
    #Calculate Reynolds stresses
    du_dy_c3 = np.gradient(struct_mdl3.u[i_ph_c3_mdl,:], struct_mdl3.y, axis=1)
    rs_mdl_c3 = struct_mdl3.nu_t[i_ph_c3_mdl,:]*du_dy_c3/struct_mdl3.U0m**2
    rs_meas_c3 = -struct_meas3.uv[:,i_ph_c3_meas]/struct_meas3.U0m**2
    
    #Calculate the boundary layer thickness
    delta_c3 = case_tbl.loc[case, "a"]*3*np.pi/4\
            * np.sqrt(2/case_tbl.loc[case, "Re"])
    
if replot_tasks["C3"]:
    if not 'struct_mdl3' in globals():
        print("Data missing for Case 3 - Plots not replotted")

    #Plot u/U_om over y/a
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    ya_max = 1e-2*(np.ceil(np.max(struct_meas3.y_uuvv
                                /case_tbl.loc[case, "a"])/1e-2))
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl3.u[i_ph_c3_mdl[i],:]/struct_mdl3.U0m,
                struct_mdl3.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        ax[i].scatter(struct_meas3.u[:,i_ph_c3_meas[i]]/struct_meas3.U0m,
                   struct_meas3.y_u/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        # ax.set_xlabel(r'$\frac{\overline{u}}{U_{0m}}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        fig.supxlabel(r'$\frac{\overline{u}}{U_{0m}}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_ylim([0, ya_max])
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
    
    fname = exp_fld+"Case_3_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot u/U_om over y/a (semilog)
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    ya_max = 1e-2*(np.ceil(np.max(struct_meas3.y_uuvv
                                /case_tbl.loc[case, "a"])/1e-2))
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl3.u[i_ph_c3_mdl[i],:]/struct_mdl3.U0m,
                struct_mdl3.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        ax[i].scatter(struct_meas3.u[:,i_ph_c3_meas[i]]/struct_meas3.U0m,
                   struct_meas3.y_u/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        # ax.set_xlabel(r'$\frac{\overline{u}}{U_{0m}}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        fig.supxlabel(r'$\frac{\overline{u}}{U_{0m}}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_ylim([0, ya_max])
        ax[i].set_yscale('log')
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
    
    fname = exp_fld+"Case_3_u_vs_y_log"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot k/U_om^2 over y/a (separate plots)
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    xmax = 1e-3*(np.ceil(np.max(struct_mdl3.k[i_ph_c3_mdl,:]
                                /struct_mdl3.U0m**2)/1e-3)+1)
    
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl3.k[i_ph_c3_mdl[i],:]/struct_mdl3.U0m**2,
                struct_mdl3.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        
        ax[i].scatter(k_meas_c3[:,i_ph_c3_meas[i]]/struct_meas3.U0m**2,
                   struct_meas3.y_uuvv/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])
        
        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        fig.supxlabel(r'$\frac{k}{U_{0m}^2}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        # ax[i].set_xlabel(r'$\frac{k}{U_{0m}^2}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_xlim([0, xmax])
        ax[i].set_ylim([0, ya_max])
        ax[i].set_xticks(np.arange(0,xmax+1e-3, 2e-3))
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
        
    fname = exp_fld+f"Case_3_k_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot Reynolds stresses over y/a (separate plots)
    fig, ax = plt.subplots(1,4,sharey=True)
    xmin = 5e-4*(np.floor(min(np.min(rs_mdl_c3), np.min(rs_meas_c3))/5e-4))
    xmax = 5e-4*(np.ceil(np.max(rs_mdl_c3)/5e-4))
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(rs_mdl_c3[i,:],
                struct_mdl3.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        
        ax[i].scatter(rs_meas_c3[:,i],
                   struct_meas3.y_uv/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        fig.supxlabel(r'$\frac{-\overline{u^\prime v^\prime}}{U_{0m}^2}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        # ax[i].set_xlabel(r'$\frac{-\overline{u^\prime v^\prime}}{U_{0m}^2}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_xlim([xmin, xmax])
        ax[i].set_ylim([0, ya_max])
        # ax[i].set_xticks(np.arange(xmin,xmax, 5e-4))
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
        
    fname = exp_fld+f"Case_3_rs_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*struct_mdl3.U0m^2) over omega*t
    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(np.rad2deg(omegat_c3[i_ph_c3_mdl_cont]-start_ang),
            struct_mdl3.tau0[i_ph_c3_mdl_cont]/struct_mdl3.U0m**2,
            label = r"MatRANS Model",
            zorder=2)
    ax.scatter(struct_meas3.omegat_tau0[i_ph_c3_meas_cont],
               struct_meas3.tau0[i_ph_c3_meas_cont]/struct_meas3.U0m**2,
               label = r"Data from Jensen et. al.", 
               zorder=2, **mss["+"])
    
    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim ([-5,365])
    ax.set_xticks(np.arange(0,361,10),
                  labels=np.arange(0,361,10), rotation="vertical")
    ax.grid(zorder=1)
    ax.legend(loc="upper right")
    
    fname = exp_fld+"Case_3_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    print("Case 3 successfully plotted")
else:
    print("Plots for Case 3 not replotted")   
#%% Case 4: Turbulent wave boundary layer, rough wall
if 'struct_mdl4' in globals():
    case="4"  
    
    #Preparation
    omegat_c4 = omega*struct_mdl4.t
    
    #Detemine indices of datapoints for specified phase angles
    i_ph_c4_mdl = [np.argwhere(omegat_c4>=start_ang+np.deg2rad(pa-.5))[0][0] 
                       for pa in phase_angles]
    i_ph_c4_meas = [np.argwhere(struct_meas4.omegat>=pa-.5)[0][0] 
                 for pa in phase_angles]
    
    #Determine indices for the first half of the fifth cylcle
    i_ph_c4_mdl_cont = np.argwhere((omegat_c4>=start_ang-np.deg2rad(.5)) 
                           & (omegat_c4<=start_ang+np.deg2rad(360))).flatten()
    i_ph_c4_meas_cont = np.argwhere(struct_meas4.omegat_tau0<=360).flatten()
    
    #Calculate turbulent kinetic energy
    k_meas_c4 =.65 * (struct_meas4.uu + struct_meas4.vv)  #Eq. 10.21
    
    #Calculate Reynolds stresses
    du_dy_c4 = np.gradient(struct_mdl4.u[i_ph_c4_mdl,:], struct_mdl4.y, axis=1)
    rs_mdl_c4 = struct_mdl4.nu_t[i_ph_c4_mdl,:]*du_dy_c4/struct_mdl4.U0m**2
    rs_meas_c4 = -struct_meas4.uv[:,i_ph_c4_meas]/struct_meas4.U0m**2
    
    #Calculate the boundary layer thickness
    delta_c4 = case_tbl.loc[case, "a"]*3*np.pi/4\
            * np.sqrt(2/case_tbl.loc[case, "Re"])
    
if replot_tasks["C4"]:
    if not 'struct_mdl4' in globals():
        print("Data missing for Case 4 - Plots not replotted")

    #Plot u/U_om over y/a
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    ya_max = 1e-2*(np.ceil(np.max(struct_meas4.y_uuvv
                                /case_tbl.loc[case, "a"])/1e-2))
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl4.u[i_ph_c4_mdl[i],:]/struct_mdl4.U0m,
                struct_mdl4.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        ax[i].scatter(struct_meas4.u[:,i_ph_c4_meas[i]]/struct_meas4.U0m,
                   struct_meas4.y_u/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        # ax.set_xlabel(r'$\frac{\overline{u}}{U_{0m}}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        fig.supxlabel(r'$\frac{\overline{u}}{U_{0m}}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_ylim([0, ya_max])
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
    
    fname = exp_fld+"Case_4_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)

    #Plot u/U_om over y/a (semilog)
    fig, ax = plt.subplots(1,4,sharey=True, sharex=True)
    ya_max = 1e-2*(np.ceil(np.max(struct_meas4.y_uuvv
                                /case_tbl.loc[case, "a"])/1e-2))
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl4.u[i_ph_c4_mdl[i],:]/struct_mdl4.U0m,
                struct_mdl4.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        ax[i].scatter(struct_meas4.u[:,i_ph_c4_meas[i]]/struct_meas4.U0m,
                   struct_meas4.y_u/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        # ax.set_xlabel(r'$\frac{\overline{u}}{U_{0m}}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        fig.supxlabel(r'$\frac{\overline{u}}{U_{0m}}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_ylim([0, ya_max])
        ax[i].set_yscale('log')
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
    
    fname = exp_fld+"Case_4_u_vs_y_log"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)


    #Plot k/U_om^2 over y/a (separate plots)
    fig, ax = plt.subplots(1,4,sharey=True)
    xmax = 1e-3*(np.ceil(np.max(struct_mdl4.k[i_ph_c4_mdl,:]
                                /struct_mdl4.U0m**2)/1e-3)+1)
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(struct_mdl4.k[i_ph_c4_mdl[i],:]/struct_mdl4.U0m**2,
                struct_mdl4.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        
        ax[i].scatter(k_meas_c4[:,i_ph_c4_meas[i]]/struct_meas4.U0m**2,
                   struct_meas4.y_uuvv/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])
    
        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        fig.supxlabel(r'$\frac{k}{U_{0m}^2}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        # ax[i].set_xlabel(r'$\frac{k}{U_{0m}^2}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_xlim([0, xmax])
        ax[i].set_ylim([0, ya_max])
        # ax[i].set_xticks(np.arange(0,xmax+1e-3, 2.5e-3))
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
        
    fname = exp_fld+f"Case_4_k_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot Reynolds stresses over y/a (separate plots)
    fig, ax = plt.subplots(1,4,sharey=True)
    xmin = 1e-3*(np.floor(min(np.min(rs_mdl_c4), np.min(rs_meas_c4))/1e-3))
    xmax = 1e-3*(np.ceil(min(np.max(rs_mdl_c4), np.max(rs_meas_c4))/1e-3)+1)
    for i,phase_i in enumerate(phase_angles):
        ax[i].plot(rs_mdl_c4[i,:],
                struct_mdl4.y/case_tbl.loc[case, "a"],
                label = r"MatRANS Model",
                zorder=2)
        ax[i].scatter(rs_meas_c4[:,i],
                   struct_meas4.y_uv/case_tbl.loc[case, "a"],
                   label = r"Data from Jensen et. al.", 
                   zorder=2, **mss["+"])

        #Formatting
        if i == 0:
            ax[i].set_title(r"$\omega t = " + f"{phase_i}"
                            + r"\:\unit{\degree}$", y=1.04)
        else:
            ax[i].set_title(r"$" + f"{phase_i}" + r"\:\unit{\degree}$", y=1.04)
        fig.supxlabel(r'$\frac{-\overline{u^\prime v^\prime}}{U_{0m}^2}$',
                      fontsize = 1.5*mpl.rcParams['axes.labelsize'],
                      y=0, va="top")
        # ax[i].set_xlabel(r'$\frac{-\overline{u^\prime v^\prime}}{U_{0m}^2}$',
        #               fontsize = 1.5*mpl.rcParams['axes.labelsize'])
        if i == 0:
            ax[i].set_ylabel(r'$y/a$')
        ax[i].grid(zorder=1)
        ax[i].set_xlim([xmin, xmax])
        ax[i].set_ylim([0, ya_max])
        ax[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        
        if i == len(phase_angles)-1:
            ax[i].legend(loc="upper left", bbox_to_anchor=(1.05,1))
        
    fname = exp_fld+f"Case_4_rs_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    #Plot tau_0/(rho*struct_mdl4.U0m^2) over omega*t
    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(np.rad2deg(omegat_c4[i_ph_c4_mdl_cont]-start_ang),
            struct_mdl4.tau0[i_ph_c4_mdl_cont]/struct_mdl4.U0m**2,
            label = r"MatRANS Model",
            zorder=2)
    ax.scatter(struct_meas4.omegat_tau0[i_ph_c4_meas_cont],
               struct_meas4.tau0[i_ph_c4_meas_cont]/struct_meas4.U0m**2,
               label = r"Data from Jensen et. al.", 
               zorder=2, **mss["+"])

    #Formatting
    ax.set_ylabel(r'$\frac{\tau_0}{\rho \cdot U_{0m}^2}$',
                   fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_xlabel(r'$\omega t\:\unit{[\degree]}$')
    ax.set_xlim ([-5,365])
    ax.set_xticks(np.arange(0,361,10),
                  labels=np.arange(0,361,10), rotation="vertical")
    ax.grid(zorder=1)
    ax.legend(loc="upper right")

    fname = exp_fld+"Case_4_tau0_vs_omegat"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    print("Case 4 successfully plotted")
else:
    print("Plots for Case 4 not replotted")   
    
#%% Task 7: TKE Comparison for Case 3 & 4
if replot_tasks["T7"]:
    if not ('struct_mdl3' in globals() and 'struct_mdl4' in globals()):
        print("For Task 7, Task 3 & 4 need to be calculated")
        
    #Plot k/U_om^2 over y/a
    ya_max = 1e-2*(np.ceil(np.max(struct_meas3.y_uuvv
                                /case_tbl.loc[case, "a"])/1e-2))
    fig, ax = plt.subplots()
    ax.plot(struct_mdl3.k[i_ph_c3_mdl[2],:]/struct_mdl3.U0m**2,
            struct_mdl3.y/case_tbl.loc["3", "a"],
            label = r"Model -- Smooth wall",
            zorder=2)
    ax.plot(struct_mdl4.k[i_ph_c4_mdl[2],:]/struct_mdl4.U0m**2,
            struct_mdl4.y/case_tbl.loc["4", "a"],
            label = r"Model -- Rough wall",
            zorder=2)
    ax.scatter(k_meas_c3[:,i_ph_c3_meas[2]]/struct_meas3.U0m**2,
               struct_meas3.y_uuvv/case_tbl.loc["3", "a"],
               label = r"Data from Jensen et. al. -- Smooth wall", 
               zorder=2, **mss["+"])
    ax.scatter(k_meas_c4[:,i_ph_c4_meas[2]]/struct_meas4.U0m**2,
               struct_meas4.y_uuvv/case_tbl.loc["4", "a"],
               label = r"Data from Jensen et. al. -- Rough wall", 
               zorder=2, **mss["1"])

    #Formatting
    ax.set_xlabel(r'$\frac{k}{U_{0m}^2}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(r'$y/a$')
    ax.grid(zorder=1)
    ax.ticklabel_format(axis="both", style='scientific', 
                           scilimits=(0, 0))
    ax.set_ylim([0, ya_max])
    # ax.set_yscale("log")
    ax.legend(loc="upper right", ncols=2)
    
    fname = exp_fld+"Task_7_k_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    
    #Plot u/U_om over y/a
    ya_max = 1e-2*(np.ceil(np.max(struct_meas3.y_u
                                /case_tbl.loc[case, "a"])/1e-2))
    fig, ax = plt.subplots()
    ax.plot(struct_mdl3.u[i_ph_c3_mdl[2],:]/struct_mdl3.U0m,
            struct_mdl3.y/case_tbl.loc["3", "a"],
            label = r"Model -- Smooth wall",
            zorder=2)
    ax.plot(struct_mdl4.u[i_ph_c4_mdl[2],:]/struct_mdl4.U0m,
            struct_mdl4.y/case_tbl.loc["4", "a"],
            label = r"Model -- Rough wall",
            zorder=2)
    ax.scatter(struct_meas3.u[:,i_ph_c3_meas[2]]/struct_meas3.U0m,
               struct_meas3.y_u/case_tbl.loc["3", "a"],
               label = r"Data from Jensen et. al. -- Smooth wall", 
               zorder=2, **mss["+"])
    ax.scatter(struct_meas4.u[:,i_ph_c4_meas[2]]/struct_meas4.U0m,
               struct_meas4.y_u/case_tbl.loc["4", "a"],
               label = r"Data from Jensen et. al. -- Rough wall", 
               zorder=2, **mss["1"])

    #Formatting
    ax.set_xlabel(r'$\frac{u}{U_{0m}}$',
                  fontsize = 1.5*mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(r'$y/a$')
    ax.grid(zorder=1)
    ax.ticklabel_format(axis="both", style='scientific', 
                           scilimits=(0, 0))
    ax.set_ylim([0, ya_max])
    # ax.set_yscale("log")
    ax.legend(loc="upper left", ncols=2)
    
    fname = exp_fld+"Task_7_u_vs_y"
    fig.savefig(fname=fname+".svg")
    fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                   # inclusion in LaTeX
    plt.close(fig)
    
    print("Task 7 successfully plotted")
else:
    print("Plots for Task 7 not replotted")  
