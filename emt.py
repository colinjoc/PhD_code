import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
"""
This code calculates sheet resistances using an effective medium theory derived in the manuscript: "Effective medium theory for the conductivity of disor-
dered metallic nanowire networks" by O'Callaghan et al.
"""

def Nx_theory(W,C,L,nw):
    """
    calculate the number of nodes (junctions and wire segments) between electrodes, equation 29. W = distance between electrodes, C = 1 = constant
    L = wire length (in micrometer), nw = wire density 
    
    """
    lavg = L*nw/(2*a*L**2*nw**2 + nw) # average segment length, eqn 18. 

    nj = a*L**2*nw**2 # junction density, equation 21.

    Pi = (2 * nj - nw)/(3 * nj + nw) # relative probability of inner wire segmant, from equations 15.

    return W*C/(L*Pi) * np.log(6*L**2*nj*Pi)


def Ny_theory(H,L,nw):
    """
    calculate number of parallel paths, equation 26. H = length of electrodes, L = wire length, nw = wire density
    """
    return (H*L*nw)/math.pi

def gm_old(gj,gi,L,nw):
    """
    calculate effective conductance, equation 22. gj = junction conductance, gi = average wire segment conductance (note, not the conductance of an entire wire), L = wire length, nw = wire density.
    """
    ge = (-3*gi-gj+gi*nw*a*(L**2)-gj*nw*a*(L**2) + ( ( 3*gi+gj-gi*nw*a*(L**2)+gj*nw*a*(L**2))**2 + 12.*gi*gj*(-1 + nw*a*L**2)*(1 + 3*nw*a*L**2))**0.5 )/(2+6*nw*a*(L**2))
    return ge

def Re_old(C,W,H,L,nw,gj,rho,A0):
    """
    Calculate the sheet resistance, equation 23. Note eqn 23 should read nw^2 and not nw^1 in log expression. C is fitting constant in eqn 29, W = seperation between electrodes (units um), H = electrode length (units um), L = wire length (um), nw = wire denisty (#wires/(um^2)), 
    gj = junction conductance (units Ohm), rho = resistivity (Ohm * um), A0 = cross sectional area of wire (um^2).

    """    
    
    Nx = Nx_theory(C,W,L,nw) # Calculate nx with function Nx_theory()
    Ny = Ny_theory(H,L,nw) # calculate ny with Ny_theory()

    lavg = L*nw/(2*a*L**2*nw**2 + nw) # average wire segment length
    gi_avg = A0/(lavg*rho) # conductance of average wire segment 
    Rm = 1./gm_old(gj,gi_avg,L,nw) # calculate sheet resistance with Rm()
    return Rm*Nx/Ny # equation 23. 

# constant
a = 0.2827 # nj = a*L^2*nw^2, a = P*math.pi/2 in equation 21. 

rho = 0.0226 #  Ohm * um
A0 = 0.025**2*math.pi
W = 20. # electrde seperation, (um)
H = 20. #electrode height (um)
L = 6.7 #wire length (um)
nw = 0.2089 # wire density
gj = 1./11 # junction conductance
C = 1.#constant

print "Sheet resistance for nw = 0.2089 wires/um^2: ", Re_old(C,W,H,L,nw,gj,rho,A0) 

# example, plot sheet resistance vs wire density as in figure 5b.
dens_range = np.arange(0.15,1,0.01) # list of wire densities between 0.15 and 1, steps of 0.01
plt.plot(dens_range, Re_old(1,W,H,L,dens_range,1./11,rho,A0),label="gj = 11.",color = "blue") # blue line in Fig. 5b
plt.plot(dens_range, Re_old(1,W,H,L,dens_range,10**6.,rho,A0),label="gj = $10^6$.",color = "black",linestyle = "dashed") # dashed line in Fig. 5b. Use a really high conductance as cant use infinity for numerical reasons
plt.xlabel("$n_w$ (wires/$\mu m^{-2}$)",size = 20); plt.ylabel("$R_s$ ($\Omega$/sq)",size = 20) # axis labels
plt.legend() # show legend
plt.show() # show plot
#plt.savefig("wire_density.png")


plt.clf() # clear previously plot figures
# example, sheet resistance dependance on wire length.
len_range = np.arange(5,40,1) # list of wire lengths between 5 and 40 um, steps of 1
plt.plot(len_range, Re_old(1,W,H,len_range,0.3,1./11,rho,A0),label="$n_w$ = 0.3",color = "green") 
plt.plot(len_range, Re_old(1,W,H,len_range,0.5,1./11,rho,A0),label="$n_w$ = 0.5",color = "black", linestyle = "dashed") 
plt.xlabel("$L$ ($\mu m$)",size = 20); plt.ylabel("$R_s$ ($\Omega$/sq)",size = 20) # axis labels
plt.legend() # show legend
plt.show() # show plot
#plt.savefig("wire_length.png")


