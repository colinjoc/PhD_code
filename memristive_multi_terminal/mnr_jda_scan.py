from multi_terminal import MultiTerminalNw
import numpy as np  
from cvxopt import matrix, solvers
from cvxopt.base import sparse
from cvxopt.base import matrix as m
from cvxopt.lapack import *
from cvxopt.blas import *
import cvxopt.misc as misc
from symmlq import *


def mnr_scan(Nw,in_node,out_node):
    t0 = time.time()

    def Gfun(x,y,trans='N'):
        ''' Function that passes matrix A to the symmlq routine which solves Ax=B.'''
        gemv(Amatrix,x,y,trans)

    resist_info = open("mnr_conductance_curve_nodes_%s_%s.txt"%(in_node,out_node),"w")
    num_i = np.arange(0.001,5,0.001)
    Nw.mmr_mr_graph_build()
    mr_matrix = Nw.mnr_mr_matrix
    mr_mnr_matrix = mr_matrix
    neigh_mr = Nw.mnr_neigh
    nwires_plus_leads = len(Nw.coords)

    alph = 0.05#3.5248
    expon = 1.1
    Roff = 10000.0
    Ron = 11.0


    for ic in num_i:
        
        # Initiating fixed current vector
        iv = np.zeros(len(mr_matrix))
        iv[in_node] = +ic
        iv[out_node] = -ic
        Imatrix = m(iv)

        tol=1e-10
        show=False
        maxit=None


        mr_matrix_form = m(mr_matrix)
        Amatrix = mr_matrix_form
        elec_pot_mr = symmlq( Imatrix, Gfun, show=show, rtol=tol, maxit=maxit)


        resistance = (elec_pot_mr[0][in_node] - elec_pot_mr[0][out_node])/ic
        #resistance = (pot[0]-pot[1])/ic

        print('Sheet Resistance: ', resistance, ic)
        resist_info.write('%s   %s\n' % (ic, 1.0/resistance))

        #ax2 = plt.subplot2grid((1,2),(0,1),colspan=3)


        # Loop over all inter-wire connections 
        for i in range(len(neigh_mr[0])):

            diffmr = abs(elec_pot_mr[0][neigh_mr[0][i]] - elec_pot_mr[0][neigh_mr[1][i]])

            # current passing through the junction
            jcurr = diffmr/(abs(1.0/mr_matrix[neigh_mr[0][i],neigh_mr[1][i]]))

            # resistance updated as a function of its current
            new_resis=1.0/(alph*jcurr**expon)
            
            # thresholds (the junction resistance cannot be bigger than Roff or smaller than Ron)
            if new_resis > Roff:
                new_resis = Roff
            if new_resis < Ron:
                new_resis = Ron
                Ron_list.append([neigh_mr[0][i],neigh_mr[1][i],ival])
            # modify resistance of the junction
            mr_matrix[neigh_mr[0][i],neigh_mr[1][i]] = -1.0/new_resis
            mr_matrix[neigh_mr[1][i],neigh_mr[0][i]] = -1.0/new_resis

        # Reset diagonal elements before updating Mr!
        np.fill_diagonal(mr_matrix, 0.0)

        sum_rows_mr = mr_matrix.sum(1)
        np.fill_diagonal(mr_matrix, abs(sum_rows_mr))  

    Ron_save = open("mnr_Ron_list_nodes_%s_%s_scan_end_%s.txt"%(in_node,out_node,ival),"w")
    for row in Ron_list:
        Ron_save.write("%s %s %s\n" %(row[0],row[1], row[2]))
    Ron_save.close()

    resist_info.close()
    tf = time.time()
    print("TOTAL TIME: %s" %(tf-t0))



def save_Ron(Ron_list,in_node,out_node,ival):
    Ron_save = open("jda_Ron_list_nodes_%s_%s_scan_end_%s.txt"%(in_node,out_node,ival),"w")
    for row in Ron_list:
        Ron_save.write("%s %s %s\n" %(row[0],row[1], row[2]))
    Ron_save.close()


def jda_scan(Nw,in_node,out_node):
    def Gfun(x,y,trans='N'):
        ''' Function that passes matrix A to the symmlq routine which solves Ax=B.'''
        gemv(Amatrix,x,y,trans)
    resist_info = open("jda_conductance_curve_nodes_%s_%s.txt"%(in_node,out_node),"w")
    num_i = np.arange(0.001,5,0.001)
    Nw.mmr_mr_graph_build()
    mr_matrix = Nw.jda_mr_matrix
    mr_jda_matrix = mr_matrix
    nwires_plus_leads = len(Nw.coords)

    alph = 0.05#3.5248
    expon = 1.1
    Roff = 10000.0
    Ron = 11.0

    for ival in num_i:
        Ron_list = []
        r_model_activated=[]
        # Initiating fixed current vector
        iv = np.zeros(len(mr_matrix))
        iv[in_node] = +ival
        iv[out_node] = -ival
        Imatrix = m(iv)

        mr_matrix_form = m(mr_matrix)
        Amatrix = mr_matrix_form

        tol=1e-10
        show=False
        maxit=None

        elec_pot_mr = symmlq( Imatrix, Gfun, show=False, rtol=tol, maxit=maxit)

        resistance = abs(elec_pot_mr[0][in_node] - elec_pot_mr[0][out_node])/ival
        conductance = 1.0/resistance

        print('Sheet Resistance: %s, ic: %s ' %(resistance, ival))
        resist_info.write('%s   %s\n' % (ival, conductance))

        #have to update Gr!

        # Loop over all inter-wire connections 
        for i in range(nwires_plus_leads):
            for j in range(i+1,nwires_plus_leads):    
                if mr_jda_matrix[i][j]!=0:
                    diffmr = abs(elec_pot_mr[0][i] - elec_pot_mr[0][j])

                    # current passing through the junction
                    jcurr = diffmr/(abs(1.0/mr_jda_matrix[i][j]))

                    # resistance updated as a function of its current
                    new_resis=1.0/(alph*jcurr**expon)

                    # thresholds (the junction resistance cannot be bigger than Roff or smaller than Ron)
                    if new_resis > Roff:
                        new_resis = Roff
                    if new_resis <= Ron:
                        new_resis = Ron
                        Ron_list.append([i,j,ival])
                    # modify resistance of the junction
                    mr_jda_matrix[i][j] = -1.0/new_resis
                    mr_jda_matrix[j][i] = -1.0/new_resis
        np.fill_diagonal(mr_jda_matrix, 0.0)
        sum_rows_mr = mr_jda_matrix.sum(1)
        # Place the sum in the diagonal of Mr.
        np.fill_diagonal(mr_jda_matrix, abs(sum_rows_mr))  
    resist_info.close()    
    save_Ron(Ron_list,in_node,out_node,ival)    

electrode_nodes = [1,3]
Nw = MultiTerminalNw(20.,0.3)
Nw.plotNwpaths()

in_node =electrode_nodes[0]
out_node = electrode_nodes[1]
jda_scan(Nw,in_node,out_node)



