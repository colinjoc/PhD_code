#!/usr/bin/env python
##############################################################################################
# This is a self consistent capacitance model simulation. Want to look at the frequency distributions of avalaunches.
##############################################################################################

from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import cascaded_union
import scipy.stats as stats
from scipy.misc import imread
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_edge_cut, minimum_node_cut
import k_shortest_paths as kpaths
import math
import random
import json
import os
import numpy as np
from collections import Counter, defaultdict
from itertools import islice, product, combinations
import pickle

from cvxopt import matrix, solvers
from cvxopt.base import sparse
from cvxopt.base import matrix as m
from cvxopt.lapack import *
from cvxopt.blas import *
import cvxopt.misc as misc
from symmlq import *

def Gfun(x,y,trans='N'):
    ''' Function that passes matrix A to the symmlq routine which solves Ax=B.'''
    gemv(Amatrix,x,y,trans)

def distance(x1,y1,x2,y2):
    return math.sqrt( (x1 - x2)**2 + (y1 - y2)**2  )

#------------------------------ parameters for the run trhough -------------------------------
# sample
box_length = 55
elec_length = box_length+10
lead_sep = box_length
density = 0.3
wire_length = 6.7
samples = 1000

short_path_num = 1

# Parameters for capacitance Gaussian distribution and charge scan
num_steps= 2000
step_size = 0.05

q0 = step_size
qf = num_steps*step_size
qstep = step_size

num_q = np.arange(q0,qf,qstep)

cav = 10.0
lower_c = 1.0
upper_c = np.inf
sigma = 0.0
if sigma != 0:
    cap_dist = stats.truncnorm((lower_c - cav) / sigma, (upper_c - cav) / sigma, loc=cav, scale=sigma)
else:
    cap0 = cav

print random.random(),np.random.sample()

#---------------------------------------------------------------------------------------------
# Parameters for solving the linear system of equations
tol=1e-10
show=False
maxit=None

nactivated_file	= open("capacitance_avalaunch_size_%s.txt"%box_length,"a")
time_file = open("capacitance_time_size_%s.txt"%box_length,"a")
for s in range(0,samples):	
    print s
    area = box_length**2
    box_x = box_length
    box_y = box_length
    num_junc = 0 # counts number if junctions

    nwires = area*density
    nwires_plus_leads = int(nwires+2)

    #----------------------- Create random stick coordinates -----------------------------
    # Sorting initial points of nwire sticks. have n = nwires in vector, each of value in range [0,1). muliply by legth of box to give x1,y1 coords.
    x1 = np.random.rand(int(nwires))*box_x
    y1 = np.random.rand(int(nwires))*box_y


    length_array = np.array([wire_length]*int(nwires))


    # Sorting the angles in radians (from 0 to 2 *pi). Then transforming them to degrees. np.random.rand(n) reurns n randomnmbers between [0,1).
    theta1 = np.random.rand(int(nwires))*2.0*math.pi
    theta1_degree = theta1 * 180.0 / math.pi

    # Sorting final points of nwire sticks with respect to theta1 and lengths.

    x2 = length_array * np.cos(theta1) + x1
    y2 = length_array * np.sin(theta1) + y1
    # Adding on the initial coordinate list (x1,y1) the points corresponding to the contact leads.
    x1 = np.insert(x1, 0, 0.0)
    x1 = np.insert(x1, 0,0)


    # Adding on the final coordinate list (x2,y2) the points corresponding to the contact leads.
    x2 = np.insert(x2, 0, 0.0)
    x2 = np.insert(x2, 0,0)
    ypostop = box_y/2 + elec_length/2
    yposbot = box_y/2 - elec_length/2
    y1 = np.insert(y1, 0,ypostop)
    y1 = np.insert(y1, 0,ypostop)
    y2 = np.insert(y2, 0,yposbot)
    y2 = np.insert(y2, 0, yposbot)



    xposleft = box_x/2-lead_sep/2
    xposright = box_x/2+lead_sep/2

    x1[0]= xposleft
    x2[0] = xposleft	
    x1[1] = xposright
    x2[1] = xposright

    # Zip (x1,y1)-(x2,y2) in accordance to shapely format 
    coords1 = zip(x1,y1)
    coords2 = zip(x2,y2)
    coords = zip(coords1,coords2)
    mlines = MultiLineString(coords)



    # all pair wire combination
    lines_comb = combinations(mlines, 2)

    # list storing True or False for pair intersection
    intersection_check = [pair[0].intersects(pair[1]) for pair in lines_comb]

    # list storing the indexes of intersection_check where the intersection between two wires is TRUE
    intersections = [i for i, x in enumerate(intersection_check) if x]

    # full list containing all non-repeated combinations of wires
    combination_index = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(mlines), 2))

    # list storing the connection (wire_i, wire_j) 
    intersection_index = [combination_index[intersections[i]] for i in range(len(intersections))]

    # checking the coordinates for interesection points
    inter_point_coll = []


    # total number of intersections
    nintersections = len(intersection_index)


    # dictionary containing wire index: [list of wires connected to it]
    wire_touch_list = defaultdict(list)
    for k, v in intersection_index:
        wire_touch_list[k].append(v)
        wire_touch_list[v].append(k)

    # dictionary containing wire index: [label nodes following MNR mapping]
    wire_touch_label_list = defaultdict(list)
    each_wire_inter_point_storage = defaultdict(list)
    new_pos_vec = defaultdict(list)
    label = 2

    # Starting creating the new node labelling according to MNR mapping
    for i in iter(wire_touch_list.viewitems()):
        for j in range(len(i[1])):
            cpoint = mlines[i[0]].intersection(mlines[i[1][j]])
            inter_point_coll.append(cpoint)
            npoint = (cpoint.x,cpoint.y)
            each_wire_inter_point_storage[i[0]].append(npoint)
            
            if i[0] > 1:
                wire_touch_label_list[i[0]].append(label)
                new_pos_vec[label].append(npoint)
                label += 1
            else:
                wire_touch_label_list[i[0]].append(i[0])
                if i[0] == 0:
                    new_pos_vec[0].append(npoint)
                if i[0] == 1:
                    new_pos_vec[1].append(npoint)


    maxl = label # dimension of the resistance matrix


    # flattening intersection_index for counting the amount of occurances of wire i
    flat = list(sum(intersection_index, ())) 
    conn_per_wire = Counter(flat)

    # checking for isolated wires
    complete_list = range(nwires_plus_leads)
    isolated_wires = [x for x in complete_list if not x in flat]


    # list containing the length segments of each wire (if it has a junction)
    each_wire_length_storage = [[] for _ in range(nwires_plus_leads)]  

    # Routine that obtains the segment lengths on each wire
    for i in each_wire_inter_point_storage:
        
        point_ini = Point(mlines[i].coords[0])  # Initial point of the wire
        point_fin = Point(mlines[i].coords[1])  # Final point of the wire
        wlength = point_ini.distance(point_fin) # Whole length
        wire_points = each_wire_inter_point_storage[i]

        dist = [0.0]*(len(wire_points)+1)
        for j in range(len(wire_points)):
            point = Point(wire_points[j])
            dist[j] = point_ini.distance(point)

        dist[-1] = wlength  # Whole length stored on the last component of dist vector.
        dist.sort() # Sorting in crescent order

        dist_sep = [0.0]*len(dist)
        dist_sep[0] = dist[0]
        dist_sep[1:len(dist)] = [dist[k]-dist[k-1] for k in range(1,len(dist))] # Segment lengths calculated for a particular wire
        each_wire_length_storage[i].append(dist_sep)

    # starting building resistance matrix

    # Matrix storing coordinates of intersection points
    interpos_matrix = np.zeros((maxl,maxl),dtype=object)
    innerpos_matrix = np.zeros(maxl,dtype=object)
    pos_vec = np.zeros(maxl,dtype=object)

    # list to store all junction resistances

    # file containing information of mr_matrix
    #matrix_mr = open('matrix_mr.dat', 'w')

    # start weighted graph. mnr graph
    G = nx.Graph()
    Gr = nx.Graph()

    # non-weighted graph
    H = nx.Graph()

    # store ONLY inner edges
    Gi = nx.Graph()
    factor = 10000.0  

    # Procedure to build the resistance matrix (mr_matrix) which assumes that the wires are not in the same potential.
    mnr_jda_junction_pairs=[]
    for iwire in xrange(nwires_plus_leads):
        # if each_wire_inter_point_storage[iwire] is not empty, the procedure can start.
        if each_wire_inter_point_storage[iwire]:
            # First we obtain the capacitance matrix elements related to the internal "capacitances"...
            # This procedure is similar to the one used above for building multilinestring segments.
            # Scan all the junction coordinate points stored in each_wire_inter_point_storage[iwire].
            for j, pointj in enumerate(each_wire_inter_point_storage[iwire]):
                # Reserve a particular point of this list.
                point = Point(pointj)
                # Scan all the junction coordinate points stored in each_wire_inter_point_storage[iwire].
                for i, pointw in enumerate(each_wire_inter_point_storage[iwire]):
                    # Reserve another point of this list.
                    comp_pointw = Point(pointw)
                    # Calculate the distance between point - comp_pointw
                    inter_dist = point.distance(comp_pointw)
                    # A 4 digit precision for this distance must be imposed otherwise, a comparison between exact numbers can fail.
                    round_inter_dist = round(inter_dist, 4)
                    # Check if each_wire_length_storage[iwire] contains a segment length that matches round_inter_dist.
                    # If it does, we found a capacitance matrix element correspondent to an inner "capacitance".
                    for il in each_wire_length_storage[iwire][0]:
                        value = float(il)
                        value = round(value,4)
                        if value == round_inter_dist and value != 0:
                            if iwire != 0 and iwire != 1:# and mr_matrix[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] == 0.0:
                                inner_resis = (float(value))
                                # ELEMENT FOR mr_matrix FOUND! Its labels are stored in wire_touch_label_list.
                                innerpos_matrix[wire_touch_label_list[iwire][i]] = comp_pointw
                                innerpos_matrix[wire_touch_label_list[iwire][j]] = point
                                Gi.add_edge(wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j])

                                check_node = G.has_node(wire_touch_label_list[iwire][i])
                                
                                if not(check_node):
                                    posi = (comp_pointw.x, comp_pointw.y)
                                    G.add_node(wire_touch_label_list[iwire][i], pos=posi)
                                    pos=nx.get_node_attributes(G,'pos')
                                
                                check_node = G.has_node(wire_touch_label_list[iwire][j])
                                if not(check_node):
                                    posj = (point.x, point.y)
                                    G.add_node(wire_touch_label_list[iwire][j], pos=posj)
                                    pos=nx.get_node_attributes(G,'pos')

                                eg = (wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j])
                                check_edge = G.has_edge(*eg)

                                if not(check_edge):
                                    G.add_edge(wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j],weight=inner_resis)

                                check_node = Gr.has_node(wire_touch_label_list[iwire][i])
                                
                                if not(check_node):
                                    posi = (comp_pointw.x, comp_pointw.y)
                                    Gr.add_node(wire_touch_label_list[iwire][i], pos=posi)
                                    pos=nx.get_node_attributes(Gr,'pos')
                                
                                check_node = Gr.has_node(wire_touch_label_list[iwire][j])
                                if not(check_node):
                                    posj = (point.x, point.y)
                                    Gr.add_node(wire_touch_label_list[iwire][j], pos=posj)
                                    pos=nx.get_node_attributes(Gr,'pos')

                                eg = (wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j])
                                check_edge = Gr.has_edge(*eg)


                # Procedure to find capacitance matrix elements for the junctions...
                # Scan the list (wire_touch_list) which stores the label of wires to which iwire is connected.
                for k, label in enumerate(wire_touch_list[iwire]):
                    # For a particular wire (labelled as label) in wire_touch_list, scan its junction coordinate points stored in (each_wire_inter_point_storage[label].
                    for kk, pointk in enumerate(each_wire_inter_point_storage[label]):
                        # Reserve one of the junction points.
                        pointk = Point(pointk)
                        # Calculate the distance between point - pointk
                        inter_dist = point.distance(pointk)
                        # A 4 digit precision for this distance must be imposed otherwise, a comparison between exact numbers can fail.
                        round_inter_dist = round(inter_dist, 4)
                        # If round_inter_dist is ZERO, it means we FOUND a junction capacitance element that is stored in mr_matrix.
                        # Its value is computed from the Gaussian distribution.
                        if round_inter_dist == 0:# and mr_matrix[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]]== 0:

                            interpos_matrix[iwire, label] = pointk
                            interpos_matrix[label,iwire] = pointk
                            pos_vec[wire_touch_label_list[label][kk]] = pointk
                            pos_vec[wire_touch_label_list[iwire][j]] = pointk

                            check_node = G.has_node(wire_touch_label_list[iwire][j])
                            if not(check_node):
                                posj = (point.x, point.y)
                                G.add_node(wire_touch_label_list[iwire][j], pos=posj)
                                pos=nx.get_node_attributes(G,'pos')

                            check_node = G.has_node(wire_touch_label_list[label][kk])
                            if not(check_node):
                                posk = (pointk.x, pointk.y)
                                G.add_node(wire_touch_label_list[label][kk], pos=posk)
                                pos=nx.get_node_attributes(G,'pos')

                            eg = (wire_touch_label_list[iwire][j],wire_touch_label_list[label][kk])
                            check_edge = G.has_edge(*eg)

                            if not(check_edge):
                                mnr_jda_junction_pairs.append([[iwire,label],[wire_touch_label_list[iwire][j],wire_touch_label_list[label][kk]]])

                            check_node = Gr.has_node(wire_touch_label_list[iwire][j])
                            if not(check_node):
                                posj = (point.x, point.y)
                                Gr.add_node(wire_touch_label_list[iwire][j], pos=posj)
                                pos=nx.get_node_attributes(Gr,'pos')

                            check_node = Gr.has_node(wire_touch_label_list[label][kk])
                            if not(check_node):
                                posk = (pointk.x, pointk.y)
                                Gr.add_node(wire_touch_label_list[label][kk], pos=posk)
                                pos=nx.get_node_attributes(Gr,'pos')

                            eg = (wire_touch_label_list[iwire][j],wire_touch_label_list[label][kk])
                            check_edge = Gr.has_edge(*eg)



    # ---------------------------- Important ---------------------------------------
    # mnr_jda_junction_pairs contains a list of junction edges in mnr and in jda. From the wire_touch_list we shall build a JDA matrix.
    # The graph G has all nodes inputted and neighbouring nodes on the same wire are connected normally. The junction edges are not included, they will be included when a capacitor blows in the JDA model C.

    const = 1.0
    # Lists to store capacitance and breakdown voltages
    cap_list = []
    cap_list_all = []
    vb_list = []
    point_count = 0

    # Breakdown voltages also within Gaussian distribution
    vb_av = 1.0
    lower_vb = 0.001
    upper_vb = np.inf
    sigma_vb = 0.0 
    if sigma_vb != 0:
        vb_dist = stats.truncnorm((lower_vb - vb_av) / sigma_vb, (upper_vb - vb_av) / sigma_vb, loc=vb_av, scale=sigma_vb)
    else:
        vb0 = vb_av

    # Capacitance matrix 
    mc_matrix = np.zeros((nwires_plus_leads,nwires_plus_leads))
    mr_jda_matrix = np.zeros((nwires_plus_leads,nwires_plus_leads))
    # Breakdown voltage matrix
    vb_matrix = np.zeros((nwires_plus_leads,nwires_plus_leads))

    #building mc_matrix from wire_touch_list. 
    for wire1 in wire_touch_list:
        for wire2 in wire_touch_list[wire1]:
            cap = cap0 # add in distribution for capacitance here
            mc_matrix[wire1][wire2]=-cap
            mc_matrix[wire2][wire1]=-cap
            if sigma_vb != 0:
                y = vb_dist.rvs()
            else:
                # y = random.uniform(0.001, 5.0)
                y = const / cap

            vb_matrix[wire1,wire2] = y
            vb_matrix[wire2,wire1] = y
            vb_list.append(y)
            
    # Adds the non-zero elements on each row of Mc.
    sum_rows_mc = mc_matrix.sum(1) 
    sum_rows_mr_jda = mr_jda_matrix.sum(1) 

    # Place the sum in the diagonal of Mc.
    np.fill_diagonal(mc_matrix, abs(sum_rows_mc))  
    np.fill_diagonal(mr_jda_matrix, abs(sum_rows_mr_jda))  


    # array storing the index of the non-zero elements of vb_matrix (only on its upper diagonal part)
    neigh_triu = np.nonzero(np.triu(vb_matrix))

    # Parameters for solving the linear system of equations
    tol=1e-9
    show=False
    maxit=None

    # Activated junctions list
    inter_point_act = []

    # counter of activated junctions
    nactivated = 0
    nactivated_total = 0
    nactivated_list = []

    # Store amount of junctions activated at this charge
    nactivated_per_charge = []

    #
    ncapacitors = nintersections

    # Time of an avalanche
    time_list = []
    time = 0

    # Tells if leads 0 and 1 has a path

    avalaunche_activations = []

    #Internal energy before
    inter_energy_before = 0.0
    path_flag = False
    c_model_activated_all=[]; c_pickle = []
    old_q = 0
    for q in num_q:
        time_count = 0
        if nx.has_path(G,0,1):
                if path_flag ==False:
                    path_form_charge = old_q; path_flag = True
        else:        
            old_q = q
            c_model_activated=[]
            # Initiating charge vector
            qv = np.zeros(nwires_plus_leads)
            qv[0] = +q
            qv[1] = -q
            Qmatrix = m(qv)

            # mc_matrix will be updated while noups=True
            noups = True

            # Store positions of activated junctions at this charge
            inter_point_act_charge = []

            # Counts how many junctions were activated at this charge
            nactivated_total = 0
            
            # Counts the time of the avalanche
            time = 0

            # Loop for tries (mc_matrix might be updated at the same charge)
            while(noups):
    	        time_count+=1
                mc_matrix_form = m(mc_matrix)
                Amatrix = mc_matrix_form
                elec_pot_mc = symmlq( Qmatrix, Gfun, show=show, rtol=tol, maxit=maxit)

                # Set potential vector in array format
                elec_pot_mc_orib = [value for value in elec_pot_mc[0]]
                elec_pot_mc_ori = np.asarray(elec_pot_mc_orib)

                # Calculating internal energy E = 0.5 * (U, CV) or E = 0.5 VQ
                mc13 = np.dot(elec_pot_mc_ori, qv)
                inter_energy = 0.5 * mc13

                # obtain equivalent capacitance between electrodes
                cap_eq = q/(elec_pot_mc[0][0] - elec_pot_mc[0][1])
                work = 0.5 * cap_eq * (elec_pot_mc[0][0] - elec_pot_mc[0][1])**2

                # Reset diagonal elements before updating Mc!
                np.fill_diagonal(mc_matrix, 0.0)

                # Counts how many junctions were activated at this try
                nactivated = 0
                
                # Transfer intersection points activated on the step before
                inter_point_act_before = inter_point_act[:]

                # Store intersection points of junctions activated NOW
                inter_point_act_now = []
            
                # Store diff = |U[i]-U[j]|
                diff_list = []

                # Loop over all inter-wire connections
                for i in range(len(neigh_triu[0])):

                    # Obtain diff = |U[i]-U[j]| for a pair of connected i,j wires
                    diff = abs(elec_pot_mc[0][neigh_triu[0][i]]-elec_pot_mc[0][neigh_triu[1][i]])
                    diff_list.append(diff)

                    # Obtain intersection point of wires i,j
                    ipos = interpos_matrix[neigh_triu[0][i],neigh_triu[1][i]]
                    # if diff[i,j] > vb_matrix[i,j] and |mc_matrix[i,j]| > tol, junction i,j is activated! nactivated adds 1.
                    if diff > vb_matrix[neigh_triu[0][i],neigh_triu[1][i]] and diff < 1e4 and abs(mc_matrix[neigh_triu[0][i],neigh_triu[1][i]])>=lower_c:
                        nactivated += 1
                        
                        # Store the activated position
                        inter_point_act.append(ipos)
                        inter_point_act_charge.append(ipos)
                        inter_point_act_now.append(ipos)
                        
                        for pair in mnr_jda_junction_pairs:
                             if pair[0] == [ neigh_triu[0][i],neigh_triu[1][i]] or pair[0] == [ neigh_triu[1][i],neigh_triu[0][i]]:
                                 node1 = pair[1][0]; node2=pair[1][1]
                                 G.add_edge(node1,node2,weight=11.0)
                                 flag = True
                                 for c_row in c_model_activated_all:
                                     for pair in c_row:
                                         if [node1,node2] == pair or [node2,node1] == pair: 
                                             flag = False
                                 if flag: 
                                     c_model_activated.append([node1,node2])

                        # Update mc_matrix[i,j] element (BUT ONLY OFF-DIAGONAL ELEMENT)
                        mc_matrix[neigh_triu[0][i],neigh_triu[1][i]] = -1e-3
                        mc_matrix[neigh_triu[1][i],neigh_triu[0][i]] = -1e-3


                # adds the amount of activated junctions on this TRY
                nactivated_total += nactivated        
                ncapacitors -= nactivated
                
                # DO NOT FORGET TO UPDATE DIAGONAL ELEMENTS OF MC_MATRIX
                # Adds the non-zero elements on each row of Mc.
                sum_rows_mc = mc_matrix.sum(1)

                # Place the sum in the diagonal of Mc.
                np.fill_diagonal(mc_matrix, abs(sum_rows_mc))      

                # Average of diff = |U[i]-U[j]| 
                diff_array = np.asarray(diff_list)
                diff_av = np.mean(diff_array)
                #print 'Average of diff = ', diff_av

                # if the amount of activated junctions at this TRY is zero, while(noups) loop can stop
                # Otherwise, the time of the avalanche is incremented
                if nactivated == 0:
                    noups = False
                    #print 'Final U - Initial U = ', inter_energy, ' - ', inter_energy_before
                    delta_energy = inter_energy - inter_energy_before
                    #print 'delta_energy = ', delta_energy
                    inter_energy_before = inter_energy
                else:
                    time += 1

            # Store the amount of activated junctions this current
            nactivated_list.append(nactivated)
            if nactivated_total!=0:
                nactivated_file.write("%s\n" %nactivated_total)
                print s, "file write"
                time_file.write("%s\n" %time)
            nactivated_per_charge.append(nactivated_total)
            if c_model_activated!=[]:	
                c_model_activated_all.append(c_model_activated)
            c_model_activated= [q,c_model_activated]
            c_pickle.append(c_model_activated)

time_file.close()
nactivated_file.close()	

