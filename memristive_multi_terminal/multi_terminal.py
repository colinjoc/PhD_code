"""
Written by Colin O'Callaghan. This code defines a class MultiTermninalNw to generate a multi terminal nanowire network.
See mnr_jda_scan.py for implementation of numerical simulations on the nanowire network.
This is python 2.7, there is an issue with how the code is written and shapely in python3. 
"""
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_edge_cut, minimum_node_cut
import math
import random
import json
import os
import numpy as np
from collections import Counter, defaultdict
from itertools import islice, product, combinations

from cvxopt import matrix, solvers
from cvxopt.base import sparse
from cvxopt.base import matrix as m
from cvxopt.lapack import *
from cvxopt.blas import *
import cvxopt.misc as misc
from symmlq import *
import time

class MultiTerminalNw:
    np.random.seed(0)
    #random.random(0)
    def __init__(self,box_length,density):
        wire_length = 7.0        
        area = box_length**2
        box_x = box_length
        box_y = box_length
        num_junc = 0 # counts number if junctions
        #------- Files containing information of the particular sample.------
        nwires = int(area*density)
        self.nwires = nwires
        #----------------------- Create random stick coordinates -----------------------------
        # Sorting initial points of nwire sticks. have n = nwires in vector, each of value in range [0,1). muliply by legth of box to give x1,y1 coords.
        x1 = np.random.rand(nwires)*box_x
        y1 = np.random.rand(nwires)*box_y

        length_array = np.zeros(nwires)
        length_array.fill(wire_length)

        # Sorting the angles in radians (from 0 to 2 *pi). Then transforming them to degrees. np.random.rand(n) reurns n randomnmbers between [0,1).
        theta1 = np.random.rand(nwires)*2.0*math.pi
        theta1_degree = theta1 * 180.0 / math.pi

        # Sorting final points of nwire sticks with respect to theta1 and lengths.

        x2 = length_array * np.cos(theta1) + x1
        y2 = length_array * np.sin(theta1) + y1
    
        # Zip (x1,y1)-(x2,y2) in accordance to shapely format 
        coords1 = zip(x1,y1)
        coords2 = zip(x2,y2)
        coords = zip(coords1,coords2)
        mlines = MultiLineString(coords)
        self.coords = coords
        self.box_length = box_length        
        self.four_electrode()

    def four_electrode(self):
        box_length = self.box_length
        gap_size = 1.0/3*box_length
        self.electrodes = [0,1,2,3]
        for xcoord in [0,box_length]:
        
            for ycoords in [[0,gap_size],[2*gap_size,box_length]]:
                elec_coord = [((xcoord,ycoords[0]),(xcoord,ycoords[1]))]
                self.coords = elec_coord+self.coords

    def mmr_mr_graph_build(self):
        
        Roff = 10000.0
        rho0 = 22.63676*(0.001) # Nano Ohm m
        # Characteristic diameter
        D0 = 60.0*(0.001) # Nm
        # Cross section areas
        A0 = math.pi * (D0 / 2.0)**2
        
        mlines = MultiLineString(self.coords)
        nwires_plus_leads = int(len(self.coords))
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
        inter_point_coll = [pair[0].intersection(pair[1]) for pair in combinations(mlines, 2)]
        # eliminating empty shapely points from the previous list
        no_empty_inter_point_coll = [inter_point_coll[intersections[i]] for i in range(len(intersections))]
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
        label = 4

        # Starting creating the new node labelling according to MNR mapping
        for i in iter(wire_touch_list.viewitems()):            
            for j in range(len(i[1])):
                cpoint = mlines[i[0]].intersection(mlines[i[1][j]])
                npoint = (cpoint.x,cpoint.y)

                each_wire_inter_point_storage[i[0]].append(npoint)
                if i[0] >= len(self.electrodes):
                    wire_touch_label_list[i[0]].append(label)
                    label += 1
                else:
                    wire_touch_label_list[i[0]].append(i[0])
                maxl = label # dimension of the capacitance matrix
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
        G = nx.Graph()
        G.add_nodes_from(range(2*nintersections))
        mr_matrix_plus = np.zeros((2*nintersections,2*nintersections))
        inner_count = 0
        inter_count = 0
        # Same procedure to build mr_matrix_plus as described above (out of the loop)
        mr_jda_matrix = np.zeros((nwires_plus_leads,nwires_plus_leads))
        for wire1 in wire_touch_list:
            for wire2 in wire_touch_list[wire1]:
                # mr_jda_res
                mr_jda_matrix[wire1,wire2] = -1.0/Roff
                mr_jda_matrix[wire2,wire1] =  mr_jda_matrix[wire1,wire2]
                        
        mr_matrix_info = np.zeros((2*nintersections,2*nintersections))

        for iwire in xrange(nwires_plus_leads):
            if each_wire_inter_point_storage[iwire]:
                for j, pointj in enumerate(each_wire_inter_point_storage[iwire]):
                    point = Point(pointj)
                    for i, pointw in enumerate(each_wire_inter_point_storage[iwire]):
                        comp_pointw = Point(pointw)
                        inter_dist = point.distance(comp_pointw)
                        round_inter_dist = round(inter_dist, 4)
                        for il in each_wire_length_storage[iwire][0]:
                            value = float(il)
                            value = round(value,4)
                            if value == round_inter_dist and value != 0:
                                inner_resis = (float(value) * rho0 / A0)
                                if not(iwire in self.electrodes)  and mr_matrix_plus[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] == 0.0:                                
                                    mr_matrix_plus[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] = -1.0/inner_resis
                                    mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[iwire][i]] = -1.0/inner_resis
                                    G.add_edge(wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j])
                                    inner_count += 1
                    for k, label in enumerate(wire_touch_list[iwire]):
                        for kk, pointk in enumerate(each_wire_inter_point_storage[label]):
                            pointk = Point(pointk)
                            inter_dist = point.distance(pointk)
                            round_inter_dist = round(inter_dist, 4)
                            if round_inter_dist == 0 and mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] == 0:
                                G.add_edge(wire_touch_label_list[label][kk],wire_touch_label_list[iwire][j])
                                r0 = -1/Roff
                                mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = r0
                                mr_matrix_plus[wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j]] = r0              
                                mr_matrix_info[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = 1.0
                                mr_matrix_info[wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j]] = 1.0

           
        self.mnr_neigh = np.nonzero(np.triu(mr_matrix_info))
        sum_rows_mr_plus = mr_matrix_plus.sum(1)
        np.fill_diagonal(mr_matrix_plus, abs(sum_rows_mr_plus))
        mr_nozero_rows_plus = mr_matrix_plus[~(mr_matrix_plus==0).all(1),:]
        mr_nonconnected_plus = mr_nozero_rows_plus[:,~(mr_nozero_rows_plus==0).all(0)]
        self.nintersections = nintersections
        self.mnr_mr_matrix = mr_nonconnected_plus
        self.jda_mr_matrix = mr_jda_matrix
        self.mrm_graph = G
        
    def plotNw(self):

        plt.clf()
        plt.axis('off')
        mlines =  MultiLineString(self.coords)
        flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
        for i,line in enumerate(mlines):
            xl, yl = line.xy
            if i>3:
                plt.plot(xl, yl, color="black", lw=0.5, zorder=0)
            else:
                plt.plot(xl, yl, color="red", lw=0.5, zorder=1,linewidth = 2)

        #plt.figure(figsize=(1000,1000))
        plt.savefig('network.png', format='png',dpi=1200)
        #plt.show()

    def plotNwpaths(self):

        plt.clf()
        plt.axis('off')
        mlines =  MultiLineString(self.coords)
        flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
        for i,line in enumerate(mlines):
            xl, yl = line.xy
            if i>3:
                plt.plot(xl, yl, color="black", lw=0.5, zorder=0)
            else:
                plt.plot(xl, yl, color="red", lw=0.5, zorder=2,linewidth = 2)

        #plt.figure(figsize=(1000,1000))
        import itertools

        my_list = [0,1,2,3]
        already = []
        for pair in itertools.product(my_list, repeat=2):
            if pair[0]!=pair[1] and not(pair[::-1] in already):
                print(pair)
                already.append(pair)
                
                x1 = self.coords[pair[0]][0][0]                
                x2 = self.coords[pair[1]][0][0]                
                
                y1 = (self.coords[pair[0]][0][1] + self.coords[pair[0]][1][1])/2
                y2 = (self.coords[pair[1]][0][1] + self.coords[pair[1]][1][1])/2
    
                plt.plot([x1,x2],[y1,y2], linestyle = "--", lw=2, zorder=1)


        #plt.savefig('network.png', format='png',dpi=1200)
        plt.show()





