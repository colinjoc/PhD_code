##########################################################################
##	This code creates a box of dimensions given below and then creates 	##
##	a random network inside this box. 
## sheet resistance is calculated for different nanowire material properties.
##########################################################################


from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import cascaded_union
from scipy.misc import comb
from itertools import product
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from itertools import islice
from cvxopt import matrix, solvers
from cvxopt.base import sparse
from cvxopt.base import matrix as m
from cvxopt.lapack import *
from cvxopt.blas import *
import cvxopt.misc as misc
from symmlq import *
import networkx as nx
from itertools import islice, combinations
from collections import Counter, defaultdict
import random
#---------------------- parameters ---------------------
A0 = math.pi*((42*0.001)/2)**2 #cross section (in um^2)
wire_length= 6.7 #average wire length, in um
density = 0.4


#density = 0.5 # wires per um^2
box_length = 20.0 # nwn is box_length^2 um^2 in size

#samples = 3
elec_length=box_length # electrode length 
box_y = box_length
lead_sep = box_length

dens_temp=[]
avg_res_temp=[]
st_dev_temp=[]


# Name of file to save resistance between nodes data into

# ------------------------ functions ----------------------
def Gfun(x,y,trans='N'):
    ''' Function that passes matrix A to the symmlq routine which solves Ax=B.'''
    gemv(Amatrix,x,y,trans)

def distance(x1,y1,x2,y2):
    return math.sqrt( (x1 - x2)**2 + (y1 - y2)**2  )
#------------------------------------------------------
# Box size: box_x \pm tol x box_y \pm tol

# Parameters of the Gamma distribution of wire lengths. lmax is the upper limit of the distribution.
wsk = 2.5
wsb = 2.7
lower_l = 2.2
upper_l = np.inf
sigmal = 2.0
lmean = 7.0
dist = False

# fix random number generator seeds
random.seed(11)
np.random.seed(11)

#material properties
material_properties = {
                        "Ag":{"Rj": 11.0, "rho": 19.26*0.001},
                        "Cu":{"Rj": 205.7, "rho": 20.1*0.001},
                        "Ni":{"Rj":81.4 , "rho": 62*0.001},
                        "AgCu":{"Rj":23.0 , "rho":30.4*0.001}                    
                        }


area = box_length**2
box_x = box_length
box_y = box_length
num_junc = 0 # counts number if junctions
#------- Files containing information of the particular sample.------
nwires = area*density
        
#----------------------- Create random stick coordinates -----------------------------
# Sorting initial points of nwire sticks. have n = nwires in vector, each of value in range [0,1). muliply by legth of box to give x1,y1 coords.
x1 = np.random.rand(int(nwires))*box_x
y1 = np.random.rand(int(nwires))*box_y
length_array = np.zeros(int(nwires))      
   
   


if dist == True:
	lengths = stats.truncnorm((lower_l - lmean) / sigmal, (upper_l - lmean) / sigmal, loc=lmean, scale=sigmal)
	length_array = lengths.rvs(size=nwires)
else:
	length_array.fill(wire_length)

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
nwires_plus_leads = int(nwires+2)

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
label = 2

# Starting creating the new node labelling according to MNR mapping
for i in iter(wire_touch_list.viewitems()):
	for j in range(len(i[1])):
		cpoint = mlines[i[0]].intersection(mlines[i[1][j]])
		npoint = (cpoint.x,cpoint.y)
		each_wire_inter_point_storage[i[0]].append(npoint)
	
		if i[0] > 1:
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


for material in ["Ag","Cu","Ni","AgCu"]:
    print "y"
    R_junc = material_properties[material]["Rj"]
    rho0 = material_properties[material]["rho"]
    G = nx.Graph() # a class from the networkx module. we can create a networkx graph to perform specific graph related operations.
    G.add_nodes_from(range(2*nintersections)) # add the nodes to the graph. Note we are using the MNR mapping so the number of nodes is twice the number of intersections
    mr_matrix_plus = np.zeros((2*nintersections,2*nintersections)) # initialise the Kirchhoff matrix.

    for iwire in xrange(nwires_plus_leads): # cycle through wires
	    if each_wire_inter_point_storage[iwire]: # intersection points for this wire
		    for j, pointj in enumerate(each_wire_inter_point_storage[iwire]): # pointj is the intersection point, j is its index in the list each_wire_inter_point_storage[iwire]
			    point = Point(pointj) # a specific shapely class
                # this for loop determines inner-wire resistances
			    for i, pointw in enumerate(each_wire_inter_point_storage[iwire]): # cycle through the intersections on iwire again, looking for nearest neighbours here
				    comp_pointw = Point(pointw) 
				    inter_dist = point.distance(comp_pointw) # checks the distance between the intersection points
				    round_inter_dist = round(inter_dist, 4) # round to 4 decimal places, enough for following comparison
				    for il in each_wire_length_storage[iwire][0]: # length of wire segments on wire iwire
					    value = float(il) 
					    value = round(value,4)
					    if value == round_inter_dist and value != 0: #  checking that |comp_pointw - pointj| matches a wire segment length that was calculated previously. If they do then comp_pointw and  pointj are nearest neighbours
					        inner_resis = (float(value) * rho0 / A0) # calculate the inner-wire resistance of the wire segment
					        
					        if iwire != 0 and iwire != 1 and mr_matrix_plus[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] == 0.0: #removing the wires 0 and 1, these are electrodes. also check that the resistor has not already been defined, this saves a bit of time.                               
					            mr_matrix_plus[wire_touch_label_list[iwire][i], wire_touch_label_list[iwire][j]] = -1.0/inner_resis #defining the two conductances, as it is a symmetric matrix.
					            mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[iwire][i]] = -1.0/inner_resis
					            G.add_edge(wire_touch_label_list[iwire][i],wire_touch_label_list[iwire][j]) # add the edge into the networkx class. 
                
                    
			    for k, label in enumerate(wire_touch_list[iwire]): #  this loop determines junction resistances
				    for kk, pointk in enumerate(each_wire_inter_point_storage[label]): # pointk is a new intersection point on the wire, k is its index in the list each_wire_inter_point_storage[label]
					    pointk = Point(pointk) 
					    inter_dist = point.distance(pointk) 
					    round_inter_dist = round(inter_dist, 4) # again getting distances between pointj and pointk
					    if round_inter_dist == 0 and mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] == 0:  # if |pointj-pointk| = 0 then they are connected by a junction resistor
					        G.add_edge(wire_touch_label_list[label][kk],wire_touch_label_list[iwire][j]) # adding a junction resistor to the networkx graph
					        r0 = -1/R_junc 
					        mr_matrix_plus[wire_touch_label_list[iwire][j], wire_touch_label_list[label][kk]] = r0 # defining the conductance in the symmetric matrix
					        mr_matrix_plus[wire_touch_label_list[label][kk], wire_touch_label_list[iwire][j]] = r0                         

    sum_rows_mr_plus = mr_matrix_plus.sum(1) # get the sum of each row of the kirchhoff matrix
    np.fill_diagonal(mr_matrix_plus, abs(sum_rows_mr_plus)) # fill in the diagonal of kirchhoff matrix
    mr_nozero_rows_plus = mr_matrix_plus[~(mr_matrix_plus==0).all(1),:] # determine rows corresponding to non-connected nodes
    mr_nonconnected_plus = mr_nozero_rows_plus[:,~(mr_nozero_rows_plus==0).all(0)] # remove the non-connected nodes, this is done for numerical reasons

    # some parameters for the symmlq numerical solving routine. til is tolerance, make it smaller for more accuracy, show plots some info to terminal during the routine, maxit limits the routine to a maximum number of iterations
    tol=1e-10
    show=False
    maxit=None

    i0 = 1.0 #  test current through the network
    ic = np.zeros(mr_nonconnected_plus.shape[0]) # initialise the current vector
    ic[0] = +i0 # current injected into electrode 0
    ic[1] = -i0 # current extracted at electrode 1

    Imatrix = m(ic) # some function to get ic into correct format for symmlq
    Amatrix = m(mr_nonconnected_plus) # # some function to get mr_connected_plus into correct format for symmlq
    elec_pot_mr = symmlq( Imatrix, Gfun, show=show, rtol=tol, maxit=maxit) # solve the set of linear equations. elec_pot_mr[0] contains the potential of each node

    resistance = ((elec_pot_mr[0][0] - elec_pot_mr[0][1]))/i0 # resistance of the network between electrodes

    print( material, resistance)



