# Code for Example 6 of our paper at https://arxiv.org/abs/2310.20310.
# This code is for the Linear finite element spatial discretization of
# this problem with Backward Euler, Crank-Nicholson and implicit leapfrog
# time discretizations.

import numpy as np
import pydec as pydec
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import matplotlib.pyplot as plt; plt.ion()
import modepy as modepy
import itertools as itrs
import mayavi.mlab as mlab
import os.path as pth
import sys as sys
import tqdm as tqdm
import pypardiso as pypard
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

##  Inputs  ##
# Some common utility variables
float_tol = 1e-12    # Tolerance for floating point comparisons
sigma = 1
epsilon = 2
mu = 1

# Time Parameters
dt = 1e-2; T_min = 0; T_max = 3
number_of_plot_times = 9   # How many time steps to sample for plotting
filename_prefix = "Maxwells"

# Meshes
mesh_dir = "meshes/unit_cube/"
mesh_no = 1

# FE Space Choice
fe_order = "Linear"
p_string = "example6"

# Computation Choices
plot_solutions = True
compute_energy = True
save_data = True
save_figs = True

# Time Discretization Methods (Default is Crank Nicholson if 0, 1, or 2 not
# specified)
# 0: Backward Euler
# 1: Crank-Nicholson
# 2: Implicit Leapfrog
time_discretization_to_use = 2    # Specify 0, 1, 2

if time_discretization_to_use == 0:
    use_backward_euler = True
    use_crank_nicholson = False; use_leap_frog = False
    method_string = "be"
elif time_discretization_to_use == 1:
    use_crank_nicholson = True
    use_backward_euler = False; use_leap_frog = False
    method_string = "cn"
elif time_discretization_to_use == 2:
    use_leap_frog = True
    use_backward_euler = False; use_crank_nicholson = False
    method_string = "implicit_lf"
else:
    use_crank_nicholson = True
    use_backward_euler = False; use_leap_frog = False
    method_string = "cn"

# Data and Figures
data_dir = "data/3d/"; data_prestring = p_string + "_" + method_string + "_"
figs_dir = "figs/3d/"; figs_prestring = p_string + "_" + method_string + "_"
savefig_options = {"bbox_inches": 'tight'}

# Visualization choices
camera_view = {"azimuth": 45.0, "elevation": 54.735, "distance": 3.25,
                   "focalpoint": np.array([0.5, 0.5, 0.5])} 

# Analytical Solutions and Right Hand Side
# Analytical p
def p_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return 0

# Analytical E
def E_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([np.sin(y)*np.sin(z)*np.cos(t), 
                     np.sin(x)*np.sin(z)*np.cos(t),
                     np.sin(x)*np.sin(y)*np.cos(t)])
    


# Analytical H
def H_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([np.sin(x)*(np.cos(z) - np.cos(y))*np.sin(t),
                     np.sin(y)*(np.cos(x) - np.cos(z))*np.sin(t),
                     np.sin(z)*(np.cos(y) - np.cos(x))*np.sin(t)])
    
# Analytical f_p
def fp_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return 0

# Analytical f_E
def fE_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([0, 0, 0])

# Analytical f_H
def fH_analytical(v, t):
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([0, 0, 0])

# Boundary data and Boundary Conditions
boundary_normals = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 1, 0]), 
                    np.array([0, 0, -1]), np.array([0, 0, 1])]

# Cross product of the Analytical vector field E with the outward normal to
# each of the boundary faces
def E_boundary(v, t, normal):
    return np.cross(E_analytical(v, t), normal)

# Dot product of the Analytical vector field H with the outward normal to each
# of the boundary faces
def H_boundary(v, t, normal):
    return np.dot(H_analytical(v, t), normal)

## End of Inputs  ##


##  Time Steps Setup  ##
# Preprocessing for time steps
number_of_time_steps = int(np.ceil((T_max - T_min)/dt)) + 1
computation_times = np.arange(0, number_of_time_steps) * dt
if number_of_time_steps < number_of_plot_times:
    number_of_plot_times = number_of_time_steps    # For consistency
if use_leap_frog == True:
    plot_time_steps = np.sort(list(set(list(
        np.concatenate((np.arange(0, number_of_time_steps, 
                                      (number_of_time_steps + 1)/(number_of_plot_times - 1), 
                                      dtype=int), [number_of_time_steps - 1]))))))
    plot_times =  np.concatenate(([0],(plot_time_steps[1:] - 0.5) * dt))
    plot_times_H = plot_time_steps * dt
else:
    plot_time_steps = np.sort(list(set(list(
        np.concatenate((np.arange(0, number_of_time_steps, 
                                      (number_of_time_steps + 1)/(number_of_plot_times - 1), 
                                      dtype=int), [number_of_time_steps - 1]))))))
    plot_times = plot_time_steps * dt

solution_time_steps = np.sort(list(set(list(
    np.concatenate((np.arange(0, number_of_time_steps, 
                                  (number_of_time_steps + 1)/(number_of_time_steps - 1), 
                                  dtype=int), [number_of_time_steps - 1]))))))
solution_times = solution_time_steps * dt

##  Quadrature Setup  ##
# Quadrature rule choices in 2d
dims = 2
order = 6
quad2d = modepy.XiaoGimbutasSimplexQuadrature(order, dims)
qnodes_2d = quad2d.nodes.T
qweights_2d = quad2d.weights

# Vertices of standard triangle on which quadrature nodes are specified
Ts_v = np.array([[-1, -1], [1, -1], [-1, 1]])

# Express the quadrature nodes in barycentric coordinates
A_2d = np.vstack((Ts_v.T, np.ones(3)))
B_2d = np.hstack((qnodes_2d, np.ones((qnodes_2d.shape[0], 1)))).T
qnodes_bary_2d = np.linalg.solve(A_2d, B_2d).T

# Area of this standard triangle for quadrature
vol_std_tri = np.sum(qweights_2d)

# Quadrature rule choices in 3d
dims = 3
order = 6
quad3d = modepy.XiaoGimbutasSimplexQuadrature(order, dims)
qnodes = quad3d.nodes.T
qweights = quad3d.weights

# Vertices of the reference tetrahedron on which quadrature nodes are specified
Tets_v = np.array([[-1, -1, -1],[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

# Express the quadrature nodes in barycentric coordinates
A_3d = np.vstack((Tets_v.T, np.ones(4)))
B_3d = np.hstack((qnodes, np.ones((qnodes.shape[0], 1)))).T
qnodes_bary = np.linalg.solve(A_3d, B_3d).T

# Area of the standard tetrahedron
vol_std_tet = np.sum(qweights)


## Finite Element Setup and Utility Functions 
# Barycentric Gradients on a triangle with specified vertices
def dl(vertices_Tet):
    return pydec.barycentric_gradients(vertices_Tet)

# Lagrange Basis (order = 1) on a physical simplex
def W0(ell_T):
    return ell_T[0]
def W1(ell_T):
    return ell_T[1]
def W2(ell_T):
    return ell_T[2]
def W3(ell_T):
    return ell_T[3]

# Gradient of Lagrange Basis (order = 1)
def grad_W0(dl_T):
    return dl_T[0]
def grad_W1(dl_T):
    return dl_T[1]
def grad_W2(dl_T):
    return dl_T[2]
def grad_W3(dl_T):
    return dl_T[3]

# Edge Whitney basis (order = 1) on a physical simplex
def E_W01(ell_Tet,dl_Tet):
    return(ell_Tet[0]*dl_Tet[1]-ell_Tet[1]*dl_Tet[0])
def E_W02(ell_Tet,dl_Tet):
    return(ell_Tet[0]*dl_Tet[2]-ell_Tet[2]*dl_Tet[0])
def E_W03(ell_Tet,dl_Tet):
    return(ell_Tet[0]*dl_Tet[3]-ell_Tet[3]*dl_Tet[0])
def E_W12(ell_Tet,dl_Tet):
    return(ell_Tet[1]*dl_Tet[2]-ell_Tet[2]*dl_Tet[1])
def E_W13(ell_Tet,dl_Tet):
    return(ell_Tet[1]*dl_Tet[3]-ell_Tet[3]*dl_Tet[1])
def E_W23(ell_Tet,dl_Tet):
    return(ell_Tet[2]*dl_Tet[3]-ell_Tet[3]*dl_Tet[2])

# Curl of Edge Whitney basis (order = 1)
def curl_E_W01(dl_Tet):
    return 2 * np.cross(dl_Tet[0],dl_Tet[1])
def curl_E_W02(dl_Tet):
    return 2 * np.cross(dl_Tet[0],dl_Tet[2])
def curl_E_W03(dl_Tet):
    return 2 * np.cross(dl_Tet[0],dl_Tet[3])
def curl_E_W12(dl_Tet):
    return 2 * np.cross(dl_Tet[1],dl_Tet[2])
def curl_E_W13(dl_Tet):
    return 2 * np.cross(dl_Tet[1],dl_Tet[3])
def curl_E_W23(dl_Tet):
    return 2 * np.cross(dl_Tet[2],dl_Tet[3])

# Face Whitney basis (order = 1) on physical simplex
def F_W012(ell_Tet,dl_Tet):
    return np.cross(ell_Tet[0]*dl_Tet[1],dl_Tet[2])-np.cross(ell_Tet[1]*dl_Tet[0],dl_Tet[2])+np.cross(ell_Tet[2]*dl_Tet[0],dl_Tet[1])
def F_W013(ell_Tet,dl_Tet):
    return np.cross(ell_Tet[0]*dl_Tet[1],dl_Tet[3])-np.cross(ell_Tet[1]*dl_Tet[0],dl_Tet[3])+np.cross(ell_Tet[3]*dl_Tet[0],dl_Tet[1])
def F_W023(ell_Tet,dl_Tet):
    return np.cross(ell_Tet[0]*dl_Tet[2],dl_Tet[3])-np.cross(ell_Tet[2]*dl_Tet[0],dl_Tet[3])+np.cross(ell_Tet[3]*dl_Tet[0],dl_Tet[2])
def F_W123(ell_Tet,dl_Tet):
    return np.cross(ell_Tet[1]*dl_Tet[2],dl_Tet[3])-np.cross(ell_Tet[2]*dl_Tet[1],dl_Tet[3])+np.cross(ell_Tet[3]*dl_Tet[1],dl_Tet[2])

# General divergence function for computing particular divergences of Face Whitney basis 
# on a physical simplex
# Note: This function computes the divegence of the following general form:
#       Divergence((ell_0**a * ell_1**b * ell_2**c * ell_3**d * Gradient[ell_Tet][s]) $\times$ Gradient[ell_Tet][t])
# where a, b, c, d are nonnegative integers, and s, t belongs to {0, 1, 2, 3}.
def div_F_Wbasis(ell_Tet,dl_Tet,a,b,c,d,s,t):
    val = np.cross(((a*(ell_Tet[0]**(a-1))*(ell_Tet[1]**b)*(ell_Tet[2]**c)*(ell_Tet[3]**d)*dl_Tet[0])+
                    (b*(ell_Tet[0]**a)*(ell_Tet[1]**(b-1))*(ell_Tet[2]**c)*(ell_Tet[3]**d)*dl_Tet[1])+
                    (c*(ell_Tet[0]**a)*(ell_Tet[1]**b)*(ell_Tet[2]**(c-1))*(ell_Tet[3]**d)*dl_Tet[2])+
                    (d*(ell_Tet[0]**a)*(ell_Tet[1]**b)*(ell_Tet[2]**c)*(ell_Tet[3]**(d-1))*dl_Tet[3])) , dl_Tet[s])
    return np.dot(val,dl_Tet[t])

# Divergence of Face Whitney basis on physical tetrahedron
def div_F_W012(ell_Tet,dl_Tet):
    return div_F_Wbasis(ell_Tet,dl_Tet,1,0,0,0,1,2) - div_F_Wbasis(ell_Tet,dl_Tet,0,1,0,0,0,2) + div_F_Wbasis(ell_Tet,dl_Tet,0,0,1,0,0,1)
def div_F_W013(ell_Tet,dl_Tet):
    return div_F_Wbasis(ell_Tet,dl_Tet,1,0,0,0,1,3) - div_F_Wbasis(ell_Tet,dl_Tet,0,1,0,0,0,3) + div_F_Wbasis(ell_Tet,dl_Tet,0,0,0,1,0,1)
def div_F_W023(ell_Tet,dl_Tet):
    return div_F_Wbasis(ell_Tet,dl_Tet,1,0,0,0,2,3) - div_F_Wbasis(ell_Tet,dl_Tet,0,0,1,0,0,3) + div_F_Wbasis(ell_Tet,dl_Tet,0,0,0,1,0,2)
def div_F_W123(ell_Tet,dl_Tet):
    return div_F_Wbasis(ell_Tet,dl_Tet,0,1,0,0,2,3) - div_F_Wbasis(ell_Tet,dl_Tet,0,0,1,0,1,3) + div_F_Wbasis(ell_Tet,dl_Tet,0,0,0,1,1,2)

# Lists referencing the various basis functions
V_Ws = [W0, W1, W2, W3]
grad_V_Ws = [grad_W0, grad_W1, grad_W2, grad_W3]
E_Ws = [E_W01, E_W02, E_W03, E_W12, E_W13, E_W23]
Eb_Ws = [E_W01, E_W02, E_W12]
curl_E_Ws = [curl_E_W01, curl_E_W02, curl_E_W03, curl_E_W12, curl_E_W13, curl_E_W23]
F_Ws = [F_W012, F_W013, F_W023, F_W123]
div_F_Ws = [div_F_W012, div_F_W013, div_F_W023, div_F_W123]

##  Finite Element System Computations  ##
# Mass Matrix 00: Inner product of Lagrange Basis and Lagrange Basis
def Mass_00(Tet, Tet_index):
    M = np.zeros((len(V_Ws), len(V_Ws)))
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    vertices_Tet = sc.vertices[Tet]
    dl_Tet = dl(vertices_Tet)

    # The required inner product is computed via quadrature
    for i in range(len(V_Ws)):    # Loop over vertices (p basis)
        for j in range(len(V_Ws)):    # Loop over vertices (p basis)
            integral = 0
            for k, qp_b in enumerate(qnodes_bary):
                integral += np.dot(V_Ws[i](qp_b), V_Ws[j](qp_b)) * qweights[k]
            integral *= vol_phy_tet/vol_std_tet
            M[i, j] = integral

    return M

# Stiffness Matrix 01: Inner product of Gradient of Lagrange Basis and Edge Whitney Basis 
def Stiff_01(Tet, Tet_index):
    S = np.zeros((len(grad_V_Ws), len(E_Ws)))
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    dl_Tet = dl(sc.vertices[Tet])
    
    # The required inner product is computed via quadrature
    for i in range(len(grad_V_Ws)):    # Loop over vertices (p basis)
        for j in range(len(E_Ws)):    # Loop over edges (E basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(grad_V_Ws[i](dl_Tet), 
                                   E_Ws[j](qp_b, dl_Tet)) * qweights[kq]
            integral *= vol_phy_tet/vol_std_tet
            S[i, j] = integral

    return S

# Mass Matrix 11: Inner product of Edge Whitney Basis and Edge Whitney Basis
def Mass_11(Tet, Tet_index):
    M = np.zeros((len(E_Ws), len(E_Ws)))
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    vertices_Tet = sc.vertices[Tet]
    dl_Tet = dl(vertices_Tet)

    # The required inner product is computed via quadrature
    for i in range(len(E_Ws)):    # Loop over edges (E basis)
        for j in range(len(E_Ws)):    # Loop over edges (E basis)
            integral = 0
            for k, qp_b in enumerate(qnodes_bary):
                integral += np.dot(E_Ws[i](qp_b, dl_Tet),
                                   E_Ws[j](qp_b, dl_Tet)) * qweights[k]
            integral *= vol_phy_tet/vol_std_tet
            M[i, j] = integral

    return M

# Mass Matrix b1b1: Inner product of Edge Whitney Basis and Edge Whitney Basis restricted to the boundary
def Mass_b1b1(dl_Face, vol_phy_Face, normal):
    M = np.zeros((len(Eb_Ws), len(Eb_Ws)))

    # The required inner product is computed via quadrature
    for i in range(len(Eb_Ws)):    # Loop over edges (E basis)
        for j in range(len(Eb_Ws)):    # Loop over edges (E basis)
            integral = 0
            for k, qp_b in enumerate(qnodes_bary_2d):
                integral += np.dot(np.cross(Eb_Ws[i](qp_b, dl_Face), normal),
                                   np.cross(Eb_Ws[j](qp_b, dl_Face), normal)) * qweights_2d[k]
            integral *= vol_phy_Face/vol_std_tri
            M[i, j] = integral

    return M

# Stiffness Matrix 12: Inner product of Curl of Edge Whitney Basis and Face Whitney Basis 
def Stiff_12(Tet, Tet_index):
    S = np.zeros((len(curl_E_Ws), len(F_Ws)))
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    dl_Tet = dl(sc.vertices[Tet])
    
    # The required inner product is computed via quadrature
    for i in range(len(curl_E_Ws)):    # Loop over edges (E basis)
        for j in range(len(F_Ws)):    # Loop over faces (H basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(curl_E_Ws[i](dl_Tet), 
                                   F_Ws[j](qp_b, dl_Tet)) * qweights[kq]
            integral *= vol_phy_tet/vol_std_tet
            S[i, j] = integral

    return S

# Stiffness Matrix 22: Inner product of divergence of Face Whitney Basis with divergence of Face Whitney basis
def Mass_22(Tet, Tet_index):
    M = np.zeros((len(F_Ws), len(F_Ws)))
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    dl_Tet = dl(sc.vertices[Tet])

    # The required inner product is computed via quadrature
    for i in range(len(F_Ws)):    # Loop over faces (H basis)
        for j in range(len(F_Ws)):    # Loop over faces (H basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(F_Ws[i](qp_b, dl_Tet), 
                                   F_Ws[j](qp_b, dl_Tet)) * qweights[kq]
            integral *= vol_phy_tet/vol_std_tet
            M[i, j] = integral

    return M

# Interpolation for p
def p_interpolation(p_Tet, ell_interp, dl_Tet):
    # Compute the interpolant at the interpolation point
    return np.sum([p_Tet[i] * V_Ws[i](ell_interp) for i in range(len(V_Ws))], axis=0)

# Interpolation for E
def E_interpolation(E_Tet, ell_interp, dl_Tet):
    # Compute the interpolant at the interpolation point
    return np.sum([E_Tet[i] * E_Ws[i](ell_interp, dl_Tet) for i in range(len(E_Ws))], axis=0)

# Interpolation for H
def H_interpolation(H_Tet, ell_interp, dl_Tet):
    # Compute the interpolant at the interpolation point
    return np.sum([H_Tet[i] * F_Ws[i](ell_interp, dl_Tet) for i in range(len(F_Ws))], axis=0)

# Loop over the meshes for solution of the mixed finite element discretization of the problem
# on the specified problem domain, and for computation of the L2 errors of the solution in 
# comparison with the analytical solutions

# Lists to store errors and energies
p_error_L2 = []; E_error_L2 = []; H_error_L2 = []
p_norm_L2 = []; E_norm_L2 = []; H_norm_L2 = []
p_comp_norm_L2 = []; E_comp_norm_L2 = []; H_comp_norm_L2 = []
L2_energy = []; comp_L2_energy = []

# Set up the mixed finite element discretization on the problem domain
Vertices = np.loadtxt(pth.join(mesh_dir, "vertices" + str(mesh_no) + ".txt"))
Tetrahedrons = np.loadtxt(pth.join(mesh_dir, "tets" + str(mesh_no) + ".txt"), dtype=int)

sc = pydec.simplicial_complex(Vertices,Tetrahedrons)
Vertices = Vertices[sc[0].simplices.ravel()]

# A preprocessing to remove any potentially duplicate vertices in the mesh provided by a
# mesh generator
vertex_vertex_map = {}
for v, k in enumerate(sc[0].simplices.ravel()):
    vertex_vertex_map[k] = v

for Tet_index, Tet in enumerate(Tetrahedrons):
    for i in range(4):
        Tetrahedrons[Tet_index][i] = vertex_vertex_map[Tet[i]]

# Use PyDEC to setup the simplicial complex and related operators
sc = pydec.simplicial_complex(Vertices, Tetrahedrons)
N0 = sc[0].num_simplices
N1 = sc[1].num_simplices
N2 = sc[2].num_simplices
N3 = sc[3].num_simplices

print("============================================================")
print("Spatial Discretization using Linear Polynomial Basis.")
print("Initial Time: ", T_min, ", Final Time: ", T_max, ", Time Step Size: ", dt)
print("============================================================")

print("\nComputing over mesh " + str(mesh_no) + "..."); sys.stdout.flush()
# Obtain the boundary faces of the mesh
boundary_face_indices = (sc[2].d.T * np.ones(N3)).nonzero()[0]
boundary_faces = sc[2].simplices[boundary_face_indices]
boundary_face_identifiers = np.zeros_like(boundary_face_indices)

# Obtain left, right, front, back, top and bottom boundary faces
boundary_left_faces = []; boundary_right_faces = []
boundary_back_faces = []; boundary_front_faces = []
boundary_bottom_faces = []; boundary_top_faces = []

for bfi_index, bfi in enumerate(boundary_face_indices):
    v0, v1, v2 = Vertices[sc[2].simplices[bfi]]
    bv = (v0 + v1 + v2)/3
    if abs(bv[0]) <= float_tol:
        boundary_left_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 0
    if abs(bv[1]) <= float_tol:
        boundary_back_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 2
    if abs(bv[2]) <= float_tol:
        boundary_bottom_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 4
    if abs(1 - bv[0]) <= float_tol:
        boundary_right_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 1
    if abs(1 - bv[1]) <= float_tol:
        boundary_front_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 3
    if abs(1 - bv[2]) <= float_tol:
        boundary_top_faces.append(bfi)
        boundary_face_identifiers[bfi_index] = 5

boundary_left_faces = np.array(boundary_left_faces)
boundary_back_faces = np.array(boundary_back_faces) 
boundary_bottom_faces = np.array(boundary_bottom_faces)
boundary_right_faces = np.array(boundary_right_faces)
boundary_front_faces = np.array(boundary_front_faces) 
boundary_top_faces = np.array(boundary_top_faces)

boundary_faces_descriptor = [boundary_left_faces, boundary_right_faces, boundary_back_faces, boundary_front_faces, 
                             boundary_bottom_faces, boundary_top_faces]

# Impose the boundary condition E \times n = a given function on the boundary for E
# Obtain the boundary edges of the mesh
boundary_edge_indices = []
for face in boundary_faces:
    boundary_edge_indices.extend([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                        for v0, v1 in itrs.combinations(face, 2)])
boundary_edge_indices = np.sort(list(set(boundary_edge_indices)))
boundary_edges = sc[1].simplices[boundary_edge_indices]

N1b = boundary_edge_indices.shape[0]    # Number of edges in the boundary

# Reindex the boundary edges to ensure that they are contiguously indexed for the following computations
boundary_edge_index_map = {v:k for k, v in enumerate(boundary_edge_indices)}

# Obtain the boundary vertices of the mesh
boundary_vertex_indices = np.array(list(set(list(np.ravel(sc[1].simplices[boundary_edge_indices])))))

# Create a surface triangular mesh of the boundary for visualization
boundary_vertex_index_map = {v:k for k, v in enumerate(boundary_vertex_indices)}
boundary_triangles = np.array([[boundary_vertex_index_map[v0], boundary_vertex_index_map[v1], boundary_vertex_index_map[v2]] 
                               for v0, v1, v2 in boundary_faces])
boundary_vertices = sc.vertices[boundary_vertex_indices]

# Create data structures for the mass matrix of Whitney (edge) vector fields restricted to the boundary mesh
rowsb1b1 = []; columnsb1b1 = []; Mb1b1_data = []

# Compute the boundary mass matrix for the Whitney (edge) basis
for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
    T_index = sc[2].d.T[Face_index].indices[0]
    T = list(sc[3].simplices[T_index])
    vertices_T = sc.vertices[T]
    edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                for v0, v1 in itrs.combinations(Face, 2)])
    Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
    dl_T = dl(vertices_T)
    dl_Face = dl_T[Face_local_indices]
    vol_phy_Face = sc[2].primal_volume[Face_index]
    bfi = boundary_face_identifiers[index]
    boundary_normal = boundary_normals[bfi]

    Mb1b1 = Mass_b1b1(dl_Face, vol_phy_Face, boundary_normal)
    # Integrate the local boundary mass matrices into data structures for the global boundary mass matrix
    for i in range(3):    # Loop over edges (sigma basis)
        for j in range(3):    # Loop over edges (sigma basis)
            rowsb1b1.append(boundary_edge_index_map[edge_indices_Face[i]])
            columnsb1b1.append(boundary_edge_index_map[edge_indices_Face[j]])
            Mb1b1_data.append(Mb1b1[i, j])
    
# Setup the Global Boundary Mass Matrix as SciPy sparse matrices
Mb1b1_g = sprs.coo_matrix((Mb1b1_data, (rowsb1b1, columnsb1b1)), (N1b, N1b), dtype='float')
Mb1b1_g = Mb1b1_g.tocsr()

# Data structures to define the various Mass and Stiffness matrices
rows00 = []; columns00 = []; M00_data = []
rows01 = []; columns01 = []; S01_data = []
rows11 = []; columns11 = []; M11_data = []
rows12 = []; columns12 = []; S12_data = []
rows22 = []; columns22 = []; M22_data = []

# Initialize solution and right hand side vectors
p_0 = np.zeros(N0); E_0 = np.zeros(N1); H_0 = np.zeros(N2)

# Obtain the mass and stiffness matrices on each tetrahedron and integrate into the global ones
print("\tbuilding stiffness and mass matrix..."); sys.stdout.flush()
for Tet_index, Tet in enumerate(tqdm.tqdm(sc[3].simplices)):
    vertices_Tet = sc.vertices[Tet]
    vol_phy_tet = sc[3].primal_volume[Tet_index]
    dl_Tet = dl(vertices_Tet)
    integral_scaling = vol_phy_tet/vol_std_tet

    # Obtain the indices of edges of this tetrahedron
    edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                        for v0, v1 in itrs.combinations(Tet, 2)])
    
    # Obtain the indices of faces of this tetrahedron
    faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                        for v0, v1, v2 in itrs.combinations(Tet, 3)])

    # Obtain the local mass and stiffness matrices
    M00 = Mass_00(Tet, Tet_index)
    S01 = Stiff_01(Tet, Tet_index)
    M11 = Mass_11(Tet, Tet_index)    
    S12 = Stiff_12(Tet, Tet_index)
    M22 = Mass_22(Tet, Tet_index)
    
    # Integrate the local matrices into data structures for the global matrices
    for i in range(4):    # Loop over vertices
        for j in range(4):    # Loop over vertices
            rows00.append(Tet[i])
            columns00.append(Tet[j])
            M00_data.append(M00[i, j])

        for j in range(6):    # Loop over edges
            rows01.append(Tet[i])
            columns01.append(edges_Tet[j])
            S01_data.append(S01[i, j])

    for i in range(6):    # Loop over edges
        for j in range(6):    # Loop over edges
            rows11.append(edges_Tet[i])
            columns11.append(edges_Tet[j])
            M11_data.append(M11[i, j])

        for j in range(4):    # Loop over faces
            rows12.append(edges_Tet[i])
            columns12.append(faces_Tet[j])
            S12_data.append(S12[i, j])

    for i in range(4):    # Loop over faces
        for j in range(4):    # Loop over faces
            rows22.append(faces_Tet[i])
            columns22.append(faces_Tet[j])
            M22_data.append(M22[i, j])
        
    for i in range(6):
        E_integral = 0
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_Tet[:, 0])
            y_q = np.dot(qp_b, vertices_Tet[:, 1])
            z_q = np.dot(qp_b, vertices_Tet[:, 2])
            # Computing Initial Vector at t = T_min
            E_integral += np.dot(E_analytical([x_q, y_q, z_q], T_min) , E_Ws[i](qp_b, dl_Tet)) * qweights[kq]
        E_integral *= integral_scaling
        E_0[edges_Tet[i]] += E_integral

    for i in range(4):
        H_integral = 0
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_Tet[:, 0])
            y_q = np.dot(qp_b, vertices_Tet[:, 1])
            z_q = np.dot(qp_b, vertices_Tet[:, 2])
            # Computing Initial Vector at t = T_min
            H_integral += np.dot(H_analytical([x_q, y_q, z_q], T_min) , F_Ws[i](qp_b, dl_Tet)) * qweights[kq]
        H_integral *= integral_scaling
        H_0[faces_Tet[i]] += H_integral

# Setup the Global Mass and Stiffness Matrices as SciPy sparse matrices
M00_g = sprs.coo_matrix((M00_data, (rows00, columns00)), (N0, N0), dtype='float')
M00_g = M00_g.tocsr()
S01_g = sprs.coo_matrix((S01_data, (rows01, columns01)), (N0, N1), dtype='float')
S01_g = S01_g.tocsr()
M11_g = sprs.coo_matrix((M11_data, (rows11, columns11)), (N1, N1), dtype='float')
M11_g = M11_g.tocsr()
S12_g = sprs.coo_matrix((S12_data, (rows12, columns12)), (N1, N2), dtype='float')
S12_g = S12_g.tocsr()
M22_g = sprs.coo_matrix((M22_data, (rows22, columns22)), (N2, N2), dtype='float')
M22_g = M22_g.tocsr()

# Some clean up
del (rows00, columns00, M00_data, rows01, columns01, S01_data, rows11, columns11, M11_data,
    rows12, columns12, S12_data, rows22, columns22, M22_data, rowsb1b1, columnsb1b1, Mb1b1_data)

# Computing Initial Vectors
print("\n\tsolving for initial condition vectors..."); sys.stdout.flush()
p = np.zeros((number_of_time_steps, N0))
E = np.zeros((number_of_time_steps, N1))
H = np.zeros((number_of_time_steps, N2))

# Computing Initial Vectors
# Initial p
for i, v in enumerate(sc.vertices):
    p_0[i] = p_analytical(v, t=0)
p[0] = p_0
p0_initial = p[0].copy()

# Initial E and H
# Compute the right hand side for the boundary condition on E and also compute the boundary
# coefficients for the boundary condition on H
bb_E = np.zeros(N1b)
H_bc = np.zeros(boundary_faces.shape[0])
Hbasis_trace_factor = 2

for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
    T_index = sc[2].d.T[Face_index].indices[0]
    T = list(sc[3].simplices[T_index])
    vertices_T = sc.vertices[T]
    vertices_Face = sc.vertices[Face]
    edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                for v0, v1 in itrs.combinations(Face, 2)])
    Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
    dl_T = dl(vertices_T)
    dl_Face = dl_T[Face_local_indices]
    vol_phy_Face = sc[2].primal_volume[Face_index]
    bfi = boundary_face_identifiers[index]
    boundary_normal = boundary_normals[bfi]
    v0, v1, v2 = vertices_Face
    e_vector01 = v1 - v0
    e_vector02 = v2 - v0
    face_normal_vector = np.cross(e_vector01, e_vector02)
    sign = np.sign(np.dot(face_normal_vector, boundary_normal))
    integral_scaling_2d = vol_phy_Face/vol_std_tri

    # Right hand side vector: Inner product of the Edge Whitney basis with the boundary function
    H_integral = 0
    for kq, qp_b in enumerate(qnodes_bary_2d):
        xq_phy = np.dot(qp_b, vertices_Face[:, 0])
        yq_phy = np.dot(qp_b, vertices_Face[:, 1])
        zq_phy = np.dot(qp_b, vertices_Face[:, 2])
        bb_integral = np.zeros(3)
        for i in range(3):    # Loop over edges (E boundary basis)
            bb_E[boundary_edge_index_map[edge_indices_Face[i]]] += (integral_scaling_2d * 
                np.dot(E_boundary([xq_phy, yq_phy, zq_phy], T_min, boundary_normal), 
                        np.cross(Eb_Ws[i](qp_b, dl_Face), boundary_normal)) * qweights_2d[kq])
        H_integral += H_boundary([xq_phy, yq_phy, zq_phy], T_min, boundary_normal) * qweights_2d[kq]
    H_integral *= sc[2].primal_volume[Face_index]/vol_std_tri
    H_bc[index] = sign * H_integral

# Solve for the coefficients for the Whitney (Edges) basis for the boundary edges for the E x n 
# boundary condition
solver_bE = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
solver_bE.set_iparm(34, 4)    # 
solver_bE.set_iparm(60, 2)    # 
solver_bE.factorize(Mb1b1_g)

# Obtain the linear system solution for E coefficients on the boundary
E_bc = solver_bE.solve(Mb1b1_g, bb_E)

# Scale the H coefficients on the boundary by the integral of the face basis on this triangle 
H_bc *= Hbasis_trace_factor

# Setup boundary conditions for computation of initial E
M11_g_bndry = M11_g.copy()
M11_g_bndry[boundary_edge_indices] = 0
M11_g_bndry[boundary_edge_indices, boundary_edge_indices] = 1
E_0[boundary_edge_indices] = E_bc

# Solve for initial E
solverM11_bndry = pypard.PyPardisoSolver()
solverM11_bndry.set_iparm(34, 4)
solverM11_bndry.set_iparm(60, 2)
solverM11_bndry.factorize(M11_g_bndry)
E[0] = solverM11_bndry.solve(M11_g_bndry, E_0)
E0_initial = E[0].copy()

# Setup boundary conditions for computation of intial H
M22_g_bndry = M22_g.copy()
M22_g_bndry[boundary_face_indices] = 0
M22_g_bndry[boundary_face_indices, boundary_face_indices] = 1
H_0[boundary_face_indices] = H_bc

# Solve for initial H
solverM22_bndry = pypard.PyPardisoSolver()
solverM22_bndry.set_iparm(34, 4)
solverM22_bndry.set_iparm(60, 2)
solverM22_bndry.factorize(M22_g_bndry)
H[0] = solverM22_bndry.solve(M22_g_bndry, H_0)
H0_initial = H[0].copy()

# Set up solvers
# Solver for p
solverM00 = pypard.PyPardisoSolver()
solverM00.set_iparm(34, 4)
solverM00.set_iparm(60, 2)
solverM00.factorize(M00_g)

# Solver for E
solverM11 = pypard.PyPardisoSolver()
solverM11.set_iparm(34, 4)
solverM11.set_iparm(60, 2)
solverM11.factorize(M11_g)

# Solver for H
solverM22 = pypard.PyPardisoSolver()
solverM22.set_iparm(34, 4)
solverM22.set_iparm(60, 2)
solverM22.factorize(M22_g)


# Loop over timesteps to find the solution
print("\n\tcomputing solution over the time steps..."); sys.stdout.flush()
# Backward Euler
if use_backward_euler:  
    print("\t\tusing Backward Euler with time step of %1.4f"%dt)

    # Setup the linear system for the solution of E and H
    S_LHS = sprs.bmat([[1/dt*M00_g, -epsilon*S01_g, None], 
                       [-S01_g.T, -epsilon/dt*M11_g, S12_g], 
                       [None, S12_g.T, mu/dt * M22_g]], 
                       format='csr')

    # Modify this system matrix to account for boundary conditions
    bc_zeromatrix_vertices = sprs.csr_matrix((boundary_vertex_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_edges = sprs.csr_matrix((boundary_edge_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_faces = sprs.csr_matrix((boundary_face_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1
    S_LHS[N0 + N1 + boundary_face_indices] = bc_zeromatrix_faces
    S_LHS[N0 + N1 + boundary_face_indices, N0 + N1 + boundary_face_indices] = 1

    # Setup linear system solver for the solution of E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    for time_step in tqdm.tqdm(range(1, number_of_time_steps)):
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2)
        for Tet_index, Tet in enumerate(sc[3].simplices):
            vertices_Tet = sc.vertices[Tet]
            vol_phy_tet = sc[3].primal_volume[Tet_index]
            dl_Tet = dl(vertices_Tet)
            integral_scaling = vol_phy_tet/vol_std_tet

            # Obtain the indices of edges of this tetrahedron
            edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(Tet, 2)])
            
            # Obtain the indices of faces of this tetrahedron
            faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                for v0, v1, v2 in itrs.combinations(Tet, 3)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_Tet))
            fE_integral = np.zeros(len(edges_Tet))
            fH_integral = np.zeros(len(faces_Tet))
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_Tet[:, 0])
                y_q = np.dot(qp_b, vertices_Tet[:, 1])
                z_q = np.dot(qp_b, vertices_Tet[:, 2])
                for i in range(6):
                    fE_integral[i] += np.dot(fE_analytical([x_q, y_q, z_q], (time_step) * dt), 
                                             E_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                for i in range(4):
                    fp_integral[i] += np.dot(fp_analytical([x_q, y_q, z_q], (time_step) * dt), 
                                             V_Ws[i](qp_b)) * qweights[kq]
                    fH_integral[i] += np.dot(fH_analytical([x_q, y_q, z_q], (time_step) * dt), 
                                             F_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[Tet] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_Tet] += fE_integral
            fH_integral *= integral_scaling
            b_H[faces_Tet] += fH_integral

        # Setup right hand side intermediate variables
        bp_RHS = b_p + 1/dt*M00_g*p[time_step - 1] 
        bE_RHS = -b_E - epsilon/dt*M11_g*E[time_step - 1] 
        bH_RHS = b_H + mu/dt*M22_g*H[time_step - 1]

        # Obtain boundary coefficients for E and H
        bb_E = np.zeros(N1b)
        H_bc = np.zeros(boundary_faces.shape[0])
        Hbasis_trace_factor = 2

        # Compute the right hand side for the boundary condition on E and also compute the boundary
        # coefficients for the boundary condition on H
        for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
            T_index = sc[2].d.T[Face_index].indices[0]
            T = list(sc[3].simplices[T_index])
            vertices_T = sc.vertices[T]
            vertices_Face = sc.vertices[Face]
            edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                        for v0, v1 in itrs.combinations(Face, 2)])
            Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
            dl_T = dl(vertices_T)
            dl_Face = dl_T[Face_local_indices]
            vol_phy_Face = sc[2].primal_volume[Face_index]
            bfi = boundary_face_identifiers[index]
            boundary_normal = boundary_normals[bfi]
            v0, v1, v2 = vertices_Face
            e_vector01 = v1 - v0
            e_vector02 = v2 - v0
            face_normal_vector = np.cross(e_vector01, e_vector02)
            sign = np.sign(np.dot(face_normal_vector, boundary_normal))
            integral_scaling_2d = vol_phy_Face/vol_std_tri

            # Right hand side vector: Inner product of the Edge Whitney basis with the boundary function
            H_integral = 0
            for kq, qp_b in enumerate(qnodes_bary_2d):
                xq_phy = np.dot(qp_b, vertices_Face[:, 0])
                yq_phy = np.dot(qp_b, vertices_Face[:, 1])
                zq_phy = np.dot(qp_b, vertices_Face[:, 2])
                for i in range(3):    # Loop over edges (E boundary basis)
                    bb_E[boundary_edge_index_map[edge_indices_Face[i]]] += (integral_scaling_2d * 
                        np.dot(E_boundary([xq_phy, yq_phy, zq_phy], time_step * dt, boundary_normal), 
                               np.cross(Eb_Ws[i](qp_b, dl_Face), boundary_normal)) * qweights_2d[kq])
                H_integral += H_boundary([xq_phy, yq_phy, zq_phy], time_step * dt, boundary_normal) * qweights_2d[kq]
            H_integral *= sc[2].primal_volume[Face_index]/vol_std_tri
            H_bc[index] = sign * H_integral

        # Solve for the coefficients for the Whitney (Edges) basis for the boundary edges for the E x n 
        # boundary condition
        solver_bE = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
        solver_bE.set_iparm(34, 4)    # 
        solver_bE.set_iparm(60, 2)    # 
        solver_bE.factorize(Mb1b1_g)

        # Obtain the linear system solution for E coefficients on the boundary
        E_bc = solver_bE.solve(Mb1b1_g, bb_E)
        
        # Scale the H coefficients on the boundary by the integral of the face basis on this triangle 
        H_bc *= Hbasis_trace_factor

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, time_step * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Impose boundary conditions on the right hand side at this time step
        bp_RHS[boundary_vertex_indices] = p_bc
        bE_RHS[boundary_edge_indices] = E_bc
        bH_RHS[boundary_face_indices] = H_bc

        # Setup the right hand side matrix
        b_RHS = np.concatenate((bp_RHS, bE_RHS, bH_RHS))

        # Obtain the linear system solution for E and H
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0 + N1]
        H[time_step] = x[N0 + N1:]


# Crank Nicholson
if use_crank_nicholson:
    print("\t\tusing Crank Nicholson with time step of %1.4f"%dt)

    # Setup the linear system for the solution of E and H
    S_LHS = sprs.bmat([[1/dt*M00_g, -epsilon*S01_g/2, None], 
                       [-S01_g.T/2, -epsilon/dt*M11_g, S12_g/2], 
                       [None, S12_g.T/2, mu/dt*M22_g]], 
                       format='csr')

    # Modify this system matrix to account for boundary conditions
    bc_zeromatrix_vertices = sprs.csr_matrix((boundary_vertex_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_edges = sprs.csr_matrix((boundary_edge_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_faces = sprs.csr_matrix((boundary_face_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1
    S_LHS[N0 + N1 + boundary_face_indices] = bc_zeromatrix_faces
    S_LHS[N0 + N1 + boundary_face_indices, N0 + N1 + boundary_face_indices] = 1

    # Setup linear system solver for the solution of E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    for time_step in tqdm.tqdm(range(1, number_of_time_steps)):
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2)
        for Tet_index, Tet in enumerate(sc[3].simplices):
            vertices_Tet = sc.vertices[Tet]
            vol_phy_tet = sc[3].primal_volume[Tet_index]
            dl_Tet = dl(vertices_Tet)
            integral_scaling = vol_phy_tet/vol_std_tet

            # Obtain the indices of edges of this tetrahedron
            edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(Tet, 2)])
            
            # Obtain the indices of faces of this tetrahedron
            faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                for v0, v1, v2 in itrs.combinations(Tet, 3)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_Tet))
            fE_integral = np.zeros(len(edges_Tet))
            fH_integral = np.zeros(len(faces_Tet))
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_Tet[:, 0])
                y_q = np.dot(qp_b, vertices_Tet[:, 1])
                z_q = np.dot(qp_b, vertices_Tet[:, 2])
                for i in range(6):
                    fE_integral[i] += np.dot((fE_analytical([x_q, y_q, z_q], (time_step) * dt) +
                                              fE_analytical([x_q, y_q, z_q], (time_step - 1) * dt)), 
                                             E_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                for i in range(4):
                    fp_integral[i] += np.dot((fp_analytical([x_q, y_q, z_q], (time_step) * dt) +
                                              fp_analytical([x_q, y_q, z_q], (time_step - 1) * dt)), 
                                             V_Ws[i](qp_b)) * qweights[kq]
                    fH_integral[i] += np.dot((fH_analytical([x_q, y_q, z_q], (time_step) * dt) +
                                              fH_analytical([x_q, y_q, z_q], (time_step - 1) * dt)), 
                                             F_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[Tet] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_Tet] += fE_integral
            fH_integral *= integral_scaling
            b_H[faces_Tet] += fH_integral

        # Setup right hand side intermediate variables
        bp_RHS = b_p/2 + 1/dt*M00_g*p[time_step - 1] + epsilon*S01_g/2*E[time_step - 1]
        bE_RHS = -b_E/2 + S01_g.T/2*p[time_step - 1] - epsilon/dt*M11_g*E[time_step - 1] - S12_g/2*H[time_step - 1]
        bH_RHS = b_H/2 - S12_g.T/2*E[time_step - 1] + mu/dt*M22_g*H[time_step - 1]

        # Obtain boundary coefficients for E and H
        bb_E = np.zeros(N1b)
        H_bc = np.zeros(boundary_faces.shape[0])
        Hbasis_trace_factor = 2

        # Compute the right hand side for the boundary condition on E and also compute the boundary
        # coefficients for the boundary condition on H
        for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
            T_index = sc[2].d.T[Face_index].indices[0]
            T = list(sc[3].simplices[T_index])
            vertices_T = sc.vertices[T]
            vertices_Face = sc.vertices[Face]
            edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                        for v0, v1 in itrs.combinations(Face, 2)])
            Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
            dl_T = dl(vertices_T)
            dl_Face = dl_T[Face_local_indices]
            vol_phy_Face = sc[2].primal_volume[Face_index]
            bfi = boundary_face_identifiers[index]
            boundary_normal = boundary_normals[bfi]
            v0, v1, v2 = vertices_Face
            e_vector01 = v1 - v0
            e_vector02 = v2 - v0
            face_normal_vector = np.cross(e_vector01, e_vector02)
            sign = np.sign(np.dot(face_normal_vector, boundary_normal))
            integral_scaling_2d = vol_phy_Face/vol_std_tri

            # Right hand side vector: Inner product of the Edge Whitney basis with the boundary function
            H_integral = 0
            for kq, qp_b in enumerate(qnodes_bary_2d):
                xq_phy = np.dot(qp_b, vertices_Face[:, 0])
                yq_phy = np.dot(qp_b, vertices_Face[:, 1])
                zq_phy = np.dot(qp_b, vertices_Face[:, 2])
                bb_integral = np.zeros(3)
                for i in range(3):    # Loop over edges (E basis)
                    bb_E[boundary_edge_index_map[edge_indices_Face[i]]] += (integral_scaling_2d * 
                        np.dot(E_boundary([xq_phy, yq_phy, zq_phy], time_step * dt, boundary_normal), 
                               np.cross(Eb_Ws[i](qp_b, dl_Face), boundary_normal)) * qweights_2d[kq])
                H_integral += H_boundary([xq_phy, yq_phy, zq_phy], time_step * dt, boundary_normal) * qweights_2d[kq]
            H_integral *= sc[2].primal_volume[Face_index]/vol_std_tri
            H_bc[index] = sign * H_integral

        # Solve for the coefficients for the Whitney (Edges) basis for the boundary edges for the E x n 
        # boundary condition
        solver_bE = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
        solver_bE.set_iparm(34, 4)    # 
        solver_bE.set_iparm(60, 2)    # 
        solver_bE.factorize(Mb1b1_g)

        # Obtain the linear system solution for E coefficients on the boundary
        E_bc = solver_bE.solve(Mb1b1_g, bb_E)
        
        # Scale the H coefficients on the boundary by the integral of the face basis on this triangle 
        H_bc *= Hbasis_trace_factor

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, time_step * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Impose boundary conditions on the right hand side at this time step
        bp_RHS[boundary_vertex_indices] = p_bc
        bE_RHS[boundary_edge_indices] = E_bc
        bH_RHS[boundary_face_indices] = H_bc

        # Setup the right hand side matrix
        b_RHS = np.concatenate((bp_RHS, bE_RHS, bH_RHS))

        # Obtain the linear system solution for E and H
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0 + N1]
        H[time_step] = x[N0 + N1:]

# Implicit Leap-Frog
if use_leap_frog:  
    print("\t\tusing implicit Leap Frog with time step of %1.4f"%dt)

    # Setup the linear system for the solution of p, E and H
    S_LHS = sprs.bmat([[1/dt*M00_g, -epsilon/2*S01_g, None], 
                       [1/2*S01_g.T, epsilon/dt*M11_g, -1/2*S12_g],
                       [None, 1/2*S12_g.T, mu/dt*M22_g]], format='csr')

    # Modify this system matrix to account for boundary conditions
    bc_zeromatrix_vertices = sprs.csr_matrix((boundary_vertex_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_edges = sprs.csr_matrix((boundary_edge_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    bc_zeromatrix_faces = sprs.csr_matrix((boundary_face_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1
    S_LHS[N0 + N1 + boundary_face_indices] = bc_zeromatrix_faces
    S_LHS[N0 + N1 + boundary_face_indices, N0 + N1 + boundary_face_indices] = 1

    # Setup linear system solver for the solution of p, E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    # Initial right hand sides
    b0_p = np.zeros(N0); b0_E = np.zeros(N1); b0_H = np.zeros(N2)
    for Tet_index, Tet in enumerate(sc[3].simplices):
        vertices_Tet = sc.vertices[Tet]
        vol_phy_tet = sc[3].primal_volume[Tet_index]
        dl_Tet = dl(vertices_Tet)
        integral_scaling = vol_phy_tet/vol_std_tet

        # Obtain the indices of edges of this tetrahedron
        edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                            for v0, v1 in itrs.combinations(Tet, 2)])
        
        # Obtain the indices of faces of this tetrahedron
        faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                            for v0, v1, v2 in itrs.combinations(Tet, 3)])
        
        # Computing values of f_p, f_E and f_H at current time step
        fp_integral = np.zeros(len(vertices_Tet))
        fE_integral = np.zeros(len(edges_Tet))
        fH_integral = np.zeros(len(faces_Tet))
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_Tet[:, 0])
            y_q = np.dot(qp_b, vertices_Tet[:, 1])
            z_q = np.dot(qp_b, vertices_Tet[:, 2])
            for i in range(4):
                fp_integral[i] += np.dot(fp_analytical([x_q, y_q, z_q], 0), 
                                            V_Ws[i](qp_b)) * qweights[kq]
                fH_integral[i] += np.dot(fH_analytical([x_q, y_q, z_q], dt/2), 
                                            F_Ws[i](qp_b, dl_Tet)) * qweights[kq]
            for i in range(6):
                fE_integral[i] += np.dot(fE_analytical([x_q, y_q, z_q], 0), 
                                            E_Ws[i](qp_b, dl_Tet)) * qweights[kq]
        
        fp_integral *= integral_scaling
        b0_p[Tet] += fp_integral
        fE_integral *= integral_scaling
        b0_E[edges_Tet] += fE_integral
        fH_integral *= integral_scaling
        b0_H[faces_Tet] += fH_integral

    # Obtain boundary coefficients for E0
    bb0_E = np.zeros(N1b)
    H0_bc = np.zeros(boundary_faces.shape[0])
    Hbasis_trace_factor = 2

    # Compute the right hand side for the boundary condition on E0 and H0
    H_integral = 0
    for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
        T_index = sc[2].d.T[Face_index].indices[0]
        T = list(sc[3].simplices[T_index])
        vertices_T = sc.vertices[T]
        vertices_Face = sc.vertices[Face]
        edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                    for v0, v1 in itrs.combinations(Face, 2)])
        Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
        dl_T = dl(vertices_T)
        dl_Face = dl_T[Face_local_indices]
        vol_phy_Face = sc[2].primal_volume[Face_index]
        bfi = boundary_face_identifiers[index]
        boundary_normal = boundary_normals[bfi]
        v0, v1, v2 = vertices_Face
        e_vector01 = v1 - v0
        e_vector02 = v2 - v0
        face_normal_vector = np.cross(e_vector01, e_vector02)
        sign = np.sign(np.dot(face_normal_vector, boundary_normal))
        integral_scaling_2d = vol_phy_Face/vol_std_tri

        # Right hand side vector: Inner product of the Edge Whitney basis with the boundary function
        for kq, qp_b in enumerate(qnodes_bary_2d):
            xq_phy = np.dot(qp_b, vertices_Face[:, 0])
            yq_phy = np.dot(qp_b, vertices_Face[:, 1])
            zq_phy = np.dot(qp_b, vertices_Face[:, 2])
            for i in range(3):    # Loop over edges (E boundary basis)
                bb0_E[boundary_edge_index_map[edge_indices_Face[i]]] += (integral_scaling_2d * 
                    np.dot(E_boundary([xq_phy, yq_phy, zq_phy], 0, boundary_normal), 
                            np.cross(Eb_Ws[i](qp_b, dl_Face), boundary_normal)) * qweights_2d[kq])
            H_integral += H_boundary([xq_phy, yq_phy, zq_phy], dt/2, boundary_normal) * qweights_2d[kq]
            H_integral *= sc[2].primal_volume[Face_index]/vol_std_tri
            H0_bc[index] = sign * H_integral

    # Solve for the coefficients for the Whitney (Edges) basis for the boundary edges for the E x n 
    # boundary condition
    solver_bE = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    solver_bE.set_iparm(34, 4)    # 
    solver_bE.set_iparm(60, 2)    # 
    solver_bE.factorize(Mb1b1_g)

    # Obtain the linear system solution for E coefficients on the boundary
    E0_bc = solver_bE.solve(Mb1b1_g, bb0_E)

    # Scale the H coefficients on the boundary by the integral of the face basis on this triangle 
    H0_bc *= Hbasis_trace_factor

    # Incorporate boundary conditions in p
    p0_bc = np.array([p_analytical(v, 0) for v in sc.vertices[boundary_vertex_indices]])

    # Compute right hand side at this time step
    b0_RHS = np.concatenate((b0_p, b0_E, b0_H)) + sprs.bmat([[2/dt*M00_g, epsilon/4*S01_g, None], 
                                                         [-1/4*S01_g.T, 2/dt*epsilon*M11_g, 1/2*S12_g],
                                                         [None, -1/4*S12_g.T, 1/dt*mu*M22_g]], 
                                                         format='csr') * np.concatenate((p[0], E[0], H[0]))

    # Impose boundary conditions on the right hand side at this time step
    b0_RHS[boundary_vertex_indices] = p0_bc
    b0_RHS[N0 + boundary_edge_indices] = E0_bc
    b0_RHS[N0 + N1 + boundary_face_indices] = H0_bc
    
    S0_LHS = sprs.bmat([[2/dt*M00_g, -epsilon/4*S01_g, None],
                        [-1/4*S01_g.T, 2/dt*epsilon*M11_g, -1/2*S12_g],
                        [None, 1/4*S12_g.T, mu/dt*M22_g]], format='csr')

    # Modify this system matrix to account for boundary conditions
    bc_zeromatrix_vertices = sprs.csr_matrix((boundary_vertex_indices.shape[0], S0_LHS.shape[0]), dtype=S0_LHS.dtype)
    bc_zeromatrix_edges = sprs.csr_matrix((boundary_edge_indices.shape[0], S0_LHS.shape[0]), dtype=S0_LHS.dtype)
    bc_zeromatrix_faces = sprs.csr_matrix((boundary_face_indices.shape[0], S_LHS.shape[0]), dtype=S_LHS.dtype)
    S0_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S0_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S0_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S0_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1
    S0_LHS[N0 + N1 + boundary_face_indices] = bc_zeromatrix_faces
    S0_LHS[N0 + N1 + boundary_face_indices, N0 + N1 + boundary_face_indices] = 1
    
    # Setup linear system solver for the solution of p0, E0 and H0
    S0_LHS.eliminate_zeros()    # Sparsify again
    pEH0_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH0_solver.set_iparm(34, 4)    # 
    pEH0_solver.set_iparm(60, 2)    # 
    pEH0_solver.factorize(S0_LHS)

    # Obtain the linear system solution for p0, E0 at t = 1/2
    x0 = pEH0_solver.solve(S0_LHS, b0_RHS)
    p[1] = x0[:N0]
    E[1] = x0[N0:N0+N1]
    H[1] = x0[N0+N1:]

    for time_step in tqdm.tqdm(range(2, number_of_time_steps)):
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2)
        for Tet_index, Tet in enumerate(sc[3].simplices):
            vertices_Tet = sc.vertices[Tet]
            vol_phy_tet = sc[3].primal_volume[Tet_index]
            dl_Tet = dl(vertices_Tet)
            integral_scaling = vol_phy_tet/vol_std_tet

            # Obtain the indices of edges of this tetrahedron
            edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(Tet, 2)])
            
            # Obtain the indices of faces of this tetrahedron
            faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                for v0, v1, v2 in itrs.combinations(Tet, 3)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_Tet))
            fE_integral = np.zeros(len(edges_Tet))
            fH_integral = np.zeros(len(faces_Tet))
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_Tet[:, 0])
                y_q = np.dot(qp_b, vertices_Tet[:, 1])
                z_q = np.dot(qp_b, vertices_Tet[:, 2])
                for i in range(6):
                    fE_integral[i] += np.dot(fE_analytical([x_q, y_q, z_q], (time_step-1) * dt), 
                                             E_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                for i in range(4):
                    fp_integral[i] += np.dot(fp_analytical([x_q, y_q, z_q], (time_step-1) * dt), 
                                             V_Ws[i](qp_b)) * qweights[kq]
                    fH_integral[i] += np.dot(fH_analytical([x_q, y_q, z_q], (time_step-1/2) * dt), 
                                             F_Ws[i](qp_b, dl_Tet)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[Tet] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_Tet] += fE_integral
            fH_integral *= integral_scaling
            b_H[faces_Tet] += fH_integral

        # Obtain boundary coefficients for E and H
        bb_E = np.zeros(N1b)
        H_bc = np.zeros(boundary_faces.shape[0])
        Hbasis_trace_factor = 2

        # Compute the right hand side for the boundary condition on E and also compute the boundary
        # coefficients for the boundary condition on H
        for index, (Face_index, Face) in enumerate(zip(boundary_face_indices, boundary_faces)):
            T_index = sc[2].d.T[Face_index].indices[0]
            T = list(sc[3].simplices[T_index])
            vertices_T = sc.vertices[T]
            vertices_Face = sc.vertices[Face]
            edge_indices_Face = np.array([sc[1].simplex_to_index[pydec.simplex([v0, v1])] 
                                        for v0, v1 in itrs.combinations(Face, 2)])
            Face_local_indices = np.array([T.index(vi) for vi in list(Face)])
            dl_T = dl(vertices_T)
            dl_Face = dl_T[Face_local_indices]
            vol_phy_Face = sc[2].primal_volume[Face_index]
            bfi = boundary_face_identifiers[index]
            boundary_normal = boundary_normals[bfi]
            v0, v1, v2 = vertices_Face
            e_vector01 = v1 - v0
            e_vector02 = v2 - v0
            face_normal_vector = np.cross(e_vector01, e_vector02)
            sign = np.sign(np.dot(face_normal_vector, boundary_normal))
            integral_scaling_2d = vol_phy_Face/vol_std_tri

            # Right hand side vector: Inner product of the Edge Whitney basis with the boundary function
            H_integral = 0
            for kq, qp_b in enumerate(qnodes_bary_2d):
                xq_phy = np.dot(qp_b, vertices_Face[:, 0])
                yq_phy = np.dot(qp_b, vertices_Face[:, 1])
                zq_phy = np.dot(qp_b, vertices_Face[:, 2])
                for i in range(3):    # Loop over edges (E boundary basis)
                    bb_E[boundary_edge_index_map[edge_indices_Face[i]]] += (integral_scaling_2d * 
                        np.dot(E_boundary([xq_phy, yq_phy, zq_phy], (time_step-1) * dt, boundary_normal), 
                               np.cross(Eb_Ws[i](qp_b, dl_Face), boundary_normal)) * qweights_2d[kq])
                H_integral += H_boundary([xq_phy, yq_phy, zq_phy], (time_step-1/2) * dt, boundary_normal) * qweights_2d[kq]
            H_integral *= sc[2].primal_volume[Face_index]/vol_std_tri
            H_bc[index] = sign * H_integral

        # Solve for the coefficients for the Whitney (Edges) basis for the boundary edges for the E x n 
        # boundary condition
        solver_bE = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
        solver_bE.set_iparm(34, 4)    # 
        solver_bE.set_iparm(60, 2)    # 
        solver_bE.factorize(Mb1b1_g)

        # Obtain the linear system solution for E coefficients on the boundary
        E_bc = solver_bE.solve(Mb1b1_g, bb_E)
        
        # Scale the H coefficients on the boundary by the integral of the face basis on this triangle 
        H_bc *= Hbasis_trace_factor

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, (time_step-1) * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Setup right hand side for intermediate variables p, E and H
        b_RHS = np.concatenate((b_p, b_E, b_H)) + sprs.bmat([[1/dt*M00_g, epsilon/2*S01_g, None], 
                                                             [-1/2*S01_g.T, epsilon/dt*M11_g, 1/2*S12_g], 
                                                             [None, -1/2*S12_g.T, mu/dt*M22_g]], 
                                                             format='csr') * np.concatenate((p[time_step - 1], 
                                                                                             E[time_step - 1], 
                                                                                             H[time_step - 1]))

        # Impose boundary conditions on the right hand side at this time step
        b_RHS[boundary_vertex_indices] = p_bc
        b_RHS[N0 + boundary_edge_indices] = E_bc
        b_RHS[N0 + N1 + boundary_face_indices] = H_bc
        
        # Obtain the linear system solution for p and E
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0+N1]
        H[time_step] = x[N0+N1:]

# Visualization of Solutions and Error Computation
if plot_solutions == True:
    print("\n\tinterpolating and plotting solutions, and computing error over time steps..."); sys.stdout.flush()
    if use_leap_frog == True:
        for pts_index, plot_time_step in enumerate(plot_time_steps):
            plot_time = plot_times[pts_index]
            plot_time_H = plot_times_H[pts_index]
            print("\t\tt = %1.3f"%plot_time)

            # Visualize the interpolated vector field at this point in barycentric coordinates on each tetrahedron
            l_bases = np.array([[1/4, 1/4, 1/4, 1/4], [1/2, 1/6, 1/6, 1/6], [1/6, 1/2, 1/6, 1/6], [1/6, 1/6, 1/2, 1/6], 
                                [1/6, 1/6, 1/6, 1/2]])

            # Set up some data structures to store the interpolated vector field
            bases = []; E_arrows = []; H_arrows = []
            p_l2error = 0; E_l2error = 0; H_l2error = 0
            
            p_plot_time = p[plot_time_steps[pts_index]]
            E_plot_time = E[plot_time_steps[pts_index]]
            H_plot_time = H[plot_time_steps[pts_index]]

            # Loop over all tetrahedrons and compute the interpolated vector field
            for Tet_index, Tet in enumerate(tqdm.tqdm(sc[3].simplices)):
                vertices_Tet = sc.vertices[Tet]
                vol_phy_tet = sc[3].primal_volume[Tet_index]
                dl_Tet = dl(vertices_Tet)
                integral_scaling = vol_phy_tet/vol_std_tet

                # Obtain the indices of edges of this tetrahedron
                edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(Tet, 2)])
            
                # Obtain the indices of faces of this tetrahedron
                faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                    for v0, v1, v2 in itrs.combinations(Tet, 3)])

                # Obtain the restriction of discrete p to this tetrahedron
                p_Tet = p_plot_time[Tet]
                
                # Obtain the restriction of discrete E to this tetrahedron
                E_Tet = E_plot_time[edges_Tet]

                # Obtain the restriction of discrete H to this tetrahedron
                H_Tet = H_plot_time[faces_Tet]

                # Obtain the interpolated E and H on this tetrahedron
                bases += [np.dot(vertices_Tet.T, l_base) for l_base in l_bases]
                E_arrows += [E_interpolation(E_Tet, l_base, dl_Tet) for l_base in l_bases]
                H_arrows += [H_interpolation(H_Tet, l_base, dl_Tet) for l_base in l_bases]

                p_l2error_Tet = 0; E_l2error_Tet = 0; H_l2error_Tet = 0

                for kq, qp_b in enumerate(qnodes_bary):
                    xq_phy = np.dot(qp_b, vertices_Tet[:, 0])
                    yq_phy = np.dot(qp_b, vertices_Tet[:, 1])
                    zq_phy = np.dot(qp_b, vertices_Tet[:, 2])

                    p_interpolated_Tet = p_interpolation(p_Tet, qp_b, dl_Tet)
                    p_analytical_Tet = p_analytical([xq_phy, yq_phy, zq_phy], plot_time)

                    E_interpolated_Tet = E_interpolation(E_Tet, qp_b, dl_Tet)
                    E_analytical_Tet = E_analytical([xq_phy, yq_phy, zq_phy], plot_time)

                    H_interpolated_Tet = H_interpolation(H_Tet, qp_b, dl_Tet)
                    H_analytical_Tet = H_analytical([xq_phy, yq_phy, zq_phy], plot_time_H)
                    
                    # Computing errors of E and H at plot times
                    p_l2error_Tet += np.dot(p_analytical_Tet - p_interpolated_Tet, 
                                            p_analytical_Tet - p_interpolated_Tet) * qweights[kq]
                    E_l2error_Tet += np.dot(E_analytical_Tet - E_interpolated_Tet, 
                                            E_analytical_Tet - E_interpolated_Tet) * qweights[kq]
                    H_l2error_Tet += np.dot(H_analytical_Tet - H_interpolated_Tet, 
                                            H_analytical_Tet - H_interpolated_Tet) * qweights[kq]
                                    
                p_l2error_Tet *= integral_scaling
                E_l2error_Tet *= integral_scaling
                H_l2error_Tet *= integral_scaling

                p_l2error += p_l2error_Tet
                E_l2error += E_l2error_Tet
                H_l2error += H_l2error_Tet
                
            p_error_L2.append(np.sqrt(p_l2error))
            E_error_L2.append(np.sqrt(E_l2error))
            H_error_L2.append(np.sqrt(H_l2error))

            bases = np.array(bases); E_arrows = np.array(E_arrows); H_arrows = np.array(H_arrows)
            
            # Obtain the analytical vector field for E at the interpolation point
            E_true_arrows = np.array([E_analytical([x_phy, y_phy, z_phy], plot_time) for x_phy, y_phy, z_phy in bases])

            # Obtain the analytical scalar field for E at the interpolation point
            H_true_arrows = np.array([H_analytical([x_phy, y_phy, z_phy], plot_time_H) for x_phy, y_phy, z_phy in bases])

            # Plot the interpolated solution E on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], E_arrows[:, 0], E_arrows[:, 1], E_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Computed solution for E at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "E_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

            # Plot the analytical vector field E on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], E_true_arrows[:, 0], E_true_arrows[:, 1], E_true_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Analytical solution for E at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "E_analytical_" + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

            # Plot the interpolated solution H on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], H_arrows[:, 0], H_arrows[:, 1], H_arrows[:, 2])    
            mlab.view(**camera_view) 
            # mlab.title("Computed solution for H at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "H_computed_" + fe_order + "_" +
                                        "t%1.4f"%plot_time_H + "_" + str(mesh_no) + ".png"))

            # Plot the analytical vector field H on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], H_true_arrows[:, 0], H_true_arrows[:, 1], H_true_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Analytical solution for H at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "H_analytical_" + "_" + 
                                        "t%1.4f"%plot_time_H + "_" + str(mesh_no) + ".png"))

    else:
        for pts_index, plot_time_step in enumerate(plot_time_steps):
            plot_time = plot_times[pts_index]
            print("\t\tt = %1.3f"%plot_time)

            # Visualize the interpolated vector field at this point in barycentric coordinates on each tetrahedron
            l_bases = np.array([[1/4, 1/4, 1/4, 1/4], [1/2, 1/6, 1/6, 1/6], [1/6, 1/2, 1/6, 1/6], [1/6, 1/6, 1/2, 1/6], 
                                [1/6, 1/6, 1/6, 1/2]])

            # Set up some data structures to store the interpolated vector field
            bases = []; E_arrows = []; H_arrows = []
            p_l2error = 0; E_l2error = 0; H_l2error = 0
            
            p_plot_time = p[plot_time_steps[pts_index]]
            E_plot_time = E[plot_time_steps[pts_index]]
            H_plot_time = H[plot_time_steps[pts_index]]

            # Loop over all tetrahedrons and compute the interpolated vector field
            for Tet_index, Tet in enumerate(tqdm.tqdm(sc[3].simplices)):
                vertices_Tet = sc.vertices[Tet]
                vol_phy_tet = sc[3].primal_volume[Tet_index]
                dl_Tet = dl(vertices_Tet)
                integral_scaling = vol_phy_tet/vol_std_tet

                # Obtain the indices of edges of this tetrahedron
                edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(Tet, 2)])
            
                # Obtain the indices of faces of this tetrahedron
                faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                    for v0, v1, v2 in itrs.combinations(Tet, 3)])

                # Obtain the restriction of discrete p to this tetrahedron
                p_Tet = p_plot_time[Tet]
                
                # Obtain the restriction of discrete E to this tetrahedron
                E_Tet = E_plot_time[edges_Tet]

                # Obtain the restriction of discrete H to this tetrahedron
                H_Tet = H_plot_time[faces_Tet]

                # Obtain the interpolated E and H on this tetrahedron
                bases += [np.dot(vertices_Tet.T, l_base) for l_base in l_bases]
                E_arrows += [E_interpolation(E_Tet, l_base, dl_Tet) for l_base in l_bases]
                H_arrows += [H_interpolation(H_Tet, l_base, dl_Tet) for l_base in l_bases]

                p_l2error_Tet = 0; E_l2error_Tet = 0; H_l2error_Tet = 0

                for kq, qp_b in enumerate(qnodes_bary):
                    xq_phy = np.dot(qp_b, vertices_Tet[:, 0])
                    yq_phy = np.dot(qp_b, vertices_Tet[:, 1])
                    zq_phy = np.dot(qp_b, vertices_Tet[:, 2])

                    p_interpolated_Tet = p_interpolation(p_Tet, qp_b, dl_Tet)
                    p_analytical_Tet = p_analytical([xq_phy, yq_phy, zq_phy], plot_time)

                    E_interpolated_Tet = E_interpolation(E_Tet, qp_b, dl_Tet)
                    E_analytical_Tet = E_analytical([xq_phy, yq_phy, zq_phy], plot_time)

                    H_interpolated_Tet = H_interpolation(H_Tet, qp_b, dl_Tet)
                    H_analytical_Tet = H_analytical([xq_phy, yq_phy, zq_phy], plot_time)
                    
                    # Computing errors of E and H at plot times
                    p_l2error_Tet += np.dot(p_analytical_Tet - p_interpolated_Tet, 
                                            p_analytical_Tet - p_interpolated_Tet) * qweights[kq]
                    E_l2error_Tet += np.dot(E_analytical_Tet - E_interpolated_Tet, 
                                            E_analytical_Tet - E_interpolated_Tet) * qweights[kq]
                    H_l2error_Tet += np.dot(H_analytical_Tet - H_interpolated_Tet, 
                                            H_analytical_Tet - H_interpolated_Tet) * qweights[kq]
                                    
                p_l2error_Tet *= integral_scaling
                E_l2error_Tet *= integral_scaling
                H_l2error_Tet *= integral_scaling

                p_l2error += p_l2error_Tet
                E_l2error += E_l2error_Tet
                H_l2error += H_l2error_Tet
                
            p_error_L2.append(np.sqrt(p_l2error))
            E_error_L2.append(np.sqrt(E_l2error))
            H_error_L2.append(np.sqrt(H_l2error))

            bases = np.array(bases); E_arrows = np.array(E_arrows); H_arrows = np.array(H_arrows)
            
            # Obtain the analytical vector field for E at the interpolation point
            E_true_arrows = np.array([E_analytical([x_phy, y_phy, z_phy], plot_time) for x_phy, y_phy, z_phy in bases])

            # Obtain the analytical scalar field for E at the interpolation point
            H_true_arrows = np.array([H_analytical([x_phy, y_phy, z_phy], plot_time) for x_phy, y_phy, z_phy in bases])

            # Plot the interpolated solution E on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], E_arrows[:, 0], E_arrows[:, 1], E_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Computed solution for E at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "E_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

            # Plot the analytical vector field E on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], E_true_arrows[:, 0], E_true_arrows[:, 1], E_true_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Analytical solution for E at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "E_analytical_" + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

            # Plot the interpolated solution H on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], H_arrows[:, 0], H_arrows[:, 1], H_arrows[:, 2])    
            mlab.view(**camera_view) 
            # mlab.title("Computed solution for H at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "H_computed_" + fe_order + "_" +
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

            # Plot the analytical vector field H on the mesh
            mlab.figure(size=(1024,768), bgcolor=(1, 1, 1))
            mlab.triangular_mesh(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
                                    boundary_triangles, color=(0, 0, 0), line_width=0.75, representation='wireframe')
            mlab.quiver3d(bases[:, 0], bases[:, 1], bases[:, 2], H_true_arrows[:, 0], H_true_arrows[:, 1], H_true_arrows[:, 2])
            mlab.view(**camera_view) 
            # mlab.title("Analytical solution for H at t=%1.4f"%plot_time)
            if save_figs:
                mlab.savefig(pth.join(figs_dir, figs_prestring + "H_analytical_" + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".png"))

if compute_energy == True:
    print("\n\tcomputing L2 norms and L2 energy over time steps..."); sys.stdout.flush()
    for pts_index, solution_time_step in enumerate(tqdm.tqdm(solution_time_steps)):
        solution_time = solution_times[pts_index]

        # Set up some data structures to store the interpolated data for energy
        p_l2norm = 0; E_l2norm = 0; H_l2norm = 0
        p_comp_l2norm = 0; E_comp_l2norm = 0; H_comp_l2norm = 0

        p_solution_time = p[solution_time_step]
        E_solution_time = E[solution_time_step]
        H_solution_time = H[solution_time_step]

        # Loop over all tetrahedrons and compute the L2 norms of interpolated functions as needed
        for Tet_index, Tet in enumerate(sc[3].simplices):
            vertices_Tet = sc.vertices[Tet]
            vol_phy_tet = sc[3].primal_volume[Tet_index]
            dl_Tet = dl(vertices_Tet)
            integral_scaling = vol_phy_tet/vol_std_tet

            # Obtain the indices of edges of this tetrahedron
            edges_Tet = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                            for v0, v1 in itrs.combinations(Tet, 2)])
        
            # Obtain the indices of faces of this tetrahedron
            faces_Tet = np.array([sc[2].simplex_to_index[pydec.simplex((v0, v1, v2))] 
                                for v0, v1, v2 in itrs.combinations(Tet, 3)])

            # Obtain the restriction of p to this triangle
            p_Tet = p_solution_time[Tet]
            
            # Obtain the restriction of discrete E to this tetrahedron
            E_Tet = E_solution_time[edges_Tet]

            # Obtain the restriction of discrete H to this tetrahedron
            H_Tet = H_solution_time[faces_Tet]

            p_l2norm_Tet = 0; E_l2norm_Tet = 0; H_l2norm_Tet = 0
            p_comp_l2norm_Tet = 0; E_comp_l2norm_Tet = 0; H_comp_l2norm_Tet = 0
            for kq, qp_b in enumerate(qnodes_bary):
                xq_phy = np.dot(qp_b, vertices_Tet[:, 0])
                yq_phy = np.dot(qp_b, vertices_Tet[:, 1])
                zq_phy = np.dot(qp_b, vertices_Tet[:, 2])

                p_interpolated_Tet = p_interpolation(p_Tet, qp_b, dl_Tet)
                p_analytical_Tet = p_analytical([xq_phy, yq_phy, zq_phy], solution_time)
                
                E_interpolated_Tet = E_interpolation(E_Tet, qp_b, dl_Tet)
                E_analytical_Tet = E_analytical([xq_phy, yq_phy, zq_phy], solution_time)

                H_interpolated_Tet = H_interpolation(H_Tet, qp_b, dl_Tet)
                H_analytical_Tet = H_analytical([xq_phy, yq_phy, zq_phy], solution_time)

                # Computing norms of analytical E and H at solution times
                p_l2norm_Tet += np.dot(p_analytical_Tet, p_analytical_Tet) * qweights[kq]
                E_l2norm_Tet += np.dot(E_analytical_Tet, E_analytical_Tet) * qweights[kq]
                H_l2norm_Tet += np.dot(H_analytical_Tet, H_analytical_Tet) * qweights[kq]

                # Computing norms of interpolated E and H at solution times
                p_comp_l2norm_Tet += np.dot(p_interpolated_Tet, p_interpolated_Tet) * qweights[kq]
                E_comp_l2norm_Tet += np.dot(E_interpolated_Tet, E_interpolated_Tet) * qweights[kq]
                H_comp_l2norm_Tet += np.dot(H_interpolated_Tet, H_interpolated_Tet) * qweights[kq]

            p_l2norm_Tet *= integral_scaling
            E_l2norm_Tet *= integral_scaling
            H_l2norm_Tet *= integral_scaling
            p_comp_l2norm_Tet *= integral_scaling
            E_comp_l2norm_Tet *= integral_scaling
            H_comp_l2norm_Tet *= integral_scaling

            p_l2norm += p_l2norm_Tet
            E_l2norm += E_l2norm_Tet
            H_l2norm += H_l2norm_Tet
            p_comp_l2norm += p_comp_l2norm_Tet
            E_comp_l2norm += E_comp_l2norm_Tet
            H_comp_l2norm += H_comp_l2norm_Tet
        
        p_norm_L2.append(np.sqrt(p_l2norm))
        E_norm_L2.append(np.sqrt(E_l2norm))
        H_norm_L2.append(np.sqrt(H_l2norm))
        p_comp_norm_L2.append(np.sqrt(p_comp_l2norm))
        E_comp_norm_L2.append(np.sqrt(E_comp_l2norm))
        H_comp_norm_L2.append(np.sqrt(H_comp_l2norm))
        L2_energy.append(p_l2norm + epsilon*E_l2norm + mu*H_l2norm)
        comp_L2_energy.append(p_comp_l2norm + epsilon*E_comp_l2norm + mu*H_comp_l2norm)

p_error_L2 = np.array(p_error_L2); E_error_L2 = np.array(E_error_L2); H_error_L2 = np.array(H_error_L2)
p_norm_L2 = np.array(p_norm_L2); E_norm_L2 = np.array(E_norm_L2); H_norm_L2 = np.array(H_norm_L2)
p_comp_norm_L2 = np.array(p_comp_norm_L2); E_comp_norm_L2 = np.array(E_comp_norm_L2); H_comp_norm_L2 = np.array(H_comp_norm_L2)
L2_energy = np.array(L2_energy); comp_L2_energy = np.array(comp_L2_energy)

# Print errors and parameters
print("\nL2 norm of analytical scalar field p: ", p_norm_L2)
print("L2 norm of computed scalar field p: ", p_comp_norm_L2)
print("L2 error in scalar field E: ", p_error_L2)

print("\nL2 norm of analytical vector field E: ", E_norm_L2)
print("L2 norm of computed vector field E: ", E_comp_norm_L2)
print("L2 error in vector field E: ", E_error_L2)

print("\nL2 norm of analytical vector field H: ", H_norm_L2)
print("L2 norm of computed vector field H: ", H_comp_norm_L2)
print("L2 error in vector field H: ", H_error_L2)

print("\nL2 Energy of analytical fields: ", L2_energy)
print("L2 Energy of computed fields: ", comp_L2_energy)

# Save data
if save_data:
    np.savetxt(pth.join(data_dir, data_prestring + "pl2analytical_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               p_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "El2analytical_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               E_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "Hl2analytical_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               H_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "pl2computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               p_comp_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "El2computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               E_comp_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "Hl2computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               H_comp_norm_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "pl2error_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               p_error_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "El2error_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               E_error_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "Hl2error_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               H_error_L2, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "plot_times.txt"), 
               plot_times, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "energy_analytical_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               L2_energy, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "energy_computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               comp_L2_energy, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "solution_times.txt"), 
               solution_times, fmt="%1.16e")

    np.savetxt(pth.join(data_dir, data_prestring + "p_computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               p, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "E_computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               E, fmt="%1.16e")
    np.savetxt(pth.join(data_dir, data_prestring + "H_computed_" + fe_order + "_" + str(mesh_no) + ".txt"), 
               H, fmt="%1.16e")




    
