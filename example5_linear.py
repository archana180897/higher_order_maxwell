# Code for Example 5 of our paper at https://arxiv.org/abs/2310.20310.
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
epsilon = 1
mu = 1

# Time Parameters
dt = 1e-2; T_min = 0; T_max = 3.14
number_of_plot_times = 9   # How many time steps to sample for plotting
filename_prefix = "Maxwells"

# Meshes
mesh_dir = "meshes/unit_square/"
mesh_no = 1

# FE Space Choice
fe_order = "Linear"
p_string = "example5"

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
data_dir = "data/2d/"; data_prestring = p_string + "_" + method_string + "_"
figs_dir = "figs/2d/"; figs_prestring = p_string + "_" + method_string + "_"
savefig_options = {"bbox_inches": 'tight'}

# Visualization choices
camera_view = {"azimuth": 45.0, "elevation": 54.735, "distance": 3.25,
                   "focalpoint": np.array([0.5, 0.5, 0.5])} 

# Analytical Solutions and Right Hand Side
# Analytical p
def p_analytical(v, t):
    x = v[0]
    y = v[1]
    return (np.cos(x) + np.cos(y)) * np.sin(t)

# Analytical E
def E_analytical(v, t):
    x = v[0]
    y = v[1]
    return np.array([np.sin(np.pi*(np.sqrt(2)*t - (x+y))) - np.sin(x)*np.cos(t), 
                     -np.sin(np.pi*(np.sqrt(2)*t - (x+y))) - np.sin(y)*np.cos(t)])

# Analytical H
def H_analytical(v, t):
    x = v[0]
    y = v[1]
    return -np.sqrt(2)*np.sin(np.pi*(np.sqrt(2)*t - (x+y)))

# Analytical f_p
def fp_analytical(v, t):
    x = v[0]
    y = v[1]
    return 0

# Analytical f_E
def fE_analytical(v, t):
    x = v[0]
    y = v[1]
    return np.array([0, 0])

# Analytical f_H
def fH_analytical(v, t):
    x = v[0]
    y = v[1]
    return 0

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
# Quadrature rule choices
dims = 2
order = 4
quad2d = modepy.XiaoGimbutasSimplexQuadrature(order, dims)
qnodes = quad2d.nodes.T
qweights = quad2d.weights

# Vertices of standard triangle on which quadrature nodes are specified
Ts_v = np.array([[-1, -1], [1, -1], [-1, 1]])

# Express the quadrature nodes in barycentric coordinates
A = np.vstack((Ts_v.T, np.ones(3)))
B = np.hstack((qnodes, np.ones((qnodes.shape[0], 1)))).T
qnodes_bary = np.linalg.solve(A, B).T

# Area of this standard triangle for quadrature
vol_std_tri = np.sum(qweights)


## Finite Element Setup and Utility Functions 
# Barycentric Gradients on a triangle with specified vertices
def dl(vertices_T):
    return pydec.barycentric_gradients(vertices_T)

# Lagrange Basis (order = 1) on a physical simplex
def W0(ell_T):
    return ell_T[0]
def W1(ell_T):
    return ell_T[1]
def W2(ell_T):
    return ell_T[2]

# Gradient of Lagrange Basis (order = 1)
def grad_W0(dl_T):
    return dl_T[0]
def grad_W1(dl_T):
    return dl_T[1]
def grad_W2(dl_T):
    return dl_T[2]

# Whitney basis (order = 1) on physical simplex
# Bases associated with the edges
def W01(ell_T, dl_T):
    return ell_T[0]*dl_T[1] - ell_T[1]*dl_T[0]
def W02(ell_T, dl_T):
    return ell_T[0]*dl_T[2] - ell_T[2]*dl_T[0]
def W12(ell_T, dl_T):
    return ell_T[1]*dl_T[2] - ell_T[2]*dl_T[1]

# General rotation function for computing particular rotations of Whitney basis 
# on a physical simplex
# Note: This function computes the rotation of the following general form:
#       Rotation(ell_0**a * ell_1**b * ell_2**c * Gradient[ell_t])
# where a, b, c are nonnegative integers, and t belongs to {0, 1, 2}.
def rot_Wbasis(ell_T, dl_T, a, b, c, t):
    return (dl_T[t,1] * (a * ell_T[0]**(a-1) * ell_T[1]**b * ell_T[2]**c * dl_T[0,0] + 
                          b * ell_T[0]**a * ell_T[1]**(b-1) * ell_T[2]**c * dl_T[1,0] + 
                          c * ell_T[0]**a * ell_T[1]**b * ell_T[2]**(c-1) * dl_T[2,0]) - 
            dl_T[t,0] * (a * ell_T[0]**(a-1) * ell_T[1]**b * ell_T[2]**c * dl_T[0,1] +  
                         b * ell_T[0]**a * ell_T[1]**(b-1) * ell_T[2]**c * dl_T[1,1] + 
                         c * ell_T[0]**a * ell_T[1]**b * ell_T[2]**(c-1) * dl_T[2,1]))

# Rotation of Whitney basis on physical triangle
def rot_W01(ell_T, dl_T):
    return rot_Wbasis(ell_T, dl_T, a=1, b=0, c=0, t=1) - rot_Wbasis(ell_T, dl_T, a=0, b=1, c=0, t=0)
def rot_W02(ell_T, dl_T):
    return rot_Wbasis(ell_T, dl_T, a=1, b=0, c=0, t=2) - rot_Wbasis(ell_T, dl_T, a=0, b=0, c=1, t=0)
def rot_W12(ell_T, dl_T):
    return rot_Wbasis(ell_T, dl_T, a=0, b=1, c=0, t=2) - rot_Wbasis(ell_T, dl_T, a=0, b=0, c=1, t=1)

# Discontinuous Galerkin (order = 1) basis on a physical triangle
def DG012(ell_T, vol_T):
    return 1/vol_T

# Lists referencing the various basis functions
V_Ws = [W0, W1, W2]
grad_V_Ws = [grad_W0, grad_W1, grad_W2]
E_Ws = [W01, W02, W12]
rot_E_Ws = [rot_W01, rot_W02, rot_W12]
T_DGs = [DG012]

##  Finite Element System Computations  ##
# Mass Matrix 00: Inner product of Lagrange Basis and Lagrange Basis
def Mass_00(T, T_index):
    M = np.zeros((len(V_Ws), len(V_Ws)))
    vol_phy_tri = sc[2].primal_volume[T_index]

    # The required inner product is computed via quadrature
    for i in range(len(V_Ws)):    # Loop over vertices (p basis)
        for j in range(len(V_Ws)):    # Loop over vertices (p basis)
            integral = 0
            for k, qp_b in enumerate(qnodes_bary):
                integral += np.dot(V_Ws[i](qp_b),
                                   V_Ws[j](qp_b)) * qweights[k]
            integral *= vol_phy_tri/vol_std_tri
            M[i, j] = integral

    return M

# Stiffness Matrix 01: Inner product of Gradient of Lagrange Basis and Whitney Basis 
def Stiff_01(T, T_index):
    S = np.zeros((len(grad_V_Ws), len(E_Ws)))
    vol_phy_tri = sc[2].primal_volume[T_index]
    vertices_T = sc.vertices[T]
    dl_T = dl(vertices_T)
    integral_scaling = vol_phy_tri/vol_std_tri 
    
    # The required inner product is computed via quadrature
    for i in range(len(grad_V_Ws)):    # Loop over vertices (p basis)
        for j in range(len(E_Ws)):    # Loop over edges (E basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(grad_V_Ws[i](dl_T), 
                                   E_Ws[j](qp_b, dl_T)) * qweights[kq]
            integral *= integral_scaling
            S[i, j] = integral

    return S

# Mass Matrix 11: Inner product of Edge Whitney Basis and Edge Whitney Basis
def Mass_11(T, T_index):
    M = np.zeros((len(E_Ws), len(E_Ws)))
    vol_phy_tri = sc[2].primal_volume[T_index]
    vertices_T = sc.vertices[T]
    dl_T = dl(vertices_T)
    integral_scaling = vol_phy_tri/vol_std_tri

    # The required inner product is computed via quadrature
    for i in range(len(E_Ws)):    # Loop over edges (E basis)
        for j in range(len(E_Ws)):    # Loop over edges (E basis)
            integral = 0
            for k, qp_b in enumerate(qnodes_bary):
                integral += np.dot(E_Ws[i](qp_b, dl_T),
                                   E_Ws[j](qp_b, dl_T)) * qweights[k]
            integral *= integral_scaling
            M[i, j] = integral

    return M

# Stiffness Matrix 12: Inner product of Curl of Edge Whitney Basis and Discontinuous Galerkin
def Stiff_12(T, T_index):
    S = np.zeros((len(rot_E_Ws), len(T_DGs)))
    vol_phy_tri = sc[2].primal_volume[T_index]
    dl_T = dl(sc.vertices[T])
    integral_scaling = vol_phy_tri/vol_std_tri
    
    # The required inner product is computed via quadrature
    for i in range(len(rot_E_Ws)):    # Loop over edges (E basis)
        for j in range(len(T_DGs)):    # Loop over faces (H basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(rot_E_Ws[i](qp_b, dl_T), 
                                   T_DGs[j](qp_b, vol_phy_tri)) * qweights[kq]
            integral *= integral_scaling
            S[i, j] = integral

    return S

# Stiffness Matrix 22: Inner product of Discontinuous Galerkin with Discontinuous Galerkin
def Mass_22(T, T_index):
    M = np.zeros((len(T_DGs), len(T_DGs)))
    vol_phy_tri = sc[2].primal_volume[T_index]
    dl_T = dl(sc.vertices[T])
    integral_scaling = vol_phy_tri/vol_std_tri

    # The required inner product is computed via quadrature
    for i in range(len(T_DGs)):    # Loop over faces (H basis)
        for j in range(len(T_DGs)):    # Loop over faces (H basis)
            integral = 0
            for kq, qp_b in enumerate(qnodes_bary):
                integral += np.dot(T_DGs[i](qp_b, vol_phy_tri), 
                                   T_DGs[j](qp_b, vol_phy_tri)) * qweights[kq]
            integral *= integral_scaling
            M[i, j] = integral

    return M

# Interpolation for p
def p_interpolation(p_T, ell_interp, dl_T):
    # Compute the interpolant at the interpolation point
    return np.sum([p_T[i] * V_Ws[i](ell_interp) for i in range(len(V_Ws))], axis=0)

# Interpolation for E
def E_interpolation(E_T, ell_interp, dl_T):
    # Compute the interpolant at the interpolation point
    return np.sum([E_T[i] * E_Ws[i](ell_interp, dl_T) for i in range(len(E_Ws))], axis=0)

# Interpolation for H
def H_interpolation(H_T, ell_interp, vol_T):
    # Compute the interpolant at the interpolation point
    return np.sum([H_T * T_DGs[i](ell_interp, vol_T) for i in range(len(T_DGs))], axis=0)

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
Triangles = np.loadtxt(pth.join(mesh_dir, "triangles" + str(mesh_no) + ".txt"), dtype=int)

sc = pydec.simplicial_complex(Vertices, Triangles)
Vertices = Vertices[sc[0].simplices.ravel()]

# A preprocessing to remove any potentially duplicate vertices in the mesh provided by a
# mesh generator
vertex_vertex_map = {}
for v, k in enumerate(sc[0].simplices.ravel()):
    vertex_vertex_map[k] = v

for T_index, T in enumerate(Triangles):
    for i in range(3):
        Triangles[T_index][i] = vertex_vertex_map[T[i]]

# Use PyDEC to setup the simplicial complex and related operators
sc = pydec.simplicial_complex(Vertices, Triangles)
N0 = sc[0].num_simplices
N1 = sc[1].num_simplices
N2 = sc[2].num_simplices
boundary_2 = sc[1].d.T

print("============================================================")
print("Spatial Discretization using Linear Polynomial Basis.")
print("Initial Time: ", T_min, ", Final Time: ", T_max, ", Time Step Size: ", dt)
print("============================================================")

print("\nComputing over mesh " + str(mesh_no) + "..."); sys.stdout.flush()
# Obtain the boundary edges of the mesh
boundary_edge_indices = (sc[1].d.T * np.ones(N2)).nonzero()[0]
boundary_edges = sc[1].simplices[boundary_edge_indices]

# Obtain the boundary vertices of the mesh
boundary_vertex_indices = []
for edge in boundary_edges:
    boundary_vertex_indices.append(edge[0])
    boundary_vertex_indices.append(edge[1])
boundary_vertex_indices = np.array(list(set(boundary_vertex_indices)))

# Obtain left, right, top and bottom boundary vertices
boundary_left_vertices = []; boundary_right_vertices = []
boundary_top_vertices = []; boundary_bottom_vertices = []

for bv in boundary_vertex_indices:
    if abs(Vertices[bv, 0]) <= float_tol:
        boundary_left_vertices.append(bv)
    if abs(Vertices[bv, 1]) <= float_tol:
        boundary_bottom_vertices.append(bv)
    if abs(1 - Vertices[bv, 0]) <= float_tol:
        boundary_right_vertices.append(bv)
    if  abs(1 - Vertices[bv, 1]) <= float_tol:
        boundary_top_vertices.append(bv)

# The corners of the square will be in two boundaries. We shall assign each of the corners
# to only the left and right boundaries
bv_left_top = (set(boundary_top_vertices).intersection(set(boundary_left_vertices))).pop()
boundary_top_vertices.remove(bv_left_top)
bv_left_bottom = (set(boundary_bottom_vertices).intersection(set(boundary_left_vertices))).pop()
boundary_bottom_vertices.remove(bv_left_bottom)
bv_right_top = (set(boundary_top_vertices).intersection(set(boundary_right_vertices))).pop()
boundary_top_vertices.remove(bv_right_top)
bv_right_bottom = (set(boundary_bottom_vertices).intersection(set(boundary_right_vertices))).pop()
boundary_bottom_vertices.remove(bv_right_bottom)

boundary_left_vertices = np.array(boundary_left_vertices)
boundary_top_vertices = np.array(boundary_top_vertices)
boundary_right_vertices = np.array(boundary_right_vertices) 
boundary_bottom_vertices = np.array(boundary_bottom_vertices)

# Obtain left, right, top and bottom boundary edges
boundary_left_edges = []; boundary_right_edges = []
boundary_top_edges = []; boundary_bottom_edges = []

for bei in boundary_edge_indices:
    v0, v1 = Vertices[sc[1].simplices[bei]]
    bv = (v0 + v1)/2
    if abs(bv[0]) <= float_tol:
        boundary_left_edges.append(bei)
    if abs(bv[1]) <= float_tol:
        boundary_bottom_edges.append(bei)
    if abs(1 - bv[0]) <= float_tol:
        boundary_right_edges.append(bei)
    if abs(1 - bv[1]) <= float_tol:
        boundary_top_edges.append(bei)

boundary_left_edges = np.array(boundary_left_edges)
boundary_top_edges = np.array(boundary_top_edges)
boundary_right_edges = np.array(boundary_right_edges) 
boundary_bottom_edges = np.array(boundary_bottom_edges)

# Define a function to compute the normal trace of the dot product of an analytical function 
# with a Whitney basis on a boundary edge
def normal_boundary_trace(analytical_function, boundary_edges_local, normal_direction, time):
    boundary_edges_local = np.asarray(boundary_edges_local)
    E_trace_boundary_edges = np.zeros(boundary_edges_local.shape[0])
    for edge_index, edge in enumerate(boundary_edges_local):
        vi, vj = sc[1].simplices[edge]
        T_index = boundary_2[edge].indices[0]
        edge_vector = sc.vertices[vj] - sc.vertices[vi]
        def E_trace_on_edge (t):
            E_edge_evals = []
            for t_ in np.asarray(t):
                E_edge_evals.append(
                    np.cross(analytical_function(sc.vertices[vi] + t_ * edge_vector, time), normal_direction))
            return np.array(E_edge_evals)
        edge_length = sc[1].primal_volume[edge]
        s = boundary_2[edge, T_index]
        E_trace_boundary_edges[edge_index] = -s * edge_length * integrate.fixed_quad(E_trace_on_edge, 0, 1, n=5)[0]
    return E_trace_boundary_edges

# Compute normal trace of analytical solution E on each of the boundaries
E0_left_boundary = normal_boundary_trace(E_analytical, boundary_left_edges, np.array([-1, 0]), 0)
E0_right_boundary = normal_boundary_trace(E_analytical, boundary_right_edges, np.array([1, 0]), 0)
E0_top_boundary = normal_boundary_trace(E_analytical, boundary_top_edges, np.array([0, 1]), 0)
E0_bottom_boundary = normal_boundary_trace(E_analytical, boundary_bottom_edges, np.array([0, -1]), 0)

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
for T_index, T in enumerate(tqdm.tqdm(sc[2].simplices)):
    vertices_T = sc.vertices[T]
    vol_phy_tri = sc[2].primal_volume[T_index]
    dl_T = dl(vertices_T)
    integral_scaling = vol_phy_tri/vol_std_tri 

    # Obtain the indices of edges of this tetrahedron
    edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                        for v0, v1 in itrs.combinations(T, 2)])
    
    # Obtain the local mass and stiffness matrices
    M00 = Mass_00(T, T_index)
    S01 = Stiff_01(T, T_index)
    M11 = Mass_11(T, T_index)    
    S12 = Stiff_12(T, T_index)
    M22 = Mass_22(T, T_index)
    
    # Integrate the local matrices into data structures for the global matrices
    for i in range(3):    # Loop over vertices/edges
        for j in range(3):    # Loop over vertices/edges
            rows00.append(T[i])
            columns00.append(T[j])
            M00_data.append(M00[i, j])

            rows01.append(T[i])
            columns01.append(edges_T[j])
            S01_data.append(S01[i, j])

            rows11.append(edges_T[i])
            columns11.append(edges_T[j])
            M11_data.append(M11[i, j])

        for j in range(1):    # Loop over faces
            rows12.append(edges_T[i])
            columns12.append(T_index)
            S12_data.append(S12[i, j])

    for i in range(1):    # Loop over faces
        for j in range(1):    # Loop over faces
            rows22.append(T_index)
            columns22.append(T_index)
            M22_data.append(M22[i, j])
        
    for i in range(3):
        E_integral = 0
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_T[:, 0])
            y_q = np.dot(qp_b, vertices_T[:, 1])

            # Computing Initial Vector at t = T_min
            E_integral += np.dot(E_analytical([x_q, y_q], T_min) , E_Ws[i](qp_b, dl_T)) * qweights[kq]
        E_integral *= integral_scaling
        E_0[edges_T[i]] += E_integral

    for i in range(1):
        H_integral = 0
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_T[:, 0])
            y_q = np.dot(qp_b, vertices_T[:, 1])

            # Computing Initial Vector at t = T_min
            H_integral += np.dot(H_analytical([x_q, y_q], T_min) , T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
        H_integral *= integral_scaling
        H_0[T_index] += H_integral

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
     rows12, columns12, S12_data, rows22, columns22, M22_data)

# Computing Initial Vectors
print("\n\tsolving for initial condition vectors..."); sys.stdout.flush()
p = np.zeros((number_of_time_steps, N0))
E = np.zeros((number_of_time_steps, N1))
H = np.zeros((number_of_time_steps, N2))
H_true = np.zeros((number_of_time_steps, N2))

# Initial p
for i, v in enumerate(sc.vertices):
    p_0[i] = p_analytical(v, t=0)
p[0] = p_0
p0_initial = p[0].copy()

# Initial E and H
# Setup boundary conditions for computation of initial E
M11_g_bndry = M11_g.copy()
M11_g_bndry[boundary_edge_indices] = 0
M11_g_bndry[boundary_edge_indices, boundary_edge_indices] = 1
E_0[boundary_left_edges] = E0_left_boundary
E_0[boundary_right_edges] = E0_right_boundary
E_0[boundary_top_edges] = E0_top_boundary
E_0[boundary_bottom_edges] = E0_bottom_boundary

# Solve for initial E
solverM11_bndry = pypard.PyPardisoSolver()
solverM11_bndry.set_iparm(34, 4)
solverM11_bndry.set_iparm(60, 2)
solverM11_bndry.factorize(M11_g_bndry)
E[0] = solverM11_bndry.solve(M11_g_bndry, E_0)
E0_initial = E[0].copy()

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
H[0] = solverM22.solve(M22_g, H_0)
H_true[0] = H[0].copy()

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
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1

    # Setup linear system solver for the solution of E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    for time_step in tqdm.tqdm(range(1, number_of_time_steps)):
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2); H_h = np.zeros(N2)
        for T_index, T in enumerate(sc[2].simplices):
            vertices_T = sc.vertices[T]
            vol_phy_tri = sc[2].primal_volume[T_index]
            dl_T = dl(vertices_T)
            integral_scaling = vol_phy_tri/vol_std_tri

            # Obtain the indices of edges of this tetrahedron
            edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(T, 2)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_T))
            fE_integral = np.zeros(len(edges_T))
            fH_integral = np.zeros(1)
            H_integral = np.zeros(1)
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_T[:, 0])
                y_q = np.dot(qp_b, vertices_T[:, 1])
                for i in range(3):
                    fp_integral[i] += np.dot(fp_analytical([x_q, y_q], (time_step) * dt), 
                                             V_Ws[i](qp_b)) * qweights[kq]
                    fE_integral[i] += np.dot(fE_analytical([x_q, y_q], (time_step) * dt), 
                                             E_Ws[i](qp_b, dl_T)) * qweights[kq]
                for i in range(1):
                    fH_integral[i] += np.dot(fH_analytical([x_q, y_q], (time_step) * dt), 
                                             T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    H_integral[i] += np.dot(H_analytical([x_q, y_q], (time_step) * dt), 
                                            T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[T] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_T] += fE_integral
            fH_integral *= integral_scaling
            b_H[T_index] += fH_integral
            H_integral *= integral_scaling
            H_h[T_index] += H_integral

        # Setup right hand side intermediate variables
        bp_RHS = b_p + 1/dt*M00_g*p[time_step - 1] 
        bE_RHS = -b_E - epsilon/dt*M11_g*E[time_step - 1] 
        bH_RHS = b_H + mu/dt*M22_g*H[time_step - 1]

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, time_step * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Compute normal trace of analytical solution E on each of the boundaries
        E_left_boundary = normal_boundary_trace(E_analytical, boundary_left_edges, np.array([-1, 0]), time_step*dt)
        E_right_boundary = normal_boundary_trace(E_analytical, boundary_right_edges, np.array([1, 0]), time_step*dt)
        E_top_boundary = normal_boundary_trace(E_analytical, boundary_top_edges, np.array([0, 1]), time_step*dt)
        E_bottom_boundary = normal_boundary_trace(E_analytical, boundary_bottom_edges, np.array([0, -1]), time_step*dt)

        # Impose boundary conditions on the right hand side at this time step
        bp_RHS[boundary_vertex_indices] = p_bc
        bE_RHS[boundary_left_edges] = E_left_boundary
        bE_RHS[boundary_right_edges] = E_right_boundary
        bE_RHS[boundary_top_edges] = E_top_boundary
        bE_RHS[boundary_bottom_edges] = E_bottom_boundary

        # Setup the right hand side matrix
        b_RHS = np.concatenate((bp_RHS, bE_RHS, bH_RHS))

        # Obtain the linear system solution for E and H
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0 + N1]
        H[time_step] = x[N0 + N1:]

        # Obtain the L2 projection of the analytical density field for u to the space spanned by Discontinuous
        # Galerkin basis (order = 1) 
        H_true[time_step] = solverM22.solve(M22_g, H_h)


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
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1

    # Setup linear system solver for the solution of E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    for time_step in tqdm.tqdm(range(1, number_of_time_steps)):
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2); H_h = np.zeros(N2)
        for T_index, T in enumerate(sc[2].simplices):
            vertices_T = sc.vertices[T]
            vol_phy_tri = sc[2].primal_volume[T_index]
            dl_T = dl(vertices_T)
            integral_scaling = vol_phy_tri/vol_std_tri

            # Obtain the indices of edges of this tetrahedron
            edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(T, 2)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_T))
            fE_integral = np.zeros(len(edges_T))
            fH_integral = np.zeros(1)
            H_integral = np.zeros(1)
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_T[:, 0])
                y_q = np.dot(qp_b, vertices_T[:, 1])
                for i in range(3):
                    fp_integral[i] += np.dot((fp_analytical([x_q, y_q], (time_step) * dt) + 
                                              fp_analytical([x_q, y_q], (time_step - 1) * dt)), 
                                              V_Ws[i](qp_b)) * qweights[kq]
                    fE_integral[i] += np.dot((fE_analytical([x_q, y_q], (time_step) * dt) + 
                                              fE_analytical([x_q, y_q], (time_step - 1) * dt)), 
                                              E_Ws[i](qp_b, dl_T)) * qweights[kq]
                for i in range(1):
                    fH_integral[i] += np.dot((fH_analytical([x_q, y_q], (time_step) * dt) + 
                                              fH_analytical([x_q, y_q], (time_step - 1) * dt)), 
                                              T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    H_integral[i] += np.dot(H_analytical([x_q, y_q], (time_step) * dt), 
                                            T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[T] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_T] += fE_integral
            fH_integral *= integral_scaling
            b_H[T_index] += fH_integral
            H_integral *= integral_scaling
            H_h[T_index] += H_integral

        # Setup right hand side intermediate variables
        bp_RHS = b_p/2 + 1/dt*M00_g*p[time_step - 1] + epsilon*S01_g/2*E[time_step - 1]
        bE_RHS = -b_E/2 + S01_g.T/2*p[time_step - 1] - epsilon/dt*M11_g*E[time_step - 1] - S12_g/2*H[time_step - 1]
        bH_RHS = b_H/2 - S12_g.T/2*E[time_step - 1] + mu/dt*M22_g*H[time_step - 1]

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, time_step * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Compute normal trace of analytical solution E on each of the boundaries
        E_left_boundary = normal_boundary_trace(E_analytical, boundary_left_edges, np.array([-1, 0]), time_step*dt)
        E_right_boundary = normal_boundary_trace(E_analytical, boundary_right_edges, np.array([1, 0]), time_step*dt)
        E_top_boundary = normal_boundary_trace(E_analytical, boundary_top_edges, np.array([0, 1]), time_step*dt)
        E_bottom_boundary = normal_boundary_trace(E_analytical, boundary_bottom_edges, np.array([0, -1]), time_step*dt)

        # Impose boundary conditions on the right hand side at this time step
        bp_RHS[boundary_vertex_indices] = p_bc
        bE_RHS[boundary_left_edges] = E_left_boundary
        bE_RHS[boundary_right_edges] = E_right_boundary
        bE_RHS[boundary_top_edges] = E_top_boundary
        bE_RHS[boundary_bottom_edges] = E_bottom_boundary

        # Setup the right hand side matrix
        b_RHS = np.concatenate((bp_RHS, bE_RHS, bH_RHS))

        # Obtain the linear system solution for E and H
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0 + N1]
        H[time_step] = x[N0 + N1:]

        # Obtain the L2 projection of the analytical density field for u to the space spanned by Discontinuous
        # Galerkin basis (order = 1) 
        H_true[time_step] = solverM22.solve(M22_g, H_h)


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
    S_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1

    # Setup linear system solver for the solution of p, E and H
    S_LHS.eliminate_zeros()    # Sparsify again
    pEH_solver = pypard.PyPardisoSolver()    # PyParadiso is a wrapper for Intel MKL's Paradiso
    pEH_solver.set_iparm(34, 4)    # 
    pEH_solver.set_iparm(60, 2)    # 
    pEH_solver.factorize(S_LHS)

    # Initial right hand sides
    b0_p = np.zeros(N0); b0_E = np.zeros(N1); b0_H = np.zeros(N2)
    for T_index, T in enumerate(sc[2].simplices):
        vertices_T = sc.vertices[T]
        vol_phy_tri = sc[2].primal_volume[T_index]
        dl_T = dl(vertices_T)
        integral_scaling = vol_phy_tri/vol_std_tri

        # Obtain the indices of edges of this tetrahedron
        edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                            for v0, v1 in itrs.combinations(T, 2)])
        
        # Computing values of f_p, f_E and f_H at time step '0'
        fp_integral = np.zeros(len(vertices_T))
        fE_integral = np.zeros(len(edges_T))
        fH_integral = np.zeros(1)
        for kq, qp_b in enumerate(qnodes_bary):
            x_q = np.dot(qp_b, vertices_T[:, 0])
            y_q = np.dot(qp_b, vertices_T[:, 1])
            for i in range(3):
                fp_integral[i] += np.dot(fp_analytical([x_q, y_q], 0), 
                                         V_Ws[i](qp_b)) * qweights[kq]
                fE_integral[i] += np.dot(fE_analytical([x_q, y_q], 0), 
                                         E_Ws[i](qp_b, dl_T)) * qweights[kq]
            for i in range(1):
                fH_integral[i] += np.dot(fH_analytical([x_q, y_q], dt/2), 
                                          T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                
        fp_integral *= integral_scaling
        b0_p[T] += fp_integral
        fE_integral *= integral_scaling
        b0_E[edges_T] += fE_integral
        fH_integral *= integral_scaling
        b0_H[T_index] += fH_integral

    # Incorporate boundary conditions in p
    p0_bc = np.array([p_analytical(v, 0) for v in sc.vertices[boundary_vertex_indices]])

    # Compute normal trace of analytical solution E on each of the boundaries
    E0_left_boundary = normal_boundary_trace(E_analytical, boundary_left_edges, np.array([-1, 0]), 0)
    E0_right_boundary = normal_boundary_trace(E_analytical, boundary_right_edges, np.array([1, 0]), 0)
    E0_top_boundary = normal_boundary_trace(E_analytical, boundary_top_edges, np.array([0, 1]), 0)
    E0_bottom_boundary = normal_boundary_trace(E_analytical, boundary_bottom_edges, np.array([0, -1]), 0)
    
    # Compute right hand side at this time step
    b0_RHS = np.concatenate((b0_p, b0_E, b0_H)) + sprs.bmat([[2/dt*M00_g, epsilon/4*S01_g, None], 
                                                         [-1/4*S01_g.T, 2/dt*epsilon*M11_g, 1/2*S12_g],
                                                         [None, -1/4*S12_g.T, 1/dt*mu*M22_g]], 
                                                         format='csr') * np.concatenate((p[0], E[0], H[0]))

    # Impose boundary conditions on the right hand side at this time step
    b0_RHS[boundary_vertex_indices] = p0_bc
    b0_RHS[N0 + boundary_left_edges] = E0_left_boundary
    b0_RHS[N0 + boundary_right_edges] = E0_right_boundary
    b0_RHS[N0 + boundary_top_edges] = E0_top_boundary
    b0_RHS[N0 + boundary_bottom_edges] = E0_bottom_boundary

    S0_LHS = sprs.bmat([[2/dt*M00_g, -epsilon/4*S01_g, None],
                        [-1/4*S01_g.T, 2/dt*epsilon*M11_g, -1/2*S12_g],
                        [None, 1/4*S12_g.T, mu/dt*M22_g]], format='csr')

    # Modify this system matrix to account for boundary conditions
    bc_zeromatrix_vertices = sprs.csr_matrix((boundary_vertex_indices.shape[0], S0_LHS.shape[0]), dtype=S0_LHS.dtype)
    bc_zeromatrix_edges = sprs.csr_matrix((boundary_edge_indices.shape[0], S0_LHS.shape[0]), dtype=S0_LHS.dtype)
    S0_LHS[boundary_vertex_indices] = bc_zeromatrix_vertices
    S0_LHS[boundary_vertex_indices, boundary_vertex_indices] = 1
    S0_LHS[N0 + boundary_edge_indices] = bc_zeromatrix_edges
    S0_LHS[N0 + boundary_edge_indices, N0 + boundary_edge_indices] = 1
    
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
        b_p = np.zeros(N0); b_E = np.zeros(N1); b_H = np.zeros(N2); H_h = np.zeros(N2)
        for T_index, T in enumerate(sc[2].simplices):
            vertices_T = sc.vertices[T]
            vol_phy_tri = sc[2].primal_volume[T_index]
            dl_T = dl(vertices_T)
            integral_scaling = vol_phy_tri/vol_std_tri

            # Obtain the indices of edges of this tetrahedron
            edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(T, 2)])
            
            # Computing values of f_p, f_E and f_H at current time step
            fp_integral = np.zeros(len(vertices_T))
            fE_integral = np.zeros(len(edges_T))
            fH_integral = np.zeros(1)
            H_integral = np.zeros(1)
            for kq, qp_b in enumerate(qnodes_bary):
                x_q = np.dot(qp_b, vertices_T[:, 0])
                y_q = np.dot(qp_b, vertices_T[:, 1])
                for i in range(3):
                    fp_integral[i] += np.dot(fp_analytical([x_q, y_q], (time_step-1) * dt), 
                                             V_Ws[i](qp_b)) * qweights[kq]
                    fE_integral[i] += np.dot(fE_analytical([x_q, y_q], (time_step-1) * dt), 
                                             E_Ws[i](qp_b, dl_T)) * qweights[kq]
                for i in range(1):
                    fH_integral[i] += np.dot(fH_analytical([x_q, y_q], (time_step-1/2) * dt), 
                                             T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    H_integral[i] += np.dot(H_analytical([x_q, y_q], (time_step-1/2) * dt), 
                                            T_DGs[i](qp_b, vol_phy_tri)) * qweights[kq]
                    
            fp_integral *= integral_scaling
            b_p[T] += fp_integral
            fE_integral *= integral_scaling
            b_E[edges_T] += fE_integral
            fH_integral *= integral_scaling
            b_H[T_index] += fH_integral
            H_integral *= integral_scaling
            H_h[T_index] += H_integral

        # Setup right hand side for intermediate variables p, E and H
        b_RHS = np.concatenate((b_p, b_E, b_H)) + sprs.bmat([[1/dt*M00_g, epsilon/2*S01_g, None], 
                                                             [-1/2*S01_g.T, epsilon/dt*M11_g, 1/2*S12_g], 
                                                             [None, -1/2*S12_g.T, mu/dt*M22_g]], 
                                                             format='csr') * np.concatenate((p[time_step - 1], 
                                                                                             E[time_step - 1], 
                                                                                             H[time_step - 1]))

        # Incorporate boundary conditions in p
        p_bc = np.array([p_analytical(v, (time_step-1) * dt) for v in sc.vertices[boundary_vertex_indices]])

        # Compute normal trace of analytical solution E on each of the boundaries
        E_left_boundary = normal_boundary_trace(E_analytical, boundary_left_edges, np.array([-1, 0]), (time_step-1)*dt)
        E_right_boundary = normal_boundary_trace(E_analytical, boundary_right_edges, np.array([1, 0]), (time_step-1)*dt)
        E_top_boundary = normal_boundary_trace(E_analytical, boundary_top_edges, np.array([0, 1]), (time_step-1)*dt)
        E_bottom_boundary = normal_boundary_trace(E_analytical, boundary_bottom_edges, np.array([0, -1]), (time_step-1)*dt)

        # Impose boundary conditions on the right hand side at this time step
        b_RHS[boundary_vertex_indices] = p_bc
        b_RHS[N0 + boundary_left_edges] = E_left_boundary
        b_RHS[N0 + boundary_right_edges] = E_right_boundary
        b_RHS[N0 + boundary_top_edges] = E_top_boundary
        b_RHS[N0 + boundary_bottom_edges] = E_bottom_boundary
        
        # Obtain the linear system solution for p and E
        x = pEH_solver.solve(S_LHS, b_RHS)
        p[time_step] = x[:N0]
        E[time_step] = x[N0:N0+N1]
        H[time_step] = x[N0+N1:]

        # Obtain the L2 projection of the analytical density field for u to the space spanned by Discontinuous
        # Galerkin basis (order = 1) 
        H_true[time_step] = solverM22.solve(M22_g, H_h)


# Visualization of Solutions and Error Computation
if plot_solutions == True:
    print("\n\tinterpolating and plotting solutions, and computing error over time steps..."); sys.stdout.flush()
    if use_leap_frog == True:
        for pts_index, plot_time_step in enumerate(plot_time_steps):
            plot_time = plot_times[pts_index]
            plot_time_H = plot_times_H[pts_index]
            print("\t\tt = %1.3f"%plot_time)

            # Visualize the interpolated vector field at this point in barycentric coordinates on each tetrahedron
            l_bases = [[1/3, 1/3, 1/3]]

            # Set up some data structures to store the interpolated vector field
            bases = []; p_interp = []; E_arrows = []; H_interp = []
            p_l2error = 0; E_l2error = 0; H_l2error = 0
            
            p_plot_time = p[plot_time_steps[pts_index]]
            E_plot_time = E[plot_time_steps[pts_index]]
            H_plot_time = H[plot_time_steps[pts_index]]
            # H_T = np.zeros(N2); H_true_T = np.zeros(N2)
            H_true_T = H_true[plot_time_step]
            H_true_comp = np.zeros(N2)

            # Loop over all triangles and compute the L2 error and interpolated functions as needed
            for T_index, T in enumerate(tqdm.tqdm(sc[2].simplices)):
                vertices_T = sc.vertices[T]
                vol_phy_tri = sc[2].primal_volume[T_index]
                dl_T = dl(vertices_T)
                integral_scaling = vol_phy_tri/vol_std_tri

                # Obtain the indices of edges of this tetrahedron
                edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(T, 2)])

                # Obtain the restriction of p to this triangle
                p_T = p_plot_time[T]

                # Obtain the restriction of discrete E to this triangle
                E_T = E_plot_time[edges_T]
                
                # Obtain the interpolated E on this triangle
                bases += [np.dot(vertices_T.T, l_base) for l_base in l_bases]
                E_arrows += [E_interpolation(E_T, l_base, dl_T) for l_base in l_bases]

                # Obtain the restriction of discrete H to this triangle and interpolate it
                H_T = H_plot_time[T_index]

                # Obtain the interpolated H on this triangle
                H_interp += [H_interpolation(H_T, l_base, vol_phy_tri) for l_base in l_bases]

                p_l2error_T = 0; E_l2error_T = 0; H_l2error_T = 0

                for kq, qp_b in enumerate(qnodes_bary):
                    xq_phy = np.dot(qp_b, vertices_T[:, 0])
                    yq_phy = np.dot(qp_b, vertices_T[:, 1])

                    p_interpolated_T = np.array(p_interpolation(p_T, qp_b, dl_T))
                    p_analytical_T = np.array(p_analytical([xq_phy, yq_phy], plot_time))
                    
                    E_interpolated_T = np.array(E_interpolation(E_T, qp_b, dl_T))
                    E_analytical_T = np.array(E_analytical([xq_phy, yq_phy], plot_time))

                    H_interpolated_T = np.array(H_interpolation(H_T, qp_b, vol_phy_tri))
                    H_analytical_T = np.array(H_analytical([xq_phy, yq_phy], plot_time_H))
                    
                    # Computing errors of E and H at plot times
                    p_l2error_T += np.dot(p_analytical_T - p_interpolated_T, 
                                        p_analytical_T - p_interpolated_T) * qweights[kq]
                    E_l2error_T += np.dot(E_analytical_T - E_interpolated_T, 
                                        E_analytical_T - E_interpolated_T) * qweights[kq]
                    H_l2error_T += np.dot(H_analytical_T - H_interpolated_T, 
                                        H_analytical_T - H_interpolated_T) * qweights[kq]
                                    
                p_l2error_T *= integral_scaling
                E_l2error_T *= integral_scaling
                H_l2error_T *= integral_scaling

                p_l2error += p_l2error_T
                E_l2error += E_l2error_T
                H_l2error += H_l2error_T

            p_error_L2.append(np.sqrt(p_l2error))
            E_error_L2.append(np.sqrt(E_l2error))
            H_error_L2.append(np.sqrt(H_l2error))

            p_interp = p_plot_time
            bases = np.array(bases); E_arrows = np.array(E_arrows)
            H_interp = np.array(H_interp)

            # Obtain the analytical scalar field for p at the interpolation point
            p_true = np.array([p_analytical([x_phy, y_phy], plot_time) for x_phy, y_phy in sc.vertices])
            
            # Obtain the analytical vector field for E at the interpolation point
            E_true_arrows = np.array([E_analytical([x_phy, y_phy], plot_time) for x_phy, y_phy in bases])

            # Obtain the analytical scalar field for H at the interpolation point
            H_true_comp = np.array([H_interpolation(H_true_T[T_index], l_base, sc[2].primal_volume[T_index]) 
                                    for l_base in l_bases for T_index in range(N2)])

            # Plot the interpolated solution p on the mesh
            # We shall plot the restriction of p to the vertices of the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], p_plot_time, 
                                triangles=Triangles, shading='gouraud', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cbar = fig.colorbar(tpc, cax=cax, ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_useMathText(True)

            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nComputed $p$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "p_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time  + "_" + str(mesh_no) + ".pdf"), **savefig_options)

            # Plot the analytical scalar function p on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], p_true, 
                                triangles=Triangles, shading='gouraud', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cbar = fig.colorbar(tpc, cax=cax, ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_useMathText(True)

            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nAnalytical $p$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "p_analytical_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)
                
            # Plot the interpolated solution E on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            ax.quiver(bases[::4, 0], bases[::4, 1], E_arrows[::4, 0], E_arrows[::4, 1],
                color=(0.3, 0.3, 0.3), scale = 20, width=0.0035)
            ax.set_aspect('equal'); plt.axis('off')
            ax.set_title("\nComputed $E$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "E_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"),
                        **savefig_options)

            # Plot the analytical vector field E on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            ax.quiver(bases[::4, 0], bases[::4, 1], E_true_arrows[::4, 0], E_true_arrows[::4, 1],
                color=(0.3, 0.3, 0.3), scale = 20, width=0.0035)
            ax.set_aspect('equal'); plt.axis('off')
            ax.set_title("\nAnalytical $E$ at $t=$%1.4f"%plot_time, fontsize=20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "E_analytical_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)
                
            # Plot the interpolated density field H on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], facecolors=H_interp, 
                                triangles=Triangles, shading='flat', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            if pts_index == len(plot_time_steps) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2%")
                cbar = fig.colorbar(tpc, cax=cax, ax=ax)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_useMathText(True)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nComputed $H$ at $t=$%1.4f"%plot_time_H, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "H_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time_H + "_" + str(mesh_no) + ".pdf"), **savefig_options)

            # Plot the projection of the analytical density field H on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], facecolors=H_true_comp, 
                                triangles=Triangles, shading='flat', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            if pts_index == len(plot_time_steps) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2%")
                cbar = fig.colorbar(tpc, cax=cax, ax=ax)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_useMathText(True)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nAnalytical $H$ at $t=$%1.4f"%plot_time_H, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "H_analytical_" +
                                        "t%1.4f"%plot_time_H + "_" + str(mesh_no) + ".pdf"), **savefig_options)

    else:
        for pts_index, plot_time_step in enumerate(plot_time_steps):
            plot_time = plot_times[pts_index]
            print("\t\tt = %1.3f"%plot_time)

            # Visualize the interpolated vector field at this point in barycentric coordinates on each tetrahedron
            l_bases = [[1/3, 1/3, 1/3]]

            # Set up some data structures to store the interpolated vector field
            bases = []; p_interp = []; E_arrows = []; H_interp = []
            p_l2error = 0; E_l2error = 0; H_l2error = 0
            
            p_plot_time = p[plot_time_steps[pts_index]]
            E_plot_time = E[plot_time_steps[pts_index]]
            H_plot_time = H[plot_time_steps[pts_index]]
            H_true_T = H_true[plot_time_step]
            H_true_comp = np.zeros(N2)

            # Loop over all triangles and compute the L2 error and interpolated functions as needed
            for T_index, T in enumerate(tqdm.tqdm(sc[2].simplices)):
                vertices_T = sc.vertices[T]
                vol_phy_tri = sc[2].primal_volume[T_index]
                dl_T = dl(vertices_T)
                integral_scaling = vol_phy_tri/vol_std_tri

                # Obtain the indices of edges of this tetrahedron
                edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                                for v0, v1 in itrs.combinations(T, 2)])

                # Obtain the restriction of p to this triangle
                p_T = p_plot_time[T]
                
                # Obtain the restriction of discrete E to this triangle
                E_T = E_plot_time[edges_T]
                
                # Obtain the interpolated E on this triangle
                bases += [np.dot(vertices_T.T, l_base) for l_base in l_bases]
                E_arrows += [E_interpolation(E_T, l_base, dl_T) for l_base in l_bases]

                # Obtain the restriction of discrete H to this triangle and interpolate it
                H_T = H_plot_time[T_index]

                # Obtain the interpolated H on this triangle
                H_interp += [H_interpolation(H_T, l_base, vol_phy_tri) for l_base in l_bases]

                p_l2error_T = 0; E_l2error_T = 0; H_l2error_T = 0

                for kq, qp_b in enumerate(qnodes_bary):
                    xq_phy = np.dot(qp_b, vertices_T[:, 0])
                    yq_phy = np.dot(qp_b, vertices_T[:, 1])

                    p_interpolated_T = np.array(p_interpolation(p_T, qp_b, dl_T))
                    p_analytical_T = np.array(p_analytical([xq_phy, yq_phy], plot_time))
                    
                    E_interpolated_T = np.array(E_interpolation(E_T, qp_b, dl_T))
                    E_analytical_T = np.array(E_analytical([xq_phy, yq_phy], plot_time))

                    H_interpolated_T = np.array(H_interpolation(H_T, qp_b, vol_phy_tri))
                    H_analytical_T = np.array(H_analytical([xq_phy, yq_phy], plot_time))
                    
                    # Computing errors of E and H at plot times
                    p_l2error_T += np.dot(p_analytical_T - p_interpolated_T, 
                                        p_analytical_T - p_interpolated_T) * qweights[kq]
                    E_l2error_T += np.dot(E_analytical_T - E_interpolated_T, 
                                        E_analytical_T - E_interpolated_T) * qweights[kq]
                    H_l2error_T += np.dot(H_analytical_T - H_interpolated_T, 
                                        H_analytical_T - H_interpolated_T) * qweights[kq]
                                    
                p_l2error_T *= integral_scaling
                E_l2error_T *= integral_scaling
                H_l2error_T *= integral_scaling

                p_l2error += p_l2error_T
                E_l2error += E_l2error_T
                H_l2error += H_l2error_T

            p_error_L2.append(np.sqrt(p_l2error))
            E_error_L2.append(np.sqrt(E_l2error))
            H_error_L2.append(np.sqrt(H_l2error))

            p_interp = p_plot_time
            bases = np.array(bases); E_arrows = np.array(E_arrows)
            H_interp = np.array(H_interp)

            # Obtain the analytical scalar field for p at the interpolation point
            p_true = np.array([p_analytical([x_phy, y_phy], plot_time) for x_phy, y_phy in sc.vertices])
            
            # Obtain the analytical vector field for E at the interpolation point
            E_true_arrows = np.array([E_analytical([x_phy, y_phy], plot_time) for x_phy, y_phy in bases])

            # Obtain the analytical scalar field for H at the interpolation point
            H_true_comp = np.array([H_interpolation(H_true_T[T_index], l_base, sc[2].primal_volume[T_index]) 
                                    for l_base in l_bases for T_index in range(N2)])

            # Plot the interpolated solution p on the mesh
            # We shall plot the restriction of p to the vertices of the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], p_plot_time, 
                                triangles=Triangles, shading='gouraud', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cbar = fig.colorbar(tpc, cax=cax, ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_useMathText(True)

            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nComputed $p$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "p_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time  + "_" + str(mesh_no) + ".pdf"), **savefig_options)

            # Plot the analytical scalar function p on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], p_true, 
                                triangles=Triangles, shading='gouraud', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cbar = fig.colorbar(tpc, cax=cax, ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_useMathText(True)

            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nAnalytical $p$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "p_analytical_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)
                
            # Plot the interpolated solution E on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            ax.quiver(bases[::4, 0], bases[::4, 1], E_arrows[::4, 0], E_arrows[::4, 1],
                color=(0.3, 0.3, 0.3), scale = 20, width=0.0035)
            ax.set_aspect('equal'); plt.axis('off')
            ax.set_title("\nComputed $E$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "E_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"),
                        **savefig_options)

            # Plot the analytical vector field E on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            ax.quiver(bases[::4, 0], bases[::4, 1], E_true_arrows[::4, 0], E_true_arrows[::4, 1],
                color=(0.3, 0.3, 0.3), scale = 20, width=0.0035)
            ax.set_aspect('equal'); plt.axis('off')
            ax.set_title("\nAnalytical $E$ at $t=$%1.4f"%plot_time, fontsize=20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "E_analytical_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)
                
            # Plot the interpolated density field H on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], facecolors=H_interp, 
                                triangles=Triangles, shading='flat', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            if pts_index == len(plot_time_steps) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2%")
                cbar = fig.colorbar(tpc, cax=cax, ax=ax)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_useMathText(True)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nComputed $H$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "H_computed_" + fe_order + "_" + 
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)

            # Plot the projection of the analytical density field H on the mesh
            plt.figure(); fig = plt.gcf(); ax = plt.gca()
            plt.triplot(Vertices[:, 0], Vertices[:, 1], Triangles, color=(0.6, 0.6, 0.6), linewidth=0.5)
            tpc = plt.tripcolor(Vertices[:, 0], Vertices[:, 1], facecolors=H_true_comp, 
                                triangles=Triangles, shading='flat', edgecolors='k')

            # Create an Axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # Plot the colorbar only for the last time step
            if pts_index == len(plot_time_steps) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2%")
                cbar = fig.colorbar(tpc, cax=cax, ax=ax)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_useMathText(True)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title("\nAnalytical $H$ at $t=$%1.4f"%plot_time, fontsize = 20)
            if save_figs:
                plt.savefig(pth.join(figs_dir, figs_prestring + "H_analytical_" +
                                        "t%1.4f"%plot_time + "_" + str(mesh_no) + ".pdf"), **savefig_options)


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

        # Loop over all triangles and compute the L2 norms of interpolated functions as needed
        for T_index, T in enumerate(sc[2].simplices):
            vertices_T = sc.vertices[T]
            vol_phy_tri = sc[2].primal_volume[T_index]
            dl_T = dl(vertices_T)
            integral_scaling = vol_phy_tri/vol_std_tri

            # Obtain the indices of edges of this tetrahedron
            edges_T = np.array([sc[1].simplex_to_index[pydec.simplex((v0, v1))] 
                            for v0, v1 in itrs.combinations(T, 2)])
            
            # Obtain the restriction of p to this triangle
            p_T = p_solution_time[T]

            # Obtain the restriction of discrete E to this triangle
            E_T = E_solution_time[edges_T]

            # Obtain the restriction of discrete H to this triangle
            H_T = H_solution_time[T_index]

            p_l2norm_T = 0; E_l2norm_T = 0; H_l2norm_T = 0
            p_comp_l2norm_T = 0; E_comp_l2norm_T = 0; H_comp_l2norm_T = 0
            for kq, qp_b in enumerate(qnodes_bary):
                xq_phy = np.dot(qp_b, vertices_T[:, 0])
                yq_phy = np.dot(qp_b, vertices_T[:, 1])

                p_interpolated_T = np.array(p_interpolation(p_T, qp_b, dl_T))
                p_analytical_T = np.array(p_analytical([xq_phy, yq_phy], solution_time))
                
                E_interpolated_T = np.array(E_interpolation(E_T, qp_b, dl_T))
                E_analytical_T = np.array(E_analytical([xq_phy, yq_phy], solution_time))

                H_interpolated_T = np.array(H_interpolation(H_T, qp_b, vol_phy_tri))
                H_analytical_T = np.array(H_analytical([xq_phy, yq_phy], solution_time))

                # Computing norms of analytical E and H at solution times
                p_l2norm_T += np.dot(p_analytical_T, p_analytical_T) * qweights[kq]
                E_l2norm_T += np.dot(E_analytical_T, E_analytical_T) * qweights[kq]
                H_l2norm_T += np.dot(H_analytical_T, H_analytical_T) * qweights[kq]

                # Computing norms of interpolated E and H at solution times
                p_comp_l2norm_T += np.dot(p_interpolated_T, p_interpolated_T) * qweights[kq]
                E_comp_l2norm_T += np.dot(E_interpolated_T, E_interpolated_T) * qweights[kq]
                H_comp_l2norm_T += np.dot(H_interpolated_T, H_interpolated_T) * qweights[kq]

            p_l2norm_T *= integral_scaling
            E_l2norm_T *= integral_scaling
            H_l2norm_T *= integral_scaling
            p_comp_l2norm_T *= integral_scaling
            E_comp_l2norm_T *= integral_scaling
            H_comp_l2norm_T *= integral_scaling

            p_l2norm += p_l2norm_T
            E_l2norm += E_l2norm_T
            H_l2norm += H_l2norm_T
            p_comp_l2norm += p_comp_l2norm_T
            E_comp_l2norm += E_comp_l2norm_T
            H_comp_l2norm += H_comp_l2norm_T
        
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

print("\nL2 norm of analytical density field H: ", H_norm_L2)
print("L2 norm of computed density field H: ", H_comp_norm_L2)
print("L2 error in density field H: ", H_error_L2)

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


