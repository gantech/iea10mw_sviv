import netCDF4 as nc
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import yaml

def calc_interp_matrix():
    """Find matrix that interpolates from
    finite element nodes to quadrature points"""

    data = yaml.load(open("../../template/forced_motion/openfast_run/00_IEA-10.0-198-RWT.BD1.sum.yaml"), Loader=yaml.Loader)

    # Node initial position and rotation
    node_x0 = np.array(data['Init_Nodes_E1'])[:,:3]

    # Node locations are based on Gauss-Legendre-Lobatto quadrature
    # node_xi = lglnodes(node_x0.shape[0]-1, 1e-12)[0]
    node_xi = 2*(node_x0[:,2] - node_x0[0,2]) / np.ptp(node_x0[:,2])-1

    # QP initial position and rotation
    qp_x0 = np.array(data['Init_QP_E1'])[:,:3]

    # QP locations should be based on eta values from stations in BD input file
    qp_xi = 2*(qp_x0[:,2] - qp_x0[0,2]) / np.ptp(qp_x0[:,2])-1

    # Calculate shape function matrix
    N = np.zeros([qp_xi.size, node_xi.size])
    for k, xi in enumerate(qp_xi):
        for i, xi_i in enumerate(node_xi):
            num = 1
            den = 1
            for j, xi_j in enumerate(node_xi):
                if i != j:
                    num *= xi - xi_j
                    den *= xi_i - xi_j
            N[k, i] = num / den

    return N


def get_eigenvalues_and_vectors():
    #Linearization matrices
    A = np.loadtxt('../../template/forced_motion/openfast_run/00_IEA-10.0-198-RWT.1.BD1.lin',skiprows=5043,max_rows=120)
    B = np.loadtxt('../../template/forced_motion/openfast_run/00_IEA-10.0-198-RWT.1.BD1.lin',skiprows=5164,max_rows=120)
    # Calculation of M, C, K matrices from the linearization matrices
    invM = B[60:, 18:84] # This gives us 66 nodes. Need to reorder them.
    invM = invM[:, [3,4,5,36,37,38, 6,7,8,39,40,41, 9,10,11,42,43,44, 12,13,14,45,46,47, 15,16,17,48,49,50, 18,19,20,51,52,53, 21,22,23,54,55,56, 24,25,26,57,58,59, 27,28,29,60,61,62, 30,31,32,63,64,65]] # Reorder to match the expected order
    M = np.linalg.inv(invM)
    M = 0.5 * (M + np.transpose(M))  # Ensure M is symmetric
    minus_invm_k = A[60:, :60]
    minus_invm_c = A[60:, 60:]
    K = -np.matmul(M, minus_invm_k)
    C = -np.matmul(M, minus_invm_c)
    eigvals_km, eigvecs_km = eigh(K,M)
    eigvals_km = np.sqrt(np.absolute(eigvals_km[:]))/2.0/np.pi
    return eigvals_km, eigvecs_km, M

def compose_wm(c1, c2, transC1=1.0):
    """Compose two vectors of Wiener-Milenkovic parameters. Optionally transpose the first rotation c1."""

    c1Plusc2 = np.zeros(np.shape(c1))

    c10 = 2.0 - 0.125*np.sum(c1 * c1, axis=0)
    c20 = 2.0 - 0.125*np.sum(c2 * c2, axis=0)
    delta1 = (4.0-c10)*(4.0-c20)
    delta2 = c10*c20 - transC1 * np.sum(c1 * c2, axis=0)

    # Use np.where to handle the conditional assignment vectorized
    premult_fac = np.where(delta2 < 0,
                          -4.0 / (delta1 - delta2),
                          4.0 / (delta1 + delta2))

    c1Plusc2 = premult_fac * (c10 * c2 + c20 * transC1 * c1 + transC1 * np.cross(c1, c2, axis=0))

    return c1Plusc2

def applyWM(c,v,transpose=1):
    """Apply a rotation defined by Wiener-Milenkovic parameters in 'c' to a vector 'v'. Optionally transpose the rotation """

    magC = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
    c0 = 2.0-0.125*magC
    nu = 2.0/(4.0-c0)
    cosPhiO2 = 0.5*c0*nu
    cCrossV = np.cross(c,v)
    return v + transpose * nu * cosPhiO2 * cCrossV + 0.5 * nu * nu * np.cross(c,cCrossV)

interp_matrix = np.linalg.pinv(calc_interp_matrix())
eigvals, eigvecs, M = get_eigenvalues_and_vectors()
a = nc.Dataset('turb_00_output.nc', 'r')

bld_disp = a['bld_disp'][:,0,:,:]
#bld_disp = bld_disp - bld_disp[0]

bld_ref_pos = a['bld_ref_pos'][0,:,:]

bld_root_orient = a['bld_root_orient'][0,0,:]

# Apply peak finding algorithm
bld_disp_tip = bld_disp[:, 1, -1]  # Assuming the tip is the last node in the blade
peaks, _ = find_peaks(bld_disp_tip-bld_disp_tip[0], height=0, distance=20)

bld_orient = a['bld_orient'][:, 0, :, :]
ntsteps = np.shape(bld_orient)[0]
n_fenodes = np.shape(bld_orient)[2]
for it in range(peaks[-5],peaks[-1]):
    for inode in range(n_fenodes):
        bld_disp[it,:,inode] = applyWM(bld_root_orient, bld_disp[it,:,inode])

bld_disp[peaks[-5]:peaks[-1],0,:] = bld_disp[peaks[-5]:peaks[-1],0,:] - np.mean(bld_disp[peaks[-5]:peaks[-1], 0, :], axis=0)
bld_disp[peaks[-5]:peaks[-1],1,:] = bld_disp[peaks[-5]:peaks[-1],1,:] - np.mean(bld_disp[peaks[-5]:peaks[-1], 1, :], axis=0)
bld_disp[peaks[-5]:peaks[-1],2,:] = bld_disp[peaks[-5]:peaks[-1],2,:] - np.mean(bld_disp[peaks[-5]:peaks[-1], 2, :], axis=0)


bld_tot_disp = np.zeros( (ntsteps, 60) )
bld_trans_disp = np.zeros( (ntsteps, 30) )

for it in range(ntsteps):
    wm_def = compose_wm(bld_orient[0, :, :], bld_orient[it, :, :], transC1=-1.0)
    # wm_root = wm_def[:,0]
    # wm_def = compose_wm(np.tile(wm_root, (np.shape(wm_def)[1],1)).T, wm_def, transC1=-1.0).T
    wm_def_fenodes = np.dot(interp_matrix, wm_def.T)[1:,:].T
    # wm_def_fenodes = compose_wm(np.tile(wm_root, (np.shape(wm_def_fenodes)[1],1)).T, wm_def_fenodes)[:,1:]  # Exclude the root node
    phi = 4.0 * np.arctan(0.25 * np.linalg.norm(wm_def_fenodes, axis=0))
    rotdisp = phi * wm_def_fenodes / np.linalg.norm(wm_def_fenodes, axis=0)
    trans_disp = np.dot(interp_matrix, bld_disp[it, :, :].T)[1:,:].T  # Exclude the root node
    bld_tot_disp[it, :] = np.r_[trans_disp, rotdisp].T.flatten()
    bld_trans_disp[it, :] = trans_disp.flatten()


bld_tot_disp = bld_tot_disp - np.mean(bld_tot_disp[peaks[-5]:peaks[-1]], axis=0)  # Center the data
bld_trans_disp = bld_trans_disp - np.mean(bld_trans_disp[peaks[-5]:peaks[-1]], axis=0)  # Center the data

eigvecs_trans = eigvecs[(np.arange(60) % 6) < 3,:]

modal_weight = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(eigvecs), eigvecs)),
            np.transpose(eigvecs)
            ),
        bld_tot_disp[peaks[-5]:peaks[-1]].T)

modal_weight_trans = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(eigvecs_trans), eigvecs_trans)),
            np.transpose(eigvecs_trans)
            ),
        bld_trans_disp[peaks[-5]:peaks[-1]].T)

# Save the arrays to NPZ format (compressed NumPy arrays)
np.savez('mode_data.npz',
            bld_disp=bld_disp,
            bld_tot_disp=bld_tot_disp,
            modal_weight=modal_weight,
            modal_weight_trans=modal_weight_trans,
            peaks=peaks,
            eigvecs=eigvecs,
            eigvals=eigvals)

print(f"Data saved successfully. Peaks found: {len(peaks)}")



# fig = plt.figure(figsize=(10, 6))
# plt.plot(a['time'][:], bld_rotdisp[:, 2, -1], label='Blade Rotational Displacement')
# plt.legend()
# plt.title("Blade Orientation and Rotational Displacement")
# plt.xlabel("Time")
# plt.ylabel("Displacement")
# plt.grid(True)
# plt.show()






# # Get the mode shape at the time of the last peak
# peak_time_idx = peaks[-1]
# mode_shape = a['bld_disp'][peak_time_idx, 0, :, :] - a['bld_disp'][0, 0, :, :]  # All DOFs for blade 0 at the peak time
# mode_shape[0,:] = a['bld_disp'][peak_time_idx, 0, 0, :] - np.mean(a['bld_disp'][peaks[-6]:peaks[-1], 0, 0, :], axis=0)
# mode_shape[2,:] = a['bld_disp'][peak_time_idx, 0, 2, :] - np.mean(a['bld_disp'][peaks[-6]:peaks[-1], 0, 2, :], axis=0)

# plt.plot(np.mean(a['bld_disp'][peaks[-6]:peaks[-1], 0, 0, :], axis=0) - a['bld_disp'][0, 0, 0, :], label='Mean Blade Displacement')
# plt.show()

# # Plot all 6 dimensions of the mode shape
# plt.figure(figsize=(12, 8))
# dof_names = ['tx', 'ty', 'tz']  # Assuming these are the 6 DOFs


# m = yaml.load(open('../../template/forced_motion/openfast_run/edge_mode.yaml'), Loader=yaml.FullLoader)
# mode_shape_km = np.matmul(np.array(m['mode']['interp_matrix']), -np.array(m['mode']['shape'])*np.sin(np.array(m['mode']['phase'])))
# phase_km = np.matmul(np.array(m['mode']['interp_matrix']), np.array(m['mode']['phase']))

# print(mode_shape[1,-1])
# print(mode_shape_km[-1,1])

# for i in range(mode_shape.shape[0]):
#     plt.subplot(1, 3, i+1)
#     plt.plot(np.arange(mode_shape.shape[1]), mode_shape[i, :] / mode_shape[1,-1], 'o-', label='FSI')
#     plt.plot(mode_shape_km[:,i], label='Mode Shape')
#     plt.title(f'{dof_names[i]} Mode Shape')
#     plt.xlabel('Node Index')
#     plt.ylabel('Displacement')
#     plt.grid(True)
# plt.legend(loc=0)
# plt.tight_layout()
# plt.savefig('mode_shape_plots.png')
# plt.show()
