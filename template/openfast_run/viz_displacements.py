import netCDF4 as nc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from WienerMilenkovic import *
#from palettable.colorbrewer.qualitative import Paired_6
import yaml
from scipy.linalg import eig
import numpy as np
from scipy.spatial.transform import Rotation

# Rotation class extended to support Wiener-Milenkovic Parameters (CRV)
class R(Rotation):
    def as_crv(self) -> np.ndarray:
        q = R.from_matrix(self.as_matrix()).as_quat()  # Normalize before quat
        e, e0 = q[0:3], q[3]
        return 4*e/(1+e0)

    def from_crv(c: np.ndarray):
        c0 = 2 - np.dot(c, c)/8
        e0 = c0/(4-c0)
        e = c/(4-c0)
        return R.from_quat([e[0], e[1], e[2], e0])

    def as_hmat(self) -> np.ndarray:
        cc = self.as_crv()
        cf1 = cc[0]/4
        cf2 = cc[1]/4
        cf3 = cc[2]/4
        cq = cf1 * cf1 + cf2 * cf2 + cf3 * cf3
        ocq = 1 + cq
        aa = 2 * ocq * ocq
        cb0 = 2 * (1 - cq) / aa
        cb1 = cc[0]/aa
        cb2 = cc[1]/aa
        cb3 = cc[2]/aa
        return np.array([
            [cb1 * cf1 + cb0, cb1 * cf2 - cb3, cb1 * cf3 + cb2],
            [cb2 * cf1 + cb3, cb2 * cf2 + cb0, cb2 * cf3 - cb1],
            [cb3 * cf1 - cb2, cb3 * cf2 + cb1, cb3 * cf3 + cb0]
        ])

class OpenfastFSI:

    def __init__(self, filename):

        t = nc.Dataset(filename)

        #Blade nodes
        self.bld_rloc = t['bld_rloc']
        self.bld_ref_pos = t['bld_ref_pos']
        self.bld_ref_orient = t['bld_ref_orient']

        self.bld_disp = t['bld_disp']
        self.bld_orient = t['bld_orient']

        #Blade root nodes
        self.bld_root_ref_pos = t['bld_root_ref_pos']
        self.bld_root_ref_orient = t['bld_root_ref_orient']

        self.bld_root_disp = t['bld_root_disp']
        self.bld_root_orient = t['bld_root_orient']

        #Hub nodes
        self.hub_ref_pos = t['hub_ref_pos']
        self.hub_ref_orient = t['hub_ref_orient']

        self.hub_orient = t['hub_orient']
        self.hub_disp = t['hub_disp']

        ntsteps = t.dimensions['n_tsteps'].size

        self.nblds = t.dimensions['n_blds'].size
        self.nbld_nodes = t.dimensions['n_bld_nds'].size
        self.ntwr_nds = t.dimensions['n_twr_nds'].size

    def calc_local_chord_dir(self, it):
        """Calculate local chord direction at a given time step
        Args:
            it (int): Time step
        Return:
            None

        """
        self.bld_chord_dir = np.zeros_like(self.bld_ref_pos)
        for b in range(self.nblds):
          for i in range(self.nbld_nodes):
              self.bld_chord_dir[b,:,i] = applyWM(self.bld_orient[it,b,:,i], [0,1,0], -1)

    def get_local_chord_dir(self, ib, r):
        """
        Args:
            ib (int): Blade number (0,1,2)
            r (double): Radial location
        Return:
            chord_dir (np.array): Chord direction at given radial location
        """
        return np.array([ np.interp(r, self.bld_rloc[ib,:], self.bld_chord_dir[ib,i,:]) for i in range(3)])


    def calc_disp_blade_rotation(self, it):
        """Calculate displacements due to blade rotation
        Args:
            it (int): Time step
        Return:
            None

        """

        self.bld_disp_rotation = np.zeros_like(self.bld_ref_pos)
        self.bld_orient_rotation = np.zeros_like(self.bld_ref_pos)
        for b in range(self.nblds):
          for i in range(self.nbld_nodes):

            #Deal with translational displacements first
            bld_hub_ref_frame = applyWM(
                self.hub_ref_orient,
                self.bld_ref_pos[b, :, i] - self.hub_ref_pos[:] )


            bld_hub_final_frame = applyWM(
                self.hub_orient[it,:], bld_hub_ref_frame, -1)

            tmp = bld_hub_final_frame - (self.bld_ref_pos[b, :, i] - self.hub_ref_pos[:])
            self.bld_disp_rotation[b, :, i] = tmp + self.hub_disp[it, :]

            #Now rotational displacements
            #Get twist in blade root reference frame
            bld_root_ref_frame = applyWM(self.bld_root_ref_orient[b,:],
                                         compose(
                                             self.bld_root_ref_orient[b,:],
                                             self.bld_ref_orient[b,:,i], -1) )

            bld_root_orient_m_pitch = compose(
                applyWM(
                    self.hub_orient[it,:],
                    self.bld_root_ref_orient[b,:]),
                self.hub_orient[it,:])


            # pitch_axis = applyWM(self.bld_root_orient[it,b,:],
            #                      [0,0,1], -1)
            # wm_pitch = 4.0 * np.tan(0.25 * np.radians(45.0)) * pitch_axis
            # bld_root_orient_m_pitch = compose(wm_pitch, self.bld_root_orient[it,b,:], -1)

            #Get twist in final blade root frame
            bld_root_final_frame = applyWM(bld_root_orient_m_pitch,
                                           bld_root_ref_frame, -1)


            self.bld_orient_rotation[b,:,i] = compose(bld_root_orient_m_pitch, bld_root_final_frame)



            # bld_root_ref_frame = applyWM(self.hub_ref_orient[:],
            #                              compose(
            #                                  self.hub_ref_orient[:],
            #                                  self.bld_ref_orient[b,:,i], -1))

            # bld_root_final_frame = applyWM(self.hub_orient[it,:],
            #                                bld_root_ref_frame, -1)
            # self.bld_orient_rotation[b,:,i] = compose(self.hub_orient[it,:], bld_root_final_frame)

            print(self.bld_orient[it,b,:,i], ' ', self.bld_ref_orient[b,:,i])


    def get_blade_deflection(self, it):
        """Get deflections only due to blade displacement
        Args:
            it (int): Time step
        Return:
            None
        """

        self.calc_disp_blade_rotation(it)

        return np.c_[ self.bld_disp[it,:,:,:] - self.bld_disp_rotation, self.bld_orient[it,:,:,:] - self.bld_orient_rotation]


    def viz_displacements(self, it):
        """Get deflections only due to blade displacement
        Args:
            it (int): Time step
        Return:
            None
        """

        self.calc_disp_blade_rotation(it)

        bcolors = Paired_6.mpl_colors

        xp_ref = np.zeros_like(self.bld_ref_pos)
        yp_ref = np.zeros_like(self.bld_ref_pos)
        zp_ref = np.zeros_like(self.bld_ref_pos)

        xp_disp = np.zeros_like(self.bld_ref_pos)
        yp_disp = np.zeros_like(self.bld_ref_pos)
        zp_disp = np.zeros_like(self.bld_ref_pos)

        xp_disp_rot = np.zeros_like(self.bld_ref_pos)
        yp_disp_rot = np.zeros_like(self.bld_ref_pos)
        zp_disp_rot = np.zeros_like(self.bld_ref_pos)

        for b in range(self.nblds):
            for i in range(self.nbld_nodes):
                xp_ref[b,:,i] = applyWM(self.bld_ref_orient[b,:,i], [1,0,0], -1)
                yp_ref[b,:,i] = applyWM(self.bld_ref_orient[b,:,i], [0,1,0], -1)
                zp_ref[b,:,i] = applyWM(self.bld_ref_orient[b,:,i], [0,0,1], -1)

                xp_disp_rot[b,:,i] = applyWM(self.bld_orient_rotation[b,:,i], [1,0,0], -1)
                yp_disp_rot[b,:,i] = applyWM(self.bld_orient_rotation[b,:,i], [0,1,0], -1)
                zp_disp_rot[b,:,i] = applyWM(self.bld_orient_rotation[b,:,i], [0,0,1], -1)

                xp_disp[b,:,i] = applyWM(self.bld_orient[it,b,:,i], [1,0,0], -1)
                yp_disp[b,:,i] = applyWM(self.bld_orient[it,b,:,i], [0,1,0], -1)
                zp_disp[b,:,i] = applyWM(self.bld_orient[it,b,:,i], [0,0,1], -1)

        bld_ref_pos = np.r_ [self.bld_ref_pos[0, :, :].transpose(),
                             self.bld_ref_pos[1, :, :].transpose(),
                             self.bld_ref_pos[2, :, :].transpose()]

        bld_disp_rotation = np.r_ [self.bld_disp_rotation[0, :, :].transpose(),
                                   self.bld_disp_rotation[1, :, :].transpose(),
                                   self.bld_disp_rotation[2, :, :].transpose()]

        bld_disp = np.r_ [self.bld_disp[it, 0, :, :].transpose(),
                          self.bld_disp[it, 1, :, :].transpose(),
                          self.bld_disp[it, 2, :, :].transpose()]

        xp_d_rot = np.r_[ xp_disp_rot[0, :, :].transpose(),
                        xp_disp_rot[1, :, :].transpose(),
                        xp_disp_rot[2, :, :].transpose() ]

        yp_d_rot = np.r_[ yp_disp_rot[0, :, :].transpose(),
                        yp_disp_rot[1, :, :].transpose(),
                        yp_disp_rot[2, :, :].transpose() ]

        zp_d_rot = np.r_[ zp_disp_rot[0, :, :].transpose(),
                        zp_disp_rot[1, :, :].transpose(),
                        zp_disp_rot[2, :, :].transpose() ]

        xp_d = np.r_[ xp_disp[0, :, :].transpose(),
                      xp_disp[1, :, :].transpose(),
                      xp_disp[2, :, :].transpose() ]

        yp_d = np.r_[ yp_disp[0, :, :].transpose(),
                      yp_disp[1, :, :].transpose(),
                      yp_disp[2, :, :].transpose() ]

        zp_d = np.r_[ zp_disp[0, :, :].transpose(),
                      zp_disp[1, :, :].transpose(),
                      zp_disp[2, :, :].transpose() ]

        all_bld_disp = np.c_[ bld_ref_pos, bld_disp, bld_disp_rotation, xp_d_rot, yp_d_rot, zp_d_rot, xp_d, yp_d, zp_d]
        np.savetxt('bld_disp.csv',
                   all_bld_disp,
                   delimiter=',',
                   header='ref_x, ref_y, ref_z, disp_x, disp_y, disp_z, disp_rot_x, disp_rot_y, disp_rot_z, xp_d_rot_x, xp_d_rot_y, xp_d_rot_z, yp_d_rot_x, yp_d_rot_y, yp_d_rot_z, zp_d_rot_x, zp_d_rot_y, zp_d_rot_z, xpd_x, xpd_y, xpd_z, ypd_x, ypd_y, ypd_z, zpd_x, zpd_y, zpd_z')


    def calc_interp_matrix(self):
        """Find matrix that interpolates from
        finite element nodes to quadrature points"""

        data = yaml.load(open("IEA-10.0-198-RWT.BD1.sum.yaml"), Loader=yaml.Loader)

        # Node initial position and rotation
        node_x0 = np.array(data['Init_Nodes_E1'])[:,:3]
        node_r0 = np.array(data['Init_Nodes_E1'])[:,3:]

        # Node locations are based on Gauss-Legendre-Lobatto quadrature
        # node_xi = lglnodes(node_x0.shape[0]-1, 1e-12)[0]
        node_xi = 2*(node_x0[:,2] - node_x0[0,2]) / np.ptp(node_x0[:,2])-1

        # QP initial position and rotation
        qp_x0 = np.array(data['Init_QP_E1'])[:,:3]
        qp_r0 = np.array(data['Init_QP_E1'])[:,3:]

        # QP locations should be based on eta values from stations in BD input file
        # lines = open('IEA-15-240-RWT_BeamDyn_blade.dat').readlines()[10::15]
        # qp_eta = np.array(list(map(str.strip,lines)),dtype=float)
        # qp_eta = np.array(sorted(qp_eta.tolist() + ((qp_eta[1:] + qp_eta[:-1]) / 2).tolist()))
        # qp_xi = 2*(qp_eta - qp_eta[0]) / np.ptp(qp_eta)-1
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

    def interp_rotations_to_qpts(self, disp):
        """Interpolate rotations to quadrature points"""

        data = yaml.load(open("IEA-10.0-198-RWT.BD1.sum.yaml"), Loader=yaml.Loader)

        # Node initial position and rotation
        node_x0 = np.array(data['Init_Nodes_E1'])[:,:3]
        node_r0 = np.array(data['Init_Nodes_E1'])[:,3:]

        disp = np.r_[np.zeros((1,6)), disp]
        assert(np.shape(node_x0)[0] == np.shape(disp)[0])

        # Node locations are based on Gauss-Legendre-Lobatto quadrature
        # node_xi = lglnodes(node_x0.shape[0]-1, 1e-12)[0]
        node_xi = 2*(node_x0[:,2] - node_x0[0,2]) / np.ptp(node_x0[:,2])-1

        # QP initial position and rotation
        qp_x0 = np.array(data['Init_QP_E1'])[:,:3]
        qp_r0 = np.array(data['Init_QP_E1'])[:,3:]

        # QP locations should be based on eta values from stations in BD input file
        # lines = open('IEA-15-240-RWT_BeamDyn_blade.dat').readlines()[10::15]
        # qp_eta = np.array(list(map(str.strip,lines)),dtype=float)
        # qp_eta = np.array(sorted(qp_eta.tolist() + ((qp_eta[1:] + qp_eta[:-1]) / 2).tolist()))
        # qp_xi = 2*(qp_eta - qp_eta[0]) / np.ptp(qp_eta)-1
        qp_xi = 2*(qp_x0[:,2] - qp_x0[0,2]) / np.ptp(qp_x0[:,2])-1

        # Node translation displacements (to be specified)
        node_u = disp[:,:3]

        # Node rotation displacements (to be specified)
        node_r = disp[:,3:]

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

        # Interpolate qp initial position (should match input)
        qp_x0_test = np.matmul(N, node_x0)

        # Calculate qp initial rotations (should match input)
        rr = R.from_crv(node_r0[0])  # Root rotation
        n_r0_wo_root = np.array([(rr.inv() * R.from_crv(r)).as_crv() for r in node_r0])
        qp_r0_test = np.array([(rr * R.from_crv(r))
                          for r in np.matmul(N, n_r0_wo_root)])

        # Calculate quadrature point translation displacement
        qp_u = np.matmul(N, node_u)

        # Get quadrature point rotation displacement
        rr = R.from_crv(node_r[0])  # Root rotation
        n_r_wo_root = np.array([(rr.inv() * R.from_crv(r)).as_crv() for r in node_r])
        qp_ur = np.array([(rr * R.from_crv(r)).as_crv() for r in np.matmul(N, n_r_wo_root)])

        # Calculate global rotation for quadrature points
        qp_r = np.zeros_like(qp_ur)
        for i in range(qp_ur.shape[0]):
            qp_r[i] = (R.from_crv(qp_ur[i])*R.from_crv(qp_r0[i])).as_crv()

        # Calculate global position for quadrature points
        # This assumes the blade root hasn't moved or rotated from reference
        qp_x = qp_x0 + qp_u

        return np.c_[qp_u, qp_r]

    def calc_edge_mode(self):

        """Calculate edge mode """

        with open('IEA-10.0-198-RWT.BD1.sum.yaml') as f:
            data = list(yaml.load_all(f, Loader=yaml.loader.SafeLoader))[-1]

        #All computations below at FEM nodes - not quadrature points
        init_loc = np.array(data['Init_Nodes_E1'])

        A = np.loadtxt('00_IEA-10.0-198-RWT.1.lin',skiprows=3444,max_rows=120)
        eigvals, eigvecs = eig(A)

        #First edge mode is at location -2

        #Get first edge mode scaled to 1m displacement in edge direction
        edge_mode = np.abs(eigvecs[:60,-2].reshape((10,6))) * 1.0 / np.abs(eigvecs[60-5,-2])
        edge_mode = np.r_[np.zeros((1,6)), edge_mode]

        edge_mode_phase = np.angle(eigvecs[:60,-2].reshape((10,6)))
        edge_mode_phase = np.r_[np.zeros((1,6)), edge_mode_phase]

        print(np.array(sorted(eigvals[:].imag))/2.0/np.pi)
        print(eigvals[:].imag/2.0/np.pi)
        edge_freq = (eigvals[-2].imag)/2.0/np.pi

        return [edge_freq, edge_mode, edge_mode_phase]

    def write_mode_shape_to_file(self, filename):
        edge_freq, edge_mode, edge_mode_phase = self.calc_edge_mode()
        N = self.calc_interp_matrix()
        Nlist = []
        for i in N:
            Nlist.append(i.tolist())

        yaml_node = {
            'mode': {
                'freq': float(edge_freq),
                'shape': edge_mode.tolist(),
                'phase': edge_mode_phase.tolist(),
                'interp_matrix': Nlist
            }
        }
        yaml.dump(yaml_node, open(filename,'w'), default_flow_style=False)

    def convert_xdot_to_xdot_omega(self):

        """Replace the time derivatives of Wiener-Milenkovic parameters in xdot into angular velocity """

        edge_freq, edge_mode = self.calc_edge_mode()
        edge_mode_qp = self.interp_rotations_to_qpts(edge_mode)
        n_nodes = np.shape(edge_mode)[0]

        time = np.linspace(0,1.0/edge_freq,100)

        for it,t in enumerate(time):
            #Get edge mode shape at current time
            edge_mode_t = edge_mode * np.sin(2.0 * np.pi * edge_freq * t)
            edge_mode_t_dot = 2.0 * np.pi * edge_freq * edge_mode * np.cos(2.0 * np.pi * edge_freq * t)


    def calc_struct_power_edge(self):

        with open('IEA-10.0-198-RWT.BD1.sum.yaml') as f:
            data = list(yaml.load_all(f, Loader=yaml.loader.SafeLoader))[-1]

        #All computations below at FEM nodes - not quadrature points
        init_loc = np.array(data['Init_Nodes_E1'])

        #Only load the set of nodes after the first one - Boundary condition for clamping root node
        Mfull = np.array(data['M_BD'])[6:, 6:]
        Kfull = np.array(data['K_BD'])[6:, 6:]

        damp_coeffs = np.repeat(np.array([0.00299005, 0.00218775, 0.00084171, 0.00218775, 0.00299005, 0.00084171]), 10)
        Cfull = np.dot(np.identity(Kfull.shape), damp_coeffs)

        eigvals, eigvecs = eigh(Kfull, Mfull, subset_by_index=[0,59])

        #Get first edge mode scaled to 1m displacement in edge direction
        edge_mode = eigvecs[:,0].reshape((10,6)) * 1.0 / eigvecs[-6,0]

        edge_freq = np.sqrt(eigvals[0])/2.0/np.pi
        time = np.linspace(0,1.0/edge_freq,100)

        for it,t in enumerate(time):
            #Get edge mode shape at current time
            edge_mode_t = edge_mode * np.cos(2.0 * np.pi * edge_freq * t)



    def viz_edgewise_modeshape(self):

        with open('IEA-10.0-198-RWT.BD1.sum.yaml') as f:
            data = list(yaml.load_all(f, Loader=yaml.loader.SafeLoader))[-1]

        #All computations below at FEM nodes - not quadrature points
        init_loc = np.array(data['Init_Nodes_E1'])

        A = np.loadtxt('IEA-10.0-198-RWT.1.lin',skiprows=317,max_rows=120)
        eigvals, eigvecs = eig(A)

        #First edge mode is at location -2

        #Get first edge mode scaled to 1m displacement in edge direction
        edge_mode = np.abs(eigvecs[:60,-2].reshape((10,6))) * 1.0 / np.abs(eigvecs[60-5,-2])
        edge_mode_phase = np.angle(eigvecs[:60,-2].reshape((10,6)))

        edge_freq = np.sqrt(eigvals[-2].imag)/2.0/np.pi
        time = np.linspace(0,1.0/edge_freq,100)

        for it,t in enumerate(time):

            #Get edge mode shape at current time
            edge_mode_t = edge_mode * np.cos(2.0 * np.pi * edge_freq * t + edge_mode_phase)

            for enode in edge_mode_t:
                phi = np.linalg.norm(enode[3:])
                nvec = enode[3:]/phi
                enode[3:] = 4.0 *np.tan(0.25 * phi) * nvec

            #First viz at the FE nodes
            node_pts_shape = (edge_mode.shape[0],3)
            xp_ref = np.zeros(node_pts_shape)
            yp_ref = np.zeros(node_pts_shape)
            zp_ref = np.zeros(node_pts_shape)

            xp_disp = np.zeros(node_pts_shape)
            yp_disp= np.zeros(node_pts_shape)
            zp_disp = np.zeros(node_pts_shape)

            for i in range(node_pts_shape[0]):
                xp_ref[i] = applyWM(-init_loc[i+1,3:], [1,0,0], -1)
                yp_ref[i] = applyWM(-init_loc[i+1,3:], [0,1,0], -1)
                zp_ref[i] = applyWM(-init_loc[i+1,3:], [0,0,1], -1)

                xp_disp[i] = applyWM( compose(edge_mode_t[i,3:], -init_loc[i+1,3:],-1), [1,0,0], -1)
                yp_disp[i] = applyWM( compose(edge_mode_t[i,3:], -init_loc[i+1,3:],-1), [0,1,0], -1)
                zp_disp[i] = applyWM( compose(edge_mode_t[i,3:], -init_loc[i+1,3:],-1), [0,0,1], -1)

            all_bld_disp = np.c_[init_loc[1:,:3]+self.bld_root_ref_pos[0,:], xp_ref, yp_ref, zp_ref, edge_mode_t[:,:3], xp_disp, yp_disp, zp_disp]

            np.savetxt('edge_mode_{:03d}.csv'.format(it),
                       all_bld_disp,
                       delimiter=',',
                       header='ref_x,ref_y,ref_z,ref_rotx_0,ref_rotx_1,ref_rotx_2,ref_roty_0,ref_roty_1,ref_roty_2,ref_rotz_0,ref_rotz_1,ref_rotz_2,trans_disp_x,trans_disp_y,trans_disp_z,disp_rotx_0,disp_rotx_1,disp_rotx_2,disp_roty_0,disp_roty_1,disp_roty_2,disp_rotz_0,disp_rotz_1,disp_rotz_2')


            # #Now redo viz at the quadrature points
            # edge_mode_t_qp = self.interp_rotations_to_qpts(edge_mode_t)


            # qpts_shape = np.transpose(self.bld_ref_orient[0]).shape
            # xp_ref = np.zeros(qpts_shape)
            # yp_ref = np.zeros(qpts_shape)
            # zp_ref = np.zeros(qpts_shape)

            # xp_disp = np.zeros(qpts_shape)
            # yp_disp= np.zeros(qpts_shape)
            # zp_disp = np.zeros(qpts_shape)

            # for i in range(qpts_shape[0]):
            #     xp_ref[i] = applyWM(self.bld_ref_orient[0,:,i], [1,0,0], -1)
            #     yp_ref[i] = applyWM(self.bld_ref_orient[0,:,i], [0,1,0], -1)
            #     zp_ref[i] = applyWM(self.bld_ref_orient[0,:,i], [0,0,1], -1)

            #     xp_disp[i] = applyWM( compose(edge_mode_t_qp[i,3:], -self.bld_ref_orient[0,:,i],-1), [1,0,0], -1)
            #     yp_disp[i] = applyWM( compose(edge_mode_t_qp[i,3:], -self.bld_ref_orient[0,:,i],-1), [0,1,0], -1)
            #     zp_disp[i] = applyWM( compose(edge_mode_t_qp[i,3:], -self.bld_ref_orient[0,:,i],-1), [0,0,1], -1)

            # all_bld_disp = np.c_[self.bld_ref_pos[0].transpose(), xp_ref, yp_ref, zp_ref, edge_mode_t_qp[:,:3], xp_disp, yp_disp, zp_disp]

            # np.savetxt('edge_mode_qp_{:03d}.csv'.format(it),
            #            all_bld_disp,
            #            delimiter=',',
            #            header='ref_x, ref_y, ref_z, ref_rotx_0, ref_rotx_1, ref_rotx_2, ref_roty_0, ref_roty_1, ref_roty_2, ref_rotz_0, ref_rotz_1, ref_rotz_2, trans_disp_x, trans_disp_y, trans_disp_z, disp_rotx_0, disp_rotx_1, disp_rotx_2, disp_roty_0, disp_roty_1, disp_roty_2, disp_rotz_0, disp_rotz_1, disp_rotz_2')


def calc_aoa34_wm(X, dXdt, c, pitch_axis, uinfty):
    """
    Calculate aoa at 3/4 chord given positions in Wiener-Milenkovic parameters

    Args:
        X (np.array): Position vector [0:2], Wiener-Milenkovic parameters [3:5]
        dXdt (np.array): Translational velocity [0:2], Rate of change of Wiener-Milenkovic parameters [3:5]
        c (double): Chord
        uinfty (np.array): Velocity vector in the free-stream

    Return:
        aoa (double): Angle of attack at 3/4 chord

    """

    omega = np.dot(calcH(X[3:]), dXdt[3:]) #Angular velocity at the pitch axis
    loc_chord = applyWM(X[3:], [1,0,0], -1)
    vec_pa_34c = loc_chord * c * (0.75 - pitch_axis) #Vector in inertial coordinates from pitch axis to 3/4 chord
    af_vel_34c = np.cross(omega, vec_pa_34c) #Velocity of the airfoil at the 3/4 chord due to airfoil motion
    rel_vel = uinfty - af_vel_34c
    loc_norm = applyWM(X[3:], [0,1,0]) #Vector along direction in plane of airfoil normal to chord
    aoa = np.degrees(np.atan2(np.dot(rel_vel, loc_norm), np.dot(rel_vel, loc_chord)))




if __name__=="__main__":

    of_fsi = OpenfastFSI('turb_00_output.nc')

    # for tstep in range(-1440, -10):
    #     of_fsi.calc_local_chord_dir(tstep)
    #     # print(of_fsi.get_local_chord_dir(0,13))
    #     # print(of_fsi.get_local_chord_dir(0,19))
    #     # print(of_fsi.get_local_chord_dir(0,33))
    #     print(of_fsi.get_local_chord_dir(0,37))

    #of_fsi.viz_displacements(0)
    #of_fsi.convert_xdot_to_xdot_omega()

    #of_fsi.viz_edgewise_modeshape()
    of_fsi.write_mode_shape_to_file('edge_mode.yaml')
