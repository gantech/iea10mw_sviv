import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
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

a = nc.Dataset('turb_00_output.nc')
# Extract all arrays from mode_data.npz
with np.load('mode_data.npz') as data:
    bld_disp = data['bld_disp']
    bld_tot_disp = data['bld_tot_disp']
    modal_weight_time = data['modal_weight']
    modal_weight_trans_time = data['modal_weight_trans']
    peaks = data['peaks']
    eigvecs = data['eigvecs']
    eigvals = data['eigvals']

    # bld_tot_disp = bld_tot_disp - np.mean(bld_tot_disp[peaks[-5]:peaks[-1]], axis=0)  # Center the data
    for i in range(60):
        print(f'MAC - Mode {i} - Freq {eigvals[i]} = {np.corrcoef(bld_tot_disp[peaks[-1],:], eigvecs[:,i])}')

    modal_weight = np.sqrt(2.0) * np.sqrt(np.mean(modal_weight_time**2, axis=1))  # RMS of modal weights
    modal_weight_trans = np.sqrt(2.0) * np.sqrt(np.mean(modal_weight_trans_time**2, axis=1))  # RMS of modal weights

    # Create empty figure with 3 subplots from left to right
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Set up titles and labels for first subplot
    for i in range(10):
        axes[0, 0].plot(modal_weight[i] * eigvecs[::6,i])
    #axes[0, 0].plot(np.cumsum(modal_weight * eigvecs[-6,:]))
    axes[0, 0].set_title('Blade Tip X Displacement')
    axes[0, 0].set_xlabel('Mode Index')
    axes[0, 0].set_ylabel('Displacement X')
    axes[0, 0].grid(True)

    # Set up titles and labels for second subplot
    for i in range(10):
        axes[0, 1].plot(modal_weight[i] * eigvecs[1::6,i])
    #axes[0, 1].plot(np.cumsum(modal_weight * eigvecs[-5,:]))
    axes[0, 1].set_title('Blade Tip Y Displacement')
    axes[0, 1].set_xlabel('Mode Index')
    axes[0, 1].set_ylabel('Displacement Y')
    axes[0, 1].grid(True)

    # Set up titles and labels for third subplot
    for i in range(10):
        axes[0, 2].plot(modal_weight[i] * eigvecs[2::6,i])
    #axes[0, 2].plot(np.cumsum(modal_weight * eigvecs[-4,:]))
    axes[0, 2].set_title('Blade Tip Z Displacement')
    axes[0, 2].set_xlabel('Mode Index')
    axes[0, 2].set_ylabel('Displacement Z')
    axes[0, 2].grid(True)

    # Set up titles and labels for first subplot
    for i in range(10):
        axes[1, 0].plot(modal_weight[i] * eigvecs[3::6,i])
    #axes[1, 0].plot(np.cumsum(modal_weight * eigvecs[-3,:]))
    axes[1, 0].set_title('Blade Tip X Rotational Displacement')
    axes[1, 0].set_xlabel('Mode Index')
    axes[1, 0].set_ylabel('Displacement X rot')
    axes[1, 0].grid(True)

    # Set up titles and labels for second subplot
    for i in range(10):
        axes[1, 1].plot(modal_weight[i] * eigvecs[4::6,i])
    #axes[1, 1].plot(np.cumsum(modal_weight * eigvecs[-2,:]))
    axes[1, 1].set_title('Blade Tip Y Rotational Displacement')
    axes[1, 1].set_xlabel('Mode Index')
    axes[1, 1].set_ylabel('Displacement Y rot')
    axes[1, 1].grid(True)

    # Set up titles and labels for third subplot
    for i in range(10):
        axes[1, 2].plot(modal_weight[i] * eigvecs[5::6,i])
    #axes[1, 2].plot(np.cumsum(modal_weight * eigvecs[-1,:]))
    axes[1, 2].set_title('Blade Tip Z Rotational Displacement')
    axes[1, 2].set_xlabel('Mode Index')
    axes[1, 2].set_ylabel('Displacement Z rot')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('modal_analysis.png', dpi=300)

    # interp_matrix = calc_interp_matrix()

    # yaml_node = {
    #     'mode': {
    #         'n_modes': 2,
    #         'amplitude': [float(modal_weight[1]), float(modal_weight[0])],
    #         'freq': [float(eigvals[1]),
    #                  float(eigvals[1])],
    #         'shape': [np.r_[np.zeros((1,6)), eigvecs[:,1].reshape(10,6)].tolist(),
    #                   np.r_[np.zeros((1,6)), eigvecs[:,0].reshape(10,6)].tolist() ],
    #         'phase': [ np.zeros((11,6)).tolist(),
    #                   np.zeros((11,6)).tolist() ],
    #         'interp_matrix': interp_matrix.tolist()
    #     }
    # }

    # yaml_node = {
    #     'mode': {
    #         'n_modes': 60,
    #         'amplitude': [-float(modal_weight[i]) for i in range(60)],
    #         'freq': [float(eigvals[1]) for i in range(60)],
    #         'shape': [np.r_[np.zeros((1,6)), eigvecs[:,i].reshape(10,6)].tolist() for i in range(60)],
    #         'phase': [ (0.5*np.pi*np.ones((11,6))).tolist() for _ in range(60) ],
    #         'interp_matrix': interp_matrix.tolist()
    #     }
    # }

    # nalu_file = yaml.load(open('iea10mw-nalu.yaml'), Loader=yaml.Loader)
    # nalu_file['realms'][0]['mode_shape_analysis']['mode'] = yaml_node['mode']
    # yaml.dump(nalu_file, open('iea10mw-nalu.yaml','w'), default_flow_style=False)
    # yaml.dump(yaml_node, open('modes.yaml','w'), default_flow_style=False)

    print(bld_tot_disp.shape)
    idx = -5
    # Perform FFT on blade total displacement data
    n = len(bld_tot_disp[peaks[-5]:peaks[-1], idx])

    # # Create figure with two subplots
    # plt.figure(figsize=(16, 6))
    # # Left subplot for time domain signal
    # plt.subplot(1, 2, 1)
    # plt.plot(a['time'][peaks[-5]:peaks[-1]], bld_tot_disp[peaks[-5]:peaks[-1], idx])
    # plt.title('Blade Tip Displacement (Time Domain)')
    # plt.xlabel('Time Step')
    # plt.ylabel('Displacement')
    # plt.grid(True)

    # # Perform FFT on blade total displacement data
    # fft_result = np.fft.fft(bld_tot_disp[peaks[-5]:peaks[-1], idx] - np.mean(bld_tot_disp[peaks[-5]:peaks[-1], idx]))
    # # Calculate frequency bins
    # sample_rate = 1/0.005  # Assuming timestep of 0.005s from filename
    # freq = np.fft.fftfreq(n, d=0.005)

    # # Plot the FFT result (only first half as the rest is symmetric for real signals)
    # plt.subplot(1, 2, 2)
    # plt.plot(freq[:n//2], np.abs(fft_result[:n//2]))
    # plt.title(f'FFT of Blade Tip Displacement in y direction')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    fig = plt.figure()
    plt.bar(np.linspace(1, 60, 60), modal_weight)
    plt.title("Average Modal Weight")
    plt.xlabel("Mode Index")
    plt.ylabel("Average Weight")
    plt.grid(True)
    plt.legend()


    fig = plt.figure()
    plt.bar(np.linspace(1, 60, 60), modal_weight_trans)
    plt.title("Average Modal Weight (Translational)")
    plt.xlabel("Mode Index")
    plt.ylabel("Average Weight")
    plt.grid(True)
    plt.legend()

    plt.show()
