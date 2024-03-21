import numpy as np
import csv
import warnings
import random
import gpuRIR
from scipy.optimize import minimize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from common.utils_room_acoustics import rt60_from_rirs, dpRIR_from_RIR
from dataset import Parameter, AcousticScene, ArraySetup

dualch_array_setup = ArraySetup(arrayType='planar_linear',
    orV = np.array([0.0, 1.0, 0.0]), 
    mic_scale = Parameter(0.3, 2), 
    mic_rotate = Parameter(0, 360), 
    mic_pos = np.array(((-0.05, 0.0, 0.0),
                        (0.05, 0.0, 0.0))), 
    mic_orV = np.array(((-1.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0))), 
    mic_pattern = 'omni'
)

class RandomMicSigDataset(Dataset):
    """ generate microphone signals & RIRs (specify fixed number of rooms)
	"""
    def __init__(self, room_sz, abs_weights, T60, array_setup, array_pos, num_source, source_state,min_src_array_dist, min_src_boundary_dist, nb_points,
            sourceDataset, noiseDataset, SNR, room_num=np.inf, rir_num_eachroom=50, c=343.0,
            transforms=None, return_data = ['sig', 'gt'], ):
        """
		sourceDataset: dataset with the source signals 
		num_source: Number of sources
		source_state: Static or mobile sources
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients ratios of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) noise ratio
		nb_points: Number of points to simulate along the trajectory
		c: Speed of sound 
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		return_data: ['rir'], ['sig', 'scene'] or ['sig', 'gt']

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

        assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV must be parallel to an axis"
        self.array_setup = array_setup
        self.N = array_setup.mic_pos.shape[0]  # the number of microphones
        self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        self.mic_scale = array_setup.mic_scale if type(array_setup.mic_scale) is Parameter else Parameter(array_setup.mic_scale)
        self.mic_rotate = array_setup.mic_rotate if type(array_setup.mic_rotate) is Parameter else Parameter(array_setup.mic_rotate)
        self.min_src_array_dist = min_src_array_dist # control the distance between sources and array (in cm)
        self.min_src_boundary_dist = min_src_boundary_dist
        
        self.num_source = num_source if type(num_source) is Parameter else Parameter(num_source, discrete=True)
        self.source_state = source_state
        self.nb_points = nb_points

        self.sourceDataset = sourceDataset
        self.sourceDataset_idx = Parameter(range(len(self.sourceDataset)), discrete=True)

        self.noiseDataset = noiseDataset
        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)

        self.fs = sourceDataset.fs
        self.c = c   # Speed of sound

        self.transforms = transforms
        self.return_data = return_data
        self.room_num = room_num
        self.rir_num_eachroom = rir_num_eachroom
        self.rooms = []  
        if 'rir' in self.return_data:
            gen_mode = 'rir'
        elif 'sig' in self.return_data:
            gen_mode = 'sig'
        while (len(self.rooms)<self.room_num) & (self.room_num is not np.inf):
            configs = self.getRandomScene(gen_mode=gen_mode)
            self.rooms += [configs]

    def __len__(self):
        return self.room_num 

    def __getitem__(self, idx):

        if 'rir' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='rir', room_config=self.rooms[idx])
            return acoustic_scene
        
        if 'sig' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='sig', room_config=self.rooms[idx])
            mic_signals = acoustic_scene.simulate()

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            mic_signals = mic_signals.astype(np.float32)

            if 'scene' in self.return_data:
                return mic_signals, acoustic_scene

            if 'gt' in self.return_data:
                gts = {}
                gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32)
                gts['T60'] = acoustic_scene.T60.astype(np.float32)
                gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
                gts['C50'] = acoustic_scene.C50w.astype(np.float32)
                gts['SNR'] = acoustic_scene.SNR.astype(np.float32)
                gts['SUR'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2
                gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                gts['ABS'] = (0.161*gts['VOL']/gts['T60']/gts['SUR']).astype(np.float32)
                return mic_signals, gts
                
    def saveRoomConfigurations(self, save_dir, room_range):
        if (self.rooms != []):
            row_name = ['RoomSize', 'RT60', 'AbsorptionCoefficient',]
            csv_name = save_dir + '/Room' + str(room_range[0]) + '-' + str(room_range[1]-1) + '_sz_t60_abs' + '.csv'
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_name)
                for csv_row in self.rooms:
                    writer.writerow(csv_row)

    def genTrajectory(self, room_sz, array_pos, array_setup, min_src_array_dist, min_src_boundary_dist, num_source, traj_pt_mode='time'):
        src_pos_min = np.array([0.0, 0.0, 0.0]) + np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
        src_pos_max = room_sz - np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
        if array_setup.arrayType == 'planar_linear':   
            # suited cases: annotation is symetric about end-fire direction (like TDOA, not DOA), front-back confusion not exists, half-plane is set
            # src can be at any 3D point in half-plane space
            if np.sum(array_setup.orV) > 0:
                src_pos_min[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                src_pos_min += min_src_array_dist * np.abs(array_setup.orV)
            else:
                src_pos_max[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                src_pos_max -= min_src_array_dist * np.abs(array_setup.orV)

        # if array_setup.arrayType == 'planar_linear':   
        #     # suited cases: annotation is symetric about end-fire direction (like DOA, not TDOA), front-back confusion exists, half-plane is set
        #     # src can be at any planar point in half-plane space
        #     assert (array_setup.orV==[0,1,0]) | (array_setup.orV==[0,-1,0]) | (array_setup.orV==[1,0,0]) | (array_setup.orV==[-1,0,0]), 'array orientation must along x or y axis'
        #     if array_setup.orV[0] == 1:
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif array_setup.orV[0] == -1:
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif array_setup.orV[1] == 1:
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif array_setup.orV[1] == -1:
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 

        # if array_setup.arrayType == 'planar': 
        #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
        #     # src can be at any planar point in the all-plane space
        #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
        #     direction_candidates = ['x', 'y', '-x', '-y']
        #     direction = random.sample(direction_candidates, 1)[0]
        #     if direction == 'x':
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif direction == '-x':
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif direction == 'y':
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif direction == '-y':
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
        #     else:
        #         raise Exception('Unrecognized direction~')
        #     src_pos_min[2] = array_pos[2] - 0.0
        #     src_pos_max[2] = array_pos[2] + 0.0
        #     # src_pos = np.concatenate((src_pos_min[np.newaxis, :], src_pos_max[np.newaxis, :]), axis=0)
        #     # self.plotScene(room_sz=room_sz, traj_pts=src_pos, mic_pos=array_setup.mic_pos, view='XY', save_path='./')
            
        # if array_setup.arrayType == '3D': 
        #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
        #     # src can be at some 3D point in the all-plane space
        #     assert array_setup.rotate == 0, 'array rotate must be 0'
        #     direction_candidates = ['x', 'y', '-x', '-y']
        #     direction = random.sample(direction_candidates, 1)[0]
        #     if direction == 'x':
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif direction == '-x':
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif direction == 'y':
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif direction == '-y':
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
        #     else:
        #         raise Exception('Unrecognized direction~')
        #     src_array_relative_height = 0.3
        #     src_pos_min[2] = array_pos[2] - src_array_relative_height
        #     src_pos_max[2] = array_pos[2] + src_array_relative_height
        
        else:
            raise Exception('Undefined array type~')

        for i in range(3):
            assert src_pos_min[i]<=src_pos_max[i], 'Src postion range error: '+str(src_pos_min[i])+ '>' + str(src_pos_max[i]) + '(array boundary dist >= src boundary dist + src array dist)'


        traj_pts = np.zeros((self.nb_points, 3, num_source))
        for source_idx in range(num_source):
            if self.source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts[:, :, source_idx] = np.ones((self.nb_points, 1)) * src_pos

            elif self.source_state == 'mobile': # 3D sinusoidal trjactories
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                
                Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
                if traj_pt_mode == 'time': # Specify nb_points according to time 
                    w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts[:,:,source_idx] = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts[:,:,source_idx] += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_line': # Specify nb_points according to line distance (pointing src_pos_end from src_pos_ini) 
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    nb_points = int(np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.1 + 1) # adaptive number of points, namely one point per liner 10cm
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts += A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_sin': # Specify nb_points according to direct sin distance
                    desired_dist = 0.1 # between ajacent points
                    src_ini_end_dist = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)
                    src_ini_end_dirc_vec = (src_pos_end - src_pos_ini)/src_ini_end_dist
                    w = 2*np.pi / src_ini_end_dist * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    traj_pts = []
                    line_pts = []
                    current_dist_along_dirc_vec = 0
                    while current_dist_along_dirc_vec < src_ini_end_dist:
                        # store current point
                        osc = A * np.sin(w * current_dist_along_dirc_vec)
                        line = src_pos_ini + src_ini_end_dirc_vec * current_dist_along_dirc_vec
                        pos0 = line + osc
                        traj_pts.append(pos0) 
                        line_pts.append(line)

                        # find next point
                        for factor in [1.0, 1.5, 3]:
                            res = minimize(self.dist_err_func, x0=[desired_dist / 10], bounds=[(0, desired_dist * factor)], 
                                           tol=desired_dist / 100, args=(current_dist_along_dirc_vec, src_ini_end_dirc_vec, src_pos_ini, pos0, desired_dist, A, w))
                            if res.fun < desired_dist / 100:
                                break
                        current_dist_along_dirc_vec = current_dist_along_dirc_vec + res.x[0]
                    traj_pts = np.array(traj_pts)
                    line_pts = np.array(line_pts)

                # if np.random.random(1) < 0.25:
                #     traj_pts[:,:,source_idx] = np.ones((self.nb_points,1)) * src_pos_ini
                # traj_pts[:,2,source_idx] = array_pos[2] # if sources and array are in the same horizontal plane
                    
            # self.plotScene(room_sz=room_sz, traj_pts=traj_pts , mic_pos=array_setup.mic_pos, view='XY', save_path='./')

        if traj_pt_mode != 'distance_sin':
            return traj_pts 
        else:
            return traj_pts, line_pts

    def dist_err_func(self, delta_dist_along_dirc_vec, current_dist_along_dirc_vec, ini_end_dirc_vec, pos_ini, pos0, desired_dist, A, w):
        osc = A * np.sin(w * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec))
        line = pos_ini + ini_end_dirc_vec * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec) 
        pos_current = line + osc
        dist = np.sqrt(np.sum((pos_current - pos0)**2))
        return np.abs(dist - desired_dist)
    
    # def plotScene(self, room_sz, traj_pts, mic_pos, view='3D', save_path=None):
    #     """ Plots the source trajectory and the microphones within the room
    #         Args:   traj_pts - (npoints, 3, nsrc)
    #                 mic_pos  - (nmic, 3)
    #     """
    #     assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']
    #     fig = plt.figure()
    #     nsrc = traj_pts.shape[-1]

    #     if view == '3D' or view == 'XYZ':
    #         ax = Axes3D(fig)
    #         ax.set_xlim3d(0, room_sz[0])
    #         ax.set_ylim3d(0, room_sz[1])
    #         ax.set_zlim3d(0, room_sz[2])
    #         ax.scatter(mic_pos[:,0], mic_pos[:,1], mic_pos[:,2])
    #         legends = ['Microphone array']
    #         for src_idx in range(nsrc):
    #             ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
    #             ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
    #             legends += ['Source trajectory ' + str(src_idx)]
    #         ax.legend(legends)
    #         # ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
    #         ax.set_xlabel('x [m]')
    #         ax.set_ylabel('y [m]')
    #         ax.set_zlabel('z [m]')

    #     else:
    #         ax = fig.add_subplot(111)
    #         plt.gca().set_aspect('equal', adjustable='box')

    #         if view == 'XY':
    #             ax.set_xlim(0, room_sz[0])
    #             ax.set_ylim(0, room_sz[1])
    #             ax.scatter(mic_pos[:,0], mic_pos[:,1])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx])
    #                 ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('x [m]')
    #             ax.set_ylabel('y [m]')
    #         elif view == 'XZ':
    #             ax.set_xlim(0, room_sz[0])
    #             ax.set_ylim(0, room_sz[2])
    #             ax.scatter(mic_pos[:,0], mic_pos[:,2])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,2,src_idx])
    #                 ax.text(traj_pts[0,0,src_idx], traj_pts[0,2,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('x [m]')
    #             ax.set_ylabel('z [m]')
    #         elif view == 'YZ':
    #             ax.set_xlim(0, room_sz[1])
    #             ax.set_ylim(0, room_sz[2])
    #             ax.scatter(mic_pos[:,1], mic_pos[:,2])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
    #                 ax.text(traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('y [m]')
    #             ax.set_ylabel('z [m]')

    #     # plt.show()
    #     if save_path is not None: 
    #         plt.savefig(save_path + 'room')
    #     plt.close()

    def genRIR(self, room_sz, beta, T60, array_setup, traj_pts):
        if T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )

        RIRs_sources = []
        dp_RIRs_sources = []
        num_source = traj_pts.shape[-1]
        for source_idx in range(num_source):
            RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=nb_img, Tmax=Tmax, fs=self.fs, Tdiff=Tdiff, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            dp_RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=[1,1,1], Tmax=0.1, fs=self.fs, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            RIRs_sources += [RIRs]
            dp_RIRs_sources += [dp_RIRs]

        RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
        dp_RIRs_sources = np.array(dp_RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)

        return RIRs_sources, dp_RIRs_sources
    
    def checkRIR(self, RIRs):
        ok_flag = True
        nan_flag = np.isnan(RIRs)
        inf_flag = np.isinf(RIRs)
        if (True in nan_flag):
            warnings.warn('NAN exists in RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(RIRs**2) == 0)
        if zero_flag:
            warnings.warn('RIR is all zeros~')
            ok_flag = False
        return ok_flag

    def T60isValid(self, room_sz, T60, alpha, th=0.005, eps=1e-4):
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
            (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
            (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: # avoid cases when all walls are reflective and T60 (from sabine eq) is extremely large
            valid_flag = False 
        else:
            T60_sabine = 0.161 * V / (Sa+eps) 
            valid_flag = bool(abs(T60-T60_sabine)<th)  # avoid cases when T60<T60_min, or the abs_weights is not suited
        beta_prod = np.prod(1-alpha) # avoid sparse reflections for room size estimation
        return valid_flag & bool(beta_prod!=0), T60_sabine

    def getRandomScene(self, gen_mode='sig', room_config=None):
        RIR_is_ok = False
        nan_zero_correct_flag = False
        while(RIR_is_ok==False):
            # Room
            T60_is_valid = False
            while(T60_is_valid==False):
                if room_config is None:
                    room_sz = self.room_sz.getValue()
                    T60_specify = float(self.T60.getValue())
                    abs_weights = self.abs_weights.getValue()
                else:
                    room_sz, T60_specify, abs_weights = room_config
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60_specify, abs_weights)
                T60_is_valid, T60_sabine = self.T60isValid(room_sz, T60_specify, alpha=1-beta**2)

            # Microphones
            array_pos = self.array_pos.getValue() * room_sz
            mic_scale = self.mic_scale.getValue()
            mic_rotate = self.mic_rotate.getValue()
            rotate_matrix = np.array([[np.cos(mic_rotate/180*np.pi), -np.sin(mic_rotate/180*np.pi), 0], 
                                    [np.sin(mic_rotate/180*np.pi), np.cos(mic_rotate/180*np.pi), 0], 
                                    [0, 0, 1]]) # (3, 3)
            mic_pos_rotate = np.dot(rotate_matrix, self.array_setup.mic_pos.transpose(1, 0)).transpose(1, 0)
            mic_pos = array_pos + mic_pos_rotate * mic_scale # (nch,3)
            mic_orV = np.dot(rotate_matrix, self.array_setup.mic_orV.transpose(1, 0)).transpose(1, 0)
            orV = np.dot(rotate_matrix, self.array_setup.orV)

            array_setup = ArraySetup( arrayType=self.array_setup.arrayType,
                orV = orV,
                mic_scale = mic_scale,
                mic_rotate = mic_rotate,
                mic_pos = mic_pos, 
                mic_orV = mic_orV,
                mic_pattern = self.array_setup.mic_pattern
                ) # updated with random mic_scale, mic_rotate

            # Sources
            num_source = self.num_source.getValue()

            # Trajectory points for sources
            traj_pts = self.genTrajectory(room_sz, array_pos, array_setup, self.min_src_array_dist, self.min_src_boundary_dist, num_source) #  (npoint,3,nsource)

            # RIRs for sources
            RIRs_sources, dp_RIRs_sources = self.genRIR(room_sz, beta, T60_specify, array_setup, traj_pts)
            dp_RIRs_com = dpRIR_from_RIR(rir=RIRs_sources, dp_time=2.5, fs=self.fs)
            RIR = [RIRs_sources, dp_RIRs_com, dp_RIRs_sources]
            RIR_is_ok = self.checkRIR(RIRs_sources) & self.checkRIR(dp_RIRs_sources)

            # T60 from EDC
            if RIR_is_ok:
                T60_edc = []
                R = []
                for mic_idx in range(RIRs_sources.shape[1]):
                    rt60, r = rt60_from_rirs(RIRs_sources[0, mic_idx, :, 0], self.fs)
                    T60_edc += [rt60]
                    R += [r]
                T60_edc = np.mean(T60_edc)
                R = np.mean(r)
                # if (abs(R)<0.98):
                #     print('Correlation coefficient (<0.98):', R, 'T60 calculated (EDC):', T60, 'T60 specified:', T60_gt)
                RIR_is_ok = RIR_is_ok & bool(abs(T60_edc-T60_specify)<0.05) & bool(abs(R)>0.5)
            else:
                nan_zero_correct_flag = True
            if nan_zero_correct_flag & RIR_is_ok:
                print('RIR is corrected~')

        if gen_mode == 'rir':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

        elif gen_mode == 'sig':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            
            # Source signals
            source_signal = self.sourceDataset[self.sourceDataset_idx.getValue()]
            source_signal = source_signal[:,0:num_source]

            # Noise signals
            noise_signal = self.noiseDataset.get_random_noise(mic_pos, acoustic_scene, self.sourceDataset)
            SNR = float(self.SNR.getValue())
          
            # Interpolate trajectory points
            timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
            t = np.arange(len(source_signal)) / self.fs
            trajectory = np.zeros((len(t), 3, num_source))
            for source_idx in range(num_source):
                trajectory[:,:,source_idx] = np.array([np.interp(t, timestamps, traj_pts[:,i,source_idx]) for i in range(3)]).transpose()

            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                source_signal = source_signal,
                noise_signal = noise_signal,
                SNR = SNR,
                timestamps = timestamps,
                t = t,
                trajectory = trajectory,
                c = self.c
                )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

            # acoustic_scene.source_vad = vad[:,0:num_source] # use webrtcvad
            # acoustic_scene.mic_vad = [] # use snr
            # acoustic_scene.DOA = []
            acoustic_scene.TDOA = []
            acoustic_scene.DRR = []
            acoustic_scene.C50 = []
            # acoustic_scene.C80 = []
            # acoustic_scene.dp_mic_signal = []
            # acoustic_scene.spakerID = []
        else:
            raise Exception('Unrecognized data generation mode')

        if room_config is None:
            return [room_sz, T60_sabine, abs_weights]
        else:
            return acoustic_scene
        
class RandomMicSigDatasetOri(Dataset):
    """ generating microphone signals & RIRs (not specify fix number of rooms)
	"""
    def __init__(self, room_sz, abs_weights, T60, array_setup, array_pos, num_source, source_state, min_src_array_dist, min_src_boundary_dist, nb_points,
            sourceDataset, noiseDataset, SNR, dataset_sz, c=343.0,
            transforms=None, return_data = ['sig', 'gt']):
        """
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		num_source: Number of sources
		source_state: Static or mobile sources
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients ratios of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) noise ratio
		nb_points: Number of points to simulate along the trajectory
		c: Speed of sound 
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		return_data: ['rir'], ['sig', 'scene'] or ['sig', 'gt']

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

        assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV must be parallel to an axis"
        self.array_setup = array_setup
        self.N = array_setup.mic_pos.shape[0]  # the number of microphones
        self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        self.mic_scale = array_setup.mic_scale if type(array_setup.mic_scale) is Parameter else Parameter(array_setup.mic_scale)
        self.mic_rotate = array_setup.mic_rotate if type(array_setup.mic_rotate) is Parameter else Parameter(array_setup.mic_rotate)
        self.min_src_array_dist = min_src_array_dist # control the distance between sources and array (in cm)
        self.min_src_boundary_dist = min_src_boundary_dist

        self.num_source = num_source if type(num_source) is Parameter else Parameter(num_source, discrete=True)
        self.source_state = source_state
        self.nb_points = nb_points

        self.sourceDataset = sourceDataset
        self.sourceDataset_idx = Parameter(range(len(self.sourceDataset)), discrete=True)

        self.noiseDataset = noiseDataset
        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)

        self.fs = sourceDataset.fs
        self.c = c   # Speed of sound

        self.dataset_sz = dataset_sz
        self.transforms = transforms
        self.return_data = return_data

    def __len__(self):
        return self.dataset_sz  

    def __getitem__(self, idx):

        if 'rir' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='rir')
            return acoustic_scene
        if 'sig' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='sig')
            mic_signals = acoustic_scene.simulate()
            # acoustic_scene.plotScene( view='XY', save_path='./'+str(idx)+'_')
            # self.plotScene(room_sz=acoustic_scene.room_sz, traj_pts=acoustic_scene.traj_pts, mic_pos=acoustic_scene.mic_pos, view='XY', save_path='./'+str(idx)+'_')

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            mic_signals = mic_signals.astype(np.float32)

            if 'scene' in self.return_data:
                return mic_signals, acoustic_scene

            if 'gt' in self.return_data:
                gts = {}
                gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32)
                gts['T60'] = acoustic_scene.T60.astype(np.float32)
                gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
                gts['C50'] = acoustic_scene.C50w.astype(np.float32)
                gts['SNR'] = acoustic_scene.SNR.astype(np.float32)
                gts['SUR'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2
                gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                gts['ABS'] = (0.161*gts['VOL']/gts['T60']/gts['SUR']).astype(np.float32)
                return mic_signals, gts

    def genTrajectory(self, room_sz, array_pos, array_setup, min_src_array_dist, min_src_boundary_dist, num_source, traj_pt_mode='time'):
        traj_pts = np.zeros((self.nb_points, 3, num_source))
        for source_idx in range(num_source):
            src_pos_min = np.array([0.0, 0.0, 0.0]) + np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            src_pos_max = room_sz - np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            if array_setup.arrayType == 'planar_linear':   
                # suited cases: annotation is symetric about end-fire direction (like TDOA, not DOA), front-back confusion not exists, half-plane is set
                # src can be at any 3D point in half-plane space
                if np.sum(array_setup.orV) > 0:
                    src_pos_min[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                    src_pos_min += min_src_array_dist * np.abs(array_setup.orV)
                else:
                    src_pos_max[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                    src_pos_max -= min_src_array_dist * np.abs(array_setup.orV)

            # if array_setup.arrayType == 'planar_linear':   
            #     # suited cases: annotation is symetric about end-fire direction (like DOA, not TDOA), front-back confusion exists, half-plane is set
            #     # src can be at any planar point in half-plane space
            #     assert (array_setup.orV==[0,1,0]) | (array_setup.orV==[0,-1,0]) | (array_setup.orV==[1,0,0]) | (array_setup.orV==[-1,0,0]), 'array orientation must along x or y axis'
            #     if array_setup.orV[0] == 1:
            #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
            #     elif array_setup.orV[0] == -1:
            #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
            #     elif array_setup.orV[1] == 1:
            #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
            #     elif array_setup.orV[1] == -1:
            #         src_pos_max[1] = array_pos[1] - min_src_array_dist 

            # if array_setup.arrayType == 'planar': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at any planar point in the all-plane space
            #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
            #     direction_candidates = ['x', 'y', '-x', '-y']
            #     direction = random.sample(direction_candidates, 1)[0]
            #     if direction == 'x':
            #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
            #     elif direction == '-x':
            #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
            #     elif direction == 'y':
            #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
            #     elif direction == '-y':
            #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
            #     else:
            #         raise Exception('Unrecognized direction~')

            #     src_pos_min[2] = array_pos[2] - 0.0
            #     src_pos_max[2] = array_pos[2] + 0.0
                
            #     # src_pos = np.concatenate((src_pos_min[np.newaxis, :], src_pos_max[np.newaxis, :]), axis=0)
            #     # self.plotScene(room_sz=room_sz, traj_pts=src_pos, mic_pos=array_setup.mic_pos, view='XY', save_path='./')
                
            # if array_setup.arrayType == '3D': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at some 3D point in the all-plane space
            #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
            #     direction_candidates = ['x', 'y', '-x', '-y']
            #     direction = random.sample(direction_candidates, 1)[0]
            #     if direction == 'x':
            #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
            #     elif direction == '-x':
            #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
            #     elif direction == 'y':
            #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
            #     elif direction == '-y':
            #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
            #     else:
            #         raise Exception('Unrecognized direction~')
            #     src_array_relative_height = 0.3
            #     src_pos_min[2] = array_pos[2] - src_array_relative_height
            #     src_pos_max[2] = array_pos[2] + src_array_relative_height

            else:
                raise Exception('Undefined array type~')

            for i in range(3):
                assert src_pos_min[i]<=src_pos_max[i], 'Src postion range error: '+str(src_pos_min[i])+ '>' + str(src_pos_max[i]) + '(array boundary dist >= src boundary dist + src array dist)'
                
            if self.source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts[:, :, source_idx] = np.ones((self.nb_points, 1)) * src_pos

            elif self.source_state == 'mobile': # 3D sinusoidal trjactories
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                # self.nb_points = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.01 + 1 # adaptive number of points, namely one point per liner 10cm

                Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
                if traj_pt_mode == 'time': # Specify nb_points according to time 
                    w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts[:,:,source_idx] = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts[:,:,source_idx] += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_line': # Specify nb_points according to line distance (pointing src_pos_end from src_pos_ini) 
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    nb_points = int(np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.1 + 1) # adaptive number of points, namely one point per liner 10cm
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts += A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_sin': # Specify nb_points according to direct sin distance
                    desired_dist = 0.1 # between ajacent points
                    src_ini_end_dist = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)
                    src_ini_end_dirc_vec = (src_pos_end - src_pos_ini)/src_ini_end_dist
                    w = 2*np.pi / src_ini_end_dist * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    traj_pts = []
                    line_pts = []
                    current_dist_along_dirc_vec = 0
                    while current_dist_along_dirc_vec < src_ini_end_dist:
                        # store current point
                        osc = A * np.sin(w * current_dist_along_dirc_vec)
                        line = src_pos_ini + src_ini_end_dirc_vec * current_dist_along_dirc_vec
                        pos0 = line + osc
                        traj_pts.append(pos0) 
                        line_pts.append(line)

                        # find next point
                        for factor in [1.0, 1.5, 3]:
                            res = minimize(self.dist_err_func, x0=[desired_dist / 10], bounds=[(0, desired_dist * factor)], 
                                           tol=desired_dist / 100, args=(current_dist_along_dirc_vec, src_ini_end_dirc_vec, src_pos_ini, pos0, desired_dist, A, w))
                            if res.fun < desired_dist / 100:
                                break
                        current_dist_along_dirc_vec = current_dist_along_dirc_vec + res.x[0]
                    traj_pts = np.array(traj_pts)
                    line_pts = np.array(line_pts)

                # if np.random.random(1) < 0.25:
                #     traj_pts[:,:,source_idx] = np.ones((self.nb_points,1)) * src_pos_ini
                # traj_pts[:,2,source_idx] = array_pos[2] # if sources and array are in the same horizontal plane
            
        if traj_pt_mode != 'distance_sin':
            return traj_pts 
        else:
            return traj_pts, line_pts

    def dist_err_func(self, delta_dist_along_dirc_vec, current_dist_along_dirc_vec, ini_end_dirc_vec, pos_ini, pos0, desired_dist, A, w):
        osc = A * np.sin(w * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec))
        line = pos_ini + ini_end_dirc_vec * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec) 
        pos_current = line + osc
        dist = np.sqrt(np.sum((pos_current - pos0)**2))
        return np.abs(dist - desired_dist)

    # def plotScene(self, room_sz, traj_pts, mic_pos, view='3D', save_path=None):
    #     """ Plots the source trajectory and the microphones within the room
    #         Args:   traj_pts - (npoints, 3, nsrc)
    #                 mic_pos  - (nmic, 3)
    #     """
    #     assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']
    #     fig = plt.figure()
    #     nsrc = traj_pts.shape[-1]

    #     if view == '3D' or view == 'XYZ':
    #         ax = Axes3D(fig)
    #         ax.set_xlim3d(0, room_sz[0])
    #         ax.set_ylim3d(0, room_sz[1])
    #         ax.set_zlim3d(0, room_sz[2])
    #         ax.scatter(mic_pos[:,0], mic_pos[:,1], mic_pos[:,2])
    #         legends = ['Microphone array']
    #         for src_idx in range(nsrc):
    #             ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
    #             ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
    #             legends += ['Source trajectory ' + str(src_idx)]
    #         ax.legend(legends)
    #         # ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
    #         ax.set_xlabel('x [m]')
    #         ax.set_ylabel('y [m]')
    #         ax.set_zlabel('z [m]')

    #     else:
    #         ax = fig.add_subplot(111)
    #         plt.gca().set_aspect('equal', adjustable='box')

    #         if view == 'XY':
    #             ax.set_xlim(0, room_sz[0])
    #             ax.set_ylim(0, room_sz[1])
    #             ax.scatter(mic_pos[:,0], mic_pos[:,1])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx])
    #                 ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('x [m]')
    #             ax.set_ylabel('y [m]')
    #         elif view == 'XZ':
    #             ax.set_xlim(0, room_sz[0])
    #             ax.set_ylim(0, room_sz[2])
    #             ax.scatter(mic_pos[:,0], mic_pos[:,2])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,2,src_idx])
    #                 ax.text(traj_pts[0,0,src_idx], traj_pts[0,2,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('x [m]')
    #             ax.set_ylabel('z [m]')
    #         elif view == 'YZ':
    #             ax.set_xlim(0, room_sz[1])
    #             ax.set_ylim(0, room_sz[2])
    #             ax.scatter(mic_pos[:,1], mic_pos[:,2])
    #             legends = ['Microphone array']
    #             for src_idx in range(nsrc):
    #                 ax.scatter(traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
    #                 ax.text(traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
    #                 legends += ['Source trajectory ' + str(src_idx)]
    #             ax.legend(legends)
    #             ax.set_xlabel('y [m]')
    #             ax.set_ylabel('z [m]')

    #     # plt.show()
    #     if save_path is not None: 
    #         plt.savefig(save_path + 'room')
    #     plt.close()

    def genRIR(self, room_sz, beta, T60, array_setup, traj_pts):
        if T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )

        RIRs_sources = []
        dp_RIRs_sources = []
        num_source = traj_pts.shape[-1]
 
        for source_idx in range(num_source):
            RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=nb_img, Tmax=Tmax, fs=self.fs, Tdiff=Tdiff, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            dp_RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=[1,1,1], Tmax=0.1, fs=self.fs, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            RIRs_sources += [RIRs]
            dp_RIRs_sources += [dp_RIRs]

        RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
        dp_RIRs_sources = np.array(dp_RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)

        return RIRs_sources, dp_RIRs_sources
    
    def checkRIR(self, RIRs):
        ok_flag = True
        nan_flag = np.isnan(RIRs)
        inf_flag = np.isinf(RIRs)
        if (True in nan_flag):
            warnings.warn('NAN exists in RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(RIRs**2) == 0)
        if zero_flag:
            warnings.warn('RIR is all zeros~')
            ok_flag = False
        return ok_flag

    def T60isValid(self, room_sz, T60, alpha, th=0.005, eps=1e-4):
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
            (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
            (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: # avoid cases when all walls are reflective and T60 (from sabine eq) is extremely large
            valid_flag = False 
            T60_sabine = None
        else:
            T60_sabine = 0.161 * V / (Sa+eps) 
            valid_flag = bool(abs(T60-T60_sabine)<th)  # avoid cases when T60<T60_min, or the abs_weights is not suited
        beta_prod = np.prod(1-alpha) # avoid sparse reflections for room size estimation
        return valid_flag & bool(beta_prod!=0), T60_sabine

    def getRandomScene(self, gen_mode='sig'):
        RIR_is_ok = False
        while(RIR_is_ok==False):
            # Room
            T60_is_valid = False
            while(T60_is_valid==False):
                room_sz = self.room_sz.getValue()
                T60_specify = float(self.T60.getValue())
                abs_weights = self.abs_weights.getValue()
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60_specify, abs_weights)
                T60_is_valid, T60_sabine = self.T60isValid(room_sz, T60_specify, alpha=1-beta**2)

            # Microphones
            array_pos = self.array_pos.getValue() * room_sz
            mic_scale = self.mic_scale.getValue()
            mic_rotate = self.mic_rotate.getValue()
            rotate_matrix = np.array([[np.cos(mic_rotate/180*np.pi), -np.sin(mic_rotate/180*np.pi), 0], 
                                    [np.sin(mic_rotate/180*np.pi), np.cos(mic_rotate/180*np.pi), 0], 
                                    [0, 0, 1]]) # (3, 3)
            mic_pos_rotate = np.dot(rotate_matrix, self.array_setup.mic_pos.transpose(1, 0)).transpose(1, 0)
            mic_pos = array_pos + mic_pos_rotate * mic_scale # (nch,3)
            mic_orV = np.dot(rotate_matrix, self.array_setup.mic_orV.transpose(1, 0)).transpose(1, 0)
            orV = np.dot(rotate_matrix, self.array_setup.orV)

            array_setup = ArraySetup( arrayType=self.array_setup.arrayType,
                orV = orV,
                mic_scale = mic_scale,
                mic_rotate = mic_rotate,
                mic_pos = mic_pos, 
                mic_orV = mic_orV,
                mic_pattern = self.array_setup.mic_pattern
                ) # updated with random mic_scale and mic_rotate

            # Sources
            num_source = self.num_source.getValue()

            # Trajectory points for sources
            traj_pts = self.genTrajectory(room_sz, array_pos, array_setup, self.min_src_array_dist, self.min_src_boundary_dist, num_source) #  (npoint,3,nsource)

            # RIRs for sources
            RIRs_sources, dp_RIRs_sources = self.genRIR(room_sz, beta, T60_specify, array_setup, traj_pts)
            dp_RIRs_com = dpRIR_from_RIR(rir=RIRs_sources, dp_time=2.5, fs=self.fs)
            RIR = [RIRs_sources, dp_RIRs_com, dp_RIRs_sources]
            RIR_is_ok = self.checkRIR(RIRs_sources) & self.checkRIR(dp_RIRs_sources)
            # RIR_is_ok = True

            # T60 from EDC
            if RIR_is_ok:
                T60_edc = []
                R = []
                for mic_idx in range(RIRs_sources.shape[1]):
                    rt60, r = rt60_from_rirs(RIRs_sources[0, mic_idx, :, 0], self.fs)
                    T60_edc += [rt60]
                    R += [r]
                T60_edc = np.mean(T60_edc)
                R = np.mean(r)
                # if (abs(R)<0.98):
                #     print('Correlation coefficient (<0.98):', R, 'T60 calculated (EDC):', T60, 'T60 specified:', T60_gt)
                RIR_is_ok = RIR_is_ok & bool(abs(T60_edc-T60_specify)<0.05) & bool(abs(R)>0.5)

        if gen_mode == 'rir':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

        elif gen_mode == 'sig':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            
            # Source signals
            source_signal = self.sourceDataset[self.sourceDataset_idx.getValue()]
            source_signal = source_signal[:,0:num_source]

            # Noise signals
            noise_signal = self.noiseDataset.get_random_noise(mic_pos, acoustic_scene, self.sourceDataset)
            SNR = float(self.SNR.getValue())
          
            # Interpolate trajectory points
            timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
            t = np.arange(len(source_signal)) / self.fs
            trajectory = np.zeros((len(t), 3, num_source))
            for source_idx in range(num_source):
                trajectory[:,:,source_idx] = np.array([np.interp(t, timestamps, traj_pts[:,i,source_idx]) for i in range(3)]).transpose()

            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                source_signal = source_signal,
                noise_signal = noise_signal,
                SNR = SNR,
                timestamps = timestamps,
                t = t,
                trajectory = trajectory,
                c = self.c
                )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

            # acoustic_scene.source_vad = vad[:,0:num_source] # use webrtcvad
            # acoustic_scene.mic_vad = [] # use snr
            # acoustic_scene.DOA = []
            acoustic_scene.TDOA = []
            acoustic_scene.DRR = []
            acoustic_scene.C50 = []
            # acoustic_scene.C80 = []
            # acoustic_scene.dp_mic_signal = [] # align
            # acoustic_scene.spakerID = []
        else:
            raise Exception('Unrecognized data generation mode')

        return acoustic_scene