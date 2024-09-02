""" Generate single data
"""
import numpy as np
import csv
import os
import scipy
import scipy.io
import scipy.signal
import warnings
import soundfile
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

class SpatialAcoustics():
    """ Generate random spatial acoustics configurations.
    """
    def __init__(self, ):
        pass
    
    def generate_random_spatial_acoustics(
            self,
            room_sz_range, 
            T60_range, 
            abs_weights_range, 
            c,
            ism_db,
            mic_array_cfg,
            array_pos_ratio_range,
            num_source_range, 
            source_state,
            min_src_array_dist, 
            min_src_boundary_dist, 
            nb_points,
            traj_pt_mode,
            room_cfg=None,
            seed=1,
            idx=0
            ):

        np.random.seed(seed=seed+idx)
        if room_cfg is None:
            room_cfg = self.random_room(
                room_sz_range=room_sz_range,
                T60_range=T60_range,
                abs_weights_range=abs_weights_range,
                c=c,
                ism_db=ism_db,
                )
        mic_array_cfg = self.random_mic_array(
            mic_array_cfg=mic_array_cfg, 
            array_pos_ratio_range=array_pos_ratio_range,
            room_sz=room_cfg['room_sz'],

            )
        src_traj_cfg = self.random_src_trajectory(
            room_sz=room_cfg['room_sz'], 
            mic_array_cfg=mic_array_cfg, 
            min_src_array_dist=min_src_array_dist, 
            min_src_boundary_dist=min_src_boundary_dist, 
            num_source_range=num_source_range, 
            source_state=source_state,
            nb_points=nb_points,
            traj_pt_mode=traj_pt_mode,
            array_pos=mic_array_cfg['array_pos'],
            array_orV=mic_array_cfg['array_orV'],
            )
        spatial_acoustics_cfg = {**room_cfg, **mic_array_cfg, **src_traj_cfg}

        return spatial_acoustics_cfg

    def random_room(self, room_sz_range, T60_range, abs_weights_range, c=343.0, ism_db=12, room_cfg=None):
        if room_cfg is None:
            room_sz = [np.random.uniform(*i_range) for i_range in room_sz_range]
            T60_is_valid = False
            while(T60_is_valid==False):
                T60_specify = np.random.uniform(*T60_range)
                abs_weights = [np.random.uniform(*aw_range) for aw_range in abs_weights_range]
                beta = self._beta_Sabine_estimation(room_sz, T60_specify, abs_weights)
                T60_is_valid, T60_sabine = self._t60_is_valid(room_sz=room_sz, T60=T60_specify, alpha=1-beta**2, c=c, ism_db=ism_db)
        else:
            room_sz = room_cfg['room_sz']
            T60_specify = room_cfg['T60_specify']
            abs_weights = room_cfg['abs_weights']
            beta = self._beta_Sabine_estimation(room_sz, T60_specify, abs_weights)
            T60_is_valid, T60_sabine = self._t60_is_valid(room_sz=room_sz, T60=T60_specify, alpha=1-beta**2, c=c, ism_db=ism_db)
            assert T60_is_valid, 'Invalid T60 specified in room_cfg'

        room_cfg = {
            'room_sz': room_sz,
            'T60_sabine': T60_sabine,
            'beta': beta,
            'T60_specify': T60_specify,
            # 'abs_weights': abs_weights
        }
        return room_cfg

    def _beta_Sabine_estimation(self,room_sz, T60, abs_weights=[1.0]*6):	
        '''  Estimation of the reflection coefficients needed to have the desired reverberation time.
        '''
        def t60error(x, T60, room_sz, abs_weights):
            alpha = x * abs_weights
            Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
                (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
                (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
            V = np.prod(room_sz)
            if Sa == 0: return T60 - 0 # Anechoic chamber 
            return abs(T60 - 0.161 * V / Sa) # Sabine's formula
        
        abs_weights /= np.array(abs_weights).max()
        result = minimize(t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]])		
        return np.sqrt(1 - result.x * abs_weights).astype(np.float32)
       
    def _t60_is_valid(self, room_sz, T60, alpha, c, ism_db, th=0.005, eps=1e-4):
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
             (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
             (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: # avoid cases when all walls are reflective and T60 (from sabine eq) is extremely large
            valid_flag = False 
        else:
            T60_sabine = 0.161 * V / (Sa+eps) 
            valid_flag = bool(abs(T60-T60_sabine)<th)  # avoid cases when T60<T60_min, or the abs_weights is not suited
        beta_prod = np.prod(1-alpha) # avoid sparse reflections 
        max_dist = np.sqrt(room_sz[0]**2 + room_sz[1]**2 + room_sz[2]**2)
        ism_time = ism_db/60*T60_sabine
        ism_constraint = ism_time >= 3*max_dist/c # avoid RIR is too sparse

        return valid_flag & bool(beta_prod!=0) & (ism_constraint), T60_sabine
    
    # def _save_room_configs(self, save_dir, room_range):
    #     if (self.rooms != []):
    #         row_name = ['RoomSize', 'RT60', 'AbsorptionCoefficient',]
    #         csv_name = save_dir + '/Room' + str(room_range[0]) + '-' + str(room_range[1]-1) + '_sz_t60_abs' + '.csv'
    #         with open(csv_name, 'w', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(row_name)
    #             for csv_row in self.rooms:
    #                 writer.writerow(csv_row)

    def random_mic_array(self, mic_array_cfg, array_pos_ratio_range, room_sz):
        # Microphones
        array_pos = []
        for i in range(len(room_sz)):
            lower_bound = array_pos_ratio_range[i][0] * room_sz[i]
            upper_bound = array_pos_ratio_range[i][1] * room_sz[i]
            array_pos.append(np.random.uniform(*(lower_bound, upper_bound))) 
        array_pos = np.array(array_pos)
        array_scale = np.random.uniform(*mic_array_cfg['array_scale_range'])
        array_rotate = np.random.uniform(*mic_array_cfg['array_rotate_azi_range'])
        rotate_matrix = np.array([[np.cos(array_rotate/180*np.pi), -np.sin(array_rotate/180*np.pi), 0], 
                                [np.sin(array_rotate/180*np.pi), np.cos(array_rotate/180*np.pi), 0], 
                                [0, 0, 1]]) # (3, 3)
        mic_pos_rotate = np.dot(rotate_matrix, mic_array_cfg['mic_pos_relative'].transpose(1, 0)).transpose(1, 0)
        mic_pos = array_pos + mic_pos_rotate * array_scale # (nch,3)
        mic_orV = np.dot(rotate_matrix, mic_array_cfg['mic_orV'].transpose(1, 0)).transpose(1, 0)
        orV = np.dot(rotate_matrix, mic_array_cfg['array_orV'])

        mic_array_cfg = {'array_type':mic_array_cfg['array_type'],
                        'mic_pos': mic_pos, 
                        'array_scale': array_scale, 
                        'array_rotate_azi': array_rotate, 
                        'mic_orV': mic_orV,
                        'mic_pattern': mic_array_cfg['mic_pattern'],
                        'array_orV': orV,
                        'array_pos': array_pos,
                        }
        return mic_array_cfg

    def random_src_trajectory(self, 
                              num_source_range, 
                              source_state,
                              min_src_array_dist, 
                              min_src_boundary_dist, 
                              array_pos, 
                              array_orV, 
                              mic_array_cfg, 
                              room_sz, 
                              nb_points,
                              traj_pt_mode='time'):
        num_source = np.random.randint(num_source_range[0], num_source_range[-1]+1)
        traj_pts = [] # (nb_points, 3, num_source)
        array_pos = mic_array_cfg['array_pos']
        for source_idx in range(num_source):
            src_pos_min = np.array([0.0, 0.0, 0.0]) + np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            src_pos_max = room_sz - np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            if mic_array_cfg['array_type'] == 'planar_linear':   
                # suited cases: annotation is symetric about end-fire direction (like TDOA), front-back confusion not exists, half-plane is set
                # src can be at any 3D point in half-plane space
                if np.sum(mic_array_cfg['array_orV']) > 0:
                    src_pos_min[np.nonzero(mic_array_cfg['array_orV'])] = array_pos[np.nonzero(mic_array_cfg['array_orV'])] 
                    src_pos_min += min_src_array_dist * np.abs(mic_array_cfg['array_orV'])
                else:
                    src_pos_max[np.nonzero(mic_array_cfg['array_orV'])] = array_pos[np.nonzero(mic_array_cfg['array_orV'])] 
                    src_pos_max -= min_src_array_dist * np.abs(mic_array_cfg['array_orV'])

            # if mic_array_cfg['array_type'] == 'planar_linear':   
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion exists, half-plane is set
            #     # src can be at any planar point in half-plane space
            #     orv_list = list(mic_array_cfg['array_orV'])
            #     assert (orv_list==[0,1,0]) | (orv_list==[0,-1,0]) | (orv_list==[1,0,0]) | (orv_list==[-1,0,0]), 'array orientation must along x or y axis'
            #     if mic_array_cfg['array_orV'][0] == 1:
            #         src_pos_min[0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0])
            #     elif mic_array_cfg['array_orV'][0] == -1:
            #         src_pos_max[0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0])
            #     elif mic_array_cfg['array_orV'][1] == 1:
            #         src_pos_min[1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1])
            #     elif mic_array_cfg['array_orV'][1] == -1:
            #         src_pos_max[1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1])
            #     src_pos_min[2] = np.maximum(array_pos[2] - 0.0, src_pos_min[2])
            #     src_pos_max[2] = np.minimum(array_pos[2] + 0.0, src_pos_max[2])
 
            # elif mic_array_cfg['array_type'] == 'planar': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at any planar point in the all-plane space
            #     assert mic_array_cfg['array_rotate_azi'] == 0, 'array rotate must be 0'
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
                
            #     # directions = {}
            #     # for direction in direction_candidates:
            #     #     directions[direction] = [src_pos_min, src_pos_max]
            #     #     if direction == 'x':
            #     #         directions[direction][0][0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0]) # src_pos_min[0]
            #     #     elif direction == '-x':
            #     #         directions[direction][1][0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0]) # src_pos_max[0]
            #     #     elif direction == 'y':
            #     #         directions[direction][0][1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1]) # src_pos_min[1]
            #     #     elif direction == '-y':
            #     #         directions[direction][1][1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1]) # src_pos_max[1]
            #     #     else:
            #     #         raise Exception('Unrecognized direction~')
            #     # src_pos_min[2] = np.maximum(array_pos[2] - 0.0, src_pos_min[2])
            #     # src_pos_max[2] = np.minimum(array_pos[2] + 0.0, src_pos_max[2])
            #     # # src_pos = np.concatenate((src_pos_min[np.newaxis, :], src_pos_max[np.newaxis, :]), axis=0)
            #     # direction = random.sample(direction_candidates, 1)[0]
            #     # src_pos_min = copy.deepcopy(directions[direction][0])
            #     # src_pos_max = copy.deepcopy(directions[direction][1])
                
            # elif mic_array_cfg['array_type'] == '3D': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at some 3D point in the all-plane space
            #     assert mic_array_cfg['array_rotate_azi'] == 0, 'array rotate must be 0'
            #     direction_candidates = ['x', 'y', '-x', '-y']
            #     direction = random.sample(direction_candidates, 1)[0]
            #     if direction == 'x':
            #         src_pos_min[0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0])
            #     elif direction == '-x':
            #         src_pos_max[0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0]) 
            #     elif direction == 'y':
            #         src_pos_min[1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1])
            #     elif direction == '-y':
            #         src_pos_max[1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1])  
            #     else:
            #         raise Exception('Unrecognized direction~')
            #     src_array_relative_height = 0.5
            #     src_pos_min[2] = np.maximum(array_pos[2] - src_array_relative_height, src_pos_min[2]) 
            #     src_pos_max[2] = np.minimum(array_pos[2] + src_array_relative_height, src_pos_max[2])

            else:
                raise Exception('Undefined array type~')

            # gurantee array_pos_ratio_range[i][0]*room_sz[i] >= min_src_array_dist + min_src_boundary_dist
            # array_pos_ratio_range[i][1]*room_sz[i] >= min_src_array_dist + min_src_boundary_dist
            for i in range(3):
                assert src_pos_min[i]<=src_pos_max[i], 'Src postion range error: '+str(src_pos_min[i])+ '>' + str(src_pos_max[i]) + '(array boundary dist >= src boundary dist + src array dist)'
            
            if source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts_eachsource = np.ones((nb_points, 1)) * src_pos

            # elif source_state == 'moving_longduration': # 3D sinusoidal trjactories
            #     T_eachTrajPart_range = [4, 6] # in seconds
            #     # when room_size_range = [[3, 3, 2.5], [10, 8, 6]], array_pos_ratio_range = [[0.3, 0.3, 0.2], [0.7, 0.7, 0.5]], 
            #     # min_src_array_dist = 0.5, min_src_boundary_dist = 0.3, 
            #     # the maximumum spk speed is 1ms/s-2.5m/s for Teach=4, 0.8m/s-2m/s for Teach=5, 0.67-1.6m/s for Teach=6
            #     desired_T_interval = 0.1 # consistent with run.py
            #     traj_pts_eachsource = np.zeros((1, 3))
            #     traj_pts_eachsource[0, :] = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
            #     nb_points_total = traj_pts_eachsource.shape[0]
            #     while (nb_points_total < nb_points):
            #         T_eachTrajPart = np.random.uniform(T_eachTrajPart_range[0], T_eachTrajPart_range[1])
            #         npt_eachTraj = int(T_eachTrajPart / desired_T_interval)
            #         src_pos_ini = copy.deepcopy(traj_pts_eachsource[-1, :])
            #         src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

            #         Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
            #                                 src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
            #         A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
            #         if traj_pt_mode == 'time': # Specify nb_points according to time 
            #             w = 2*np.pi / npt_eachTraj * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
            #             line_pts = np.array([np.linspace(i,j,npt_eachTraj) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
            #             osc_pts = A * np.sin(w * np.arange(npt_eachTraj)[:, np.newaxis])
            #             traj_pts_eachsource = np.concatenate((traj_pts_eachsource, line_pts + osc_pts), axis=0) # (nbpoints, 3) 
            #             nb_points_total = traj_pts_eachsource.shape[0]
            #         # print(src_pos_ini, src_pos_end, line_pts[0,:] + osc_pts[0,:],line_pts[-1,:] + osc_pts[-1,:]) 

            #     traj_pts_eachsource = traj_pts_eachsource[0:nb_points, :]

            elif source_state == 'moving': # 3D sinusoidal trjactories
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

                Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
                if traj_pt_mode == 'time': # Specify nb_points according to time 
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    line_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    osc_pts = A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                    traj_pts_eachsource = line_pts + osc_pts # (nbpoints, 3)
                
                elif traj_pt_mode == 'distance_line': # Specify nb_points according to line distance (pointing src_pos_end from src_pos_ini) 
                    desired_dist = 0.1 # between ajacent points
                    nb_points = int(np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//desired_dist + 1) # adaptive number of points, namely one point per liner 10cm
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    line_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    osc_pts = A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                    traj_pts_eachsource = line_pts + osc_pts
                
                elif traj_pt_mode == 'distance_sin': # Specify nb_points according to direct sin distance
                    desired_dist = 0.1 # between ajacent points
                    src_ini_end_dist = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)
                    src_ini_end_dirc_vec = (src_pos_end - src_pos_ini)/src_ini_end_dist
                    w = 2*np.pi / src_ini_end_dist * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts_eachsource = []
                    line_pts = []
                    current_dist_along_dirc_vec = 0
                    while current_dist_along_dirc_vec < src_ini_end_dist:
                        # store current point
                        osc = A * np.sin(w * current_dist_along_dirc_vec)
                        line = src_pos_ini + src_ini_end_dirc_vec * current_dist_along_dirc_vec
                        pos0 = line + osc
                        traj_pts_eachsource.append(pos0) 
                        line_pts.append(line)

                        # find next point
                        for factor in [1.0, 1.5, 3]:
                            res = minimize(self._dist_err_func, x0=[desired_dist / 10], bounds=[(0, desired_dist * factor)], 
                                           tol=desired_dist / 100, args=(current_dist_along_dirc_vec, src_ini_end_dirc_vec, src_pos_ini, pos0, desired_dist, A, w))
                            if res.fun < desired_dist / 100:
                                break
                        current_dist_along_dirc_vec = current_dist_along_dirc_vec + res.x[0]
                    traj_pts_eachsource = np.array(traj_pts_eachsource)
                    line_pts = np.array(line_pts)
                
            traj_pts += [traj_pts_eachsource]
        traj_pts = np.array(traj_pts).transpose(1, 2, 0)

        if traj_pt_mode != 'distance_sin':
            src_traj_cfg = {'src_traj_pts': traj_pts, 
                            # 'src_traj_pt_mode': traj_pt_mode,
                            # 'num_source': num_source,
                            }
        else:
            src_traj_cfg = {'src_traj_pts': traj_pts, 
                            'line_pts': line_pts, 
                            # 'src_traj_pt_mode': traj_pt_mode,
                            # 'num_source': num_source,
                            }
        return src_traj_cfg
        
            
    def _dist_err_func(self, delta_dist_along_dirc_vec, current_dist_along_dirc_vec, ini_end_dirc_vec, pos_ini, pos0, desired_dist, A, w):
        osc = A * np.sin(w * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec))
        line = pos_ini + ini_end_dirc_vec * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec) 
        pos_current = line + osc
        dist = np.sqrt(np.sum((pos_current - pos0)**2))
        return np.abs(dist - desired_dist)
    
    def plot_room(self, room_sz, traj_pts, mic_pos, view='3D', save_path=None):
        """ Plots the source trajectory and the microphones within the room
            Args:   traj_pts - (npt, 3, nsrc)
                    mic_pos  - (nmic, 3)
        """
        assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

        plt.switch_backend('agg')
        fig = plt.figure()
        nsrc = traj_pts.shape[-1]

        if view == '3D' or view == 'XYZ':
            ax = Axes3D(fig)
            ax.set_xlim3d(0, room_sz[0])
            ax.set_ylim3d(0, room_sz[1])
            ax.set_zlim3d(0, room_sz[2])
            ax.scatter(mic_pos[:,0], mic_pos[:,1], mic_pos[:,2])
            legends = ['Microphone array']
            for src_idx in range(nsrc):
                ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
                ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
                ax.text(traj_pts[-1,0,src_idx], traj_pts[-1,1,src_idx], traj_pts[-1,2,src_idx], 'end')
                legends += ['Source trajectory ' + str(src_idx)]
            ax.legend(legends)

            # ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(T60, SNR))
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

        else:
            ax = fig.add_subplot(111)
            plt.gca().set_aspect('equal', adjustable='box')

            if view == 'XY':
                ax.set_xlim(0, room_sz[0])
                ax.set_ylim(0, room_sz[1])
                ax.scatter(mic_pos[:,0], mic_pos[:,1])
                legends = ['Microphone array']
                for src_idx in range(nsrc):
                    ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,1,src_idx])
                    ax.text(traj_pts[0,0,src_idx], traj_pts[0,1,src_idx], 'start')
                    ax.text(traj_pts[-1,0,src_idx], traj_pts[-1,1,src_idx], 'end')
                    legends += ['Source trajectory ' + str(src_idx)]
                ax.legend(legends)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
            elif view == 'XZ':
                ax.set_xlim(0, room_sz[0])
                ax.set_ylim(0, room_sz[2])
                ax.scatter(mic_pos[:,0], mic_pos[:,2])
                legends = ['Microphone array']
                for src_idx in range(nsrc):
                    ax.scatter(traj_pts[:,0,src_idx], traj_pts[:,2,src_idx])
                    ax.text(traj_pts[0,0,src_idx], traj_pts[0,2,src_idx], 'start')
                    ax.text(traj_pts[-1,0,src_idx], traj_pts[-1,2,src_idx], 'end')
                    legends += ['Source trajectory ' + str(src_idx)]
                ax.legend(legends)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('z [m]')
            elif view == 'YZ':
                ax.set_xlim(0, room_sz[1])
                ax.set_ylim(0, room_sz[2])
                ax.scatter(mic_pos[:,1], mic_pos[:,2])
                legends = ['Microphone array']
                for src_idx in range(nsrc):
                    ax.scatter(traj_pts[:,1,src_idx], traj_pts[:,2,src_idx])
                    ax.text(traj_pts[0,1,src_idx], traj_pts[0,2,src_idx], 'start')
                    ax.text(traj_pts[-1,1,src_idx], traj_pts[-1,2,src_idx], 'end')
                    legends += ['Source trajectory ' + str(src_idx)]
                ax.legend(legends)
                ax.set_xlabel('y [m]')
                ax.set_ylabel('z [m]')

        # plt.show()
        if save_path is not None: 
            plt.savefig(save_path + 'scene')
        plt.close()
    

# Room impulse response
class RoomImpulseResponse():
    """ Generate room impulse response (RIR)
    """
    def __init__(self, fs, c, ism_db):
        self.fs = fs
        self.c = c
        self.ism_db = ism_db
    
    def generate_rir(self, room_sz, beta, T60, mic_pos, mic_orV, mic_pattern, src_traj_pts, dp_gen=False):
        import gpuRIR

        if (T60 == 0) | dp_gen:  # For direct path signal
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(self.ism_db, T60) # Use ISM until the RIRs decay ism_db dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )

        rir = []
        num_source = src_traj_pts.shape[-1]
        for source_idx in range(num_source):
            rir_per_src = gpuRIR.simulateRIR(
                room_sz=room_sz, 
                beta=beta, 
                pos_src=src_traj_pts[:,:,source_idx], 
                pos_rcv=mic_pos,
                nb_img=nb_img, 
                Tmax=Tmax, 
                fs=self.fs, 
                Tdiff=Tdiff, 
                orV_rcv=mic_orV,
                mic_pattern=mic_pattern, 
                c=self.c)
            rir += [rir_per_src]

        rir = np.array(rir).transpose(1,2,3,0) # (npt,nch,nsample,nsrc)

        return rir

    def check_rir(self, rir):
        ok_flag = True
        nan_flag = np.isnan(rir)
        inf_flag = np.isinf(rir)
        if (True in nan_flag):
            warnings.warn('NAN exists in RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(rir**2) == 0)
        if zero_flag:
            warnings.warn('RIR is all zeros~')
            ok_flag = False
        return ok_flag

    def check_rir_envelope(self, rir, t60_specify, fs):
        # additional check for T60 (or RIR envelope)
        t60_edc = []
        corr_edc = []
        for mic_idx in range(rir.shape[1]):
            t60, corr = self.__rt60_from_rirs(rir[0, mic_idx, :, 0], fs)
            t60_edc += [t60]
            corr_edc += [corr]
        t60_edc = np.mean(t60_edc)
        corr_edc = np.mean(corr)
        # if (abs(corr_edc)<0.98):
        #     print('Correlation coefficient (<0.98):', R, 'T60 calculated (EDC):', T60, 'T60 specified:', T60_gt)
        ok_flag = bool(abs(t60_edc-t60_specify)<0.05) & bool(abs(corr_edc)>0.5)

        return ok_flag, t60_edc

    def __rt60_from_rirs(self, h, Fs, vis=False):
        """ 
            Refs: https://github.com/Chutlhu/dEchorate
        """
        edc = self.__cal_edc(h)
        t60, corr = self.__cal_rt60(edc, Fs, vis=vis)
        
        return t60, corr
    
    def __cal_edc(self,RIR, eps=1e-10):
        # Schroeder Integration method
        max_idx = np.argmax(RIR)
        EDC = 10.0 * np.log10(np.cumsum(RIR[::-1]**2)[::-1]/(np.sum(RIR[max_idx:]**2)+eps)+eps)

        return EDC

    def __cal_rt60(self, EDC, fs, edc_st_list=list(range(-5,-20,-2)), edc_duration_list=list(range(-10,-30,-2)), vis=False, eps=1e-10):
        """ add extra [edc_st, edc_ed] pairs
        """
        
        def find_nearest_value(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        t60_list = []
        r_list = []
        x_list = []
        y_list = []
        for edc_st0 in edc_st_list:
            for edc_duration in edc_duration_list: 
                edc_st = find_nearest_value(EDC, edc_st0)
                edc_st = np.where(EDC == edc_st)[0][0]

                edc_ed = find_nearest_value(EDC, edc_st0+edc_duration)
                edc_ed = np.where(EDC == edc_ed)[0][0]
                
                # Perform linear regression
                if abs(edc_st-edc_ed)>1:
                    times = np.arange(len(EDC))/fs
                    x = times[edc_st:edc_ed]
                    y = EDC[edc_st:edc_ed]
    
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
                    # assert (slope != np.inf) & (slope != np.nan), 'inf/nan exists~'
                    t60_list += [-60/(slope+eps)]
                    r_list += [r_value]
                    x_list += [x]
                    y_list += [y]
                else:
                    t60_list += [np.nan]
                    r_list += [0]
                    x_list += [np.nan]
                    y_list += [np.nan]

        # Compute the final value
        idx = np.argmax(abs(np.array(r_list)))
        corr = r_list[idx]
        t60 = t60_list[idx]
        x = x_list[idx]
        y = y_list[idx]

        if vis:
            plt.switch_backend('agg')
            plt_all = plt.scatter(times, EDC, label='all', c='lightgray', marker='.', linewidth=1, zorder=10)
            plt_used = plt.scatter(x, y, label='used', c='whitesmoke', marker='.', linewidth=1, zorder=10)
            plt.legend(handles = [plt_all, plt_used])
            plt.xlabel('Time sample')
            plt.ylabel('Value')
            plt.savefig('edc_curve')
            # plt.show()

        return t60, corr

    def rir_conv_src(self, rir, src_signal, gpu_conv=False):
        ''' rir : (npt,nch,nsam,nsrc)
        '''
        if gpu_conv:
            import gpuRIR

        # Source conv. RIR
        mic_signal_srcs = []
        num_source = rir.shape[-1]
        nsample = src_signal.shape[0]
        for source_idx in range(num_source):
            rir_per_src = rir[:, :, :, source_idx]  # (npt,nch,nsample）
            npts = rir_per_src.shape[0]

            if gpu_conv:
                timestamps = np.arange(npts) / npts * nsample / self.fs 
                mic_sig_per_src = gpuRIR.simulateTrajectory(src_signal[:, source_idx], rir_per_src, timestamps=timestamps, fs=self.fs)
                mic_sig_per_src = mic_sig_per_src[0:nsample, :]

            else:
                if rir_per_src.shape[0] == 1:
                    mic_sig_per_src = self._conv(sou_sig=src_signal[:, source_idx], rir=rir_per_src[0, :, :].transpose(1, 0))
                
                else: # to be written
                    mic_sig_per_src = 0  
                    raise Exception('Uncomplete code for RIR-Source-Conv for moving source')
                    # mixeventsig = 481.6989*ctf_ltv_direct(src_signal[:, source_idx], RIRs[:, :, riridx], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))

            mic_signal_srcs += [mic_sig_per_src]

        mic_signal_srcs = np.array(mic_signal_srcs).transpose(1, 2, 0)  # (nsample,nch,nsrc)
        mic_signal = np.sum(mic_signal_srcs, axis=2)  # (nsample, nch) 
 
        return mic_signal, mic_signal_srcs
    
    def _conv(self, sou_sig, rir):
        """ Perform convolution between source signal and room impulse reponses (RIRs)
            Args:       sou_sig   - source signal (nsample, )
                        rir       - multi-channel RIR (nrirsample, nch)
            Returns:    mic_sig   - multi-channel microphone signals (nsample, nch)
        """ 
        nsample = sou_sig.shape[0]

        # mic_sig_temp = scipy.signal.convolve(sou_sig[:, np.newaxis], rir, mode='full', method='fft')
        mic_sig_temp = scipy.signal.fftconvolve(sou_sig[:, np.newaxis], rir, mode='full', axes=0)
        mic_sig = mic_sig_temp[0:nsample, :]
        
        return mic_sig


class MicrophoneSignalOrRIR():
    """ Generate microphone signal or room impulse response (RIR) according to the given acoustic configurations.
    """
    def __init__(self, ):
        pass

    def generate_rir(
        self, 
        idx,
        sa_cfgs, 
        fs,
        c,
        roomir, 
        save_to,
        ):

        sa_cfg = sa_cfgs[idx]

        # Generate room impulse response
        rir_is_ok = False
        while(rir_is_ok==False):
        # generate room impulse response
            rir = roomir.generate_rir(
                room_sz=sa_cfg['room_sz'], 
                beta = sa_cfg['beta'], 
                T60 = sa_cfg['T60_sabine'], 
                mic_pos = sa_cfg['mic_pos'], 
                mic_orV = sa_cfg['mic_orV'], 
                mic_pattern = sa_cfg['mic_pattern'], 
                src_traj_pts=sa_cfg['src_traj_pts'], 
                )
            rir_dp = roomir.generate_rir(
                room_sz=sa_cfg['room_sz'], 
                beta = sa_cfg['beta'], 
                T60 = sa_cfg['T60_sabine'], 
                mic_pos = sa_cfg['mic_pos'], 
                mic_orV = sa_cfg['mic_orV'], 
                mic_pattern = sa_cfg['mic_pattern'], 
                src_traj_pts=sa_cfg['src_traj_pts'], 
                dp_gen=True
                )
            
            # check whether the rir is valid
            rir_is_ok = roomir.check_rir(rir) & roomir.check_rir(rir_dp)
            if rir_is_ok:
                env_is_ok, T60_edc = roomir.check_rir_envelope(rir, sa_cfg['T60_specify'], fs)
                rir_is_ok = rir_is_ok & env_is_ok
        # assert rir_is_ok==True, "RIR is not valid, please check the parameters."
        sa_cfg['T60_edc'] = T60_edc
        
        # Get annotation    
        annos = self.generate_annotation(
            traj_pts=sa_cfg['src_traj_pts'],
            array_pos=sa_cfg['array_pos'],
            mic_pos=sa_cfg['mic_pos'],
            rir_srcs=rir,
            rir_srcs_dp=rir_dp,
            DOA=False, 
            TDOA=True,                             
            DRR=True,
            C50=True, 
            C80=False,
            mic_vad=False, 
            source_vad=None, 
            mic_sig=False, 
            mic_sig_dp=False,
            mic_signal_srcs_dp=False,
            gpu_conv=False,
            pt2sample=False,
            src_single_static=True,
            fs=fs,
            c=c)
        
        # Save data
        Path(save_to).mkdir(parents=True, exist_ok=True)
        save_to_file = os.path.join(save_to, str(idx) + f'.npy')
        np.save(save_to_file, arr=rir.astype(np.float32))
        save_to_file = os.path.join(save_to, str(idx) + f'_dp.npy')
        np.save(save_to_file, arr=rir_dp.astype(np.float32))
        # np.savez_compressed(save_to_file, arr=rir_dp.astype(np.float32))
        save_to_file = os.path.join(save_to, str(idx) + f'_info.npz')
        np.savez(save_to_file, **{**sa_cfg, **annos, **{'fs':fs}})

    def generate_microphone_signal(
        self, 
        idx,
        sa_cfgs, 
        fs,
        c,
        roomir, 
        srcdataset, 
        noidataset,
        snr_range, 
        save_to,
        save_dp=False,
        gpu_conv=False,
        seed=1,
        ):
        # print(seed+idx)
        np.random.seed(seed=seed+idx)
        sa_cfg = sa_cfgs[idx]

        # Generate room impulse response
        rir_is_ok = False
        while(rir_is_ok==False):
            # generate room impulse response
            rir = roomir.generate_rir(
                room_sz=sa_cfg['room_sz'], 
                beta = sa_cfg['beta'], 
                T60 = sa_cfg['T60_sabine'], 
                mic_pos = sa_cfg['mic_pos'], 
                mic_orV = sa_cfg['mic_orV'], 
                mic_pattern = sa_cfg['mic_pattern'], 
                src_traj_pts=sa_cfg['src_traj_pts'], 
                )
            rir_dp = roomir.generate_rir(
                room_sz=sa_cfg['room_sz'], 
                beta = sa_cfg['beta'], 
                T60 = sa_cfg['T60_sabine'], 
                mic_pos = sa_cfg['mic_pos'], 
                mic_orV = sa_cfg['mic_orV'], 
                mic_pattern = sa_cfg['mic_pattern'], 
                src_traj_pts=sa_cfg['src_traj_pts'], 
                dp_gen=True
                )
            
            # check whether the rir is valid
            rir_is_ok = roomir.check_rir(rir) & roomir.check_rir(rir_dp)
            if rir_is_ok:
                env_is_ok, T60_edc = roomir.check_rir_envelope(rir, sa_cfg['T60_specify'], fs)
                rir_is_ok = rir_is_ok & env_is_ok
        # assert rir_is_ok, "RIR is not valid, please check the parameters."
        sa_cfg['T60_edc'] = T60_edc
        
        # Get source signal
        src_idx = np.random.randint(0, srcdataset.__len__())
        src_sig = srcdataset.__getitem__(src_idx)
        src_sig = src_sig[:, 0:sa_cfg['src_traj_pts'].shape[-1]]
        sa_cfg['src_idx'] = src_idx

        # Generate clean or direct-path microphone signal
        mic_sig_clean, mic_sig_srcs_clean = roomir.rir_conv_src(rir, src_sig, gpu_conv=False)
        mic_sig_dp, mic_sig_srcs_dp = roomir.rir_conv_src(rir_dp, src_sig, gpu_conv=False)

        # Generate noise signal
        noi_sig = noidataset.generate_random_noise(mic_pos=sa_cfg['mic_pos'])
        snr = np.random.uniform(*snr_range)

        # Generate noisy microphone signal
        mic_sig = noidataset.add_noise(mic_sig_clean, noi_sig, snr, mic_sig_dp=mic_sig_dp) 
        sa_cfg['SNR'] = snr

        # Check whether the values of microphone signals is in the range of [-1, 1] for wav saving (soundfile.write)
        max_value = np.max(mic_sig)
        min_value = np.min(mic_sig)
        max_value_dp = np.max(mic_sig_dp)
        min_value_dp = np.min(mic_sig_dp)
        value = np.max([np.abs(max_value), np.abs(min_value), np.abs(max_value_dp), np.abs(min_value_dp)])
        mic_sig = mic_sig / value *0.9
        mic_sig_dp = mic_sig_dp / value *0.9
        mic_sig_srcs_clean = mic_sig_srcs_clean / value *0.9
        mic_sig_srcs_dp = mic_sig_srcs_dp / value *0.9

        # Get annotation
        annos = self.generate_annotation(
            traj_pts=sa_cfg['src_traj_pts'],
            array_pos=sa_cfg['array_pos'],
            mic_pos=sa_cfg['mic_pos'],
            rir_srcs=rir,
            rir_srcs_dp=rir_dp,
            DOA=False, 
            TDOA=True,                                
            DRR=True,
            C50=True, 
            C80=False,
            mic_vad=False, 
            source_vad=None,
            mic_sig=False, 
            mic_sig_dp=False,
            mic_signal_srcs_dp=False,
            gpu_conv=False,
            pt2sample=False,
            src_single_static=True,
            fs=fs,
            c=c
            )

        # Save data
        Path(save_to).mkdir(parents=True, exist_ok=True)
        save_to_file = os.path.join(save_to, str(idx) + f'.wav')
        soundfile.write(save_to_file, mic_sig, fs)
        if save_dp:
            save_to_file = os.path.join(save_to, str(idx) + f'_dp.wav')
            soundfile.write(save_to_file, mic_sig_dp, fs)
        save_to_file = os.path.join(save_to, str(idx) + f'_info.npz')
        np.savez(save_to_file, **{**sa_cfg, **annos})
 
    def generate_annotation(
        self,
        traj_pts,
        array_pos,
        mic_pos,
        rir_srcs=False,
        rir_srcs_dp=False,
        DOA=False, 
        TDOA=False,                             
        DRR=False,
        C50=False, 
        C80=False,
        mic_vad=False, # [False, 'dp_ratio', 'src_webrtc']
        source_vad=None, 
        mic_sig=False, 
        mic_sig_dp=False,
        mic_signal_srcs_dp=False,
        gpu_conv=False,
        pt2sample=False,
        src_single_static=False,
        fs=16000,
        c=343,
        eps=1e-8,
        ):
        
        annos = {}
        npt, _, num_source = traj_pts.shape 
        if pt2sample:
            nsample = mic_sig.shape[0]
            t = np.arange(nsample) / fs
            timestamps = np.arange(npt) * nsample / fs / npt
            annos['timestamps'] = timestamps
        if DOA:  # [ele, azi]
            if pt2sample:
                DOA = np.zeros((nsample, 2, num_source), dtype=np.float32)  # (nsample, 2, nsrc)
                for source_idx in range(num_source):
                    trajectory = np.array([np.interp(t, timestamps, traj_pts[:, i, source_idx]) for i in range(3)]).transpose()
                    DOA[:, :, source_idx] = self._cart2sph(trajectory - array_pos)[:, [1,0]]
                annos['DOA'] = DOA
            else:
                DOA = np.zeros((npt, 2, num_source), dtype=np.float32)  # (nb_point, 2, nsrc)
                for source_idx in range(num_source):
                    DOA[:, :, source_idx] = self._cart2sph(traj_pts[:, :, source_idx] - array_pos)[:, [1,0]]
                annos['DOA'] = DOA
                if src_single_static:
                    annos['DOA'] = DOA[0,0,0]

        if TDOA: 
            nmic = mic_pos.shape[-2]
            # if (traj_pts==0).all(): # for ACE, static source, discrete values
            #     TDOA = np.zeros((nsample, nmic-1, num_source))  # (nsample,nch-1,nsrc)
            #     nsample_find_dp = 200
            #     for source_idx in range(num_source):
            #         for ch_idx in range(1, nmic):
            #             m_dp_sample, m_dp_value, _ = find_dp_from_rir(rir_srcs[0, ch_idx, 0:nsample_find_dp, source_idx])
            #             ref_dp_sample, ref_dp_value, _ = find_dp_from_rir(rir_srcs[0, 0, 0:nsample_find_dp, source_idx])
            #             assert m_dp_sample <=100, m_dp_sample # for ACE, the silece at begining is removed
            #             TDOA[:, ch_idx-1, source_idx] = (m_dp_sample - ref_dp_sample) / fs
                        
            # else: # for simulated data, continuous values
            if len(mic_pos.shape) == 2:
                mic_pos = np.tile(mic_pos[np.newaxis, :, :], (npt, 1, 1))
            elif len(mic_pos.shape) == 3:
                pass
            else:
                raise Exception('shape of mic_pos is out of range~')
            corr_diff = np.tile(traj_pts[:, np.newaxis, :, :], (1, nmic, 1, 1)) - np.tile(mic_pos[:, :, :, np.newaxis], (1, 1, 1, num_source))
            dist = np.sqrt(np.sum(corr_diff**2, axis=2))  # (npt,3,nsrc)-(nch,3)=(npt,nch,3,nsrc)
            re_dist = dist[:, 1:, :] - np.tile(dist[:, 0:1, :], (1, nmic - 1, 1))  # (npt,nch-1,nsrc)
            TDOA = re_dist / c  # (npt,nch-1,nsrc)
            if pt2sample:
                annos['TDOA'] = np.zeros((nsample, TDOA.shape[1], num_source))  # (nsample,nch-1,nsrc)
                for source_idx in range(num_source):
                    for ch_idx in range(annos['TDOA'].shape[1]):
                        annos['TDOA'][:, ch_idx, source_idx] = np.interp(t, timestamps, TDOA[:, ch_idx, source_idx])
            else:
                annos['TDOA'] = TDOA 
                if src_single_static:
                    annos['TDOA'] = TDOA[0,0,0]
            annos['TDOA'] = annos['TDOA'].astype(np.float32)
            
        if DRR | C50 | C80:
            rir_len = rir_srcs.shape[2]
            dp_rir_len = rir_srcs_dp.shape[2]
            nmic = mic_pos.shape[-2]
            nb_traj_pts = traj_pts.shape[0]
            zeros_pad = np.zeros((nb_traj_pts, nmic, abs(rir_len - dp_rir_len), num_source))
            if rir_len >= dp_rir_len:  # When RT60=0.15s, RIR_len = dp_RIR_len
                rir_srcs_dp_pad = np.concatenate((rir_srcs_dp, zeros_pad), axis=2)  # (npt,nch,nsample,nsrc)
                rir_srcs_pad = rir_srcs
            else:
                rir_srcs_dp_pad = rir_srcs_dp
                rir_srcs_pad = np.concatenate((rir_srcs, zeros_pad), axis=2)  # (npt,nch,nsample,nsrc)

            if DRR:
                nsamp = np.max([dp_rir_len, rir_len])
                nd = np.argmax(rir_srcs_dp_pad, axis=2)  # (npt,nch,nsrc)
                nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npt,nch,nsample,nsrc)
                n0 = int(fs * 2.5 / 1000) * np.ones_like(rir_srcs_pad)
                whole_range = np.array(range(0, nsamp))
                whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (rir_srcs_pad.shape[0], rir_srcs_pad.shape[1], 1, rir_srcs_pad.shape[3]))
                dp_range = (whole_range >= (nd - n0)) & (whole_range <= (nd + n0))
                dp_range = dp_range.astype('float')
                rev_range = np.ones_like(dp_range) - dp_range
                dp_pow = np.sum(rir_srcs_pad**2 * dp_range, axis=2)
                rev_pow = np.sum(rir_srcs_pad**2 * rev_range, axis=2)
                DRR = 10 * np.log10(dp_pow / (rev_pow+eps)+eps)  # (npt,nch,nsrc)
                DRR = DRR[:, 0, :]  # reference channel (npt,nsrc)

                if pt2sample:
                    annos['DRR'] = np.zeros((nsample, num_source))
                    for source_idx in range(num_source):
                        annos['DRR'][:, source_idx] = np.interp(t, timestamps, DRR[:, source_idx])  # (nsample,nsrc)
                        # np.array([np.interp(t, timestamps, DRR[:,i,source_idx]) for i in range(nch)]).transpose() # (nsample,nch,nsrc)
                else:
                    annos['DRR'] = DRR
                    if src_single_static:
                        annos['DRR'] = DRR[0,0]
                annos['DRR'] = annos['DRR'].astype(np.float16)

            if C50:
                nsamp = np.max([dp_rir_len, rir_len])
                nd = np.argmax(rir_srcs_dp_pad, axis=2)  # (npt,nch,nsrc)
                nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npt,nch,nsample,nsrc)
                n0 = int(fs * 50 / 1000) * np.ones_like(rir_srcs_pad)
                whole_range = np.array(range(0, nsamp))
                whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (rir_srcs_pad.shape[0], rir_srcs_pad.shape[1], 1, rir_srcs_pad.shape[3]))
                early_range = (whole_range <= (nd + n0))
                early_range = early_range.astype('float')
                late_range = np.ones_like(early_range) - early_range
                early_pow = np.sum(rir_srcs_pad**2 * early_range, axis=2)
                late_pow = np.sum(rir_srcs_pad**2 * late_range, axis=2)
                C50 = 10 * np.log10(early_pow / (late_pow + eps)+eps)  # (npt,nch,nsrc)
                C50 = C50[:, 0, :]  # reference channel, (npt,nsrc)
                if pt2sample:
                    annos['C50'] = np.zeros((nsample, num_source))
                    for source_idx in range(num_source):
                        annos['C50'][:, source_idx] = np.interp(t, timestamps, C50[:, source_idx])  # (nsample,nsrc)
                else:
                    annos['C50'] = C50
                    if src_single_static:
                        annos['C50'] = C50[0,0]
                annos['C50'] = annos['C50'].astype(np.float16)

            if C80:
                nsamp = np.max([dp_rir_len, rir_len])
                nd = np.argmax(rir_srcs_dp_pad, axis=2)  # (npt,nch,nsrc)
                nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npt,nch,nsample,nsrc)
                n0 = int(fs * 80 / 1000) * np.ones_like(rir_srcs_pad)
                whole_range = np.array(range(0, nsamp))
                whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (rir_srcs_pad.shape[0], rir_srcs_pad.shape[1], 1, rir_srcs_pad.shape[3]))
                # early_range = (whole_range >= (nd - n0)) & (whole_range <= (nd + n0))
                early_range = (whole_range <= (nd + n0))
                early_range = early_range.astype('float')
                late_range = np.ones_like(early_range) - early_range
                early_pow = np.sum(rir_srcs_pad**2 * early_range, axis=2)
                late_pow = np.sum(rir_srcs_pad**2 * late_range, axis=2)
                C80 = 10 * np.log10(early_pow / (late_pow + eps)+eps)  # (npt,nch,nsrc)
                C80 = C80[:, 0, :]  # reference channel, (npt,nsrc)
                if pt2sample:
                    annos['C80']= np.zeros((nsample, num_source))
                    for source_idx in range(num_source):
                        annos['C80'][:, source_idx] = np.interp(t, timestamps, C80[:, source_idx])  # (nsample,nsrc)
                else: 
                    annos['C80'] = C80
                    if src_single_static:
                        annos['C80'] = C80[0,0] 
                annos['C80'] = annos['C80'].astype(np.float16)
        
        assert mic_vad in ['dp_ratio', 'src_webrtc', False], mic_vad
        # Use the signal-to-noise ratio (dp_mic_sig/mic_sig) to compute the VAD
        # must combine with clean silence of source signals, to avoid the random results when a smaller value divided by a smaller value
        # the denominator is the mixed signals of multiple sources, which may be problematic when the number of sources is larger
        # segment-level results approximate to webrtc
        if mic_vad == 'dp_ratio':
            sig_len = mic_sig.shape[0]
            win_len = int(fs * 0.032) # 32ms 
            win_shift_ratio = 1
            nt = int((sig_len - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
            mic_vad_sources = np.zeros((nsample, num_source))
            th = 0.001**2
            for t_idx in range(nt):
                st = int(t_idx * win_len * win_shift_ratio)
                ed = st + win_len 
                dp_mic_signal_sources_sch = mic_signal_srcs_dp[st:ed, 0, :]
                mic_signal_sources_sch = mic_sig[st:ed, 0]
                win_engergy_ratio = np.sum(dp_mic_signal_sources_sch**2, axis=0) / (np.sum(mic_signal_sources_sch**2, axis=0) + eps) 
                mic_vad_sources[st:ed, :] = win_engergy_ratio[np.newaxis, :].repeat(win_len, axis=0) 
            annos['mic_vad_src'] = mic_vad_sources.astype(np.float16)
            # np.sum(mic_vad_sources, axis=1) >= th

        if mic_vad == 'src_webrtc': 
            import gpuRIR
            mic_vad_sources = []  # binary value, for vad of separate sensor signals of sources
            for source_idx in range(num_source):
                if gpu_conv:
                    vad = gpuRIR.simulateTrajectory(source_vad[:, source_idx], rir_srcs_dp[:, :, :, source_idx], timestamps=timestamps, fs=fs)
                vad_sources = vad[0:nsample, :].mean(axis=1) > vad[0:nsample, :].max() * 1e-3
                mic_vad_sources += [vad_sources] 
            mic_vad_sources = np.array(mic_vad_sources).transpose(1, 0)
            annos['mic_vad_src'] = mic_vad_sources.astype(bool)
            # np.sum(mic_vad_sources, axis=1) > 0.5  # binary value, for vad of mixed sensor signals of sources

        return annos
    
    def _cart2sph(self, cart):
        """ cart [x,y,z] → sph [azi,ele,r] (degrees in radian)
        """
        xy2 = cart[:,0]**2 + cart[:,1]**2
        sph = np.zeros_like(cart)
        sph[:,0] = np.arctan2(cart[:,1], cart[:,0])
        sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
        sph[:,2] = np.sqrt(xy2 + cart[:,2]**2)

        return sph

    def _sph2cart(self, sph):
        """ sph [azi,ele,r] → cart [x,y,z] (degrees in radian)
        """
        if sph.shape[-1] == 2: sph = np.concatenate((sph, np.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
        x = sph[..., 2] * np.sin(sph[..., 1]) * np.cos(sph[..., 0])
        y = sph[..., 2] * np.sin(sph[..., 1]) * np.sin(sph[..., 0])
        z = sph[..., 2] * np.cos(sph[..., 1])

        return np.stack((x, y, z)).transpose(1, 0)

class RIRDataset(Dataset):
    def __init__(
        self,
        fs,
        rir_dir_list,
        load_dp=True,
        load_info=True,
        dataset_sz=None):

        if isinstance(rir_dir_list, list):
            self.rir_files = []
            for rir_dir in rir_dir_list:
                self.rir_files += list(Path(rir_dir).rglob('*_dp.npy'))
        else:
            self.rir_files = list(Path(rir_dir_list).rglob('*_dp.npy'))
        self.load_dp = load_dp
        self.load_info = load_info
        self.fs = fs
        self.dataset_sz = len(self.rir_files) if dataset_sz is None else dataset_sz

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        rir_dp_file = str(self.rir_files[idx])
        rir_file = rir_dp_file.replace('_dp.npy', '.npy')
        rir = np.load(rir_file).astype(np.float32)
        info_file = rir_file.replace('.npy', '_info.npz')
        info = np.load(info_file)
        if self.fs!= info['fs']:
            rir = scipy.signal.resample_poly(rir, self.fs, info['fs'])
        return_data = [rir]
        if self.load_dp:
            rir_dp = np.load(rir_dp_file)
            if self.fs!= info['fs']:
                rir_dp = scipy.signal.resample_poly(rir_dp, self.fs, info['fs'])
            return_data.append(rir_dp)
        if self.load_info:
            return_data.append(info)

        return return_data

    def rir_conv_src(self, rir, src_signal, gpu_conv=False):
        ''' rir : (npt,nch,nsam,nsrc)
        '''
        if gpu_conv:
            import gpuRIR

        # Source conv. RIR
        mic_signal_srcs = []
        num_source = rir.shape[-1]
        nsample = src_signal.shape[0]
        for source_idx in range(num_source):
            rir_per_src = rir[:, :, :, source_idx]  # (npt,nch,nsample）
            npts = rir_per_src.shape[0]

            if gpu_conv:
                timestamps = np.arange(npts) / npts * nsample / self.fs 
                mic_sig_per_src = gpuRIR.simulateTrajectory(src_signal[:, source_idx], rir_per_src, timestamps=timestamps, fs=self.fs)
                mic_sig_per_src = mic_sig_per_src[0:nsample, :]

            else:
                if rir_per_src.shape[0] == 1:
                    mic_sig_per_src = self._conv(sou_sig=src_signal[:, source_idx], rir=rir_per_src[0, :, :].transpose(1, 0))
                
                else: # to be written
                    mic_sig_per_src = 0  
                    raise Exception('Uncomplete code for RIR-Source-Conv for moving source')
                    # mixeventsig = 481.6989*ctf_ltv_direct(src_signal[:, source_idx], RIRs[:, :, riridx], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))

            mic_signal_srcs += [mic_sig_per_src]

        mic_signal_srcs = np.array(mic_signal_srcs).transpose(1, 2, 0)  # (nsample,nch,nsrc)
        mic_signal = np.sum(mic_signal_srcs, axis=2)  # (nsample, nch) 
 
        return mic_signal, mic_signal_srcs    

    def _conv(self, sou_sig, rir):
        """ Perform convolution between source signal and room impulse reponses (RIRs)
            Args:       sou_sig   - source signal (nsample, )
                        rir       - multi-channel RIR (nrirsample, nch)
            Returns:    mic_sig   - multi-channel microphone signals (nsample, nch)
        """ 
        nsample = sou_sig.shape[0]

        # mic_sig_temp = scipy.signal.convolve(sou_sig[:, np.newaxis], rir, mode='full', method='fft')
        mic_sig_temp = scipy.signal.fftconvolve(sou_sig[:, np.newaxis], rir, mode='full', axes=0)
        mic_sig = mic_sig_temp[0:nsample, :]
        
        return mic_sig 

class MicSigFromRIRDataset(Dataset):
    def __init__(
        self,
        rirdataset,
        srcdataset,
        noidataset,
        snr_range,
        fs,
        dataset_sz,
        seed,
        load_info,
        save_anno=False,
        save_to=None,   
        ): 
        
        self.rirdataset = rirdataset
        self.srcdataset = srcdataset
        self.noidataset = noidataset
        self.snr_range = snr_range
        self.fs = fs
        self.seed = seed
        self.save_anno = save_anno
        self.load_info = load_info
        self.save_to = save_to
        if dataset_sz is None:
            self.dataset_sz = int(1e8)
        else:
            self.dataset_sz = dataset_sz

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        np.random.seed(seed=self.seed+idx)

        # Get ramdom RIR
        rir_idx = np.random.randint(0, len(self.rirdataset))
        if self.load_info:
            rir, rir_dp, annos = self.rirdataset[rir_idx]
        else:
            rir, rir_dp = self.rirdataset[rir_idx]

        # Get random source signal
        src_idx = np.random.randint(0, len(self.srcdataset))
        src_sig = self.srcdataset[src_idx]

        # Generate clean or direct-path microphone signal
        mic_sig_clean, mic_sig_srcs_clean = self.rirdataset.rir_conv_src(rir, src_sig, gpu_conv=False)
        mic_sig_dp, mic_sig_srcs_dp = self.rirdataset.rir_conv_src(rir_dp, src_sig, gpu_conv=False)

        # Generate noise signal
        noi_sig = self.noidataset.generate_random_noise(mic_pos=annos['mic_pos'])
        snr = np.random.uniform(*self.snr_range)

        # Generate noisy microphone signal
        mic_sig = self.noidataset.add_noise(mic_sig_clean, noi_sig, snr, mic_sig_dp=mic_sig_dp) 

        # Check whether the values of microphone signals is in the range of [-1, 1] for wav saving (soundfile.write)
        max_value = np.max(mic_sig)
        min_value = np.min(mic_sig)
        max_value_dp = np.max(mic_sig_dp)
        min_value_dp = np.min(mic_sig_dp)
        value = np.max([np.abs(max_value), np.abs(min_value), np.abs(max_value_dp), np.abs(min_value_dp)])
        mic_sig = mic_sig / value
        mic_sig_dp = mic_sig_dp / value
        mic_sig_srcs_clean = mic_sig_srcs_clean / value
        mic_sig_srcs_dp = mic_sig_srcs_dp / value

        # # Save data
        # if self.save_to:
        #     Path(self.save_to).mkdir(parents=True, exist_ok=True)
        #     save_to_file = os.path.join(self.save_to, str(idx) + f'.wav')
        #     soundfile.write(save_to_file, mic_sig, fs)
        #     if self.save_anno:
        #         annos['SNR'] = snr
        #         save_to_file = os.path.join(self.save_to, str(idx) + f'_info.npz')
        #         np.savez(save_to_file, **annos)
        if self.load_info:
            vol = annos['room_sz'][0] * annos['room_sz'][1] * annos['room_sz'][2]
            sur = annos['room_sz'][0] * annos['room_sz'][1] + annos['room_sz'][0] * annos['room_sz'][2] + annos['room_sz'][1] * annos['room_sz'][2]
            annos = {
                # 'TDOA': annos['TDOA'].astype(np.float32),
                'T60': annos['T60_edc'].astype(np.float32), 
                'DRR': annos['DRR'].astype(np.float32),
                'C50': annos['C50'].astype(np.float32),
                'ABS': np.array(0.161*vol/sur/annos['T60_edc']).astype(np.float32),
                }
            return mic_sig, annos     
        else:
            return mic_sig

    def add_noise(self, mic_sig_clean, noi_sig, snr, mic_sig_dp, eps=1e-10) :
        """ Add noise to clean microphone signals with a given signal-to-noise ratio (SNR)
            Args:       mic_sig_clean - clean microphone signals without any noise (nsample, nch)
                        noi_sig       - noise signals (nsample, nch)
                        snr           - specific SNR
                        mic_sig_dp    - clean microphone signals without any noise and reverberation (nsample, nch)
            Returns:    mic_sig       - microphone signals with noise (nsample, nch)
        """ 
        nsample, _ = mic_sig_clean.shape
        if mic_sig_dp is None: # signal power includes reverberation
            av_pow = np.mean(np.sum(mic_sig_clean**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
            av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
            noi_sig_snr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
        else: # signal power do not include reverberation
            av_pow = np.mean(np.sum(mic_sig_dp**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
            av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
            noi_sig_snr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
        mic_sig = mic_sig_clean + noi_sig_snr
        
        return mic_sig  

# if __name__ == '__main__':
#     # spatialacoustics = SpatialAcoustics()
#     # fs=16000
#     # c=343.0
#     # ism_db = 12
#     # roomir = RoomImpulseResponse(fs= fs, c= c, ism_db= ism_db)

#     pass