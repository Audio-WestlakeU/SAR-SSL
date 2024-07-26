import numpy as np

mic_array_cfg_2ch = {
    'array_type': 'planar_linear',
    'array_scale_range': (0.3, 2), 
    'array_rotate_azi_range': (0, 360), 
    'mic_pos_relative': np.array(((-0.05, 0.0, 0.0),
                        (0.05, 0.0, 0.0))),  
    'mic_orV': np.array(((-1.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0))), 
    'mic_pattern': 'omni',
    'array_orV': np.array([0.0, 1.0, 0.0]),
}

mic_array_cfg_circular_4ch = {
    'array_type': 'planar_linear',
    'array_scale_range': (1, 1), 
    'array_rotate_azi_range': (0, 0), 
    'mic_pos_relative': np.array(((0.05, 0.0, 0.0),
                        (0.0, 0.05, 0.0),
                        (-0.05, 0.0, 0.0),
                        (0.0, -0.05, 0.0))), 
    'mic_orV': np.array(((1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (-1.0, 0.0, 0.0), 
                        (0.0, -1.0, 0.0))),
    'mic_pattern': 'omni',
    'array_orV': np.array([0.0, 1.0, 0.0]),
}
