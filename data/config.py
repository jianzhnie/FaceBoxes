# config.py
cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[16, 32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'apply_distort': True,
    'apply_expand': True,
    'max_expand_ratio': 1.5,
    'data_anchor_sampling_prob': 0.5,
    'resize_width': 1024,
    'resize_heigth': 1024,
    'min_face_size': 9
}
