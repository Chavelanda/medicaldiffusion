from foundation.medicalnet import resnet_gap
from foundation.stunet.get_stunet import get_stunet
from foundation.misfm.get_misfm import get_misfm
from vq_gan_3d.model import FIDEncoder


# Unifies the model and dataset configs in a single dictionary
def standard_converter(cfg):
    params = {**cfg.model, **cfg.dataset}

    return params


def cfg_to_med3d(cfg):
    params = standard_converter(cfg)

    params['no_cuda'] = not params['cuda']
    params['sample_input_D'] = params['d']
    params['sample_input_W'] = params['w']
    params['sample_input_H'] = params['h']

    return params


def cfg_to_vqvae(cfg):
    params = standard_converter(cfg)

    params['ckpt'] = params['vqvae_checkpoint']

    return params


def cfg_to_misfm(cfg):
    params = standard_converter(cfg)
    
    params['input_size'] = (params['d'], params['w'], params['h'])

    return params


EXTRACTORS = {
    'med3d': (resnet_gap, cfg_to_med3d, {'resnet_func': 'resnet50', 'pretrain_path': 'foundation/medicalnet/pretrain/', 'num_seg_classes': 2, 'sample_input_D': 128, 'sample_input_H': 128, 'sample_input_W': 128, 'no_cuda': False}),
    'vqvae': (FIDEncoder, cfg_to_vqvae, {'ckpt': None}),
    'stunet': (get_stunet, standard_converter, {}),
    'misfm': (get_misfm, cfg_to_misfm, {'input_size': (128, 128, 128)})
}


def get_extractor(cfg):
    extractor_func, cfg_converter, extractor_params = EXTRACTORS[cfg.model.extractor]

    cfg_params = cfg_converter(cfg)

    # Overwrite params from config
    for key in extractor_params:
        if key in cfg_params:
            print(f'Found {key}: substituted {extractor_params[key]} with {cfg_params[key]}')
            extractor_params[key] = cfg_params[key]

    extractor = extractor_func(**extractor_params)

    return extractor

