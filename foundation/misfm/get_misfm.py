import os

import torch

from foundation.misfm.pct_net import PCTNetFID


def get_misfm(model_path='foundation/misfm/', middle=True, input_channels=1, input_size=(128, 128, 128)):
    
    params = {'in_chns': input_channels, 'class_num': 5, 'input_size': input_size, 'feature_chns': [24, 48, 128, 256, 512], 'resolution_mode': 1, 'multiscale_pred': True}
    
    model = PCTNetFID(params, middle)

    weights_path = os.path.join(model_path, 'pctnet_pretrain.pt')
    
    model.load_state_dict(torch.load(weights_path, weights_only=False)['model_state_dict'])

    return model


if __name__=='__main__':
    
    model = get_misfm()

    # B, C, D, W, H
    x = torch.rand((2, 1,128,128,128), dtype=torch.float32)

    x = model(x)

    print(x[0, :10])
    print(x[1, :10])
