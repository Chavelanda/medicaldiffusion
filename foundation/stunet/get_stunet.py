import os

import torch
from batchgenerators.utilities.file_and_folder_operations import *

from foundation.stunet.architecture.STUNet import STUNetFID

# Requires images in range 
def get_stunet(model_path='foundation/stunet/', middle=True, **kwargs):
    plan_path = os.path.join(model_path, 'base_ep4k.model.pkl') 
    plan = load_pickle(plan_path)
    
    # print(plan['plans']['plans_per_stage'][0])

    # pool_op_kernel_sizes = plan['plans']['plans_per_stage'][0]['pool_op_kernel_sizes']
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
    conv_kernel_sizes = plan['plans']['plans_per_stage'][0]['conv_kernel_sizes']
    
    model = STUNetFID(1, 105, pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes)

    weights_path = os.path.join(model_path, 'base_ep4k.model')
    model.load_state_dict(torch.load(weights_path, weights_only=False)['state_dict'])

    return model


if __name__ == '__main__':
    model = get_stunet()

    # B, C, D, W, H
    x = torch.rand((2, 1,128,128,128), dtype=torch.float32)

    x = model(x)

    print(x.shape)

