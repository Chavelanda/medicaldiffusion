import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from fid.medicalnet import pretrained_resnet_fid

def compute_summary_stats(x):
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma

def save_stats(name, mu, sigma, path):
    np.save(path + name + '_mu.npy', mu)
    np.save(path + name + '_sigma.npy', sigma)

def load_stats(name, path):
    mu = np.load(path + name + '_mu.npy')
    sigma = np.load(path + name + '_sigma.npy')
    return mu, sigma

def compute_fid(mu1:np.array, sigma1:np.array, mu2:np.array, sigma2:np.array):
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    # Compute the sum of squared differences
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Compute the trace of the product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def fid_from_activations(x1, x2, name1, name2, path, save=False):
    mu1, sigma1 = compute_summary_stats(x1)
    mu2, sigma2 = compute_summary_stats(x2)

    if save:
        save_stats(name1, mu1, sigma1, path)
        save_stats(name2, mu2, sigma2, path)

    fid = compute_fid(mu1, sigma1, mu2, sigma2)

    return fid

def fid_from_path(path, name1, name2):
    mu1, sigma1 = load_stats(name1, path)
    mu2, sigma2 = load_stats(name2, path)

    fid = compute_fid(mu1, sigma1, mu2, sigma2)

    return fid

def fid_from_path_and_activations(x1, path, name1, name2, save=False):
    mu1, sigma1 = compute_summary_stats(x1)
    mu2, sigma2 = load_stats(name2, path)

    if save:
        save_stats(name1, mu1, sigma1, path)

    fid = compute_fid(mu1, sigma1, mu2, sigma2)

    return fid


def get_activations(model, dataloader, device):
    model.eval()

    activations = []

    print('Computing activations...')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch['data']
            data = data.to(device)
            
            output = model(data)
            activations.append(output.cpu().numpy())

    activations = np.concatenate(activations, axis=0)

    return activations

def compute_stats_from_model(model, dataloader, device, name, path, save=False):
    activations = get_activations(model, dataloader, device)
    mu, sigma = compute_summary_stats(activations)

    if save:
        save_stats(name, mu, sigma, path)

    return mu, sigma

def get_fid(model, dataloader, device, name1, name2, path, save=False):
    activations = get_activations(model, dataloader, device)
    fid = fid_from_path_and_activations(activations, path, name1, name2, save=save)
    return fid
