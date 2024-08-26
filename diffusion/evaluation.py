from scipy.stats import wasserstein_distance
import numpy as np
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor, Resize
from scipy.linalg import sqrtm
from PIL import Image

def get_features(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        features = model(image).detach().cpu().numpy()
    return features

def compute_fid(real_features, fake_features):
    """Compute the Frechet Inception Distance between two sets of features."""
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    mu_diff = mu_real - mu_fake
    cov_sqrt, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    fid = mu_diff @ mu_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return fid

# Load pretrained Inception model
model = inception_v3(pretrained=True, transform_input=False).eval()

def kl_divergence(p, q):
    """Compute the Kullback-Leibler Divergence between two distributions."""
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)  # Normalize to make it a valid probability distribution
    q = q / np.sum(q)  # Normalize to make it a valid probability distribution
    # Add a small constant to avoid log(0) issues
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

def mmd(x, y, gamma=1.0):
    """Compute the Maximum Mean Discrepancy between two distributions."""
    x = np.asarray(x)
    y = np.asarray(y)
    xx = rbf_kernel(x, x, gamma=gamma)
    yy = rbf_kernel(y, y, gamma=gamma)
    xy = rbf_kernel(x, y, gamma=gamma)
    mmd_value = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
    return mmd_value

def emd(x, y):
    """Compute the Earth Mover's Distance between two distributions."""
    return wasserstein_distance(x, y)
