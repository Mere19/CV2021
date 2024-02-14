import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

# compute distance from point x to points in X
def distance(x, X):
    x = x[None, :]
    dist = torch.cdist(X, x, p = 2)

    return dist

def distance_batch(x, X):
    dist = torch.cdist(X, x, p = 2)

    return dist

def gaussian(dist, bandwidth):
    weight = np.exp(-dist**2/(2 * bandwidth**2))

    return weight

def update_point(weight, X):
    numerator = torch.sum(weight * X, dim=0)
    denominator = torch.sum(weight)

    vec = numerator / denominator

    return vec

def update_point_batch(weight, X):
    # n = X.size()[0]
    # Xr = X.repeat([n, 1, 1])
    numerator = torch.matmul(weight, X)
    denominator = torch.sum(weight, dim = 1)
    # denominator = denominator.repeat(0, 3)
    # print(denominator.size())
    
    X_ = numerator
    X_[:,0] = numerator[:,0] / denominator
    X_[:,1] = numerator[:,1] / denominator
    X_[:,2] = numerator[:,2] / denominator

    return X_

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)

    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X, X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)

    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
mainpath = "/Users/guangzhu/Desktop/ETHz/CV/assignments/2/mean-shift_cow/"
image = rescale(io.imread(mainpath + 'cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load(mainpath + 'colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave(mainpath + 'result.png', result_image)
