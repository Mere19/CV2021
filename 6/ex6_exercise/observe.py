import numpy as np
from scipy.stats import norm
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):

    num_particles = particles.shape[0]

    # initialize particles_w
    particles_w = np.zeros(num_particles)

    # initialize random variable with normal distribution
    rv = norm(scale=sigma_observe)

    # for each particle, compute the color histogram of the corresponding bounding box
    for i in range(num_particles):
        p = particles[i]

        # center
        center_x = p[0]
        center_y = p[1]

        # compute the bounding box
        xmin = round(center_x - bbox_width / 2)
        if xmin < 0:
            xmin = 0
        xmax = round(center_x + bbox_width / 2)
        if xmax >= frame.shape[1]:
            xmax = frame.shape[1] - 1
        ymin = round(center_y - bbox_height / 2)
        if ymin < 0:
            ymin = 0
        ymax = round(center_y + bbox_height / 2)
        if ymax >= frame.shape[0]:
            ymax = frame.shape[0] - 1

        # compute histogram
        p_hist = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)

        # compute weight
        p_cost = chi2_cost(p_hist, hist)
        p_weight = rv.pdf(p_cost)
        particles_w[i] = p_weight

    return particles_w