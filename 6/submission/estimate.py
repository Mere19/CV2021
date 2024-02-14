import numpy as np

def estimate(particles, particles_w):
    # normalized weighted sum
    num_particles = particles.shape[0]

    return np.sum(particles * particles_w.reshape((num_particles, 1)), axis=0) / np.sum(particles_w)