import numpy as np

def resample(particles, particles_w):
    num_particles = particles.shape[0]
    prob = particles_w / np.sum(particles_w)
    particles_ids = np.random.choice(range(num_particles), size=num_particles, replace=True, p=prob)

    return particles[particles_ids], particles_w[particles_ids]