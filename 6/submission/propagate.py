import numpy as np

def propagate(particles, frame_height, frame_width, params):

    model = params["model"]

    if model == 1:
        num_particles = params["num_particles"]
        # velocity parameters
        sigma_velocity = params["sigma_velocity"]
        noise_velocity = np.random.normal(scale=sigma_velocity, size=(num_particles, 2))

        # position parameters
        sigma_position = params["sigma_position"]
        noise_position = np.random.normal(scale=sigma_position, size=(num_particles, 2))

        updated_particles = np.zeros_like(particles)
        # update position of the particles
        updated_particles[:, :2] = particles[:, :2] + particles[:, 2:4] + noise_position

        # update velocity of the articles
        updated_particles[:, 2:4] = particles[:, 2:4] + noise_velocity

        # check if center of particle is within the frame
        # if yes, clip it to the boundary of the frame
        for p in updated_particles:
            if p[0] < 0:
                p[0] = 0
            if p[0] >= frame_width:
                p[0] = frame_width - 1
            if p[1] < 0:
                p[1] = 0
            if p[1] >= frame_height:
                p[1] = frame_height - 1

        return updated_particles

    else:
        num_particles = params["num_particles"]

        # position parameters
        sigma_position = params["sigma_position"]
        noise_position = np.random.normal(scale=sigma_position, size=(num_particles, 2))

        updated_particles = np.zeros_like(particles)
        # update position of the particles
        updated_particles[:, :] = particles[:, :] + noise_position

        # check if center of particle is within the frame
        # if yes, clip it to the boundary of the frame
        for p in updated_particles:
            if p[0] < 0:
                p[0] = 0
            if p[0] >= frame_width:
                p[0] = frame_width - 1
            if p[1] < 0:
                p[1] = 0
            if p[1] >= frame_height:
                p[1] = frame_height - 1

        return updated_particles