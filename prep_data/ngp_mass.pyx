# ngp_mass.pyx
cimport numpy as np
cpdef void NGP_mass(np.float32_t[:,:] pos, np.float32_t[:] logM, np.float32_t[:,:,:,:] gridM, float BoxSize):
    cdef int axis, dims, coord, nMmax, jM
    cdef long i, particles
    cdef float inv_cell_size
    cdef int index[3]

    # Find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0]
    coord = pos.shape[1]
    dims = gridM.shape[0]
    nMmax = gridM.shape[3]
    inv_cell_size = dims / BoxSize

    # When computing things in 2D, use the index[2]=0 plane
    for i in range(3):
        index[i] = 0

    # Loop over all particles
    for i in range(particles):
        for axis in range(coord):
            index[axis] = <int>(pos[i, axis] * inv_cell_size + 0.5)
            index[axis] = (index[axis] + dims) % dims
        for jM in range(nMmax):
            if gridM[index[0], index[1], index[2], jM] == 0:
                gridM[index[0], index[1], index[2], jM] = logM[i]
                break
