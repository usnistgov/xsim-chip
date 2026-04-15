import numpy as np
cimport cython


# Fix the data type for our memory views into Numpy arrays
DTYPE_I16 = np.int16
DTYPE_I32 = np.int32
DTYPE_F16 = np.float16
DTYPE_F32 = np.float32
DTYPE_U8 = np.uint8
DTYPE_U16 = np.uint16


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_vert_index(float[:] vert, float[:, :] vert_arr, float TOL):

    cdef Py_ssize_t n_dim = vert.shape[0]
    cdef Py_ssize_t n_vert = vert_arr.shape[0]
    cdef Py_ssize_t vert_arr_ndim = vert_arr.shape[1]

    # Returned index. Will be -1 if vert was not found in vert_arr.
    # Otherwise, vert_i will be the row index in vert_arr corresponding
    # to the matched vertex.
    cdef int vert_i = -1

    # assert vert.shape[0] == vert_arr.shape[1]
    # assert vert.shape[0] == 3

    cdef int cur_i
    cdef float temp, di, dr, dc

    for cur_i in range(n_vert):

        if vert_arr[cur_i, 0] < -999.0:
            if vert_arr[cur_i, 1] < -999.0:
                if vert_arr[cur_i, 2] < -999.0:
                    break

        temp = vert_arr[cur_i, 0] - vert[0]
        if (temp < 0.0):
            di = -temp
        else:
            di = temp

        temp = vert_arr[cur_i, 1] - vert[1]
        if (temp < 0.0):
            dr = -temp
        else:
            dr = temp

        temp = vert_arr[cur_i, 2] - vert[2]
        if (temp < 0.0):
            dc = -temp
        else:
            dc = temp

        if (di <= TOL):
            if (dr <= TOL):
                if (dc <= TOL):
                    vert_i = cur_i
                    break

    return vert_i


@cython.boundscheck(False)
@cython.wraparound(False)
def count_unmerged_boundary_elements(unsigned char[:, :, :] img3d):
    cdef Py_ssize_t n_img = img3d.shape[0]
    cdef Py_ssize_t n_row = img3d.shape[1]
    cdef Py_ssize_t n_col = img3d.shape[2]

    cdef Py_ssize_t ii, rr, cc, ipo, imo, rpo, rmo, cpo, cmo
    cdef int ipo_safe, imo_safe, rpo_safe, rmo_safe, cpo_safe, cmo_safe
    cdef Py_ssize_t vert_count, face_count
    cdef unsigned char cur_val, adj_val

    vert_count = 0
    face_count = 0

    for ii in range(n_img):
        for rr in range(n_row):
            for cc in range(n_col):

                cur_val = img3d[ii, rr, cc]
                if cur_val < 1:
                    continue

                ipo_safe = 1
                imo_safe = 1
                rpo_safe = 1
                rmo_safe = 1
                cpo_safe = 1
                cmo_safe = 1

                ipo = ii + 1
                imo = ii - 1
                rpo = rr + 1
                rmo = rr - 1
                cpo = cc + 1
                cmo = cc - 1

                # If near the end of the array, close the volume
                if imo < 0:
                    imo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2

                if ipo >= n_img:
                    ipo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2 

                if rmo < 0:
                    rmo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2

                if rpo >= n_row:
                    rpo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2

                if cmo < 0:
                    cmo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2

                if cpo >= n_col:
                    cpo_safe = -1
                    vert_count = vert_count + 4
                    face_count = face_count + 2

                # If near a feature edge, add surface triangles  
                if imo_safe > 0:
                    adj_val = img3d[imo, rr, cc]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

                if ipo_safe > 0:
                    adj_val = img3d[ipo, rr, cc]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

                if rmo_safe > 0:
                    adj_val = img3d[ii, rmo, cc]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

                if rpo_safe > 0:
                    adj_val = img3d[ii, rpo, cc]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

                if cmo_safe > 0:
                    adj_val = img3d[ii, rr, cmo]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

                if cpo_safe > 0:
                    adj_val = img3d[ii, rr, cpo]
                    if adj_val < 1:
                        vert_count = vert_count + 4
                        face_count = face_count + 2

    return (vert_count, face_count)


@cython.boundscheck(False)
@cython.wraparound(False)
def make_boundary_mesh(unsigned char[:, :, :] img3d, bint quiet_bool):
    
    cdef Py_ssize_t n_img = img3d.shape[0]
    cdef Py_ssize_t n_row = img3d.shape[1]
    cdef Py_ssize_t n_col = img3d.shape[2]

    cdef Py_ssize_t ii, rr, cc, ipo, imo, rpo, rmo, cpo, cmo
    cdef int vert1_i, vert2_i, vert3_i, vert4_i
    cdef int ipo_safe, imo_safe, rpo_safe, rmo_safe, cpo_safe, cmo_safe

    cdef Py_ssize_t vert_nrow, face_nrow
    cdef int vert_count, face_count
    cdef unsigned char cur_val, adj_val

    cdef float TOL = <float>0.001
    cdef float job_percent = <float>0.0

    vert_nrow, face_nrow = count_unmerged_boundary_elements(img3d)
    if not quiet_bool:
        print(f"\nInitializing memory for {vert_nrow} vertices...")
        print(f"Initializing memory for {face_nrow} faces...")

    v_arr_out = np.zeros((vert_nrow, 3), dtype=DTYPE_F32)
    cdef float[:, :] v_arr_out_view = v_arr_out
    v_arr_out_view[:, :] = <float>-999.9

    f_arr_out = np.zeros((face_nrow, 3), dtype=DTYPE_I32)
    cdef int[:, :] f_arr_out_view = f_arr_out

    v_arr_local = np.zeros((4, 3), dtype=DTYPE_F32)
    cdef float[:, :] v_arr_local_view = v_arr_local

    vert_count = 0
    face_count = 0

    if not quiet_bool:
        print(f"\nSearching for adjacent boundary voxels... ")
    for ii in range(n_img):
        for rr in range(n_row):
            for cc in range(n_col):

                cur_val = img3d[ii, rr, cc]

                if cur_val < 1:
                    continue

                ipo_safe = 1
                imo_safe = 1
                rpo_safe = 1
                rmo_safe = 1
                cpo_safe = 1
                cmo_safe = 1

                ipo = ii + 1
                imo = ii - 1
                rpo = rr + 1
                rmo = rr - 1
                cpo = cc + 1
                cmo = cc - 1

                # If near the end of the array, close the volume
                if imo < 0:
                    imo_safe = -1

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii - 0.5)
                    v_arr_local_view[0,1] = <float>(rr - 0.5)
                    v_arr_local_view[0,2] = <float>(cc - 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii - 0.5)
                    v_arr_local_view[1,1] = <float>(rr + 0.5)
                    v_arr_local_view[1,2] = <float>(cc - 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii - 0.5)
                    v_arr_local_view[2,1] = <float>(rr + 0.5)
                    v_arr_local_view[2,2] = <float>(cc + 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii - 0.5)
                    v_arr_local_view[3,1] = <float>(rr - 0.5)
                    v_arr_local_view[3,2] = <float>(cc + 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert3_i
                    f_arr_out_view[face_count, 2] = vert2_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert1_i
                    f_arr_out_view[face_count, 2] = vert4_i
                    face_count = face_count + 1

                if ipo >= n_img:
                    ipo_safe = -1 

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii + 0.5)
                    v_arr_local_view[0,1] = <float>(rr - 0.5)
                    v_arr_local_view[0,2] = <float>(cc - 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii + 0.5)
                    v_arr_local_view[1,1] = <float>(rr + 0.5)
                    v_arr_local_view[1,2] = <float>(cc - 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii + 0.5)
                    v_arr_local_view[2,1] = <float>(rr + 0.5)
                    v_arr_local_view[2,2] = <float>(cc + 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii + 0.5)
                    v_arr_local_view[3,1] = <float>(rr - 0.5)
                    v_arr_local_view[3,2] = <float>(cc + 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert2_i
                    f_arr_out_view[face_count, 2] = vert3_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert4_i
                    f_arr_out_view[face_count, 2] = vert1_i
                    face_count = face_count + 1

                if rmo < 0:
                    rmo_safe = -1

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii - 0.5)
                    v_arr_local_view[0,1] = <float>(rr - 0.5)
                    v_arr_local_view[0,2] = <float>(cc - 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii + 0.5)
                    v_arr_local_view[1,1] = <float>(rr - 0.5)
                    v_arr_local_view[1,2] = <float>(cc - 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii + 0.5)
                    v_arr_local_view[2,1] = <float>(rr - 0.5)
                    v_arr_local_view[2,2] = <float>(cc + 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii - 0.5)
                    v_arr_local_view[3,1] = <float>(rr - 0.5)
                    v_arr_local_view[3,2] = <float>(cc + 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert2_i
                    f_arr_out_view[face_count, 2] = vert3_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert4_i
                    f_arr_out_view[face_count, 2] = vert1_i
                    face_count = face_count + 1

                if rpo >= n_row:
                    rpo_safe = -1

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii - 0.5)
                    v_arr_local_view[0,1] = <float>(rr + 0.5)
                    v_arr_local_view[0,2] = <float>(cc - 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii + 0.5)
                    v_arr_local_view[1,1] = <float>(rr + 0.5)
                    v_arr_local_view[1,2] = <float>(cc - 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii + 0.5)
                    v_arr_local_view[2,1] = <float>(rr + 0.5)
                    v_arr_local_view[2,2] = <float>(cc + 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii - 0.5)
                    v_arr_local_view[3,1] = <float>(rr + 0.5)
                    v_arr_local_view[3,2] = <float>(cc + 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert3_i
                    f_arr_out_view[face_count, 2] = vert2_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert1_i
                    f_arr_out_view[face_count, 2] = vert4_i
                    face_count = face_count + 1

                if cmo < 0:
                    cmo_safe = -1

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii - 0.5)
                    v_arr_local_view[0,1] = <float>(rr - 0.5)
                    v_arr_local_view[0,2] = <float>(cc - 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii - 0.5)
                    v_arr_local_view[1,1] = <float>(rr + 0.5)
                    v_arr_local_view[1,2] = <float>(cc - 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii + 0.5)
                    v_arr_local_view[2,1] = <float>(rr + 0.5)
                    v_arr_local_view[2,2] = <float>(cc - 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii + 0.5)
                    v_arr_local_view[3,1] = <float>(rr - 0.5)
                    v_arr_local_view[3,2] = <float>(cc - 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert2_i
                    f_arr_out_view[face_count, 2] = vert3_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert4_i
                    f_arr_out_view[face_count, 2] = vert1_i
                    face_count = face_count + 1

                if cpo >= n_col:
                    cpo_safe = -1

                    # Vertex 1
                    v_arr_local_view[0,0] = <float>(ii - 0.5)
                    v_arr_local_view[0,1] = <float>(rr - 0.5)
                    v_arr_local_view[0,2] = <float>(cc + 0.5)

                    # Vertex 2
                    v_arr_local_view[1,0] = <float>(ii - 0.5)
                    v_arr_local_view[1,1] = <float>(rr + 0.5)
                    v_arr_local_view[1,2] = <float>(cc + 0.5)

                    # Vertex 3
                    v_arr_local_view[2,0] = <float>(ii + 0.5)
                    v_arr_local_view[2,1] = <float>(rr + 0.5)
                    v_arr_local_view[2,2] = <float>(cc + 0.5)

                    # Vertex 4
                    v_arr_local_view[3,0] = <float>(ii + 0.5)
                    v_arr_local_view[3,1] = <float>(rr - 0.5)
                    v_arr_local_view[3,2] = <float>(cc + 0.5)

                    v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                    vert1_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                    vert2_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                    vert3_i = vert_count
                    vert_count = vert_count + 1

                    v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                    vert4_i = vert_count
                    vert_count = vert_count + 1

                    f_arr_out_view[face_count, 0] = vert1_i
                    f_arr_out_view[face_count, 1] = vert3_i
                    f_arr_out_view[face_count, 2] = vert2_i
                    face_count = face_count + 1

                    f_arr_out_view[face_count, 0] = vert3_i
                    f_arr_out_view[face_count, 1] = vert1_i
                    f_arr_out_view[face_count, 2] = vert4_i
                    face_count = face_count + 1


                # If near a feature edge, add surface triangles 
                if imo_safe > 0:
                    adj_val = img3d[imo, rr, cc]

                    if adj_val < 1:

                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii - 0.5)
                        v_arr_local_view[0,1] = <float>(rr - 0.5)
                        v_arr_local_view[0,2] = <float>(cc - 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii - 0.5)
                        v_arr_local_view[1,1] = <float>(rr + 0.5)
                        v_arr_local_view[1,2] = <float>(cc - 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii - 0.5)
                        v_arr_local_view[2,1] = <float>(rr + 0.5)
                        v_arr_local_view[2,2] = <float>(cc + 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii - 0.5)
                        v_arr_local_view[3,1] = <float>(rr - 0.5)
                        v_arr_local_view[3,2] = <float>(cc + 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert3_i
                        f_arr_out_view[face_count, 2] = vert2_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert1_i
                        f_arr_out_view[face_count, 2] = vert4_i
                        face_count = face_count + 1

                if ipo_safe > 0:
                    adj_val = img3d[ipo, rr, cc]

                    if adj_val < 1:

                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii + 0.5)
                        v_arr_local_view[0,1] = <float>(rr - 0.5)
                        v_arr_local_view[0,2] = <float>(cc - 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii + 0.5)
                        v_arr_local_view[1,1] = <float>(rr + 0.5)
                        v_arr_local_view[1,2] = <float>(cc - 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii + 0.5)
                        v_arr_local_view[2,1] = <float>(rr + 0.5)
                        v_arr_local_view[2,2] = <float>(cc + 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii + 0.5)
                        v_arr_local_view[3,1] = <float>(rr - 0.5)
                        v_arr_local_view[3,2] = <float>(cc + 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert2_i
                        f_arr_out_view[face_count, 2] = vert3_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert4_i
                        f_arr_out_view[face_count, 2] = vert1_i
                        face_count = face_count + 1

                if rmo_safe > 0:
                    adj_val = img3d[ii, rmo, cc]

                    if adj_val < 1:

                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii - 0.5)
                        v_arr_local_view[0,1] = <float>(rr - 0.5)
                        v_arr_local_view[0,2] = <float>(cc - 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii + 0.5)
                        v_arr_local_view[1,1] = <float>(rr - 0.5)
                        v_arr_local_view[1,2] = <float>(cc - 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii + 0.5)
                        v_arr_local_view[2,1] = <float>(rr - 0.5)
                        v_arr_local_view[2,2] = <float>(cc + 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii - 0.5)
                        v_arr_local_view[3,1] = <float>(rr - 0.5)
                        v_arr_local_view[3,2] = <float>(cc + 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert2_i
                        f_arr_out_view[face_count, 2] = vert3_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert4_i
                        f_arr_out_view[face_count, 2] = vert1_i
                        face_count = face_count + 1

                if rpo_safe > 0:
                    adj_val = img3d[ii, rpo, cc]

                    if adj_val < 1:

                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii - 0.5)
                        v_arr_local_view[0,1] = <float>(rr + 0.5)
                        v_arr_local_view[0,2] = <float>(cc - 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii + 0.5)
                        v_arr_local_view[1,1] = <float>(rr + 0.5)
                        v_arr_local_view[1,2] = <float>(cc - 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii + 0.5)
                        v_arr_local_view[2,1] = <float>(rr + 0.5)
                        v_arr_local_view[2,2] = <float>(cc + 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii - 0.5)
                        v_arr_local_view[3,1] = <float>(rr + 0.5)
                        v_arr_local_view[3,2] = <float>(cc + 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert3_i
                        f_arr_out_view[face_count, 2] = vert2_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert1_i
                        f_arr_out_view[face_count, 2] = vert4_i
                        face_count = face_count + 1

                if cmo_safe > 0:
                    adj_val = img3d[ii, rr, cmo]
                    
                    if adj_val < 1:
                        
                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii - 0.5)
                        v_arr_local_view[0,1] = <float>(rr - 0.5)
                        v_arr_local_view[0,2] = <float>(cc - 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii - 0.5)
                        v_arr_local_view[1,1] = <float>(rr + 0.5)
                        v_arr_local_view[1,2] = <float>(cc - 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii + 0.5)
                        v_arr_local_view[2,1] = <float>(rr + 0.5)
                        v_arr_local_view[2,2] = <float>(cc - 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii + 0.5)
                        v_arr_local_view[3,1] = <float>(rr - 0.5)
                        v_arr_local_view[3,2] = <float>(cc - 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert2_i
                        f_arr_out_view[face_count, 2] = vert3_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert4_i
                        f_arr_out_view[face_count, 2] = vert1_i
                        face_count = face_count + 1

                if cpo_safe > 0:
                    adj_val = img3d[ii, rr, cpo]

                    if adj_val < 1:
                        
                        # Vertex 1
                        v_arr_local_view[0,0] = <float>(ii - 0.5)
                        v_arr_local_view[0,1] = <float>(rr - 0.5)
                        v_arr_local_view[0,2] = <float>(cc + 0.5)

                        # Vertex 2
                        v_arr_local_view[1,0] = <float>(ii - 0.5)
                        v_arr_local_view[1,1] = <float>(rr + 0.5)
                        v_arr_local_view[1,2] = <float>(cc + 0.5)

                        # Vertex 3
                        v_arr_local_view[2,0] = <float>(ii + 0.5)
                        v_arr_local_view[2,1] = <float>(rr + 0.5)
                        v_arr_local_view[2,2] = <float>(cc + 0.5)

                        # Vertex 4
                        v_arr_local_view[3,0] = <float>(ii + 0.5)
                        v_arr_local_view[3,1] = <float>(rr - 0.5)
                        v_arr_local_view[3,2] = <float>(cc + 0.5)

                        v_arr_out_view[vert_count, :] = v_arr_local_view[0, :]
                        vert1_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[1, :]
                        vert2_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[2, :]
                        vert3_i = vert_count
                        vert_count = vert_count + 1

                        v_arr_out_view[vert_count, :] = v_arr_local_view[3, :]
                        vert4_i = vert_count
                        vert_count = vert_count + 1

                        f_arr_out_view[face_count, 0] = vert1_i
                        f_arr_out_view[face_count, 1] = vert3_i
                        f_arr_out_view[face_count, 2] = vert2_i
                        face_count = face_count + 1

                        f_arr_out_view[face_count, 0] = vert3_i
                        f_arr_out_view[face_count, 1] = vert1_i
                        f_arr_out_view[face_count, 2] = vert4_i
                        face_count = face_count + 1

        # if ((ii%5) == 0):
        #     job_percent = <float>(100.0*(ii/(n_img-1.0)))
        #     print(f"  Percent complete: {job_percent:.1f}%")

    if not quiet_bool:
        print(f"\nCreated {vert_count} boundary vertices...")
        print(f"Created {face_count} boundary faces...")

    v_arr_out = v_arr_out[0:vert_count, 0:3]
    f_arr_out = f_arr_out[0:face_count, 0:3]

    #print(f"\n(CYTHON) v_arr_out = \n{v_arr_out})")
    #print(f"\n(CYTHON) f_arr_out = \n{f_arr_out})")

    return (v_arr_out, f_arr_out)

