# Environment libraries to import
import numpy as np
import skimage.draw as sdraw
import skimage.measure as smeas
import skimage.morphology as smorph
from skimage.util import img_as_ubyte, img_as_uint, img_as_bool, img_as_float32


def draw_sphere(radius, clipped_height=None):
    """
    Draws a sphere with an option of clipping its top/bottom portions off to a
    specified height. Returns a 3D numpy array (np.uint8) where the coordinates 
    of the sphere and its volume are represented by 1's.

    radius: Integer representing the radius
    clipped_height: Integer representing the final clipped height

    returned: 3D numpy array of dtype np.uint8. 
    """
    
    sph_rad = int(np.round(np.absolute(float(radius))))

    if clipped_height is None:
        sph_height = int(sph_rad + sph_rad + 1)
    else:
        sph_height = int(np.round(np.absolute(float(clipped_height))))

    temp_sph_imgs = img_as_ubyte(sdraw.ellipsoid(a=sph_rad, b=sph_rad, c=sph_rad))

    # Remove the surrounding padding of zeros
    temp_label_arr = smeas.label(temp_sph_imgs, connectivity=2)
    temp_props_list = smeas.regionprops(temp_label_arr)
    temp_sph_imgs = img_as_ubyte(temp_props_list[0].image)

    # Make the clipped height is odd
    #if sph_height%2 == 0:
    #    sph_height -= 1

    if sph_height > temp_sph_imgs.shape[0]:
        sph_height = temp_sph_imgs.shape[0]

    clip_start = int(np.round(((temp_sph_imgs.shape[0]) - sph_height)/2.0))
    clip_end = clip_start + sph_height

    sph_imgs_out = temp_sph_imgs[clip_start:clip_end]
    sph_imgs_out[sph_imgs_out > 0] = 1

    return sph_imgs_out


def draw_cylinder(outer_radius, height, inner_radius=None):
    """
    Draws a cylinder with an option of making it hollow. Returns a 3D Numpy
    array (np.uint8) where the coordinates of the cylinder and its volume are 
    represented by 1's. 

    outer_radius: Integer representing the outer radius
    height: Integer representing the final height
    inner_radius: Integer representing the inner radius

    returned: 3D numpy array of dtype np.uint8. 
    """

    out_radius = int(np.round(np.absolute(float(outer_radius))))
    cyl_height = int(np.round(np.absolute(float(height))))

    #if cyl_height%2 == 0:
    #    cyl_height -= 1

    if cyl_height < 0:
        cyl_height = 1

    if inner_radius is not None:
        in_radius = int(np.round(np.absolute(float(inner_radius))))

        if in_radius >= out_radius:
            in_radius = out_radius - 1

    cyl_slice = smorph.disk(out_radius, dtype=np.uint8)
    cyl_slice[cyl_slice > 0] = 1

    n_rows = cyl_slice.shape[0] 
    n_cols = cyl_slice.shape[1] 

    if (inner_radius is not None) and (inner_radius > 0):
        cyl_inner_foot = smorph.disk(in_radius, dtype=np.uint8)
        n_in_rows = cyl_inner_foot.shape[0]
        n_in_cols = cyl_inner_foot.shape[1]

        mask = np.zeros((n_rows, n_cols), dtype=np.uint8)

        r_off = int(np.floor((n_rows - n_in_rows)/2.0))
        c_off = int(np.floor((n_cols - n_in_cols)/2.0))

        mask[r_off:r_off+n_in_rows, c_off:c_off+n_in_cols] = cyl_inner_foot

        cyl_slice[mask > 0] = 0

    cyl_imgs_out = np.zeros((cyl_height, n_rows, n_cols), dtype=np.uint8)
    cyl_imgs_out[:, 0:n_rows, 0:n_cols] = cyl_slice

    return cyl_imgs_out


def draw_disk(outer_radius, inner_radius=None):
    """
    Draws a 2D circle with the option of it containing a hole in the center.
    Returns a 2D numpy array (np.uint8) where the coordinates of the circle and
    its filled area are represented by 1's. 

    outer_radius: Integer representing the outer radius of the circle
    inner_radius: Integer representing the inner radius.

    returned: 2D numpy array of dtype np.uint8. 
    """
    
    out_radius = int(np.round(np.absolute(float(outer_radius))))

    if inner_radius is not None:
        in_radius = int(np.round(np.absolute(float(inner_radius))))

        if in_radius >= out_radius:
            in_radius = out_radius - 1

    cyl_slice = smorph.disk(out_radius, dtype=np.uint8)
    cyl_slice[cyl_slice > 0] = 1

    n_rows = cyl_slice.shape[0] 
    n_cols = cyl_slice.shape[1] 

    if (inner_radius is not None) and (inner_radius > 0):
        cyl_inner_foot = smorph.disk(in_radius, dtype=np.uint8)
        n_in_rows = cyl_inner_foot.shape[0]
        n_in_cols = cyl_inner_foot.shape[1]

        mask = np.zeros((n_rows, n_cols), dtype=np.uint8)

        r_off = int(np.floor((n_rows - n_in_rows)/2.0))
        c_off = int(np.floor((n_cols - n_in_cols)/2.0))

        mask[r_off:r_off+n_in_rows, c_off:c_off+n_in_cols] = cyl_inner_foot

        cyl_slice[mask > 0] = 0

    cyl_img_out = np.zeros((n_rows, n_cols), dtype=np.uint8)
    cyl_img_out[0:n_rows, 0:n_cols] = cyl_slice

    return cyl_img_out


def duplicate_feature3d_on_generated_grid2d(feat_imgs, delta_row, delta_col, 
    n_row_features, n_col_features):
    """
    Takes a 3D numpy array and duplicates it onto a generated 2D grid of points.
    Returns the duplicated images as a 3D numpy array (np.uint8).

    feat_imgs: 3D numpy array (or feature) of dtype np.uint8 to be duplicated
    delta_row: The number of pixels to translate feat_imgs in the Y-direction
    delta_col: The number of pixels to translate feat_imgs in the X-direction
    n_row_features: The number of times to duplicate feat_imgs in the Y-direction
    n_col_features: The number of times to duplicate feat_imgs in the X-direction

    returned: 3D numpy array of the now duplicated feature as dtype np.uint8.
    """

    d_row = int(np.round(np.absolute(float(delta_row))))
    d_col = int(np.round(np.absolute(float(delta_col))))
    n_row_feat = int(np.round(np.absolute(float(n_row_features))))
    n_col_feat = int(np.round(np.absolute(float(n_col_features))))

    if n_row_feat < 1:
        n_row_feat = 1

    if n_col_feat < 1:
        n_col_feat = 1

    feat_n_imgs = feat_imgs.shape[0]
    feat_n_rows = feat_imgs.shape[1]
    feat_n_cols = feat_imgs.shape[2]

    imgs_out_n_imgs = feat_n_imgs
    imgs_out_n_rows = int(d_row*(n_row_feat - 1)) + feat_n_rows
    imgs_out_n_cols = int(d_col*(n_col_feat - 1)) + feat_n_cols
    imgs_out = np.zeros((imgs_out_n_imgs, imgs_out_n_rows, imgs_out_n_cols),
        dtype=np.uint8)

    for m in np.arange(n_row_feat):
        for n in np.arange(n_col_feat):

            i0 = 0
            i1 = feat_n_imgs

            r0 = int(m*d_row)
            r1 = r0 + feat_n_rows

            c0 = int(n*d_col)
            c1 = c0 + feat_n_cols

            imgs_out[i0:i1, r0:r1, c0:c1] = feat_imgs

    return imgs_out


def duplicate_feature2d_on_generated_grid2d(feat_img, delta_row, delta_col, 
    n_row_features, n_col_features):
    """
    Takes a 2D numpy array and duplicates it onto a generated 2D grid of points.
    Returns the duplicated images as a 2D numpy array (np.uint8).

    feat_img: 2D numpy array (or feature) of dtype np.uint8 to be duplicated
    delta_row: The number of pixels to translate feat_img in the Y-direction
    delta_col: The number of pixels to translate feat_img in the X-direction
    n_row_features: The number of times to duplicate feat_img in the Y-direction
    n_col_features: The number of times to duplicate feat_img in the X-direction

    returned: 2D numpy array of the now duplicated feature as dtype np.uint8.
    """

    d_row = int(np.round(np.absolute(float(delta_row))))
    d_col = int(np.round(np.absolute(float(delta_col))))
    n_row_feat = int(np.round(np.absolute(float(n_row_features))))
    n_col_feat = int(np.round(np.absolute(float(n_col_features))))

    if n_row_feat < 1:
        n_row_feat = 1

    if n_col_feat < 1:
        n_col_feat = 1

    feat_n_rows = feat_img.shape[0]
    feat_n_cols = feat_img.shape[1]

    img_out_n_rows = int(d_row*(n_row_feat - 1)) + feat_n_rows
    img_out_n_cols = int(d_col*(n_col_feat - 1)) + feat_n_cols
    img_out = np.zeros((img_out_n_rows, img_out_n_cols), dtype=np.uint8)

    for m in np.arange(n_row_feat):
        for n in np.arange(n_col_feat):

            r0 = int(m*d_row)
            r1 = r0 + feat_n_rows

            c0 = int(n*d_col)
            c1 = c0 + feat_n_cols

            img_out[r0:r1, c0:c1] = feat_img

    return img_out


def insert_feature3d_in_img3d(feat_imgs, imgs_3d, global_target_coord,
    ignore_z_centroid=False):
    """
    Inserts a local 3D numpy array into another (larger/global) 3D numpy array.
    The centroid of the local 3D numpy array will be made to coincide with the
    specified global indices, unless ignore_z_centroid is True. Since these are
    index coordinates, the final result could be off center by plus/minus one
    index coordinate due to round-off errors. 

    feat_imgs: The local 3D numpy array, or feature, as dtype np.uint8
    imgs_3d: The global 3D numpy array as dtype np.uint8 
    global_target_coord: A 1D array containing three values representing the
        index coordinates (img, row, col) of where to position the centroid of
        feat_imgs. This is the target position. If parts of the feature lie 
        outside of the global array, they will be clipped.
    ignore_z_centroid: If True, then the "img" index of global_target_coord
        will be ignored. Instead, the "img" index will be taken to be 0.

    returned: A 3D numpy array of dtype np.uint8 that is the same shape as 
        imgs_3d which contains the (potentially clipped) 3D array, feat_imgs,
        centered at the specified coordinates.
    """

    global_cent_i = int(np.round(float(global_target_coord[0])))
    global_cent_r = int(np.round(float(global_target_coord[1])))
    global_cent_c = int(np.round(float(global_target_coord[2])))
    global_n_imgs = imgs_3d.shape[0]
    global_n_rows = imgs_3d.shape[1]
    global_n_cols = imgs_3d.shape[2]

    local_n_imgs = feat_imgs.shape[0]
    local_n_rows = feat_imgs.shape[1]
    local_n_cols = feat_imgs.shape[2]
    local_cent_i = int(np.floor(local_n_imgs/2.0))
    local_cent_r = int(np.floor(local_n_rows/2.0))
    local_cent_c = int(np.floor(local_n_cols/2.0))

    if global_cent_i < 0:
        global_cent_i = 0
    elif global_cent_i > (global_n_imgs - 1):
        global_cent_i = global_n_imgs - 1

    if global_cent_r < 0:
        global_cent_r = 0
    elif global_cent_r > (global_n_rows - 1):
        global_cent_r = global_n_rows - 1

    if global_cent_c < 0:
        global_cent_c = 0
    elif global_cent_c > (global_n_cols - 1):
        global_cent_c = global_n_cols - 1
    
    if ignore_z_centroid:
        i0 = 0
        i1 = i0 + local_n_imgs
    else:
        i0 = global_cent_i - local_cent_i
        i1 = i0 + local_n_imgs

    r0 = global_cent_r - local_cent_r
    r1 = r0 + local_n_rows

    c0 = global_cent_c - local_cent_c
    c1 = c0 + local_n_cols

    i0_loc = 0
    if i0 < 0:
        i0_loc = i0_loc + (0 - i0)
        i0 = 0 

    i1_loc = local_n_imgs
    if i1 > global_n_imgs:
        i1_loc = i1_loc - (i1 - global_n_imgs)
        i1 = global_n_imgs

    r0_loc = 0
    if r0 < 0:
        r0_loc = r0_loc + (0 - r0)
        r0 = 0 

    r1_loc = local_n_rows
    if r1 > global_n_rows:
        r1_loc = r1_loc - (r1 - global_n_rows)
        r1 = global_n_rows

    c0_loc = 0
    if c0 < 0:
        c0_loc = c0_loc + (0 - c0)
        c0 = 0 

    c1_loc = local_n_cols
    if c1 > global_n_cols:
        c1_loc = c1_loc - (c1 - global_n_cols)
        c1 = global_n_cols
        
    imgs_3d[i0:i1, r0:r1, c0:c1] = feat_imgs[i0_loc:i1_loc,
                                             r0_loc:r1_loc,
                                             c0_loc:c1_loc]

    return imgs_3d


def insert_feature2d_in_img2d(feat_img, imgs_2d, global_centroid_indices):
    """
    Inserts a local 2D numpy array into another (larger/global) 2D numpy array.
    The centroid of the local 2D numpy array will be made to coincide with the
    specified global indices. Since these are index coordinates, the final 
    result could be off center by plus/minus one index coordinate due to 
    round-off errors. 

    feat_img: The local 2D numpy array, or feature, as dtype np.uint8
    imgs_2d: The global 2D numpy array as dtype np.uint8 
    global_centroid_indices: A 1D array containing two values representing the
        index coordinates (row, col) of where to position the centroid of
        feat_img. If parts of the feature lie outside of the global array, they
        will be clipped.

    returned: A 2D numpy array of dtype np.uint8 that is the same shape as 
        imgs_2d which contains the (potentially clipped) 2D array, feat_img,
        centered at the specified coordinates.
    """

    global_cent_r = int(np.round(float(global_centroid_indices[0])))
    global_cent_c = int(np.round(float(global_centroid_indices[1])))
    global_n_rows = imgs_2d.shape[0]
    global_n_cols = imgs_2d.shape[1]

    local_n_rows = feat_img.shape[0]
    local_n_cols = feat_img.shape[1]
    local_cent_r = int(np.floor(local_n_rows/2.0))
    local_cent_c = int(np.floor(local_n_cols/2.0))

    if global_cent_r < 0:
        global_cent_r = 0
    elif global_cent_r > (global_n_rows - 1):
        global_cent_r = global_n_rows - 1

    if global_cent_c < 0:
        global_cent_c = 0
    elif global_cent_c > (global_n_cols - 1):
        global_cent_c = global_n_cols - 1

    r0 = global_cent_r - local_cent_r
    r1 = r0 + local_n_rows

    c0 = global_cent_c - local_cent_c
    c1 = c0 + local_n_cols

    r0_loc = 0
    if r0 < 0:
        r0_loc = r0_loc + (0 - r0)
        r0 = 0 

    r1_loc = local_n_rows
    if r1 > global_n_rows:
        r1_loc = r1_loc - (r1 - global_n_rows)
        r1 = global_n_rows

    c0_loc = 0
    if c0 < 0:
        c0_loc = c0_loc + (0 - c0)
        c0 = 0 

    c1_loc = local_n_cols
    if c1 > global_n_cols:
        c1_loc = c1_loc - (c1 - global_n_cols)
        c1 = global_n_cols
        
    imgs_2d[r0:r1, c0:c1] = feat_img[r0_loc:r1_loc, c0_loc:c1_loc]

    return imgs_2d


def insert_feature3d_via_grid2d(feat_imgs, imgs_3d, grid2d_arr, z_offset=0, 
    center_grid=False):
    """
    Description
    """

    grid2d_pnts = (np.round(grid2d_arr)).astype(np.int32)
    z_off = int(np.round(np.absolute(z_offset)))

    global_n_imgs = imgs_3d.shape[0]
    global_n_rows = imgs_3d.shape[1]
    global_n_cols = imgs_3d.shape[2]
    global_cent_i = int(np.floor(float(global_n_imgs/2.0)))
    global_cent_r = int(np.floor(float(global_n_rows/2.0)))
    global_cent_c = int(np.floor(float(global_n_cols/2.0)))

    if z_off >= global_n_imgs:
        z_off = global_n_imgs - 1

    if center_grid:
        r_min = np.amin(grid2d_pnts[:,0])
        r_max = np.amax(grid2d_pnts[:,0])
        c_min = np.amin(grid2d_pnts[:,1])
        c_max = np.amax(grid2d_pnts[:,1])

        # grid_cent_coord2d = np.round(np.mean(grid2d_pnts, axis=0))
        grid_cent_coord2d = np.array([(r_max - r_min)/2.0, (c_max - c_min)/2.0])
        grid_cent_coord2d = (np.round(grid_cent_coord2d)).astype(np.int32)
        row_shift = global_cent_r - grid_cent_coord2d[0]
        col_shift = global_cent_c - grid_cent_coord2d[1]

        for m, cur_coord in enumerate(grid2d_pnts):
            grid2d_pnts[m,0] = int(np.round(cur_coord[0] + row_shift))
            grid2d_pnts[m,1] = int(np.round(cur_coord[1] + col_shift))

    if z_offset != 0:
        imgs_3d_slice = (imgs_3d[z_off:global_n_imgs]).copy()
    else:
        imgs_3d_slice = imgs_3d.copy()

    for m, cur_coord in enumerate(grid2d_pnts):
        cur_coord3d = (0, cur_coord[0], cur_coord[1])
        imgs_3d_slice = insert_feature3d_in_img3d(feat_imgs, imgs_3d_slice,
            cur_coord3d, ignore_z_centroid=True)

    imgs_3d[z_off:global_n_imgs] = imgs_3d_slice.copy()

    return imgs_3d


def insert_feature2d_via_grid2d(feat_img, imgs_2d, grid2d_arr, 
    center_grid=False):
    pass


def generate_grid2d(delta_row, delta_col, n_row_points, n_col_points,
    row_offset=0, col_offset=0, staggered_rows=False, reshape2D=True):
    """
    Description
    """

    d_row = int(np.round(np.absolute(delta_row)))
    d_col = int(np.round(np.absolute(delta_col)))
    n_row_pnts = int(np.round(np.absolute(n_row_points)))
    n_col_pnts = int(np.round(np.absolute(n_col_points)))
    r_off = int(np.round(row_offset))
    c_off = int(np.round(col_offset))

    if n_row_pnts < 1:
        n_row_pnts = 1

    if n_col_pnts < 1:
        n_col_pnts = 1

    n_pnts = int(n_row_pnts*n_col_pnts)
    pnts_arr = np.zeros((n_row_pnts, n_col_pnts, 2), dtype=np.int32)

    temp_row = np.zeros((n_col_pnts, 2), dtype=np.int32)
    d_col_half = int(np.round(d_col/2.0))
    for m in np.arange(n_row_pnts):
        cur_r = int(m*d_row) + r_off
        temp_row[0:n_col_pnts, 0:2] = 0

        for n in np.arange(n_col_pnts):
            cur_c = int(n*d_col) + c_off
            temp_row[n, 0] = cur_r
            temp_row[n, 1] = cur_c

        if staggered_rows:
            if m%2 != 0:
                temp_row[:, 1] = temp_row[:, 1] + d_col_half

        pnts_arr[m, 0:n_col_pnts, 0:2] = temp_row

    if reshape2D:
        pnts_out = np.zeros((n_pnts,2), dtype=np.int32)

        i_out = 0
        for m in np.arange(n_row_pnts):
            for n in np.arange(n_col_pnts):
                pnts_out[i_out, 0] = pnts_arr[m, n, 0]
                pnts_out[i_out, 1] = pnts_arr[m, n, 1]
                i_out += 1

    else:
        pnts_out = pnts_arr

    return pnts_out


def find_common_coords2d(coord2d_arr1, coord2d_arr2, TOL=1.0E-4):
    """
    Description
    """

    n_coords1 = coord2d_arr1.shape[0]
    n_coords2 = coord2d_arr2.shape[0]

    coords_out_list = []
    for m in np.arange(n_coords1):
        cur_coord1 = coord2d_arr1[m]
        cur_r1 = cur_coord1[0]
        cur_c1 = cur_coord1[1]

        for n in np.arange(n_coords2):
            cur_coord2 = coord2d_arr2[n]
            cur_r2 = cur_coord2[0]
            cur_c2 = cur_coord2[1]

            dr = np.absolute(cur_r2 - cur_r1)
            dc = np.absolute(cur_c2 - cur_c1)

            if ((dr <= TOL) and (dc <= TOL)):
                coords_out_list.append(cur_coord1)
                break

    coords_out_arr = (np.array(coords_out_list)).astype(coord2d_arr1.dtype)
    return coords_out_arr


def del_duplicate_coords_2d(coord2d_arr1, TOL=1.0E-4):
    """
    Description
    """

    n_coords1 = coord2d_arr1.shape[0]
    coords_out_list = []
    n_coords_out = 0
    for m in np.arange(n_coords1):
        cur_coord1 = coord2d_arr1[m]
        cur_r1 = cur_coord1[0]
        cur_c1 = cur_coord1[1]

        coord_already_exists = False

        if n_coords_out > 0:
            for n in range(len(coords_out_list)):
                cur_coord2 = coords_out_list[n]
                cur_r2 = cur_coord2[0]
                cur_c2 = cur_coord2[1]

                dr = np.absolute(cur_r2 - cur_r1)
                dc = np.absolute(cur_c2 - cur_c1)

                if ((dr <= TOL) and (dc <= TOL)):
                    coord_already_exists = True
                    break

        if not coord_already_exists:
            coords_out_list.append(cur_coord1)
            n_coords_out += 1

    coords_out_arr = (np.array(coords_out_list)).astype(coord2d_arr1.dtype)
    return coords_out_arr
