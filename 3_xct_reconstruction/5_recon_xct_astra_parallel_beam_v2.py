import numpy as np
import os
import astra
import tifffile
from skimage.util import img_as_uint, img_as_ubyte, img_as_float32, invert
from skimage import exposure
import matplotlib.pyplot as plt


# Parallel beam detector parameters
det_spacing_x = 0.004 # (mm)
det_spacing_y = 0.004 # (mm)
det_row_count = 1001 # (pixels)
det_col_count = 1201 # (pixels)

# Parallel beam XCT positions (both should be positive)
source_origin = 30.0  # [mm]
origin_det = 30.0    # [mm]

# The width of each pixel in mm for each projection image
pix_size = 0.004 # [mm/pixel]

tiff_path_in = "../2_xct_simulation/sim_radios_parallel_beam_pixsz4um/radios_2400_chip_4um_parallel_16bit.tif"
tiff_path_out = "./recon_imgs_parallel_beam_pixsz4um/recon_2400_chip_4um_parallel.tif"



# Compute the angles of the projection images
total_projections = 2400
final_angle = 360.0 # [deg]
angle_step = final_angle / total_projections
angle_arr = np.linspace(start=0.0, stop=final_angle, num=total_projections, endpoint=False)
angle_arr = np.deg2rad(angle_arr) # [rad]

# Import the image stack of projection images (as a multipage tif file)
# Note, these projection images should be dark for air bright for dense materials.
print("\nReading projection images...")
img_stack = tifffile.imread(tiff_path_in)
img_stack = invert(img_stack)
img_stack = np.moveaxis(img_stack, 0, 1) # Astra has weird array indexing
img_stack = img_as_float32(img_stack)



print("\nCalculating reconstruction...")
recon_imgs = []
for m in range(det_row_count):

    cur_proj_row = (img_stack[m, :, :]).copy()

    if np.amax(cur_proj_row) <= 1.0E-6:
        zero_img = np.zeros((det_col_count, det_col_count), dtype=np.float32)
        recon_imgs.append(zero_img.copy())

        if (((m+1)%20) == 0):
            print(f"  Computed {m+1}/{det_row_count} rows...")

        continue

    proj_geom = astra.create_proj_geom('parallel', det_spacing_x/pix_size, 
        det_col_count, angle_arr)

    vol_geom = astra.create_vol_geom(det_col_count, det_col_count)

    projection_id = astra.data2d.create('-sino', proj_geom, cur_proj_row)
    reconstruction_id = astra.data2d.create('-vol', vol_geom)

    alg_cfg = astra.astra_dict('FBP_CUDA') # No iterations
    # alg_cfg = astra.astra_dict('SIRT_CUDA') # ~1000 iterations
    # alg_cfg = astra.astra_dict('CGLS_CUDA') # ~50 iterations

    alg_cfg['ProjectionDataId'] = projection_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)

    astra.algorithm.run(algorithm_id)
    # astra.algorithm.run(algorithm_id, iterations=50)

    recon_temp = astra.data2d.get(reconstruction_id)
    recon_imgs.append(recon_temp.copy())

    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projection_id)

    if (((m+1)%20) == 0):
        print(f"  Computed {m+1}/{det_row_count} rows...")

recon_imgs = np.array(recon_imgs)

# Limit and scale reconstruction. Will also convert to unsigned 8-bit.
print("\nRotating image stack and saving reconstruction...")
recon_imgs = img_as_float32(recon_imgs)

if np.amin(recon_imgs) < 0.0:
    recon_imgs = recon_imgs + np.absolute(np.amin(recon_imgs))

recon_imgs = recon_imgs / np.amax(recon_imgs)
recon_imgs = img_as_uint(recon_imgs)

# Save reconstruction in its original orientation
tifffile.imwrite(tiff_path_out, recon_imgs, photometric='minisblack',
    compression='zlib')

# Shenanigans to make the images the same orientation as the original
#
# Numpy Notes:
#   Rotate images in stack CCW: np.rot90(recon_imgs, k=1, axes=(1,2))
#
#   Flip images in stack left-right: np.flip(recon_imgs, axis=2)
#   Flip images in stack up-down: np.flip(recon_imgs, axis=1)
#   Reverse order of images in stack: np.flip(recon_imgs, axis=0)

# Save the Top reconstruction
recon_imgs_top = np.rot90(recon_imgs, k=2, axes=(1,2))
recon_imgs_top = np.flip(recon_imgs_top, axis=2)
recon_imgs_top = np.moveaxis(recon_imgs_top, [0, 1, 2], [-1, 0, 1])
recon_imgs_top = np.flip(recon_imgs_top, axis=2)

# # Contrast stretching (optional)
# perc_lower, perc_upper = np.percentile(recon_imgs_top, (0.02, 99.98))
# recon_imgs_top = exposure.rescale_intensity(recon_imgs_top, 
#     in_range=(perc_lower, perc_upper), out_range='dtype')
#
# # Move the average and clip the intensities (optional)
# recon_imgs_top = img_as_float32(recon_imgs_top)
# avg_val = np.mean(recon_imgs_top)
# offset = 0.25 - avg_val
# recon_imgs_top = recon_imgs_top + offset
# recon_imgs_top[recon_imgs_top < 0.0] = 0
# recon_imgs_top[recon_imgs_top > 1.0] = 1.0

recon_imgs_top = img_as_uint(recon_imgs_top)

path_out_root, path_out_ext = os.path.splitext(tiff_path_out)
tiff_path_out_top = path_out_root + "_Top" + path_out_ext

mid_i = int(np.floor(recon_imgs_top.shape[0]/2))
mid_r = int(np.floor(recon_imgs_top.shape[1]/2))
mid_c = int(np.floor(recon_imgs_top.shape[2]/2))
i0 = int(mid_i - np.floor(751/2))
i1 = int(i0 + 751)
r0 = int(mid_r - np.floor(751/2))
r1 = int(r0 + 751)
c0 = int(mid_c - np.floor(751/2))
c1 = int(c0 + 751)
recon_imgs_top = recon_imgs_top[i0:i1, r0:r1, c0:c1]

tifffile.imwrite(tiff_path_out_top, recon_imgs_top, photometric='minisblack',
    compression='zlib')

# Save the Side reconstruction
recon_imgs_side = np.moveaxis(recon_imgs_top, [0, 1, 2], [-1, 0, 1])
recon_imgs_side = np.rot90(recon_imgs_side, k=-1, axes=(1,2))

path_out_root, path_out_ext = os.path.splitext(tiff_path_out)
tiff_path_out_side = path_out_root + "_Side" + path_out_ext

mid_i = int(np.floor(recon_imgs_side.shape[0]/2))
mid_r = int(np.floor(recon_imgs_side.shape[1]/2))
mid_c = int(np.floor(recon_imgs_side.shape[2]/2))
i0 = int(mid_i - np.floor(751/2))
i1 = int(i0 + 751)
r0 = int(mid_r - np.floor(751/2))
r1 = int(r0 + 751)
c0 = int(mid_c - np.floor(751/2))
c1 = int(c0 + 751)
recon_imgs_side = recon_imgs_side[i0:i1, r0:r1, c0:c1]

tifffile.imwrite(tiff_path_out_side, recon_imgs_side, photometric='minisblack',
   compression='zlib')

# Save the Front reconstruction
recon_imgs_front = np.moveaxis(recon_imgs_top, [0, 1, 2], [1, 2, 0])

path_out_root, path_out_ext = os.path.splitext(tiff_path_out)
tiff_path_out_front = path_out_root + "_Front" + path_out_ext

mid_i = int(np.floor(recon_imgs_front.shape[0]/2))
mid_r = int(np.floor(recon_imgs_front.shape[1]/2))
mid_c = int(np.floor(recon_imgs_front.shape[2]/2))
i0 = int(mid_i - np.floor(751/2))
i1 = int(i0 + 751)
r0 = int(mid_r - np.floor(751/2))
r1 = int(r0 + 751)
c0 = int(mid_c - np.floor(751/2))
c1 = int(c0 + 751)
recon_imgs_front = recon_imgs_front[i0:i1, r0:r1, c0:c1]

tifffile.imwrite(tiff_path_out_front, recon_imgs_front, photometric='minisblack',
   compression='zlib')

# Clean up
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projection_id)

print(f"\nScript finished successfully!")