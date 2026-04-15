
# Environment libraries to import
import numpy as np
import tifffile
from skimage.util import img_as_ubyte, img_as_uint, img_as_bool, img_as_float32

# Local Python functions to import
import draw_chips_lib as dcl


# ========== INITIALIZATIONS ========== 

# 0: air (no material)
# 85: oxide/insulation layers
# 170: copper
# 255: solder

print("\nInitializing variables...")
n_imgs_um = 3000.0 # microns
n_rows_um = 3000.0 # microns
n_cols_um = 3000.0 # microns
pix_sz = 4.0  # um/pixel (Use either 4.0 or 2.0)

n_imgs = int(np.round(n_imgs_um/pix_sz)) # pixels
n_rows = int(np.round(n_rows_um/pix_sz)) # pixels
n_cols = int(np.round(n_cols_um/pix_sz)) # pixels

# Force the image stack to have odd dimensions so there is one pixel 
# located precisely in the center.
if (n_imgs%2) == 0:
    n_imgs += 1

if (n_rows%2) == 0:
    n_rows += 1

if (n_cols%2) == 0:
    n_cols += 1

imgs_arr = np.zeros((n_imgs,n_rows,n_cols), dtype=np.uint8)

# The indices to the center-pixel
imgs_mid_i = int(np.floor(n_imgs/2.0))
imgs_mid_r = int(np.floor(n_rows/2.0))
imgs_mid_c = int(np.floor(n_cols/2.0))


# ========== CREATE BGAS ========== 

print(f"\nCreating BGA geometries...")

bga_radius_um = 350
bga_radius_px = int(np.round(bga_radius_um/pix_sz))

bga_height_um = 500
bga_height_px = int(np.round(bga_height_um/pix_sz))

bga_cent2cent_um = 1000
bga_cent2cent_px = int(np.round(bga_cent2cent_um/pix_sz))

bga_sphere = 255*(dcl.draw_sphere(bga_radius_px, clipped_height=bga_height_px))

n_bga_rows = int(np.floor(n_rows/bga_cent2cent_px))
n_bga_cols = int(np.floor(n_cols/bga_cent2cent_px))
bga_grid_2d = dcl.generate_grid2d(bga_cent2cent_px, bga_cent2cent_px,
    n_bga_rows, n_bga_cols)

bga_imgs = np.zeros((bga_height_px, n_rows, n_cols), np.uint8)
bga_imgs = dcl.insert_feature3d_via_grid2d(bga_sphere, bga_imgs, bga_grid_2d,
    center_grid=True)

bga_imgs[0] = bga_imgs[1].copy()
bga_imgs[-1] = bga_imgs[-2].copy()

i_global_start = 0
i_global_end = bga_height_px
imgs_arr[i_global_start:i_global_end] = bga_imgs

# Store the array slice indices in case needed for later
bga_slice = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== CREATE SIX COPPER/INSULATION LAYERS ==========

print(f"\nCreating first set of copper interconnect layers...")

cu_thick_um = 25
cu_thick_px = int(np.round(cu_thick_um/pix_sz))

sod_thick_um = 20
sod_thick_px = int(np.round(sod_thick_um/pix_sz))

num_cu_layers = 6

# Layer order: 6x(sod --> cu)
sod_layer_imgs = 85*(np.ones((sod_thick_px, n_rows, n_cols), dtype=np.uint8))
cu_layer_imgs = 170*(np.ones((cu_thick_px, n_rows, n_cols), dtype=np.uint8))

beol_n_imgs = num_cu_layers*(sod_thick_px + cu_thick_px) - sod_thick_px
beol_layer_imgs = np.zeros((beol_n_imgs, n_rows, n_cols), dtype=np.uint8)

cur_i0 = 0
temp_sodii_list = [] 
temp_cuii_list = []
for m in np.arange(num_cu_layers):
    
    # Insert spin-on-dielectric (SOD) layer
    if m != 0:
        cur_i1 = cur_i0 + sod_thick_px
        beol_layer_imgs[cur_i0:cur_i1] = sod_layer_imgs[:, 0:n_rows, 0:n_cols]
        temp_sodii_list.append(np.s_[(i_global_end+cur_i0):(i_global_end+cur_i1), \
            0:n_rows, 0:n_cols])
        cur_i0 = cur_i1

    # Insert copper layer
    cur_i1 = cur_i0 + cu_thick_px
    beol_layer_imgs[cur_i0:cur_i1] = cu_layer_imgs[:, 0:n_rows, 0:n_cols]
    temp_cuii_list.append(np.s_[(i_global_end+cur_i0):(i_global_end+cur_i1), \
        0:n_rows, 0:n_cols])
    cur_i0 = cur_i1

# Go back up and add insulation layer around the BGAs
temp_img1 = imgs_arr[i_global_end-sod_thick_px:i_global_end]
temp_img1[temp_img1 <= 0] = 85
imgs_arr[i_global_end-sod_thick_px:i_global_end] = temp_img1

# Finally, insert the BEOL into the global image array
i_global_start = i_global_end
i_global_end = i_global_start + beol_n_imgs
imgs_arr[i_global_start:i_global_end] = beol_layer_imgs

# Store the array slice indices in case needed for later
# Layers: Cu1 > SOD1 > Cu2 > SOD2 > Cu3 > SOD3 > Cu4 > SOD4 > Cu5 > SOD5 > Cu6
cu_top_slice1 = temp_cuii_list[0]
sod_top_slice1 = temp_sodii_list[0]
cu_top_slice2 = temp_cuii_list[1]
sod_top_slice2 = temp_sodii_list[1]
cu_top_slice3 = temp_cuii_list[2]
sod_top_slice3 = temp_sodii_list[2]
cu_top_slice4 = temp_cuii_list[3]
sod_top_slice4 = temp_sodii_list[3]
cu_top_slice5 = temp_cuii_list[4]
sod_top_slice5 = temp_sodii_list[4]
cu_top_slice6 = temp_cuii_list[5]


# ========== CREATE THROUGH-SILICON VIAS ==========

print(f"\nCreating through-silicon vias...")

tsv_oradius_um = 100
tsv_oradius_px = int(np.round(tsv_oradius_um/pix_sz))

tsv_iradius_um = 75
tsv_iradius_px = int(np.round(tsv_iradius_um/pix_sz))

tsv_height_um = 1225
tsv_height_px = int(np.round(tsv_height_um/pix_sz))

# tsv_spacing_um = 650
# tsv_spacing_px = int(np.round(tsv_spacing_um/pix_sz))
tsv_spacing_um = 600
tsv_spacing_px = int(np.round(tsv_spacing_um/pix_sz))

# Generate the grid of centroid coordinates for the TSVs
n_tsv_rows = int(np.floor(n_rows/tsv_spacing_px))
n_tsv_cols = int(np.floor(n_cols/tsv_spacing_px))
n_tsv = n_tsv_rows*n_tsv_cols
tsv_grid_2d = dcl.generate_grid2d(tsv_spacing_px, tsv_spacing_px, n_tsv_rows, 
    n_tsv_cols, staggered_rows=False)

# Delete some percent of these grid points
rng_seed = 2060
tsv_pct_keep = 0.6

n_tsv_kept = int(np.round(tsv_pct_keep*n_tsv))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_tsv, size=n_tsv_kept)

tsv_grid_keep = np.zeros((n_tsv_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    tsv_grid_keep[m] = tsv_grid_2d[cur_i]

# Create a TSV image stack
tsv_img = 170*(dcl.draw_cylinder(tsv_oradius_px, tsv_height_px,
    inner_radius=tsv_iradius_px))
tsv_img[tsv_img <= 0] = 85

tsv_arr_imgs = 85*(np.ones((tsv_height_px, n_rows, n_cols), dtype=np.uint8))

tsv_arr_imgs = dcl.insert_feature3d_via_grid2d(tsv_img, tsv_arr_imgs, 
    tsv_grid_keep, center_grid=True)

tsv_arr_imgs[0] = tsv_arr_imgs[1].copy()
tsv_arr_imgs[-1] = tsv_arr_imgs[-2].copy()

# Finally, insert the TSV array into the global image stack
i_global_start = i_global_end
i_global_end = i_global_start + tsv_height_px
imgs_arr[i_global_start:i_global_end] = tsv_arr_imgs

# Store the array slice indices in case needed for later
tsv_slice = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== CREATE SIX MORE COPPER/INSULATION LAYERS ==========

print(f"\nCreating second set of copper interconnect layers...")

# Finally, insert the BEOL into the global image array
i_global_start = i_global_end
i_global_end = i_global_start + beol_n_imgs
imgs_arr[i_global_start:i_global_end] = beol_layer_imgs

# Store the array slice indices in case needed for later
# Layers: Cu1 > SOD1 > Cu2 > SOD2 > Cu3 > SOD3 > Cu4 > SOD4 > Cu5 > SOD5 > Cu6
temp_start = cu_top_slice1[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice1[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice1 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols]  

temp_start = sod_top_slice1[0].start + tsv_height_px + beol_n_imgs
temp_end = sod_top_slice1[0].stop + tsv_height_px + beol_n_imgs
sod_bot_slice1 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols]

temp_start = cu_top_slice2[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice2[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice2 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = sod_top_slice2[0].start + tsv_height_px + beol_n_imgs
temp_end = sod_top_slice2[0].stop + tsv_height_px + beol_n_imgs
sod_bot_slice2 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = cu_top_slice3[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice3[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice3 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = sod_top_slice3[0].start + tsv_height_px + beol_n_imgs
temp_end = sod_top_slice3[0].stop + tsv_height_px + beol_n_imgs
sod_bot_slice3 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = cu_top_slice4[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice4[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice4 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = sod_top_slice4[0].start + tsv_height_px + beol_n_imgs
temp_end = sod_top_slice4[0].stop + tsv_height_px + beol_n_imgs
sod_bot_slice4 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = cu_top_slice5[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice5[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice5 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = sod_top_slice5[0].start + tsv_height_px + beol_n_imgs
temp_end = sod_top_slice5[0].stop + tsv_height_px + beol_n_imgs
sod_bot_slice5 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 

temp_start = cu_top_slice6[0].start + tsv_height_px + beol_n_imgs
temp_end = cu_top_slice6[0].stop + tsv_height_px + beol_n_imgs
cu_bot_slice6 = np.s_[temp_start:temp_end, 0:n_rows, 0:n_cols] 


# ========== CREATE HOLES IN CU LAYERS FOR TSVs ==========

tsv_hole_img_width = tsv_oradius_px + tsv_oradius_px + 1
tsv_hole_cut_img = 170*(np.ones((cu_thick_px, tsv_hole_img_width,
    tsv_hole_img_width), dtype=np.uint8))

tsv_hole_mask = 85*(dcl.draw_cylinder(tsv_iradius_px, cu_thick_px))
tsv_hole_mask[tsv_hole_mask <= 0] = 170

tsv_hole_cut_img = dcl.insert_feature3d_via_grid2d(tsv_hole_mask, 
    tsv_hole_cut_img, np.array([[0,0]]), center_grid=True)

cu_layer_img = (imgs_arr[cu_top_slice6]).copy()

cu_layer_img = dcl.insert_feature3d_via_grid2d(tsv_hole_cut_img, cu_layer_img,
    tsv_grid_keep, center_grid=True)

cu_layer_img[0] = cu_layer_img[1].copy()
cu_layer_img[-1] = cu_layer_img[-2].copy()

imgs_arr[cu_top_slice6] = cu_layer_img.copy()
imgs_arr[cu_bot_slice1] = cu_layer_img.copy()


# ========== CREATE C4 SOLDER BUMP LAYER ==========

print(f"\nCreating C4 solder bump geometries...")

c4_radius_um = 50
c4_radius_px = int(np.round(c4_radius_um/pix_sz))

c4_height_um = 75
c4_height_px = int(np.round(c4_height_um/pix_sz))

# c4_cent2cent_um = 250
# c4_cent2cent_px = int(np.round(c4_cent2cent_um/pix_sz))
c4_cent2cent_um = 260
c4_cent2cent_px = int(np.round(c4_cent2cent_um/pix_sz))

c4_pad_radius_um = 35
c4_pad_radius_px = int(np.round(c4_pad_radius_um/pix_sz))

c4_pad_height_um = 35
c4_pad_height_px = int(np.round(c4_pad_height_um/pix_sz))

c4_lead_height_um = 90
c4_lead_height_px = int(np.round(c4_lead_height_um/pix_sz))

c4_lead_radius_um = 5
c4_lead_radius_px = int(np.round(c4_lead_radius_um/pix_sz))

# Generate the grid of centroid coordinates for the C4s
n_c4_rows = int(np.floor(n_rows/c4_cent2cent_px))
n_c4_cols = int(np.floor(n_cols/c4_cent2cent_px))
n_c4 = n_c4_rows*n_c4_cols
c4_grid_2d = dcl.generate_grid2d(c4_cent2cent_px, c4_cent2cent_px, n_c4_rows,
    n_c4_cols)

# Delete some percent of these grid points
rng_seed = 2057
c4_pct_keep = 0.70

n_c4_kept = int(np.round(c4_pct_keep*n_c4))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_c4, size=n_c4_kept)

c4_grid_keep = np.zeros((n_c4_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    c4_grid_keep[m] = c4_grid_2d[cur_i]

# Create the C4 image stack
c4_img = 255*(dcl.draw_sphere(c4_radius_px, clipped_height=c4_height_px))
c4_img[c4_img <= 0] = 85

# Create copper pad on one side of the C4 bond
c4_pad_img = 170*(dcl.draw_cylinder(c4_pad_radius_px, c4_pad_height_px))
c4_pad_img[c4_pad_img <= 0] = 255 # The background will be masked later

# Insert the pad into the C4 geometry in the top, then flip it upside down
c4_mask = c4_img.copy()

target_coord = (0, int(np.floor(c4_mask.shape[1]/2.0)), 
     int(np.floor(c4_mask.shape[2]/2.0)))

c4_mask = dcl.insert_feature3d_in_img3d(c4_pad_img, c4_mask, 
    target_coord, ignore_z_centroid=True)

c4_img[c4_mask == 170] = 170

# Add the two lead connects at the base of the pad. Could do a cylinder or hex
c4_width = int(np.round(c4_lead_radius_px*2.0 + 1.0))

#c4_lead_img = 170*(dcl.draw_cylinder(c4_lead_radius_px, c4_lead_height_px))
c4_lead_img = 170*(np.ones((c4_lead_height_px, c4_width, c4_width),
    dtype=np.uint8))

c4_lead_spacing = int(np.round(2.0*c4_lead_img.shape[1]))
c4_lead_img2 = dcl.duplicate_feature3d_on_generated_grid2d(c4_lead_img, 
    c4_lead_spacing, 0, 2, 1)

c4_mask = c4_img.copy()
c4_img = 85*np.ones((c4_mask.shape[0] + c4_lead_img2.shape[0], 
    c4_mask.shape[1], c4_mask.shape[2]), dtype=np.uint8)

c4_img[0:c4_mask.shape[0]] = np.flipud(c4_mask)
c4_img = np.flipud(c4_img)
c4_mask = c4_img.copy()

target_coord = (0, int(np.floor(c4_mask.shape[1]/2.0)), 
     int(np.floor(c4_mask.shape[2]/2.0)))

c4_mask = dcl.insert_feature3d_in_img3d(c4_lead_img2, c4_mask, 
    target_coord, ignore_z_centroid=True)

c4_img[c4_mask == 170] = 170
c4_img = np.flipud(c4_img)

# Insert the C4 solder & pad into a full layer onto the generated grid points
c4_arr_imgs = 85*(np.ones((c4_height_px + c4_lead_height_px, n_rows, n_cols),
    dtype=np.uint8))
c4_arr_imgs = dcl.insert_feature3d_via_grid2d(c4_img, c4_arr_imgs, c4_grid_keep,
    center_grid=True)

c4_arr_imgs[0] = c4_arr_imgs[1].copy()
c4_arr_imgs[-1] = c4_arr_imgs[-2].copy()

# Finally, insert the TSV array into the global image stack
i_global_start = i_global_end
i_global_end = i_global_start + c4_height_px + c4_lead_height_px
imgs_arr[i_global_start:i_global_end] = c4_arr_imgs

# Store the array slice indices in case needed for later
c4_slice = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== CREATE INTERPOSER SOLDER BUMPS ========== 

print(f"\nCreating interposer solder bump geometries...")

intpos_radius_um = 10 
intpos_radius_px = int(np.round(intpos_radius_um/pix_sz))

intpos_height_um = 42
intpos_height_px = int(np.round(intpos_height_um/pix_sz))

intpos_cent2cent_um = 128
intpos_cent2cent_px = int(np.round(intpos_cent2cent_um/pix_sz))

intpos_layer_thick_um = 8
intpos_layer_thick_px = int(np.round(intpos_layer_thick_um/pix_sz))

# Create the interposer image array for one feature
intpos_sph_height = int(np.round(intpos_height_px/3.0))
intpos_sph_radius = int(np.round(1.5*intpos_radius_px))
intpos_sph = 255*(dcl.draw_sphere(intpos_sph_radius, intpos_sph_height))
intpos_sph_width = intpos_sph.shape[1]

intpos_pad_height = int(np.round((intpos_height_px - intpos_sph.shape[0])/2.0))
intpos_pad = 170*(dcl.draw_cylinder(intpos_radius_px, intpos_pad_height))
intpos_pad_height = intpos_pad.shape[0]
intpos_height_px = intpos_pad_height + intpos_sph_height + intpos_pad_height

intpos_img = 85*(np.ones((intpos_height_px, intpos_sph_width, intpos_sph_width),
    dtype=np.uint8))

intpos_mid_i = int(np.floor(intpos_img.shape[0]/2.0))
intpos_mid_r = int(np.floor(intpos_img.shape[1]/2.0))
intpos_mid_c = int(np.floor(intpos_img.shape[2]/2.0))
intpos_target = (intpos_mid_i, intpos_mid_r, intpos_mid_c)

intpos_img = dcl.insert_feature3d_in_img3d(intpos_pad, intpos_img, intpos_target,
    ignore_z_centroid=True)

intpos_img = np.flipud(intpos_img)
intpos_img = dcl.insert_feature3d_in_img3d(intpos_pad, intpos_img, intpos_target,
    ignore_z_centroid=True)

intpos_img[intpos_img <= 0] = 85
intpos_mask = intpos_img.copy()

intpos_mask = dcl.insert_feature3d_in_img3d(intpos_sph, intpos_mask, intpos_target,
    ignore_z_centroid=False)

intpos_img[intpos_mask == 255] = 255

# Duplicate this feature on a 2D grid
intpos_total_height_px = intpos_layer_thick_px*2 + intpos_height_px
intpos_arr_imgs = 85*(np.ones((intpos_total_height_px, n_rows, n_cols),
    dtype=np.uint8))

intpos_arr_imgs[0:intpos_layer_thick_px] = 170
intpos_arr_mid_imgs = intpos_arr_imgs[intpos_layer_thick_px:\
    intpos_layer_thick_px + intpos_height_px].copy()

# Make the 2D grid points
n_intpos_rows = int(np.floor(n_rows/intpos_cent2cent_px))
n_intpos_cols = int(np.floor(n_cols/intpos_cent2cent_px))
n_intpos = n_intpos_rows*n_intpos_cols
intpos_grid_2d = dcl.generate_grid2d(intpos_cent2cent_px, intpos_cent2cent_px,
    n_intpos_rows, n_intpos_cols, staggered_rows=False)

# Delete some percent of these grid points
rng_seed = 2056
intpos_pct_keep = 0.70

n_intpos_kept = int(np.round(intpos_pct_keep*n_intpos))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_intpos, size=n_intpos_kept)

intpos_grid_keep = np.zeros((n_intpos_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    intpos_grid_keep[m] = intpos_grid_2d[cur_i]

intpos_arr_mid_imgs = dcl.insert_feature3d_via_grid2d(intpos_img, intpos_arr_mid_imgs,
    intpos_grid_keep, center_grid=True)

intpos_arr_mid_imgs[0] = intpos_arr_mid_imgs[1].copy()
intpos_arr_mid_imgs[-1] = intpos_arr_mid_imgs[-2].copy()

intpos_arr_imgs[intpos_layer_thick_px:\
    intpos_layer_thick_px + intpos_height_px] = intpos_arr_mid_imgs.copy()

intpos_arr_imgs[intpos_layer_thick_px + intpos_height_px: \
    intpos_total_height_px] = 170

i_global_start = i_global_end
i_global_end = i_global_start + intpos_total_height_px
imgs_arr[i_global_start:i_global_end] = intpos_arr_imgs

# Store the array slice indices in case needed for later
intpos_slice = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== CREATE MICRO-BUMP LAYERS ========== 

print(f"\nCreating micro-bump geometries...")

if pix_sz >= 3.5:
    micb_lead_width_um = 4 # Use with 4.0 um/pixel
else:
    micb_lead_width_um = 6 # Use with 2.0 um/pixel

micb_lead_width_px = int(np.round(micb_lead_width_um/pix_sz))

micb_lead_height_um = 48
micb_lead_height_px = int(np.round(micb_lead_height_um/pix_sz))

micb_radius_um = 10
micb_radius_px = int(np.round(micb_radius_um/pix_sz))

micb_height_um = 20
micb_height_px = int(np.round(micb_height_um/pix_sz))

micb_cent2cent_um = 50
micb_cent2cent_px = int(np.round(micb_cent2cent_um/pix_sz))

# Create the micro-bump feature without the leads first
micb_sph_height = int(np.round(0.667*micb_height_px))
micb_sph_radius = int(np.round(1.5*micb_radius_px))
micb_sph = 255*(dcl.draw_sphere(micb_sph_radius, micb_sph_height))
micb_sph_width = micb_sph.shape[1]

micb_pad_height = int(np.round((micb_height_px - micb_sph.shape[0])/2.0))
micb_pad = 170*(dcl.draw_cylinder(micb_radius_px, micb_pad_height))
micb_pad_height = micb_pad.shape[0]
micb_height_px = micb_pad_height + micb_sph_height + micb_pad_height

micb_img = 85*(np.ones((micb_height_px, micb_sph_width, micb_sph_width),
    dtype=np.uint8))

micb_mid_i = int(np.floor(micb_img.shape[0]/2.0))
micb_mid_r = int(np.floor(micb_img.shape[1]/2.0))
micb_mid_c = int(np.floor(micb_img.shape[2]/2.0))
micb_target = (micb_mid_i, micb_mid_r, micb_mid_c)

micb_img = dcl.insert_feature3d_in_img3d(micb_pad, micb_img, micb_target,
    ignore_z_centroid=True)

micb_img = np.flipud(micb_img)
micb_img = dcl.insert_feature3d_in_img3d(micb_pad, micb_img, micb_target,
    ignore_z_centroid=True)

micb_img[micb_img <= 0] = 85
micb_mask = micb_img.copy()

micb_mask = dcl.insert_feature3d_in_img3d(micb_sph, micb_mask, micb_target,
    ignore_z_centroid=False)

micb_img[micb_mask == 255] = 255

# Create the vertical lead to the micro-bump
micb_lead = 170*(np.ones((micb_lead_height_px, micb_lead_width_px, 
    micb_lead_width_px), dtype=np.uint8))

micb_full_height_px = micb_lead_height_px + micb_height_px
micb_img2 = 85*(np.ones((micb_full_height_px, micb_img.shape[1],
    micb_img.shape[2]), dtype=np.uint8))

micb_mid_i = int(np.floor(micb_img2.shape[0]/2.0))
micb_mid_r = int(np.floor(micb_img2.shape[1]/2.0))
micb_mid_c = int(np.floor(micb_img2.shape[2]/2.0))
micb_target = (micb_mid_i, micb_mid_r, micb_mid_c)

micb_img2 = dcl.insert_feature3d_in_img3d(micb_lead, micb_img2, micb_target,
    ignore_z_centroid=True)

micb_img2[micb_lead_height_px:micb_full_height_px] = micb_img

# Duplicate micro-bumps onto a 2D grid for this one layer
n_micb_rows = int(np.floor(n_rows/micb_cent2cent_px))
n_micb_cols = int(np.floor(n_cols/micb_cent2cent_px))

if (n_micb_rows%2) == 0:
    n_micb_rows -= 1

if (n_micb_cols%2) == 0:
    n_micb_cols -= 1

micb_grid_mat2d = dcl.generate_grid2d(micb_cent2cent_px, micb_cent2cent_px,
    n_micb_rows, n_micb_cols, staggered_rows=False, reshape2D=False)

micb_grid_2d = []

# Remove regions of the microbumps. Break up into 3-by-3 regions of bumps.
micb_third = int(np.floor(n_micb_rows/3))

# Keep TopRows-MiddleCols
r_min = 0
r_max = r_min + micb_third + 1
c_min = micb_third
c_max = c_min + micb_third + 1
micb_subgrid_mat2d = micb_grid_mat2d[r_min:r_max, c_min:c_max]

for m in range(micb_subgrid_mat2d.shape[0]):
    for n in range(micb_subgrid_mat2d.shape[1]):
        cur_r_coord = micb_subgrid_mat2d[m,n,0]
        cur_c_coord = micb_subgrid_mat2d[m,n,1]
        micb_grid_2d.append([cur_r_coord, cur_c_coord])

# Keep BottomRows-MiddleCols
r_min = micb_third + micb_third
r_max = r_min + micb_third + 1
c_min = micb_third
c_max = c_min + micb_third + 1
micb_subgrid_mat2d = micb_grid_mat2d[r_min:r_max, c_min:c_max]

for m in range(micb_subgrid_mat2d.shape[0]):
    for n in range(micb_subgrid_mat2d.shape[1]):
        cur_r_coord = micb_subgrid_mat2d[m,n,0]
        cur_c_coord = micb_subgrid_mat2d[m,n,1]
        micb_grid_2d.append([cur_r_coord, cur_c_coord])

# Keep MiddleRows-LeftCols
r_min = micb_third
r_max = r_min + micb_third + 1
c_min = 0
c_max = c_min + micb_third + 1
micb_subgrid_mat2d = micb_grid_mat2d[r_min:r_max, c_min:c_max]

for m in range(micb_subgrid_mat2d.shape[0]):
    for n in range(micb_subgrid_mat2d.shape[1]):
        cur_r_coord = micb_subgrid_mat2d[m,n,0]
        cur_c_coord = micb_subgrid_mat2d[m,n,1]
        micb_grid_2d.append([cur_r_coord, cur_c_coord])

# Keep MiddleRows-RightCols
r_min = micb_third
r_max = r_min + micb_third + 1
c_min = micb_third + micb_third
c_max = c_min + micb_third + 1
micb_subgrid_mat2d = micb_grid_mat2d[r_min:r_max, c_min:c_max]

for m in range(micb_subgrid_mat2d.shape[0]):
    for n in range(micb_subgrid_mat2d.shape[1]):
        cur_r_coord = micb_subgrid_mat2d[m,n,0]
        cur_c_coord = micb_subgrid_mat2d[m,n,1]
        micb_grid_2d.append([cur_r_coord, cur_c_coord])

micb_grid_2d = np.array(micb_grid_2d)

micb_arr_imgs = 85*(np.ones((micb_full_height_px, n_rows, n_cols),
    dtype=np.uint8))

micb_arr_imgs = dcl.insert_feature3d_via_grid2d(micb_img2, micb_arr_imgs,
    micb_grid_2d, center_grid=True)

i_global_start = i_global_end
i_global_end = i_global_start + micb_full_height_px
imgs_arr[i_global_start:i_global_end] = micb_arr_imgs

# Store the array slice indices in case needed for later
micb_slice1 = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== DUPLICATE THE MICRO-BUMP LAYERS ========== 

i_global_start = i_global_end
i_global_end = i_global_start + micb_full_height_px
imgs_arr[i_global_start:i_global_end] = micb_arr_imgs
micb_slice2 = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]

i_global_start = i_global_end
i_global_end = i_global_start + micb_full_height_px
imgs_arr[i_global_start:i_global_end] = micb_arr_imgs
micb_slice3 = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]

i_global_start = i_global_end
i_global_end = i_global_start + micb_full_height_px
imgs_arr[i_global_start:i_global_end] = micb_arr_imgs
micb_slice4 = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]

# Fill the remainder of the image with insulation pixels
i_global_start = i_global_end
i_global_end = n_imgs
imgs_arr[i_global_start:i_global_end] = 85
filler_end_slice = np.s_[i_global_start:i_global_end, 0:n_rows, 0:n_cols]


# ========== GO BACK AND ADD HOLES IN COPPER LAYERS ========== 

print(f"\nCreating random holes in Cu layers...")

cu_hole_radius_um = 90
cu_hole_radius_px = int(np.round(cu_hole_radius_um/pix_sz))

cu_hole_cent2cent_um = 500
cu_hole_cent2cent_px = int(np.round(cu_hole_cent2cent_um/pix_sz))

cu_hole_mask = 85*(dcl.draw_cylinder(cu_hole_radius_px, cu_thick_px))
cu_hole_mask[cu_hole_mask <= 0] = 170

n_cu_hole_rows = int(np.floor(n_rows/cu_hole_cent2cent_px))
n_cu_hole_cols = int(np.floor(n_cols/cu_hole_cent2cent_px))

if (n_cu_hole_rows%2) == 0:
    n_cu_hole_rows -= 1

if (n_cu_hole_cols%2) == 0:
    n_cu_hole_cols -= 1

n_cu_holes = n_cu_hole_rows*n_cu_hole_cols
cu_hole_grid2d = dcl.generate_grid2d(cu_hole_cent2cent_px, cu_hole_cent2cent_px,
    n_cu_hole_rows, n_cu_hole_cols)

cu_grid_cent_coord2d = np.round(np.mean(cu_hole_grid2d, axis=0))
row_shift = imgs_mid_r - cu_grid_cent_coord2d[0]
col_shift = imgs_mid_c - cu_grid_cent_coord2d[1]

for m, cur_coord in enumerate(cu_hole_grid2d):
    cu_hole_grid2d[m,0] = int(np.round(cur_coord[0] + row_shift))
    cu_hole_grid2d[m,1] = int(np.round(cur_coord[1] + col_shift))

# Now select only some percent of these grid points for the holes

# Cu Top Layer 2 & Cu Bottom Layer 5
rng_seed = 1990
cu_hole_pct_keep = 0.4

n_cu_holes_kept = int(np.round(cu_hole_pct_keep*n_cu_holes))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_cu_holes, size=n_cu_holes_kept)

cu_hole_grid_keep1 = np.zeros((n_cu_holes_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    cu_hole_grid_keep1[m] = cu_hole_grid2d[cur_i]

cu_layer_imgs = (imgs_arr[cu_top_slice2]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep1, center_grid=False)

imgs_arr[cu_top_slice2] = cu_layer_imgs

cu_layer_imgs = (imgs_arr[cu_bot_slice5]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep1, center_grid=False)

imgs_arr[cu_bot_slice5] = cu_layer_imgs

# Cu Top Layer 3 & Cu Bottom Layer 4
rng_seed = 1991
cu_hole_pct_keep = 0.4

n_cu_holes_kept = int(np.round(cu_hole_pct_keep*n_cu_holes))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_cu_holes, size=n_cu_holes_kept)

cu_hole_grid_keep2 = np.zeros((n_cu_holes_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    cu_hole_grid_keep2[m] = cu_hole_grid2d[cur_i]

cu_layer_imgs = (imgs_arr[cu_top_slice3]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep2, center_grid=False)

imgs_arr[cu_top_slice3] = cu_layer_imgs

cu_layer_imgs = (imgs_arr[cu_bot_slice4]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep2, center_grid=False)

imgs_arr[cu_bot_slice4] = cu_layer_imgs

# Cu Top Layer 4 & Cu Bottom Layer 3
rng_seed = 1989
cu_hole_pct_keep = 0.4

n_cu_holes_kept = int(np.round(cu_hole_pct_keep*n_cu_holes))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_cu_holes, size=n_cu_holes_kept)

cu_hole_grid_keep3 = np.zeros((n_cu_holes_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    cu_hole_grid_keep3[m] = cu_hole_grid2d[cur_i]

cu_layer_imgs = (imgs_arr[cu_top_slice4]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep3, center_grid=False)

imgs_arr[cu_top_slice4] = cu_layer_imgs

cu_layer_imgs = (imgs_arr[cu_bot_slice3]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep3, center_grid=False)

imgs_arr[cu_bot_slice3] = cu_layer_imgs

# Cu Top Layer 5 & Cu Bottom Layer 2
rng_seed = 1993
cu_hole_pct_keep = 0.4

n_cu_holes_kept = int(np.round(cu_hole_pct_keep*n_cu_holes))
rng = np.random.default_rng(seed=rng_seed)
rint_holes = rng.integers(low=0, high=n_cu_holes, size=n_cu_holes_kept)

cu_hole_grid_keep4 = np.zeros((n_cu_holes_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_holes):
    cu_hole_grid_keep4[m] = cu_hole_grid2d[cur_i]

cu_layer_imgs = (imgs_arr[cu_top_slice5]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep4, center_grid=False)

imgs_arr[cu_top_slice5] = cu_layer_imgs

cu_layer_imgs = (imgs_arr[cu_bot_slice2]).copy()

cu_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_hole_mask, cu_layer_imgs,
    cu_hole_grid_keep4, center_grid=False)

imgs_arr[cu_bot_slice2] = cu_layer_imgs


# ========== GO BACK AND ADD CU CONNECTS IN SOD LAYERS ========== 

print(f"\nCreating random Cu connects in SOD layers...")

cu_con_radius_um = 30
cu_con_radius_px = int(np.round(cu_con_radius_um/pix_sz))

cu_con_cent2cent_px = cu_hole_cent2cent_px

cu_con_mask = 170*(dcl.draw_cylinder(cu_con_radius_px, sod_thick_px))
cu_con_mask[cu_con_mask <= 0] = 85

n_cu_con_rows = n_cu_hole_rows + 1
n_cu_con_cols = n_cu_hole_cols + 1
n_cu_con = n_cu_con_rows*n_cu_con_cols
cu_con_grid2d_inner = dcl.generate_grid2d(cu_con_cent2cent_px, cu_con_cent2cent_px,
    n_cu_con_rows, n_cu_con_cols)

sod_grid_cent_coord2d = np.round(np.mean(cu_con_grid2d_inner, axis=0))
row_shift = imgs_mid_r - sod_grid_cent_coord2d[0]
col_shift = imgs_mid_c - sod_grid_cent_coord2d[1]

for m, cur_coord in enumerate(cu_con_grid2d_inner):
    cu_con_grid2d_inner[m,0] = int(np.round(cur_coord[0] + row_shift))
    cu_con_grid2d_inner[m,1] = int(np.round(cur_coord[1] + col_shift))

row_shift = -int(np.floor(cu_con_cent2cent_px/2.0))
col_shift = 0
cu_con_grid2d_topcp = cu_con_grid2d_inner.copy() + np.array([row_shift, col_shift])

row_shift = int(np.floor(cu_con_cent2cent_px/2.0))
col_shift = 0
cu_con_grid2d_botcp = cu_con_grid2d_inner.copy() + np.array([row_shift, col_shift])

row_shift = 0
col_shift = -int(np.floor(cu_con_cent2cent_px/2.0))
cu_con_grid2d_lftcp = cu_con_grid2d_inner.copy() + np.array([row_shift, col_shift])

row_shift = 0
col_shift = int(np.floor(cu_con_cent2cent_px/2.0))
cu_con_grid2d_rgtcp = cu_con_grid2d_inner.copy() + np.array([row_shift, col_shift])

cu_con_grid2d = np.concatenate((cu_con_grid2d_topcp, cu_con_grid2d_botcp, \
    cu_con_grid2d_lftcp, cu_con_grid2d_rgtcp, cu_con_grid2d_inner), axis=0)

cu_con_grid2d = dcl.del_duplicate_coords_2d(cu_con_grid2d, TOL=2.0)
n_cu_con = cu_con_grid2d.shape[0]

# SOD Top Layer 1 & SOD Bottom Layer 5
rng_seed = 2000
cu_con_pct_keep = 0.3

n_cu_con_kept = int(np.round(cu_con_pct_keep*n_cu_con))
rng = np.random.default_rng(seed=rng_seed)
rint_con = rng.integers(low=0, high=n_cu_con, size=n_cu_con_kept)

cu_con_grid_keep1 = np.zeros((n_cu_con_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_con):
    cu_con_grid_keep1[m] = cu_con_grid2d[cur_i]

sod_layer_imgs = (imgs_arr[sod_top_slice1]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep1, center_grid=False)

imgs_arr[sod_top_slice1] = sod_layer_imgs

sod_layer_imgs = (imgs_arr[sod_bot_slice5]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep1, center_grid=False)

imgs_arr[sod_bot_slice5] = sod_layer_imgs

# SOD Top Layer 2 & SOD Bottom Layer 4
rng_seed = 2001
cu_con_pct_keep = 0.3

n_cu_con_kept = int(np.round(cu_con_pct_keep*n_cu_con))
rng = np.random.default_rng(seed=rng_seed)
rint_con = rng.integers(low=0, high=n_cu_con, size=n_cu_con_kept)

cu_con_grid_keep2 = np.zeros((n_cu_con_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_con):
    cu_con_grid_keep2[m] = cu_con_grid2d[cur_i]

sod_layer_imgs = (imgs_arr[sod_top_slice2]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep2, center_grid=False)

imgs_arr[sod_top_slice2] = sod_layer_imgs

sod_layer_imgs = (imgs_arr[sod_bot_slice4]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep2, center_grid=False)

imgs_arr[sod_bot_slice4] = sod_layer_imgs

# SOD Top Layer 3 & SOD Bottom Layer 3
rng_seed = 2002
cu_con_pct_keep = 0.3

n_cu_con_kept = int(np.round(cu_con_pct_keep*n_cu_con))
rng = np.random.default_rng(seed=rng_seed)
rint_con = rng.integers(low=0, high=n_cu_con, size=n_cu_con_kept)

cu_con_grid_keep3 = np.zeros((n_cu_con_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_con):
    cu_con_grid_keep3[m] = cu_con_grid2d[cur_i]

sod_layer_imgs = (imgs_arr[sod_top_slice3]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep3, center_grid=False)

imgs_arr[sod_top_slice3] = sod_layer_imgs

sod_layer_imgs = (imgs_arr[sod_bot_slice3]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep3, center_grid=False)

imgs_arr[sod_bot_slice3] = sod_layer_imgs

# SOD Top Layer 4 & SOD Bottom Layer 2
rng_seed = 2003
cu_con_pct_keep = 0.3

n_cu_con_kept = int(np.round(cu_con_pct_keep*n_cu_con))
rng = np.random.default_rng(seed=rng_seed)
rint_con = rng.integers(low=0, high=n_cu_con, size=n_cu_con_kept)

cu_con_grid_keep4 = np.zeros((n_cu_con_kept,2), dtype=np.int32)
for m, cur_i in enumerate(rint_con):
    cu_con_grid_keep4[m] = cu_con_grid2d[cur_i]

sod_layer_imgs = (imgs_arr[sod_top_slice4]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep4, center_grid=False)

imgs_arr[sod_top_slice4] = sod_layer_imgs

sod_layer_imgs = (imgs_arr[sod_bot_slice2]).copy()

sod_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_con_mask, sod_layer_imgs,
    cu_con_grid_keep4, center_grid=False)

imgs_arr[sod_bot_slice2] = sod_layer_imgs


# ========== INSERT RING HOLES IN SOME CU LAYERS ========== 

ring_hole_oradius_um = 110
ring_hole_oradius_px = int(np.round(ring_hole_oradius_um/pix_sz))

ring_hole_iradius_um = 60
ring_hole_iradius_px = int(np.round(ring_hole_iradius_um/pix_sz))

cu_ring_mask = 85*(dcl.draw_cylinder(ring_hole_oradius_px, cu_thick_px,
    inner_radius=ring_hole_iradius_px))
cu_ring_mask[cu_ring_mask <= 0] = 170

ring_coords1 = dcl.find_common_coords2d(cu_con_grid_keep1, cu_con_grid_keep2, TOL=2.0)

cu_top_layer_imgs = (imgs_arr[cu_top_slice2]).copy()
cu_bot_layer_imgs = (imgs_arr[cu_bot_slice5]).copy()

cu_top_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_top_layer_imgs, ring_coords1, center_grid=False)
cu_bot_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_bot_layer_imgs, ring_coords1, center_grid=False)

imgs_arr[cu_top_slice2] = cu_top_layer_imgs.copy()
imgs_arr[cu_bot_slice5] = cu_bot_layer_imgs.copy()


ring_coords2 = dcl.find_common_coords2d(cu_con_grid_keep2, cu_con_grid_keep3, TOL=2.0)

cu_top_layer_imgs = (imgs_arr[cu_top_slice3]).copy()
cu_bot_layer_imgs = (imgs_arr[cu_bot_slice4]).copy()

cu_top_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_top_layer_imgs, ring_coords2, center_grid=False)
cu_bot_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_bot_layer_imgs, ring_coords2, center_grid=False)

imgs_arr[cu_top_slice3] = cu_top_layer_imgs.copy()
imgs_arr[cu_bot_slice4] = cu_bot_layer_imgs.copy()


ring_coords3 = dcl.find_common_coords2d(cu_con_grid_keep3, cu_con_grid_keep4, TOL=2.0)

cu_top_layer_imgs = (imgs_arr[cu_top_slice4]).copy()
cu_bot_layer_imgs = (imgs_arr[cu_bot_slice3]).copy()

cu_top_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_top_layer_imgs, ring_coords3, center_grid=False)
cu_bot_layer_imgs = dcl.insert_feature3d_via_grid2d(cu_ring_mask, 
    cu_bot_layer_imgs, ring_coords3, center_grid=False)

imgs_arr[cu_top_slice4] = cu_top_layer_imgs.copy()
imgs_arr[cu_bot_slice3] = cu_bot_layer_imgs.copy()


# ========== SAVE IMAGES ========== 

if pix_sz == 4.0:
    filepath_imgs_out = "./imgs_out_750p/simplified_chip_750p_Top.tif"
    filepath_imgs_side_out = "./imgs_out_750p/simplified_chip_750p_Side.tif"
    filepath_imgs_front_out = "./imgs_out_750p/simplified_chip_750p_Front.tif"
elif pix_sz == 2.0:
    filepath_imgs_out = "./imgs_out_1500p/simplified_chip_1500p_Top.tif"
    filepath_imgs_side_out = "./imgs_out_1500p/simplified_chip_1500p_Side.tif"
    filepath_imgs_front_out = "./imgs_out_1500p/simplified_chip_1500p_Front.tif"
else:
    filepath_imgs_out = "./imgs_out/simplified_chip_temp_Top.tif"
    filepath_imgs_side_out = "./imgs_out/simplified_chip_temp_Side.tif"
    filepath_imgs_front_out = "./imgs_out/simplified_chip_temp_Front.tif"

print(f"\nSaving Top images to:\n{filepath_imgs_out}")
tifffile.imwrite(filepath_imgs_out, imgs_arr, photometric='minisblack')

imgs_arr_side = np.moveaxis(imgs_arr, [0, 1, 2], [-1, 0, 1])
imgs_arr_side = np.rot90(imgs_arr_side, k=-1, axes=(1,2))

print(f"\nSaving Side images to:\n{filepath_imgs_side_out}")
tifffile.imwrite(filepath_imgs_side_out, imgs_arr_side, photometric='minisblack')

imgs_arr_front = np.moveaxis(imgs_arr, [0, 1, 2], [1, 2, 0])

print(f"\nSaving Front images to:\n{filepath_imgs_front_out}")
tifffile.imwrite(filepath_imgs_front_out, imgs_arr_front, photometric='minisblack')

print("\nScript finished successfully!")