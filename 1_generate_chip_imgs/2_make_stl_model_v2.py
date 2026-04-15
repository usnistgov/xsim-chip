import os.path
import numpy as np
from scipy import ndimage as ndim
import tifffile
from skimage import io
from skimage import measure as meas
from skimage import segmentation as seg
from skimage.util import img_as_ubyte, img_as_uint, img_as_float32
import trimesh

import img2stl
import imppy_lib as imp


def create_test_cube(width=5):
    cube_out = np.zeros((width,width,width), dtype=np.uint8)

    cube_out[0, 0:width, 0:width] = 255
    cube_out[-1, 0:width, 0:width] = 255

    cube_out[0:width, 0, 0:width] = 255
    cube_out[0:width, -1, 0:width] = 255

    cube_out[0:width, 0:width, 0] = 255
    cube_out[0:width, 0:width, -1] = 255

    return cube_out


# -------- USER INPUTS --------

use_marching_cubes = False

filepath_in = "./imgs_out_750p/simplified_chip_750p_Top.tif"
imgs_keep = (9999,)
voxel_size = 4.0 # um/pixel

separate_features = True
filepath_out_si = "./imgs_out_750p/feature_list_Si/simplified_chip_750p_vox_Si.stl"
filepath_out_cu = "./imgs_out_750p/feature_list_Cu/simplified_chip_750p_vox_Cu.stl"
filepath_out_sn = "./imgs_out_750p/feature_list_Sn/simplified_chip_750p_vox_Sn.stl"


# -------- IMPORT IMAGE SEQUENCE --------

filepath_in = os.path.abspath(filepath_in)
filepath_out_si = os.path.abspath(filepath_out_si)
filepath_out_cu = os.path.abspath(filepath_out_cu)
filepath_out_sn = os.path.abspath(filepath_out_sn)

imgs, imgs_props = imp.load_multipage_image(filepath_in, indices_in=imgs_keep,
   bigtiff=False, img_bitdepth_in="uint8", flipz=False, quiet_in=False)

imgs = img_as_ubyte(imgs)

# imgs = imgs[0:100]
# imgs = create_test_cube()

num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


#  -------- IMAGE SEGMENTATION/BINARIZATION --------

# # Mesh everything but the insulator
# imgs[imgs <= 100] = 0
# imgs[imgs >= 100] = 255

# Mesh insulator (primarily Si or SiO2)
imgs_si = imgs.copy()
imgs_si[imgs_si <= 50] = 0
imgs_si[imgs_si >= 100] = 0
imgs_si[imgs_si >= 50] = 255

# # Mesh interconnects (primarily Cu)
imgs_cu = imgs.copy()
imgs_cu[imgs_cu <= 100] = 0
imgs_cu[imgs_cu >= 200] = 0
imgs_cu[imgs_cu >= 100] = 255

# # Mesh solder (primarily Sn)
imgs_sn = imgs.copy()
imgs_sn[imgs_sn <= 200] = 0
imgs_sn[imgs_sn > 200] = 255


def create_mesh_of_voxel_boundaries(imgs_in, stl_filepath_out, separate_features=True, 
    use_marching_cubes=False):

    if separate_features:
        print(f"\nSeparating features...")

        label_arr = meas.label(imgs_in, connectivity=1)
        feat_list = meas.regionprops(label_arr)

        img_1feat = np.zeros(imgs_in.shape, dtype=np.uint8)

        print(f"\nLooping through {len(feat_list)} features to create individual STL files...")
        for m, cur_feat in enumerate(feat_list):
            img_1feat[:, :, :] = 0

            feat_id_str = str(int(m))
            feat_id_str = feat_id_str.zfill(5)

            filepath_root, filepath_ext = os.path.splitext(stl_filepath_out) 
            filepath_out2 = os.path.abspath(filepath_root + "_" + feat_id_str + filepath_ext)

            cur_coords = cur_feat.coords
            ii_arr = cur_coords[:, 0]
            rr_arr = cur_coords[:, 1]
            cc_arr = cur_coords[:, 2]

            img_1feat[ii_arr, rr_arr, cc_arr] = 255

            if use_marching_cubes:
                # Marching cubes algorithm
                verts, faces, normals, vals = imp.convert_voxels_to_surface(img_1feat, 
                    scale_spacing=voxel_size, g_sigdev=0.4)

                verts = np.float32(verts)
                faces = np.int32(faces)

            else:
                verts, faces = img2stl.make_boundary_mesh(img_1feat, quiet_bool=True)
                verts = (voxel_size*verts).astype(np.float32)

            # --- TriMesh Library for Merging Due to Speed and Memory Efficiency ---
            #print("\nCreating STL object and merging duplicate vertices...")
            mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh_obj.merge_vertices(merge_tex=False, merge_norm=False)

            n_verts = mesh_obj.vertices.shape[0]
            n_faces = mesh_obj.faces.shape[0]
            verts = mesh_obj.vertices
            faces = mesh_obj.faces

            #print(f"\nNumber of vertices after merging: {n_verts}")
            #print(f"Number of faces after merging: {n_faces}")

            #  -------- SAVE TRIMESH STL MODEL --------
            #print("\nSaving STL file...")
            mesh_obj.export(filepath_out2)

            if ((m+1)%10 == 0):
                print(f"  Completed {m+1}/{len(feat_list)} features...")

    else:
        if use_marching_cubes:

            # Marching cubes algorithm
            print(f"\nRunning marching cubes algorithm...")
            verts, faces, normals, vals = imp.convert_voxels_to_surface(imgs_in, 
                scale_spacing=voxel_size, g_sigdev=0.4)

            verts = np.float32(verts)
            faces = np.int32(faces)

        else:
            print(f"\nConstructing boundary mesh of the voxel model...")
            verts, faces = img2stl.make_boundary_mesh(imgs_in, quiet_bool=False)

            verts = (voxel_size*verts).astype(np.float32)


        # --- TriMesh Library for Merging Due to Speed and Memory Efficiency ---

        print("\nCreating STL object and merging duplicate vertices...")
        mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_obj.merge_vertices(merge_tex=False, merge_norm=False)

        n_verts = mesh_obj.vertices.shape[0]
        n_faces = mesh_obj.faces.shape[0]
        verts = mesh_obj.vertices
        faces = mesh_obj.faces

        print(f"\nNumber of vertices after merging: {n_verts}")
        print(f"Number of faces after merging: {n_faces}")

        #  -------- SAVE TRIMESH STL MODEL --------
        print("\nSaving STL file...")
        mesh_obj.export(stl_filepath_out)


create_mesh_of_voxel_boundaries(imgs_si, filepath_out_si, \
    separate_features=separate_features, \
    use_marching_cubes=use_marching_cubes)

create_mesh_of_voxel_boundaries(imgs_cu, filepath_out_cu, \
    separate_features=separate_features, \
    use_marching_cubes=use_marching_cubes)

create_mesh_of_voxel_boundaries(imgs_sn, filepath_out_sn, \
    separate_features=separate_features, \
    use_marching_cubes=use_marching_cubes)

print("\nScript finished successfully!")