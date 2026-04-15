import os.path
import numpy as np
import trimesh


def calc_bounds_trimesh_stl_list(stl_obj_list):

    x_max = -9.9E12
    y_max = -9.9E12
    z_max = -9.9E12
    x_min = 9.9E12
    y_min = 9.9E12
    z_min = 9.9E12

    for cur_stl in stl_obj_list:

        cur_verts = cur_stl.vertices
        cur_x_max = np.amax(cur_verts[:,0])
        cur_y_max = np.amax(cur_verts[:,1])
        cur_z_max = np.amax(cur_verts[:,2])
        cur_x_min = np.amin(cur_verts[:,0])
        cur_y_min = np.amin(cur_verts[:,1])
        cur_z_min = np.amin(cur_verts[:,2])

        if cur_x_max > x_max:
            x_max = cur_x_max

        if cur_y_max > y_max:
            y_max = cur_y_max

        if cur_z_max > z_max:
            z_max = cur_z_max

        if cur_x_min < x_min:
            x_min = cur_x_min

        if cur_y_min < y_min:
            y_min = cur_y_min

        if cur_z_min < z_min:
            z_min = cur_z_min

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def translate_trimesh_stl_list(stl_obj_list, delta_vec):

    stl_obj_list_out = []

    for cur_stl in stl_obj_list:

        new_verts = cur_stl.vertices + delta_vec
        new_stl = trimesh.Trimesh(vertices=new_verts, faces=cur_stl.faces, \
            process=False)

        stl_obj_list_out.append(new_stl)

    return stl_obj_list_out



# ========== USER PARAMETERS ==========

import_separated_stl_files = True

if import_separated_stl_files:

    sn_stl_dir_path =  "./imgs_out_750p/feature_list_Sn/"
    sn_root_filename = "simplified_chip_750p_vox_Sn_"
    n_files_sn = 7368

    cu_stl_dir_path =  "./imgs_out_750p/feature_list_Cu/"
    cu_root_filename = "simplified_chip_750p_vox_Cu_"
    n_files_cu = 7045

    si_stl_dir_path =  "./imgs_out_750p/feature_list_Si/"
    si_root_filename = "simplified_chip_750p_vox_Si_"
    n_files_si = 6

    # Importing STL models
    print(f"\nImporting STL models...")

    mesh_si_list = []
    for m in range(n_files_si):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(si_stl_dir_path + si_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = trimesh.load_mesh(cur_path)
        mesh_si_list.append(mesh_obj)

        if (((m+1)%200) == 0):
            print(f"  Imported {m+1}/{n_files_si} Si files...")
    print(f"  Imported {n_files_si}/{n_files_si} Si files...")

    mesh_cu_list = []
    for m in range(n_files_cu):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(cu_stl_dir_path + cu_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = trimesh.load_mesh(cur_path)
        mesh_cu_list.append(mesh_obj)

        if (((m+1)%200) == 0):
            print(f"  Imported {m+1}/{n_files_cu} Cu files...")
    print(f"  Imported {n_files_cu}/{n_files_cu} Cu files...")

    mesh_sn_list = []
    for m in range(n_files_sn):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(sn_stl_dir_path + sn_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = trimesh.load_mesh(cur_path)
        mesh_sn_list.append(mesh_obj)

        if (((m+1)%200) == 0):
            print(f"  Imported {m+1}/{n_files_sn} Sn files...")
    print(f"  Imported {n_files_sn}/{n_files_sn} Sn files...")

    mesh_obj_list = mesh_sn_list + mesh_cu_list + mesh_si_list

else:
    # Parameters for importing a select number of STL files 
    sn_stl_path = "./imgs_out_750p/simplified_chip_750p_vox_Sn_All.stl"
    cu_stl_path = "./imgs_out_750p/simplified_chip_750p_vox_Cu_All.stl"
    si_stl_path = "./imgs_out_750p/simplified_chip_750p_vox_Si_All.stl"

    sn_stl_path = os.path.abspath(sn_stl_path)
    cu_stl_path = os.path.abspath(cu_stl_path)
    si_stl_path = os.path.abspath(si_stl_path)

    # Importing STL models
    print(f"\nImporting STL models...")
    mesh_sn = trimesh.load_mesh(sn_stl_path)
    mesh_cu = trimesh.load_mesh(cu_stl_path)
    mesh_si = trimesh.load_mesh(si_stl_path)

    mesh_obj_list = [mesh_sn, mesh_cu, mesh_si]

print(f"\nCalculating the middle point of the model...")
bounds_arr = calc_bounds_trimesh_stl_list(mesh_obj_list)
x_min = bounds_arr[0]
x_max = bounds_arr[1]
y_min = bounds_arr[2]
y_max = bounds_arr[3]
z_min = bounds_arr[4]
z_max = bounds_arr[5]

x_mid = (x_max + x_min)/2.0
y_mid = (y_max + y_min)/2.0
z_mid = (z_max + z_min)/2.0
pnt_mid = np.array([x_mid, y_mid, z_mid])

# Should be that offset_arr = [-1500.0, -1500.0, -1500.0]
offset_arr = -pnt_mid 

# Uncomment for reverse translation if ever needed 
# offset_arr = np.array([1500.0, 1500.0, 1500.0])

print(f"\nTranslation vector: {offset_arr}")
if import_separated_stl_files:
    mesh_sn_list = translate_trimesh_stl_list(mesh_sn_list, offset_arr)
    mesh_cu_list = translate_trimesh_stl_list(mesh_cu_list, offset_arr)
    mesh_si_list = translate_trimesh_stl_list(mesh_si_list, offset_arr)

    print(f"\nSaving the new STL models...")
    # Overwrite existing files

    for m in range(n_files_si):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(si_stl_dir_path + si_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = mesh_si_list[m]
        mesh_obj.export(cur_path)

        if (((m+1)%200) == 0):
            print(f"  Exported {m+1}/{n_files_si} Si files...")
    print(f"  Exported {n_files_si}/{n_files_si} Si files...")

    for m in range(n_files_cu):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(cu_stl_dir_path + cu_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = mesh_cu_list[m]
        mesh_obj.export(cur_path)

        if (((m+1)%200) == 0):
            print(f"  Exported {m+1}/{n_files_cu} Cu files...")
    print(f"  Exported {n_files_cu}/{n_files_cu} Cu files...")

    for m in range(n_files_sn):
        file_num = str(int(m))
        file_num = file_num.zfill(5)
        cur_path = os.path.abspath(sn_stl_dir_path + sn_root_filename \
            + file_num + ".stl")

        if not os.path.exists(cur_path):
            raise IOError(cur_path)

        mesh_obj = mesh_sn_list[m]
        mesh_obj.export(cur_path)

        if (((m+1)%200) == 0):
            print(f"  Exported {m+1}/{n_files_sn} Sn files...")
    print(f"  Exported {n_files_sn}/{n_files_sn} Sn files...")

else:
    mesh_obj_list = translate_trimesh_stl_list(mesh_obj_list, offset_arr)
    mesh_sn = mesh_obj_list[0]
    mesh_cu = mesh_obj_list[1]
    mesh_si = mesh_obj_list[2]

    print(f"\nSaving the new STL models...")
    # Overwrite existing files
    mesh_sn.export(sn_stl_path_in) 
    mesh_cu.export(cu_stl_path_in)
    mesh_si.export(si_stl_path_in)

print(f"\nScript finished successfully!")