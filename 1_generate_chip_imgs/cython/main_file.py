import numpy as np
#from stl import mesh
import img2stl
import trimesh
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


print(f"\nImporting 3D image stack...")

# filepath_in = "./simplified_chip_750p_xy.tif"
# imgs_keep = (9999,)
# voxel_size = 2.0 # um/pixel

# img3d, imgs_props = imp.load_multipage_image(filepath_in, indices_in=imgs_keep,
#     bigtiff=False, img_bitdepth_in="uint8", flipz=False, quiet_in=False)

# img3d[img3d >= 100] = 255
# img3d[img3d < 100] = 0

print(f"\nCreating 5x5x5 test cube...")
img3d_cube = create_test_cube(width=5)


print(f"\nConstructing boundary mesh...")
verts, faces = img2stl.make_boundary_mesh(img3d_cube)

print("\nMerging duplicate vertices and creating STL object...")

# #Numpy-STL routine, but does not merge duplicate vertices.
# stl_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, cur_face in enumerate(faces):
#     for j in range(3):
#         stl_obj.vectors[i][j] = verts[cur_face[j],:]
# print("\nSaving STL file...")
# filepath_out = "./TEST_vox.stl"
# stl_obj.save(filepath_out)

# Use the TriMesh library to automatically merge duplicate vertices
stl_obj = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

n_verts = stl_obj.vertices.shape[0]
n_faces = stl_obj.faces.shape[0]

print(f"\nNumber of vertices after merging: {n_verts}")
print(f"Number of faces after merging: {n_faces}")

print("\nSaving STL file...")
filepath_out = "./TEST_vox.stl"
stl_obj.export(filepath_out)

print("\nScript finished successfully!")