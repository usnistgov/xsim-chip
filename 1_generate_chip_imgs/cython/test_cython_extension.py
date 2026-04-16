import numpy as np
#from stl import mesh
import img2stl
import trimesh


def create_test_cube(width=5):
    cube_out = np.zeros((width,width,width), dtype=np.uint8)

    cube_out[0, 0:width, 0:width] = 255
    cube_out[-1, 0:width, 0:width] = 255

    cube_out[0:width, 0, 0:width] = 255
    cube_out[0:width, -1, 0:width] = 255

    cube_out[0:width, 0:width, 0] = 255
    cube_out[0:width, 0:width, -1] = 255

    return cube_out

print(f"\nCreating 5x5x5 test cube...")
img3d_cube = create_test_cube(width=5)

print(f"\nConstructing boundary mesh...")
verts, faces = img2stl.make_boundary_mesh(img3d_cube)

print("\nMerging duplicate vertices and creating STL object...")
# Use the TriMesh library to automatically merge duplicate vertices
stl_obj = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

n_verts = stl_obj.vertices.shape[0]
n_faces = stl_obj.faces.shape[0]

print(f"\nNumber of vertices after merging: {n_verts}")
print(f"Number of faces after merging: {n_faces}")

# # Uncomment below to save this STL file 
# print("\nSaving STL file...")
# filepath_out = "./TEST_vox.stl"
# stl_obj.export(filepath_out)

print("\nScript finished successfully!")