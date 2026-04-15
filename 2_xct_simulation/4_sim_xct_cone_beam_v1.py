# Import packages
import os
import numpy as np
from scipy.spatial import distance
import matplotlib # To plot images
import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import LogNorm, PowerNorm # Look up table
from skimage.util import img_as_uint, img_as_ubyte, img_as_float32
import tifffile
from gvxrPython3 import gvxr # Simulate X-ray images
from gvxrPython3.utils import loadXpecgenSpectrum

font = {'family' : 'serif',
         'size'   : 14
       }
matplotlib.rc('font', **font)

# Uncomment below to write out the help information of gvxr
#import contextlib
#def write_help(func, out_file):
#    with open(out_file, 'w') as f:
#        with contextlib.redirect_stdout(f):
#            help(func)
#write_help(gvxr, "./gvxr_help_output.txt")
#write_help(json2gvxr, "./json2gvxr_help_output.txt")


# Number of X-ray projections to take about 360 deg
total_projections = 2400

# Must end with a trailing slash
imgs_out_dir = "./sim_radios_cone_beam_pixsz4um/"

# Should not start with a slash. Do not include file extension here.
imgs_out_name = "radios_2400_chip_4um_cone"



# Print the version numbers of GVXR
print(gvxr.getVersionOfCoreGVXR())
print(gvxr.getVersionOfSimpleGVXR())

gvxr.useLogFile("./gvxr_log.txt")

# Create an OpenGL context
print("\nCreating an OpenGL context")
gvxr.createOpenGLContext()

print("\nSetting up the cone-beam x-ray source...")
# Ideal point source. If no focal spot desired, comment out "gvxr.setFocalSpot()"
gvxr.usePointSource()

# [Source position x, position y, position z, "unit"]
gvxr.setSourcePosition(-30.0, 0, 0, "mm")

# [Source position x, position y, position z, spot size, unit, discretization of spot size]
gvxr.setFocalSpot(-30.0, 0, 0, 0.005, "mm", 3)

# Create an X-ray spectrum at 160 kV with a 0.5 mm Cu filter
spectrum_filtered, k_filtered, f_filtered, units = loadXpecgenSpectrum(160, filters=[["Cu", 1.0]])
gvxr.setVoltage(160, "kV")
gvxr.setmAs(10.0) # milliampere second (going to normalize the image later, so does not really matter)

energy_bins = gvxr.getEnergyBins("keV")
photon_count = np.array(gvxr.getPhotonCountEnergyBins(), dtype=np.single)
photon_count /= photon_count.sum()

plt.figure(figsize=(8,6))
plt.bar(energy_bins, photon_count, width=1)
plt.xlabel('Energy in keV')
plt.ylabel('Probability distribution of photons per keV')
plt.tight_layout()
plt.show(block=False)
plt.pause(1.0)
plt.savefig(imgs_out_dir + "xray_spectrum.png")
plt.close()

# Detector settings 
print("\nSetting up the detector...")
# [Detector position x, y, z, "unit"]
gvxr.setDetectorPosition(907.3, 0, 0, "mm")
gvxr.setDetectorUpVector(0, 0, -1)
gvxr.setDetectorNumberOfPixels(1201, 1001)
gvxr.setDetectorPixelSize(0.125, 0.125, "mm")

# # Approximate Dirac with the heat kernel. Typical range: Let 0.01 <= epsilon <= 0.2
# # This simulates pixels "bleeding" into neighboring pixels
# epsilon = 0.025
# x = np.linspace(-2,2,41)
# coeff1 = 1.0/np.sqrt(2.0*(np.pi)*np.absolute(epsilon))
# coeff2 = -(np.power(x, 2)/(2.0*epsilon))
# lsf = coeff1*np.exp(coeff2)
# lsf /= np.sum(lsf)
# gvxr.setLSF(lsf)

print(f"\nLoading the Si mesh files...")

root_dir_si = "../1_generate_chip_imgs/imgs_out_750p/feature_list_Si/"
file_root_si = "simplified_chip_750p_vox_Si_"
n_files_si = 6

for m in range(n_files_si):
    file_num = str(int(m))
    file_num = file_num.zfill(5)
    cur_path = os.path.abspath(root_dir_si + file_root_si + file_num + ".stl")

    if not os.path.exists(cur_path):
        raise IOError(cur_path)

    cur_obj_name = "Si_" + file_num
    gvxr.loadMeshFile(cur_obj_name, cur_path, "um")
    #gvxr.translateNode(cur_obj_name, -1.5, -1.5, -1.5, "mm")
    #gvxr.setElement(cur_obj_name, "Si")
    gvxr.setMixture(cur_obj_name, [14, 8], [0.467, 0.533])
    gvxr.setDensity(cur_obj_name, 2.65,"g/cm3")

print(f"  Imported {n_files_si}/{n_files_si} Si files...")

print(f"\nLoading the Sn mesh files...")

root_dir_sn = "../1_generate_chip_imgs/imgs_out_750p/feature_list_Sn/"
file_root_sn = "simplified_chip_750p_vox_Sn_"
n_files_sn = 7368

for m in range(n_files_sn):
    file_num = str(int(m))
    file_num = file_num.zfill(5)
    cur_path = os.path.abspath(root_dir_sn + file_root_sn + file_num + ".stl")

    if not os.path.exists(cur_path):
        raise IOError(cur_path)

    cur_obj_name = "Sn_" + file_num
    gvxr.loadMeshFile(cur_obj_name, cur_path, "um")
    #gvxr.translateNode(cur_obj_name, -1.5, -1.5, -1.5, "mm")
    #gvxr.setElement(cur_obj_name, "Sn")
    gvxr.setMixture(cur_obj_name, [50, 47, 29], [0.965, 0.030, 0.005])
    gvxr.setDensity(cur_obj_name, 7.38,"g/cm3")

    if (((m+1)%500) == 0):
        print(f"  Imported {m+1}/{n_files_sn} Sn files...")
print(f"  Imported {n_files_sn}/{n_files_sn} Sn files...")

print(f"\nLoading the Cu mesh files...")

root_dir_cu = "../1_generate_chip_imgs/imgs_out_750p/feature_list_Cu/"
file_root_cu = "simplified_chip_750p_vox_Cu_"
n_files_cu = 7045

for m in range(n_files_cu):
    file_num = str(int(m))
    file_num = file_num.zfill(5)
    cur_path = os.path.abspath(root_dir_cu + file_root_cu + file_num + ".stl")

    if not os.path.exists(cur_path):
        raise IOError(cur_path)

    cur_obj_name = "Cu_" + file_num
    gvxr.loadMeshFile(cur_obj_name, cur_path, "um")
    #gvxr.translateNode(cur_obj_name, -1.5, -1.5, -1.5, "mm")
    gvxr.setElement(cur_obj_name, "Cu")

    if (((m+1)%500) == 0):
        print(f"  Imported {m+1}/{n_files_cu} Cu files...")
print(f"  Imported {n_files_cu}/{n_files_cu} Cu files...")

# Calculate pixel size and magnification
source_position = gvxr.getSourcePosition("mm")
detector_position = gvxr.getDetectorPosition("mm")
number_of_pixels = np.array(gvxr.getDetectorNumberOfPixels())
detector_size = np.array(gvxr.getDetectorSize("mm"))
detector_element_spacing = detector_size / number_of_pixels
object_bbox = gvxr.getNodeAndChildrenBoundingBox("root", "mm")
object_position = [(object_bbox[0] + object_bbox[3]) / 2,
    (object_bbox[1] + object_bbox[4]) / 2,
    (object_bbox[2] + object_bbox[5]) / 2]

source_detector_distance = distance.euclidean(source_position, detector_position)
source_object_distance = distance.euclidean(source_position, object_position)
magnification = source_detector_distance / source_object_distance
pixel_size = detector_element_spacing * source_object_distance / source_detector_distance

print(f"\nImage pixel size, [dx, dy]: {pixel_size} (mm/pixel)")
print(f"\nMagnification factor: {magnification}")

print("\nComputing X-ray images for CT acquisition...")

# Define the number of projections, along with the angle step
final_angle = 360.0
angle_step = final_angle / total_projections

x_ray_tmp = (np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()).astype(np.single)
x_ray_arr = np.zeros((total_projections, x_ray_tmp.shape[0], x_ray_tmp.shape[1]), dtype=x_ray_tmp.dtype)
x_ray_arr_16 = np.zeros((total_projections, x_ray_tmp.shape[0], x_ray_tmp.shape[1]), dtype=np.uint16)

# Rotate our object by angle_step for every projection, saving it to the results array.
for m in range(0, total_projections):

    # Save current angular projection
    x_ray_arr[m] = (np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()).astype(np.single)

    # Rotate models (in degrees)
    gvxr.rotateScene(angle_step,0,0,1)

    if ((m+1)%5) == 0:
        print(f"  Computed {m+1}/{total_projections} images...")
print(f"  Computed {total_projections}/{total_projections} images...")

print("\nNormalizing image stack...")
x_ray_arr = img_as_float32(x_ray_arr)

if np.amin(x_ray_arr) < 0.0:
    x_ray_arr = x_ray_arr + np.absolute(np.amin(x_ray_arr))

x_ray_arr = x_ray_arr / np.amax(x_ray_arr)
x_ray_arr = img_as_uint(x_ray_arr)

print("\nSaving image stack...")
cur_name = imgs_out_dir + imgs_out_name + "_16bit.tif"
tifffile.imwrite(cur_name, x_ray_arr, photometric='minisblack', compression='zlib')

gvxr.terminate()
print(f"\nScript finished successfully")