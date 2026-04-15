import os.path
import numpy as np
from scipy import ndimage as ndim
import tifffile
from skimage import io
from skimage import measure as meas
from skimage.util import img_as_ubyte, img_as_uint, img_as_float32
from stl import mesh


def pad_image_boundary(img_arr_in, cval_in=0, n_pad_in=1, quiet_in=False):
    """
    Adds enlarges the image array in all directions by one extra
    row of pixels, and then pads these locations with values equal
    to cval_in. So, for a 2D image of size equal to (rows, cols), 
    the output image with extended boundaries in all directions will
    be of size (rows + 2, cols + 2). In 3D, which is a sequence of
    images, (num_imgs, rows, cols), the output will be a similarly
    extended image array, (num_imgs + 2, rows + 2, cols + 2). Note,
    the SciKit-Image library has a more powerful function if it is
    needed, called skimage.util.pad().

    ---- INPUT ARGUMENTS ---- 
    [[img_arr_in]]: A 2D or 3D Numpy array representing the image 
        sequence. If 3D, is important that this is a Numpy array and not
        a Python list of Numpy matrices. If 3D, the shape of img_arr_in
        is expected to be as (num_images, num_pixel_rows,
        num_pixel_cols). If 2D, the shape of img_arr_in is expected to
        be as (num_pixel_rows, num_pixel_cols). It is expected that the
        image(s) are single-channel (i.e., grayscale), and the data
        type of the values are np.uint8.

    cval_in: Constant padding value to be used in the extended regions
        of the image array. This should be an integer between 0 and 255,
        inclusive. 

    n_pad_in: An integer value to determine how much to pad by. For 
        example, if img_arr_in is of shape (5, 10, 20), and n_pad is
        equal to 3, then three extra rows/columns will be created on
        all sides. The resulting size would be (5+3*2, 10+3*2, 20+3*2).

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [[img_arr_out]]: The image array with extended boundaries in all
        dimensions. The input image array will be in the center of this
        output array, and the data type will be uint8. 

    ---- SIDE EFFECTS ---- 
    The input array may be affected since a deep copy is not made in 
    order to be more memory efficient. Strings are printed to standard
    output. Nothing is written to the hard drive.
    """

    # ---- Start Local Copies ----
    img_arr = img_arr_in # Makes a new view -- NOT a deep copy
    cval = cval_in
    n_pad1 = np.around(n_pad_in).astype(np.int32)
    n_pad2 = (2*n_pad1).astype(np.int32)
    quiet = quiet_in
    # ---- End Start Local Copies ----

    if not quiet_in:
        print(f"\nExtending the boundaries of the image data...\n"\
            f"    Values in the padded regions will be set to: {cval}")

    img_shape = img_arr_in.shape
    if len(img_shape) == 3:
        num_imgs = img_shape[0]
        num_rows = img_shape[1]
        num_cols = img_shape[2]

        img_arr_out = np.ones((num_imgs+n_pad2, \
            num_rows+n_pad2, num_cols+n_pad2), dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_imgs+n_pad1, n_pad1:num_rows+n_pad1, \
            n_pad1:num_cols+n_pad1] = img_arr

    elif len(img_shape) == 2:
        num_rows = img_shape[0]
        num_cols = img_shape[1]

        img_arr_out = np.ones((num_rows+n_pad2, num_cols+n_pad2), \
            dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_rows+n_pad1, n_pad1:num_cols+n_pad1] = img_arr

    else:
        if not quiet_in:
            print(f"\nERROR: Can only pad the boundary of an image array "\
                f"in 2D or 3D.\nCurrent image shape is: {img_shape}")

    if not quiet_in:
        print(f"\nSuccesfully padded the image boundaries!")

    return img_arr_out


def load_multipage_image(path_in, indices_in=[], bigtiff=False,\
                         img_bitdepth_in="uint8", flipz=False,
                         quiet_in=False):
    """
    Read a multipage image (e.g. monolithic tiff stack) from file using
    the skimage library.
    
    ---- INPUT ARGUMENTS ----
    path_in [string]: Path to tiff file to load
    
    indices_in: An optional tuple of either length 1 or length 2. If 
        length 1, then it should contain a positive integer 
        corresponding to the number of images to be kept centered about
        the middle image of the image sequence. If length 2, then the
        first and second elements will be used directly to slice the
        list corresponding to the image sequence. For example, (0, 100)
        would keep the first 100 images. Fair warning, if you provide
        invalid indices, you will get an error.

    bigtiff [bool]: If False, the OpenCV multi-page TIFF function will
        be used to import the image. This is fine for standard TIFF 
        files. However, if the TIFF file is saved with a less 
        conventional format/header, such as for BigTIFF or ImageJ 
        Hyperstack formats, then this should be set to True in order to
        use a different importer. Note, these alternative TIFF formats
        should be used anytime the TIFF file is larger than 4 GB. When
        set to True, the tifffile library will be used via a plugin 
        within the Sci-Kit Image library.
    
    img_bitdepth_in [string]: Bit depth for the reader to use. Either
        of unsigned 8-bit (uint8) or unsigned 16-bit (uint16) are nominally
        supported. 8-bit is the default (and what the rest of the codes
        currently expect).

    flipz [bool]: When True, the images are reversed in the image stack,
        which can be thought of as flipping the image stack along the Z-
        direction. This occurs after any images have been removed from
        'indices_in' above. When False, the original order of the image
        stack is maintained. If the first image corresponds to the 
        bottom of a part, then in general, flipz should be True so as
        to ensure a right-handed coordinate system. Positive X will be
        along ascending column indices, positive Y will be along 
        ascending row indices, and positive Z will be along DESCENDING
        image indices.
        
    quiet_in [bool]: Set to true to suppress any output dialog
    
    --- RETURNED ---
    [img, img_prop]

    img: OpenCV Mats n-d array containing images (can be operated on
        like a numpy array). Its shape is [num_images, num_rows, 
        num_cols].

    img_prop: List of length three that describes the image
            properties.
        img_prop[0]: Total number of pixels in the image.
        img_prop[1]: A tuple of length three that provides the number 
            of rows, columns, and pages of pixels. 
        img_prop[2]: A string that confirms the image data type. 
            Since load_image(...) converts all images to 'uint8', 
            this will  always be equal to 'uint8'.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings may be
    printed to standard output.
    
    Warning: this loads the full image stack and then slices out the
        desired parts defined in indices_in. There maybe be a way to
        do this better ...
    
    """
    
    # ---- Start Local Copies ----
    # Forcing Python to create a new string variable in memory
    # File path to the image file
    file_path = (path_in + '.')[:-1]

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    file_path = os.path.normcase(file_path)
    
    quiet = quiet_in

    img_bitdepth = img_bitdepth_in.lower()

    indices_keep = list(indices_in) # Ex: (0, 100) keeps first 100 images

    # ---- End Local Copies ----


    if not quiet:
        print(f"\nImporting multi-page TIFF stack:\n   {file_path}")

    imgs = io.imread(fname=file_path, plugin='tifffile')

    # Catch the case that the file path was incorrect, or the image is
    # corrupted
    if (len(imgs) == 0) and (not quiet):
        print(f"\nERROR: Image not loaded from filepath:\n  {file_path}\n")
        # Use "raise Exception()" here too?
        return [None, None]

    imgs = np.array(imgs) # Convert to Numpy array if not already
    
    # Contains the number of row, columns, and pages in a 3D image
    num_imgs = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    image_shape = [num_imgs, num_rows, num_cols]

    # Returns the image data type (e.g., uint8)
    image_data_type = imgs[0].dtype
    
    # If uint16, convert to uint8 image data type
    if img_bitdepth == "uint8":
        # Use SciKit-Image to be more robust; is should handle n-d arrays
        imgs = img_as_ubyte(imgs)
        
    elif img_bitdepth == "uint16":
        # Use SciKit-Image to be more robust
        imgs = img_as_uint(imgs)
        
    else:
        if not quiet:
            print("\nWARNING: Unsupported bit depth detected. Defaulting to 8-bit")
        imgs = img_as_ubyte(imgs)
    
    if len(indices_in) == 0:
        # return full image
        indx_bounds = [0, num_imgs]
        if not quiet:
            print("\nReturning full image")

    elif len(indices_in) == 1:
        n_keep = indices_keep[0]
        cur_img_len = num_imgs # Total number of images

        if n_keep == 0: # Keeping zero images doesn't make sense
            n_keep = 1  # Will just keep the middle image then
        elif n_keep > cur_img_len: # Also doesn't make sense
            n_keep = cur_img_len   # So, keep them all

        # Index of middle element, rounds down if even
        if cur_img_len % 2 == 0: # If even
            mid_indx = int(cur_img_len/2) - 1
        else: # If odd
            mid_indx = int(cur_img_len/2)

        # Force the desired number of images to keep to be odd for now
        keep_even = False
        if n_keep % 2 == 0:
            keep_even = True
            n_keep -= 1

        # Number of images to keep above and below the middle index
        indx_rad = int((n_keep - 1)/2)

        # Index bounds for the images to be kept
        indx_bounds = [mid_indx - indx_rad, mid_indx + indx_rad + 1]

        # If originally even number of images to keep, correct it here
        if keep_even:
            indx_bounds[1] = indx_bounds[1] + 1

    # If there's two numbers, user gave the upper and lower numbers
    elif len(indices_in) == 2:
        indx_bounds = [indices_keep[0], indices_keep[1]]

    else:
        indx_bounds = [0, num_imgs]
        if not quiet: 
            print("\nWARNING: 'indices_in' is malformed. Returning full image")
    
    imgs = imgs[indx_bounds[0]:indx_bounds[1],:,:]

    # Reverse the order of the image stack
    if flipz:
        if not quiet:
            print(f"\nReversing the order of the image stack (i.e.," \
                + " flipping the Z-direction)...")
        imgs = np.flip(imgs, axis=0)

    # Update image parameters in case any changes were made
    image_size = imgs.size

    num_imgs = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    image_shape = [num_imgs, num_rows, num_cols]

    image_data_type = imgs.dtype
    img_prop = [image_size, image_shape, image_data_type]

    if not quiet:
        print(f"\nSuccessfully imported image: {file_path}")

    # Return the image objects and image properties
    return [imgs, img_prop]


def convert_voxels_to_surface(img_arr_in, iso_level=125, scale_spacing=1.0,
    is_binary=True, g_sigdev=0.8, pad_boundary=True):
    """
    Applies a marching cubes algorithm to an image sequence (i.e., a
    voxel model) in order to calculate an isosurface. The algorithm used
    is the SciKit-Image algorithm, skimage.measure.marching_cubes(). It
    is based on the Lewiner et al. approach:

    Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
      Efficient implementation of Marching Cubes’ cases with topological 
      guarantees. Journal of Graphics Tools 8(2) pp. 1-15 (December 2003).
      DOI:10.1080/10867651.2003.10487582

    No additional smoothing or modification to the resultant mesh is 
    performed. If this is desired, then consider making a VTK surface
    mesh and using Laplacian smoothing, which is all available with 
    little effort in make_vtk_surf_mesh() defined below. To measure
    the surface area, use skimage.measure.mesh_surface_area(). Or, 
    create a VTK surface mesh using make_vtk_surf_mesh() and then
    retrieve the pyvista.PolyData.area property.

    ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        Specifically, intensities should range from 0 to 255. If it is
        segmented (i.e., binarized), then only values of 0 and 255 
        should be present (NOT 0 and 1).

    iso_level: A scalar number used to define the value for the iso-
        surface. For a binarized image sequence, a value in between 
        0 and 255, like 125, is recommended. For grayscale image
        sequence, any value can be used, and the corresponding 
        isosurface will be approximated.

    scale_spacing: A scalar float that represents the distance between
        the voxels before they are converted to a surface. If this is
        an X-ray CT image sequence, this represents the voxel size.

    is_binary: Set to True if the image sequence is binarized. That is,
        if it contains only values of 0 and 255. Using the binaraized
        image sequence directly into the marching cubes algorithm will
        result in a very poor isosurface. So, if is_binary is True, 
        then a small amount of Gaussian blurring is performed which makes 
        for a much smoother gradient between black and white. This
        vastly improves the accuracy of the marching cubes algorithm.

    g_sigdev: A single float that corresponds to the standard deviation
        used as input for the 3D Gaussian blur that must be applied for
        a binary image sequence. If is_binary is False, this value is
        ignored. The same standard deviation will be used for all three
        axes. Typical values range between 0.5 and 1.5. This blur helps
        to stabilize the marching cubes algorithm, but it also has the
        side effect of smoothing edges by a small amount.

    pad_boundary: Set to True to extend the boundaries of the image
        sequence in all three dimensions. More specifically, in all
        six directions of the rectangular volume defined by the image
        sequence. The six boundaries will be extended by some amount. 
        The motivation to do this is to ensure
        there are no white pixels along the boundaries of the image
        sequence. Otherwise, the resultant surface will not be a 
        closed at the boundaries. If there are no white pixels along the
        boundaries to begin with, then there is no need for padding.
        
    ---- RETURNED ----
    [[verts]]: A 2D array that contains the coordinates of the mesh 
        vertices. The array will have 3 columns, such that each row
        will provide the [X, Y, Z] coordinates of that vertex. There
        will be V unique mesh vertices. So, the size is (V, 3).

    [[faces]]: A 2D array that defines the connectivity of the 
        triangular faces. There will be F faces, and for each face,
        three nodes are defined based on the indices of verts. So, the
        size is (F, 3).

    [[normals]]: A 2D array that defines the normal vectors at each
        vertex. So, the size is (V, 3). Each normal vector contains
        a [X, Y, Z] components.

    [values]: A 1D array containing the approximate pixel values
        at each vertex. So, the size is (1, V)

    ---- SIDE EFFECTS ---- 
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    img_arr = np.transpose(img_arr_in) # New view, but not deep copy

    if is_binary:
        # Blurring something white at the edge does not create a 
        # smooth gradient to black. So, pad the edges with black
        # pixels by making the image sequence larger.
        img_arr = pad_image_boundary(img_arr, n_pad_in=9, quiet_in=True)

        # Apply a little blur to get better isosurfaces on a binary
        # image sequence. Need smoother gradients
        g_sigma = (g_sigdev, g_sigdev, g_sigdev) 
        img_arr = img_as_float32(img_arr)
        img_arr = ndim.gaussian_filter(img_arr, g_sigma)
        img_arr = img_as_ubyte(img_arr)

    if pad_boundary:
        # If an isosurface comes to the edge, the resultant mesh is left
        # with an open hole. By padding the image array with black pixels
        # (by making it a little larger), the isosurface will be closed.
        img_arr = pad_image_boundary(img_arr, n_pad_in=1, quiet_in=True)

    # Apply marching cubes algorithm
    # verts is a (V,3) array. 
    # faces is a (F,3) array.
    # normals is a (V,3) array.
    # values is a (V,) array
    verts, faces, normals, values = meas.marching_cubes(img_arr, 
        level=iso_level, spacing=(scale_spacing, scale_spacing, scale_spacing), 
        gradient_direction='descent', step_size=1, 
        allow_degenerate=False, method='lewiner')

    # Offset the vertices by the amount of padding
    if is_binary:
        verts[:,0] = verts[:,0] - 9*scale_spacing
        verts[:,1] = verts[:,1] - 9*scale_spacing
        verts[:,2] = verts[:,2] - 9*scale_spacing

    if pad_boundary:
        verts[:,0] = verts[:,0] - 1*scale_spacing
        verts[:,1] = verts[:,1] - 1*scale_spacing
        verts[:,2] = verts[:,2] - 1*scale_spacing

    return [verts, faces, normals, values]