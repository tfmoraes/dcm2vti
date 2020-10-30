import gdcm
import numpy as np
import vtk
from vtk.util import numpy_support


def to_vtk(
    n_array,
    spacing=(1.0, 1.0, 1.0),
    slice_number=0,
    orientation="AXIAL",
    origin=(0, 0, 0),
    padding=(0, 0, 0),
):
    if orientation == "SAGITTAL":
        orientation = "SAGITAL"

    try:
        dz, dy, dx = n_array.shape
    except ValueError:
        dy, dx = n_array.shape
        dz = 1

    px, py, pz = padding

    v_image = numpy_support.numpy_to_vtk(n_array.flat)

    if orientation == "AXIAL":
        extent = (
            0 - px,
            dx - 1 - px,
            0 - py,
            dy - 1 - py,
            slice_number - pz,
            slice_number + dz - 1 - pz,
        )
    elif orientation == "SAGITAL":
        dx, dy, dz = dz, dx, dy
        extent = (
            slice_number - px,
            slice_number + dx - 1 - px,
            0 - py,
            dy - 1 - py,
            0 - pz,
            dz - 1 - pz,
        )
    elif orientation == "CORONAL":
        dx, dy, dz = dx, dz, dy
        extent = (
            0 - px,
            dx - 1 - px,
            slice_number - py,
            slice_number + dy - 1 - py,
            0 - pz,
            dz - 1 - pz,
        )

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDimensions(dx, dy, dz)
    # SetNumberOfScalarComponents and SetScalrType were replaced by
    # AllocateScalars
    #  image.SetNumberOfScalarComponents(1)
    #  image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
    image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype), 1)
    image.SetExtent(extent)
    image.GetPointData().SetScalars(v_image)

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)

    return image_copy


def get_gdcm_to_numpy_typemap():
    """Returns the GDCM Pixel Format to numpy array type mapping."""
    _gdcm_np = {
        gdcm.PixelFormat.UINT8: np.uint8,
        gdcm.PixelFormat.INT8: np.int8,
        # gdcm.PixelFormat.UINT12 :np.uint12,
        # gdcm.PixelFormat.INT12  :np.int12,
        gdcm.PixelFormat.UINT16: np.uint16,
        gdcm.PixelFormat.INT16: np.int16,
        gdcm.PixelFormat.UINT32: np.uint32,
        gdcm.PixelFormat.INT32: np.int32,
        # gdcm.PixelFormat.FLOAT16:np.float16,
        gdcm.PixelFormat.FLOAT32: np.float32,
        gdcm.PixelFormat.FLOAT64: np.float64,
    }
    return _gdcm_np


def get_numpy_array_type(gdcm_pixel_format):
    """Returns a numpy array typecode given a GDCM Pixel Format."""
    return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]


# Based on http://gdcm.sourceforge.net/html/ConvertNumpy_8py-example.html
def gdcm_to_numpy(image, apply_intercep_scale=False):
    pf = image.GetPixelFormat()
    if image.GetNumberOfDimensions() == 3:
        shape = (
            image.GetDimension(2),
            image.GetDimension(1),
            image.GetDimension(0),
            pf.GetSamplesPerPixel(),
        )
    else:
        shape = image.GetDimension(1), image.GetDimension(0), pf.GetSamplesPerPixel()
    dtype = get_numpy_array_type(pf.GetScalarType())
    gdcm_array = image.GetBuffer()
    np_array = np.frombuffer(
        gdcm_array.encode("utf-8", errors="surrogateescape"), dtype=dtype
    )
    np_array.shape = shape
    np_array = np_array.squeeze()

    if apply_intercep_scale:
        shift = image.GetIntercept()
        scale = image.GetSlope()
        output = np.empty_like(np_array, np.int16)
        output[:] = scale * np_array + shift
        return output
    else:
        return np_array
