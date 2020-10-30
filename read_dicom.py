import pathlib
import sys

import converters
import gdcm
import numpy as np
import vtk

tag_patient_id = gdcm.Tag(0x0010, 0x0020)
tag_patient_name = gdcm.Tag(0x0010, 0x0010)
tag_series_number = gdcm.Tag(0x0020, 0x0011)
tag_image_rows = gdcm.Tag(0x0028, 0x0010)
tag_image_columns = gdcm.Tag(0x0028, 0x0011)
tag_image_position = gdcm.Tag(0x0020, 0x0032)


def dcm_str_to_list(dcm_str):
    return [float(i) for i in dcm_str.split("\\")]


def get_tag_value(file, tag):
    sf = gdcm.StringFilter()
    sf.SetFile(file)
    return sf.ToString(tag)


def get_files_in_folder(input_folder):
    input_folder = pathlib.Path(input_folder).absolute()
    filenames = []
    for filename in input_folder.rglob("*"):
        if filename.is_file():
            filenames.append(str(filename))
    return filenames


def split_by_patient_series(filenames):
    patient_series = {}
    for filename in filenames:
        reader = gdcm.Reader()
        reader.SetFileName(filename)
        if reader.Read():
            # Getting patient ID, patient name and serie id
            f = reader.GetFile()
            ds = f.GetDataSet()
            patient_id = get_tag_value(f, tag_patient_id)
            patient_name = get_tag_value(f, tag_patient_name)
            serie_id = get_tag_value(f, tag_series_number)
            rows = int(get_tag_value(f, tag_image_rows))
            cols = int(get_tag_value(f, tag_image_columns))
            image_position = dcm_str_to_list(get_tag_value(f, tag_image_position))

            if patient_id in patient_series:
                if serie_id in patient_series[patient_id]["series"]:
                    patient_series[patient_id]["series"][serie_id][filename] = {
                        "rows": rows,
                        "cols": cols,
                        "image_position": image_position,
                    }
                else:
                    patient_series[patient_id]["series"][serie_id] = {
                        filename: {
                            "rows": rows,
                            "cols": cols,
                            "image_position": image_position,
                        }
                    }
            else:
                patient_series[patient_id] = {
                    "patient_name": patient_name,
                    "series": {
                        serie_id: {
                            filename: {
                                "rows": rows,
                                "cols": cols,
                                "image_position": image_position,
                            }
                        }
                    },
                }

    return patient_series


def get_largest_serie(patient_series):
    largest_value = 0
    largest = None
    for patient_id in patient_series:
        for serie_id in patient_series[patient_id]["series"]:
            serie_size = len(patient_series[patient_id]["series"][serie_id])
            if serie_size > largest_value:
                largest = (patient_id, serie_id)
                largest_value = serie_size
    return largest


def calc_zspacing(filenames, serie):
    dist_sum = 0.0
    for filename1, filename2 in zip(filenames, filenames[1::]):
        pos1 = np.array(serie[filename1]["image_position"])
        pos2 = np.array(serie[filename2]["image_position"])
        dist_sum += ((pos2 - pos1) ** 2).sum() ** 0.5
    return dist_sum / (len(serie) - 1)


def read_images(filenames):
    images = []
    spacing = None
    sorter = gdcm.IPPSorter()
    sorter.Sort(list(filenames.keys()))
    sorted_filenames = sorter.GetFilenames()
    sz = calc_zspacing(sorted_filenames, filenames)
    dx = np.array([i["cols"] for i in filenames.values()])
    dy = np.array([i["rows"] for i in filenames.values()])
    dz = len(filenames)
    dimensions = (dx[0], dy[0], dz)
    assert np.unique(dx).size == 1
    assert np.unique(dy).size == 1
    dx = dx[0]
    dy = dy[0]
    np_image = np.empty(shape=(dz, dy, dx), dtype=np.int16)
    for n, filename in enumerate(sorted_filenames):
        reader = gdcm.ImageReader()
        reader.SetFileName(filename)
        if reader.Read():
            image = reader.GetImage()
            np_image[n] = converters.gdcm_to_numpy(image).reshape(dy, dx)
            if spacing is None:
                sx, sy = image.GetSpacing()[:2]
                spacing = (sx, sy, sz)
    return np_image, spacing



def save(np_image, spacing, filename):
    vtk_image = converters.to_vtk(np_image, spacing)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(vtk_image)
    writer.SetFileName(filename)
    writer.Write()


def main():
    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    filenames = get_files_in_folder(input_folder)
    patient_series = split_by_patient_series(filenames)
    patient_id, serie_id = get_largest_serie(patient_series)
    serie_filenames = patient_series[patient_id]["series"][serie_id]
    np_image, spacing = read_images(serie_filenames)
    save(np_image, spacing, output_file)


if __name__ == "__main__":
    main()
