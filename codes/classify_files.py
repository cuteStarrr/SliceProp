import nibabel as nib
import os
import numpy as np
import h5py

def crop_image(image, label, crop_size):
    """
    centor crop based on label
    :param image: nparray needs to be cropped
    :param label: corresponding label of image
    :param crop_size: final size of image
    :return: cropped image and cropped label
    """
    x,y,z = np.where(label > 0)

    return image[int(x.mean()) - int(crop_size/2):int(x.mean()) + int(crop_size/2), int(y.mean()) - int(crop_size/2):int(y.mean()) + int(crop_size/2),:], label[int(x.mean()) - int(crop_size/2):int(x.mean()) + int(crop_size/2), int(y.mean()) - int(crop_size/2):int(y.mean()) + int(crop_size/2), :]
    

def classify_files(inpath, outpath, total_file = 193, crop_size = 256):
    str_image = r"_image"
    str_label = r"_label"
    str_insuffix = r".nii.gz"
    str_outsuffix = r".h5"

    for cur_pic in range(total_file):
        image_path = inpath + str(cur_pic) + str_image + str_insuffix
        label_path = inpath + str(cur_pic) + str_label + str_insuffix
        if os.path.exists(image_path) and os.path.exists(label_path):
            image_obj = nib.load(image_path)
            label_obj = nib.load(label_path)
            image_data = image_obj.get_fdata()
            label_data = label_obj.get_fdata()

            image_data, label_data = crop_image(image_data, label_data, crop_size)
            save_path = outpath
            if label_data.max() > 2:
                save_path = save_path + "three_class/" + str(cur_pic) + str_outsuffix
            else:
                save_path = save_path + "two_class/" + str(cur_pic) + str_outsuffix

            hf = h5py.File(save_path, 'w')
            hf.create_dataset('image', data=image_data)
            hf.create_dataset('label', data=label_data)
            hf.close()

if __name__ == '__main__':
    classify_files("/data/luwenjing/dataset/ImageTBAD/", "/data/xuxin/ImageTBAD_processed/", 193, 256)

