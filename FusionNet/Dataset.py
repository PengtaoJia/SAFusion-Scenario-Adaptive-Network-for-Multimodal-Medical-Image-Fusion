from os import listdir
import torchvision.transforms as transform
import torch.utils.data as data
from PIL import Image
from os.path import join


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp",".gif"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y,cb,cr


def mri_transform():
    return transform.Compose([
        transform.Resize(256),
        transform.ToTensor(),
    ])


class Medical_Dataset(data.Dataset):
    def __init__(self,Path_MRI=None,Path_CT=None,transform=mri_transform()):
        super(Medical_Dataset,self).__init__()
        self.MRI_filenames = [join(Path_MRI, x) for x in listdir(Path_MRI) if is_image_file(x)]
        self.CT_filenames = [join(Path_CT, x) for x in listdir(Path_CT) if is_image_file(x)]
        self.MRI_name = [x for x in listdir(Path_MRI) if is_image_file(x)]
        self.CT_name = [x for x in listdir(Path_CT) if is_image_file(x)]

        self.transform=transform

    def __getitem__(self, item):
        mri,mri_cb,mri_cr = load_img(self.MRI_filenames[item])
        ct,ct_cb,ct_cr = load_img(self.CT_filenames[item])
        mri_name=self.MRI_name[item]
        if self.transform is not None:
            mri = self.transform(mri)
            ct = self.transform(ct)

            ct_cb = self.transform(ct_cb)
            ct_cr = self.transform(ct_cr)
        return mri,ct,mri_name,ct_cb,ct_cr

    def __len__(self):
        return len(self.MRI_filenames)

