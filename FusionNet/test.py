from glob import glob
from utils import *
from AutoEncoder.models.model import Encoder_Decoder
from FusionNet.FusionLayer import Fusion_network

fusion_model_path = "../checkpoints/fusion_model.pth"
auto_encoder_path = '../AutoEncoder/checkpoints/autoencoder_model.pth'
# -------------------------设置随机数种子--------------------------
setup_seed(42)
# -----------------------设置环境------------------------
device = 'cuda:0'
# -----------------------加载数据集------------------------------------

Path_MRI='../images/MRI-CT/MRI/'
Path_CT='../images/MRI-CT/CT/'
Path_MRI1='../images/MRI-PET/MRI/'
Path_PET='../images/MRI-PET/PET/'
Path_MRI2='../images/MRI-SPECT/MRI/'
Path_SPECT='../images/MRI-SPECT/SPECT/'

# ---------------------------定义模型-------------------------------------
nb_filter = [32,64, 128,256]

with torch.no_grad():
    model = Encoder_Decoder()
    model.load_state_dict(torch.load(auto_encoder_path))
    model.eval()

    fusion_model = Fusion_network(nb_filter)
    fusion_model.load_state_dict(torch.load(fusion_model_path))
    fusion_model.eval()

model = model.to(device)
fusion_model = fusion_model.to(device)

def test():
    print('Start testing.....')

    fusion_model.load_state_dict(torch.load(fusion_model_path))
    test_one_epoch(Path_MRI,Path_CT,0)
    test_one_epoch(Path_MRI1,Path_PET,1)
    test_one_epoch(Path_MRI2,Path_SPECT,2)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp",".gif"])

def test_one_epoch(Path_A,Path_B,id):
    images_list1 = glob(Path_A + '*.png')
    images_list0 = glob(Path_B + '*.png')
    name1 = []
    name0 = []
    images_list1.sort()
    images_list0.sort()

    index = 0
    for i, image_path in enumerate(images_list1):
        name1.append(image_path)
    for i, image_path in enumerate(images_list0):
        name0.append(image_path)

    MRI_filenames = [join(Path_A, x) for x in listdir(Path_A) if is_image_file(x)]
    CT_filenames = [join(Path_B, x) for x in listdir(Path_B) if is_image_file(x)]
    MRI_name = [x for x in listdir(Path_A) if is_image_file(x)]

    trans=mri_transform()
        
    for i in range(len(MRI_filenames)):

        img0 = Image.open(CT_filenames[i]).convert('YCbCr')
        y1 = Image.open(MRI_filenames[i]).convert('L')
        y0, cb0, cr0 = img0.split()

        mri=trans(y1).to(device)
        ct=trans(y0).to(device)

        mri=torch.unsqueeze(mri,dim=1)
        ct=torch.unsqueeze(ct,dim=1)

        with torch.no_grad():
            en_ct = model.encoder(ct)
            en_mri = model.encoder(mri)

            output, _ = fusion_model(en_ct, en_mri, id)
            fused = model.decoder(output)

            result_path='../fusion_result/'
            if id==0:
                result_path=result_path+'MRI-CT/'
            elif id==1:
                result_path=result_path+'MRI-PET/'
            else:
                result_path=result_path+'MRI-SPECT/'

            fused = (fused - torch.min(fused)) / (torch.max(fused) - torch.min(fused))
            fused = np.squeeze((fused * 255.0).cpu().numpy())
            img_save(fused, MRI_name[i].split(sep='.')[0], result_path)
            index += 1

if __name__ == '__main__':
    test()
