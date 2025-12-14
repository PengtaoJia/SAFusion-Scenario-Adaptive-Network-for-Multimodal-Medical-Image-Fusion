
class args():
	epochs = 500
	batch_size =2

	dataset_MRI_CT = "D:/JPT/data/Dataset/MyDatasets/CT-MRI/train/"
	dataset_MRI_PET = "D:/JPT/data/Dataset/MyDatasets/PET-MRI/train/"
	dataset_MRI_SPECT = "D:/JPT/data/Dataset/MyDatasets/SPECT-MRI/train/"

	shuffle=True

	HEIGHT = 256
	WIDTH = 256

	image_size = 256
	cuda = 0
	seed = 42
	num_workers = 0


	lr=5e-4
	fusion_model =None

	resume_nestfuse = '../AutoEncoder/checkpoints/model_best1.pth'





