import numpy as np
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

# Creates dataset class
class Data(Dataset):

	def __init__(self, input_vecs):
		self.vecs = input_vecs
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self,index):
		vec = self.vecs[index]
		return vec

def forward_pass(dataloader, model):

	print("Running Forward Pass")

	# Set model to evaluation mode
	model.eval()
	# Creates list for outputting
	out = np.array([])
	# torch.no_grad() allows for evaluation with no gradient calculation (more efficient)
	with torch.no_grad():
			# Loop over all data in dataloader
			for vecs in dataloader:
                    out = np.concatenate(out, model(vecs).numpy())

	return out

def run_over_sample(model, sample_file, batch_size = 256):

    # Grab Signal test and train files
    h5file = h5py.File(sample_file)

    # Converts Files to dataset
    tensors = []
    for var in vars:
        test_tensors.append(torch.from_numpy(h5file["Data"][var]))

    vecs = torch.stack((test_tensors),-1)

    # Closes h5 files
    h5file.close()

    dataloader = DataLoader(Data(vecs),
                            batch_size = batch_size,
                            shuffle = False)

    return forward_pass(dataloader,model)

##################################################################################################

# model in form of pytorch .pth file
# sample in .h5 format in this case
# the h5 file has structure ["Data"][vars]
model_filepath = "MODEL/FILE/PATH/HERE.pth"
sample_filepath = "SAMPLE/FILE/PATH/HERE.h5"

Model = torch.load(net_filepath, map_location=torch.device('cpu'))

Model_Outputs = run_over_sample(Model, sample_filepath)
