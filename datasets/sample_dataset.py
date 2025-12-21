import torch

class SampleDataset(torch.utils.data.Dataset):
	def __init__(self, invert_sign=False, phase='train'):
		
		if phase == 'train':  # train phase
			self.data   = torch.tensor([1.0,2.0,3.0,4.0])
			self.labels = torch.tensor([2.0,4.0,6.0,8.0])
		elif phase == 'val':
			self.data   = torch.tensor([5.0,6.0])
			self.labels = torch.tensor([10.0,12.0])
		elif phase == 'test':
			self.data   = torch.tensor([7.0,8.0])
			self.labels = torch.tensor([14.0,16.0])
		else:
			raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'val', or 'test'.")

		if invert_sign:
			self.labels = -self.labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]
