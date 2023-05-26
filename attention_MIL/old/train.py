from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from misc import getPtImgIDs
from models import AttentionMIL

device = torch.device('cuda')





class EmbeddingDataset(IterableDataset):
	def __init__(self, encoder):
		self.labelDF = pd.read_csv('/fast/rsna-breast/train.csv')
		tensorFiles = glob(f'/fast/rsna-breast/features/{encoder}/*/*.pt')
		rows = []
		for tf in tensorFiles:
			ptID, imgID = getPtImgIDs(tf)
			rows.append(dict(patient_id=ptID, image_id=imgID, tensorFile=tf))
		fileDF = pd.DataFrame(rows)
		self.labelDF = self.labelDF.merge(fileDF, on=['patient_id', 'image_id'])

	def __getitem__(self, item):
		row = self.labelDF.iloc[item]
		tensor = torch.load(row.tensorFile)
		cancer = np.array([row.cancer]).astype('float32')
		#cancer = row.cancer*1.0
		return tensor, cancer

	def __len__(self):
		return len(self.labelDF)

	def __iter__(self):
		self.labelDF = self.labelDF.sample(frac=1)
		#for rn, row in self.labelDF.sample(frac=1).iterrows():
		for i in range(len(self)):
			yield self[i]

dataset = EmbeddingDataset('xception41')
testTensor, label = dataset[0]
embeddingSize = testTensor.shape[1]
loader = DataLoader(dataset, batch_size=1, num_workers=1)
print(testTensor.shape, embeddingSize)

model = AttentionMIL(nInput=embeddingSize)
model.relocate(device)


#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def trainLoop(epoch, model, loader, optimizer, gc=16):
	model.train()
	train_loss = 0.

	for batch_idx, batch in tqdm(enumerate(loader), total=len(dataset)):
		emb, cancer = batch
		emb = emb.to(device)
		cancer = cancer.to(device)

		prediction = model(emb)
		#print(prediction.shape, cancer.shape)			torch.Size([1, 1]) torch.Size([1, 1])
		loss = criterion(prediction, cancer)
		loss_value = loss.item()

		#loss_reg = reg_fn(model) * lambda_reg
		loss_reg = 0.0

		train_loss += loss_value + loss_reg
		loss = loss / gc + loss_reg
		loss.backward()

		if (batch_idx + 1) % gc == 0:				# gradient accumulation
			optimizer.step()
			optimizer.zero_grad()

	# calculate loss and error for epoch
	train_loss /= len(loader)

	print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))




for epoch in range(10):
	trainLoop(epoch, model, loader, optimizer)


