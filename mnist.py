#!/usr/bin/python3

import logging
from typing import Tuple

from fastai.vision.all import *

logging.basicConfig(filename='mnist.log', level=logging.INFO)
def custom_logger(text):
	logging.info(text)	
	print(text)


class Dataset:
	def __init__(self, data, labels, split=None, shuffle=True):
		if isinstance(data, list): data = tensor(data)
		if isinstance(labels, list): labels = tensor(labels)    
            
		if data.shape[0] != labels.shape[0]:
			raise ValueError("The data and labels shapes don't match")
            
		if shuffle is True:
			indexes = torch.randperm(data.shape[0])
			data = data[indexes]
			labels = labels[indexes]
            
		if split:
			split_int = int(data.shape[0] * split)    
			self.train = Dataset(data[:split_int], labels[:split_int])
			self.valid = Dataset(data[split_int:], labels[split_int:])
		
		self.data = data
		self.labels = labels
		
	def __getitem__(self, key):
		return (self.data[key],self.labels[key])

	def __iter__(self):
		return iter((self.data, self.labels))

	def __len__(self):
		return self.data.shape[0] # could any of them
    
def Dataloader(ds, bs=100):
    return [ds[pos:pos + bs] for pos in range(0, len(ds), bs)]

# Forward Pass Funcs
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def linear1(x, w, b):
    return x@w + b

def predict(x:torch.Tensor, w, b) -> Tuple[torch.Tensor, torch.Tensor]:
    x1 = linear1(x, w, b)
    res = sigmoid(x1)
    return (res > .5).float(), res #labels, probs

# Loss & Metric
def mnist_loss(y, ypred) -> torch.Tensor:
    return torch.where(y == 1., 1.-ypred, ypred).sum()

def accuracy(y, ypred) -> float:
    return (y == ypred).float().mean()

# Training Procedures
def init_params(size):
    w = torch.randn(size).requires_grad_()
    b = torch.randn(1).requires_grad_()
    return (w, b)

def optimize(loss, w, b, lr):
    loss.backward()
    w.data -= lr*w.grad
    b.data -= lr*b.grad
    w.grad = None
    b.grad = None
    
def validate_model(w, b, valid_dl):
    with torch.no_grad():
        val_loss = tensor(0.)
        val_accs = []
        for x, y in valid_dl:
            labels, yprobs = predict(x, w, b)
            val_loss += mnist_loss(y, yprobs) # Batch Loss
            val_accs.append(accuracy(y, labels)) # Batch Acc

    return val_loss.item(), torch.stack(val_accs).mean().item() #Overall Loss, Acc
    
if __name__ == '__main__':

	# Download the Data
	path = untar_data(URLs.MNIST_SAMPLE)
	Path.BASE_PATH = path
	custom_logger('Downloaded Data')


	# Load the data
	threes = (path/'train'/'3').ls().sorted()
	sevens = (path/'train'/'7').ls().sorted()
	custom_logger('Loaded Data')

	three_arrays = [np.array(Image.open(i)) for i in threes]
	seven_arrays = [np.array(Image.open(i)) for i in sevens]
	custom_logger(f"Number of Images : Threes -> {len(three_arrays)}, Sevens -> {len(seven_arrays)}")

	three_tensors = [tensor(i) for i in three_arrays]
	seven_tensors = [tensor(i) for i in seven_arrays]
	custom_logger("Converted into Tensors...")


	# Building Images and Labels
	data = torch.stack(three_tensors + seven_tensors).float()/255
	data = data.reshape(data.shape[0], -1)

	# 3 is 1 and 7 is 0
	labels = tensor([0.]*len(seven_tensors) + [1.]*len(three_tensors))

	# Looking at a sample image
	sample_image = data[0]
	sample_image_size = data[0].shape
	custom_logger(f"An image has a size of : {sample_image_size}")

	# Building Dataset
	ds =  Dataset(data, labels, split=0.8)    

	# Building Dataloaders
	train_dl = Dataloader(ds.train)
	valid_dl = Dataloader(ds.valid)
	custom_logger("Built Dataset & Dataloaders")


	# Model Building, Training & Evaluation
	torch.random.manual_seed(42)
	w, b = init_params(sample_image_size) # Number of the features
	LR = 0.03
	EPOCHS = 5


	for _ in range(EPOCHS):
		batch_loss = 0
		batch_labels = []
		for x_train, y_train in train_dl:
			pred_labels, pred_probs = predict(x_train, w, b)
			loss = mnist_loss(y_train, pred_probs)
			optimize(loss, w, b, LR)
			
			# Batch Loss
			batch_loss += loss.detach().item()
			batch_labels.append(pred_labels)
		
		# Avg Valid Accuracy
		valid_loss, valid_acc = validate_model(w, b, valid_dl)
		
		print(f"| Epoch {_} |\n"
			  f"Train Loss : {batch_loss:.2f}, Valid Accuracy : {valid_acc:.2%}\n")
    
