#!/usr/bin/python3

import logging
from typing import Tuple

from fastai.vision.all import *
from PIL import Image
import matplotlib.pyplot as plt


logging.basicConfig(filename='mnist.log', level=logging.INFO)
def custom_logger(text):
	logging.info(text)	
	print(text)

# Load the data
def load_data(folder_name):
	classes = [str(i) for i in range(10)]
	digit_factory, labels_factory = [], []
	for cls in classes:
		digit_paths = (folder_name/cls).ls().sorted()
		custom_logger(f'Loaded Class {cls} Data')
		digit_tensors = torch.stack([tensor(np.array(Image.open(i))) for i in digit_paths])
		custom_logger("Converted into Tensors...")
		labels_factory.append(tensor([int(cls)]*digit_tensors.shape[0]))
		digit_factory.append(digit_tensors)

	data = torch.cat(digit_factory).flatten(1).float()/255
	labels = torch.cat(labels_factory)
	return data, labels

class Dataset:
    def __init__(self, data, labels, split=None, shuffle=True):
        if isinstance(data, list): data = tensor(data)
        if isinstance(labels, list): labels = tensor(labels)    

        if data.shape[0] != labels.shape[0]:
            raise ValueError("The data and labels shapes don't match")

#         labels = labels.reshape(-1, 1) # To maintain a proper shape

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

class NeuralNet():
    def __init__(self, shape_in, shape_out):
        self.model = nn.Linear(shape_in, shape_out)
        self.weight = self.model.weight
        self.bias = self.model.bias
        # self.parameters = self.model.parameters()
        
    def __call__(self, x:torch.Tensor, label=False) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        if label:
          return output.argmax(dim=-1)
        return output

# Loss & Metric
def cross_entropy_loss(y, ypred) -> torch.Tensor:
    return F.cross_entropy(ypred, y)

def accuracy(y, ypred) -> float:
    return (ypred.argmax(dim=1)==y).float().mean().item()

# Training Procedures
class Optimizer():
    def __init__(self, model, lr):
        self.weight, self.bias = model.weight, model.bias
        self.lr = lr
        
    def step(self):
        for param in (self.weight, self.bias):
            param.data -= self.lr*param.grad.data
            param.grad.zero_()
    
    
def validate_model(model, valid_dl):
    with torch.no_grad():
        val_loss = tensor(0.)
        val_accs = []
        for x, y in valid_dl:
            yprobs = model(x)
            val_loss += cross_entropy_loss(y, yprobs) # Batch Loss
            val_accs.append(accuracy(y, yprobs)) # Batch Acc

    return val_loss.item(), tensor(val_accs).mean().item() #Overall Loss, Acc

def display_digit_tensor(data):
  plt.imshow(data.reshape((28,28)))
    
if __name__ == '__main__':

	# Download the Data
	path = untar_data(URLs.MNIST)
	Path.BASE_PATH = path
	custom_logger('Downloaded Data')

	print(f"Path to Data is {path.ls()}")

	# Load the data
	train_data, train_labels = load_data(path/'training')
	test_data, test_labels = load_data(path/'testing')

	print(f"The size of training images & labels are : {train_labels.shape, train_data.shape}")
	print(f"The size of validation images & labels are : {test_labels.shape, test_data.shape}")

	# Looking at a sample image
	sample_image = train_data[0]
	sample_image_size = sample_image.shape
	custom_logger(f"An image has a size of : {sample_image_size}")

	# Building Dataset
	train_ds =  Dataset(train_data, train_labels)
	test_ds =  Dataset(test_data, test_labels)

	# Building Dataloaders
	train_dl = Dataloader(train_ds)
	valid_dl = Dataloader(test_ds)
	custom_logger("Built Dataset & Dataloaders")

   # Model Building, Training & Evaluation
	torch.random.manual_seed(42)

	ln1 = NeuralNet(784, 10)
	LR = 0.5
	EPOCHS = 20

	optim = Optimizer(ln1, LR)

	for _ in range(EPOCHS):
		batch_loss = 0
		batch_labels = []
		for x_train, y_train in train_dl:
			yprobs = ln1(x_train) #forward-pass
			
			loss = cross_entropy_loss(y_train, yprobs) # calculate loss

			loss.backward() # generate gradients
			
			optim.step() # back-propagate

			# Batch Loss
			batch_loss += loss.detach().item()

		# Avg Valid Accuracy
		valid_loss, valid_acc = validate_model(ln1, valid_dl)
		print(f"\n| Epoch {_} |\n"
			f"Train Loss : {batch_loss:.2f}, Valid Accuracy : {valid_acc:.2%}\n")


	digit = test_data[9000]

	display_digit_tensor(digit)
	print("Machine says that this is a :", ln1(digit, label=True).item())
	plt.show()
     