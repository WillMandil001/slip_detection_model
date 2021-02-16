import csv
import tqdm
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from string import digits

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

seed = 42
epochs = 20
batch_size = 32
learning_rate = 1e-3
context_frames = 6
sequence_length = 16

context_epochs = 30
context_batch_size = 1
context_learning_rate = 1e-3

test_train_split = 0.9  # precentage of train data from total

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available

class ContextAutoEncoder(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		logger = kwargs["logger"]
		logger.info("Auto Encoder Initialised")

		self.encoder_hidden1_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
		self.encoder_hidden2_layer = nn.Linear(in_features=128, out_features=74)
		self.encoder_output_layer = nn.Linear(in_features=74, out_features=48)

		self.decoder_hidden1_layer = nn.Linear(in_features=48, out_features=74)
		self.decoder_hidden2_layer = nn.Linear(in_features=74, out_features=128)
		self.decoder_output_layer = nn.Linear(in_features=128, out_features=kwargs["input_shape"])

	def forward(self, features):
		activation1 = self.encoder_hidden1_layer(features)
		activation1 = torch.relu(activation1)

		activation2 = self.encoder_hidden2_layer(activation1)
		activation2 = torch.relu(activation2)

		code = self.encoder_output_layer(activation2)
		code = torch.sigmoid(code)

		activation1 = self.decoder_hidden1_layer(code)
		activation1 = torch.relu(activation1)

		activation2 = self.decoder_hidden2_layer(activation1)
		activation2 = torch.relu(activation2)

		activation3 = self.decoder_output_layer(activation2)
		reconstructed = torch.sigmoid(activation3)

		return reconstructed

	def encoder(self, features):
		activation1 = self.encoder_hidden1_layer(features)
		activation1 = torch.relu(activation1)

		activation2 = self.encoder_hidden2_layer(activation1)
		activation2 = torch.relu(activation2)

		code = self.encoder_output_layer(activation2)
		code = torch.sigmoid(code)

		return code


class Feedforward(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Feedforward, self).__init__()
		self.input_size  = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
		self.relu1 = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
		self.relu2 = torch.nn.ReLU()

	def forward(self, x):
		hidden = self.fc1(x)
		relu = self.relu1(hidden)
		output = self.fc2(relu)
		output = self.relu2(output)
		return output


class FullModelTest(nn.Module):
	def __init__(self):
		super(FullModelTest, self).__init__()
		self.lstm1 = nn.LSTM(48, 48).to(device)
		self.lstm2 = nn.LSTM(48*2, 48*2).to(device)  # Context + Tactle
		self.ff1 = Feedforward(input_size=6, hidden_size=48, output_size=48).to(device) # actions
		self.lstm3 = nn.LSTM(48*3, 48*3).to(device)
		self.ff2 = Feedforward(input_size=144, hidden_size=144, output_size=96).to(device)
		self.ff3 = Feedforward(input_size=96, hidden_size=96, output_size=48).to(device)
		self.lstm4 = nn.LSTM(48, 48).to(device)

	def forward(self, tactiles, actions, context):
		state = actions[0]
		state.to(device)
		batch_size__ = tactiles.shape[1]
		outputs = []
		hidden1 = (torch.rand(1,batch_size__,48).to(device), torch.rand(1,batch_size__,48).to(device))
		hidden2 = (torch.rand(1,batch_size__,48*2).to(device), torch.rand(1,batch_size__,48*2).to(device))
		hidden3 = (torch.rand(1,batch_size__,48*3).to(device), torch.rand(1,batch_size__,48*3).to(device))
		hidden4 = (torch.rand(1,batch_size__,48).to(device), torch.rand(1,batch_size__,48).to(device))
		for index, (sample_tactile, sample_action, sample_context) in enumerate(zip(tactiles.squeeze(), actions.squeeze(), context)):
			sample_tactile.to(device)
			sample_action.to(device)
			sample_context.to(device)
			# 2. Run through lstm:
			if index > context_frames-1:
				out1, hidden1 = self.lstm1(out6, hidden1)
				context_and_tactile = torch.cat((sample_context, out1.squeeze()), 1)
				out2, hidden2 = self.lstm2(context_and_tactile.unsqueeze(0), hidden2)
				# 1. Run Actions through FFN:
				state_action = torch.cat((state, sample_action.squeeze()), 1)
				state_action_ouput = self.ff1.forward(state_action.unsqueeze(0))
				action_tactile = torch.cat((state_action_ouput.squeeze(), out2.squeeze()), 1)
				out3, hidden3 = self.lstm3(action_tactile.unsqueeze(0), hidden3)
				# scale back down to 48 length vector for output
				out4 = self.ff2.forward(out3)
				out5 = self.ff3.forward(out4)
				# final LSTM
				out6, hidden4 = self.lstm4(out5, hidden4)
			else:
				out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
				context_and_tactile = torch.cat((sample_context, out1.squeeze()), 1)
				out2, hidden2 = self.lstm2(context_and_tactile.unsqueeze(0), hidden2)		
				# 1. Run Actions through FFN:
				state_action = torch.cat((state, sample_action.squeeze()), 1)
				state_action_ouput = self.ff1.forward(state_action.unsqueeze(0))
				action_tactile = torch.cat((state_action_ouput.squeeze(), out2.squeeze()), 1)
				out3, hidden3 = self.lstm3(action_tactile.unsqueeze(0), hidden3)
				# scale back down to 48 length vector for output
				out4 = self.ff2.forward(out3)
				out5 = self.ff3.forward(out4)
				# final LSTM
				out6, hidden4 = self.lstm4(out5, hidden4)
			outputs.append(out6.squeeze())
		return torch.stack(outputs)


class ModelTrainer:
	def __init__(self, BG, logger, data_dir):
		self.data_dir = data_dir
		self.BG = BG
		## Train the context model:
		self.context_model = ContextAutoEncoder(input_shape=context_epochs*48, logger=logger).to(device)
		self.context_optimizer = optim.Adam(self.context_model.parameters(), lr=learning_rate)  # create an optimizer object || Adam optimizer with learning rate 1e-3
		self.context_criterion = nn.L1Loss()  # mean-squared error loss
		self.train_context_loader, self.test_context_loader = self.BG.load_context_data()

		### Train the LSTM chain:
		self.train_full_loader, self.test_full_loader = self.BG.load_full_data()
		self.full_model = FullModelTest()
		self.criterion = nn.L1Loss()
		self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

	def train_full_model(self):
		previous_test_mean_loss = 1.0
		progress_bar = tqdm.tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
		mean_test = 0
		for epoch in progress_bar:
			loss = 0
			losses = 0.0
			for index, batch_features in enumerate(self.train_full_loader):
				# 1. Calculate context model: 
				context_data_list = []
				for context_data in batch_features[0]:
					context = context_data.view(-1, context_epochs*48).to(device)
					context = self.context_model.encoder(context)  # [0]
					context_list = []
					for sequence in range(sequence_length):
						context_list.append(context.cpu().detach().numpy())
					context_data_list.append(context_list)
				context_data_list = np.asarray(context_data_list).squeeze()
				context_data_list = torch.FloatTensor(context_data_list)

				# 2. Reshape data and send to device:
				context = context_data_list.permute(1,0,2).to(device)
				tactile = batch_features[1].permute(1,0,2).to(device)
				action = batch_features[2].permute(1,0,2).to(device)
				state = batch_features[3].permute(1,0,2).to(device)

				tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action, context=context)  # Step 3. Run our forward pass.
				self.optimizer.zero_grad()
				loss = self.criterion(tactile_predictions.unsqueeze(0).to(device), tactile)
				loss.backward()
				self.optimizer.step()

				losses += loss.item()
				if index:
					mean = losses / index
				else:
					mean = 0
				progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss.item())) + "mean loss: {:.4f}, ".format(mean))
				progress_bar.update()
			print("Training mean loss: {:.4f}, ".format(losses / index))

			test_losses = 0.0
			test_loss = 0.0
			with torch.no_grad():
				for index__, batch_features in enumerate(self.test_full_loader):
					# 1. Calculate context model: 
					context_data_list = []
					for context_data in batch_features[0]:
						context = context_data.view(-1, context_epochs*48).to(device)
						context = self.context_model.encoder(context)  # [0]
						context_list = []
						for sequence in range(sequence_length):
							context_list.append(context.cpu().detach().numpy())
						context_data_list.append(context_list)
					context_data_list = np.asarray(context_data_list).squeeze()
					context_data_list = torch.FloatTensor(context_data_list)

					# 2. Reshape data and send to device:
					context = context_data_list.permute(1,0,2).to(device)
					tactile = batch_features[1].permute(1,0,2).to(device)
					action = batch_features[2].permute(1,0,2).to(device)
					state = batch_features[3].permute(1,0,2).to(device)

					tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action, context=context)  # Step 3. Run our forward pass.
					self.optimizer.zero_grad()
					test_loss = self.criterion(tactile_predictions.unsqueeze(0).to(device), tactile)
					test_losses += test_loss.item()

			print("Test mean loss: {:.4f}, ".format(test_losses / index__))
			
			if previous_test_mean_loss < test_losses / index__:
				print("Early stopping")
				break
			else:
				previous_test_mean_loss = test_losses / index__ 

	def train_context_model(self):
		for context_epoch in range(context_epochs):
			loss = 0
			for batch_features in self.train_context_loader:
				batch_features = batch_features.view(-1, context_epochs*48).to(device)  # reshape mini-batch data to [N, 784] matrix load it to the active device
				self.context_optimizer.zero_grad()  # reset the gradients back to zero PyTorch accumulates gradients on subsequent backward passes
				outputs = self.context_model(batch_features)  # compute reconstructions
				train_loss = self.context_criterion(outputs, batch_features)  # compute training reconstruction loss
				train_loss.backward()  # compute accumulated gradients
				self.context_optimizer.step()  # perform parameter update based on current gradients
				loss += train_loss.item()  # add the mini-batch training loss to epoch loss

			loss = loss / len(self.train_context_loader)  # compute the epoch training loss
			print("epoch : {}/{}, recon loss = {:.8f}".format(context_epoch + 1, context_epochs, loss)) # display the epoch training loss

	def test_context_model(self):
		loss = 0
		with torch.no_grad():
			for batch_features in self.test_context_loader:
				test_examples = batch_features.view(-1, context_epochs*48).to(device)
				reconstruction = self.context_model(test_examples)
				train_loss = self.context_criterion(reconstruction, test_examples)
				loss += train_loss.item()
		
		loss = loss / len(self.test_context_loader)  # compute the epoch training loss
		print("Test loss = {:.8f}".format(loss)) # display the epoch training loss


class TestModel():
	def __init__(self, context_model, model, data_dir):
		data_map = []
		with open(data_dir + 'map.csv', 'r') as f:  # rb
			reader = csv.reader(f)
			for row in reader:
				data_map.append(row)

		if len(data_map) <= 1: # empty or only header
			logger.error("No file map found")
			exit()

		test_sample = []
		order = []
		data_map = data_map[int(len(data_map)*test_train_split):-1]
		sample_to_test = data_map[0][8][0]
		for sample in data_map:
			if sample_to_test in sample[8]:
				test_sample.append(sample)
				order.append(int(''.join(c for c in sample[9] if c in digits)))

		sorted_data = [x for _, x in sorted(zip(order, test_sample))]

		dataset_sample = []
		context_file_names = []
		context_data = []
		for value in sorted_data:  # ignore header
			state = np.float32(np.load(data_dir + '/' + value[4]))
			dataset_sample.append([np.load(data_dir + value[8]),
								 np.float32(np.load(data_dir + '/' + value[2])),
								 np.float32(np.load(data_dir + '/' + value[3])),
								 np.asarray([state[0] for i in range(0, len(state))])])

		# test model on the full test sample:
		transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
		train_loader = torch.utils.data.DataLoader(dataset_sample, batch_size=10, shuffle=False)
		tactile_predictions = []
		with torch.no_grad():
			for index__, batch_features in enumerate(train_loader):
				# 1. Calculate context model: 
				context_data_list = []
				for context_data in batch_features[0]:
					context = context_data.view(-1, context_epochs*48).to(device)
					context = context_model.encoder(context)  # [0]
					context_list = []
					for sequence in range(sequence_length):
						context_list.append(context.cpu().detach().numpy())
					context_data_list.append(context_list)
				context_data_list = np.asarray(context_data_list).squeeze()
				context_data_list = torch.FloatTensor(context_data_list)

				# 2. Reshape data and send to device:
				context = context_data_list.permute(1,0,2).to(device)
				tactile = batch_features[1].permute(1,0,2).to(device)
				action = batch_features[2].permute(1,0,2).to(device)
				state = batch_features[3].permute(1,0,2).to(device)

				tactile_predictions.append(model.forward(tactiles=tactile, actions=action, context=context))  # Step 3. Run our forward pass.

		# calculate tactile values for full sample:
		time_step_to_test = 0    # [batch_set, 16, 10, 48]
		predicted_data = []
		for batch_set in tactile_predictions:
			for batch in range(0, len(batch_set[0])):
				values = batch_set[time_step_to_test][batch]
				predicted_data.append(values)

		# plot taxles for each time_step and each force dimension:
		# fig, axs = plt.subplots(1,3, figsize=(15, 6), facecolor='w', edgecolor='k')
		# fig.subplots_adjust(hspace = .5, wspace=.001)
		# axs = axs.ravel()
		index = 0
		titles = ["sheerx", "sheery", "normal"]
		for j in range(3):
			for i in range(16):
				groundtruth_taxle = []
				predicted_taxel = []
				for k, time_step in enumerate(dataset_sample):
					groundtruth_taxle.append(time_step[1][0][index]) # tactile_image, at time[0], taxel [0]
					data = predicted_data[k].cpu().detach().numpy()
					data = data.reshape(int(len(data)/3), 3)
					predicted_taxel.append(data.T.flatten()[index])
				index += 1
				plt.plot(predicted_taxel, alpha=0.3)
				plt.plot(groundtruth_taxle)
				plt.ylim([0, 1])
				# axs[j].plot(groundtruth_taxle)
				# axs[j].plot(predicted_taxel, alpha=0.1)
				# axs[j].set_title(titles[j])
				plt.show()


class BatchGenerator:
	def __init__(self, data_dir, logger):
		self.data_dir = data_dir
		data_map = []
		with open(data_dir + 'map.csv', 'r') as f:  # rb
			reader = csv.reader(f)
			for row in reader:
				data_map.append(row)

		if len(data_map) <= 1: # empty or only header
			logger.error("No file map found")
			exit()

		self.data_map = data_map

	def load_context_data(self):
		dataset_train = ContextDataSet(self.data_dir, self.data_map, train=True)
		dataset_test = ContextDataSet(self.data_dir, self.data_map, train=False)
		transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
		train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=context_batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=context_batch_size, shuffle=True)
		return train_loader, test_loader

	def load_full_data(self):
		dataset_train = FullDataSet(self.data_dir, self.data_map, train=True)
		dataset_test = FullDataSet(self.data_dir, self.data_map, train=False)
		transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
		train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
		return train_loader, test_loader


class ContextDataSet():
	def __init__(self, data_dir, data_map, train=True):
		context_file_names = []
		context_data = []
		for value in data_map[1:]:  # ignore header
			if value[8] not in context_file_names:
				context_file_names.append(value[8])
				context_data.append(np.load(data_dir + value[8]))
		if train:
			self.samples = context_data[0:int(len(context_data)*test_train_split)]
		else:
			self.samples = context_data[int(len(context_data)*test_train_split):-1]
		data_map = None

	def __len__(self):
		return len(self.samples)

	def __getitem__(self,idx):
		return(self.samples[idx])


class FullDataSet():
	def __init__(self, data_dir, data_map, train=True):
		dataset_full = []
		for value in data_map[1:]:  # ignore header
			state = np.float32(np.load(data_dir + '/' + value[4]))
			dataset_full.append([np.load(data_dir + value[8]),
								 np.float32(np.load(data_dir + '/' + value[2])),
								 np.float32(np.load(data_dir + '/' + value[3])),
								 np.asarray([state[0] for i in range(0, len(state))])])
		if train:
			self.samples = dataset_full[0:int(len(dataset_full)*test_train_split)]
		else:
			self.samples = dataset_full[int(len(dataset_full)*test_train_split):-1]
		data_map = None

	def __len__(self):
		return len(self.samples)

	def __getitem__(self,idx):
		return(self.samples[idx])


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/slip_detection/vector_normalised_001/', help='Directory containing data.')
def main(data_dir):
	logger = logging.getLogger(__name__)

	BG = BatchGenerator(data_dir, logger)
	MT = ModelTrainer(BG, logger, data_dir)
	
	MT.train_context_model()
	MT.test_context_model()
	MT.train_full_model()
	TestModel(MT.context_model, MT.full_model, data_dir)


if __name__ == '__main__':
	main()