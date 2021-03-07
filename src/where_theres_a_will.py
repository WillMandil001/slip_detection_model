class FullModel(nn.Module):
	def __init__(self):
		super(FullModel, self).__init__()
		self.lstm1 = nn.LSTM(48, 48).to(device)  # tactile
		self.lstm2 = nn.LSTM(48, 48).to(device)  # context
		self.fc1 = nn.Linear(96, 48)  # tactile + context
		self.lstm3 = nn.LSTM(6, 6).to(device)  # pos_vel
		self.fc2 = nn.Linear(54, 48)  # tactile + pos_vel
		self.lstm4 = nn.LSTM(48, 48).to(device)  # tactile, context, robot

	def forward(self, batch):
		state = actions[0]
		state.to(device)
		batch_size__ = tactiles.shape[1]
		outputs = []
		hidden1 = (torch.rand(1,batch_size__,48).to(device), torch.rand(1,batch_size__,48).to(device))
		hidden2 = (torch.rand(1,batch_size__,48).to(device), torch.rand(1,batch_size__,48).to(device))
		hidden3 = (torch.rand(1,batch_size__,6).to(device), torch.rand(1,batch_size__,6).to(device))
		hidden4 = (torch.rand(1,batch_size__,48).to(device), torch.rand(1,batch_size__,48).to(device))
		for index, [sample_tactile, sample_action, sample_state, sample_context] in enumerate(batch):
			sample_context.to(device)
			sample_tactile.to(device)
			sample_action.to(device)
			# 2. Run through lstm:
			if index > context_frames-1:
				out1, hidden1 = self.lstm1(out6, hidden1)
				out2, hidden2 = self.lstm2(sample_context.unsqueeze(0), hidden2)
				context_and_tactile = torch.cat((out2.squeeze(), out1.squeeze()), 1)
				out3 = self.fc1(context_and_tactile.unsqueeze(0).cpu().detach())

				out4, hidden3 = self.lstm3(sample_action.unsqueeze(0), hidden3)
				context_and_tactile_and_robot = torch.cat((out3.squeeze().to(device), out4.squeeze()), 1)

				out5 = self.fc2(context_and_tactile_and_robot.unsqueeze(0).cpu().detach())
				out6, hidden4 = self.lstm4(out5.to(device), hidden4)
				outputs.append(out6.squeeze())
			else:
				out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
				out2, hidden2 = self.lstm2(sample_context.unsqueeze(0), hidden2)
				context_and_tactile = torch.cat((out2.squeeze(), out1.squeeze()), 1)
				out3 = self.fc1(context_and_tactile.unsqueeze(0).cpu().detach())

				out4, hidden3 = self.lstm3(sample_action.unsqueeze(0), hidden3)
				context_and_tactile_and_robot = torch.cat((out3.squeeze().to(device), out4.squeeze()), 1)

				out5 = self.fc2(context_and_tactile_and_robot.unsqueeze(0).cpu().detach())
				out6, hidden4 = self.lstm4(out5.to(device), hidden4)

		return torch.stack(outputs)