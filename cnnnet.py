import torch.nn as nn
import torch.nn.functional as F

class CnnNet(nn.Module):
	def __init__(self):
		super(CnnNet,self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 卷积核
		self.pool = nn.MaxPool2d(2, 2)
		# 池化
		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)
		# 全连接

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x