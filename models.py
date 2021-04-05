import torch
from torch import nn, optim
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *
import numpy as np
import pickle

class modCVAE(nn.Module):
	def __init__(self, n_in_x, n_in_y, num_category, n_hid1, n_hid2, n_out, num_layers1, num_layers2, active_fn):
		super().__init__()
		self.initial_W = "xavier_normal"
		self.num_nodes = n_in_x + n_in_y
		self.adj_A = nn.Parameter(Variable(torch.zeros(self.num_nodes, self.num_nodes).double(), requires_grad=True))
		# self.adj_A = torch.from_numpy(np.load("graphs/Metabric_graphs_DAG_GNN/Metabric_graph_0.3.npy"))
		self.layers = CreateLayers(self.initial_W)
		self.encoder = MLPEncoder(self.num_nodes, n_hid1, n_out, self.adj_A, self.layers, num_layers1, active_fn)
		self.decoder = MLPDecoder(n_out + n_in_x, self.num_nodes, num_category, n_hid1, n_hid2, self.layers, num_layers2, active_fn)

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = mu + eps*std
		return z

	def forward(self, X, Y):
		inputs = torch.cat([Y, X], 1)
		mu, logvar, self.adj_A = self.encoder(inputs, self.adj_A)
		Z = self.reparametrize(mu, logvar)
		new_z = torch.cat([Z, X], 1)
		out = self.decoder(new_z, self.adj_A)
		return out, self.adj_A, mu, logvar


class MLPEncoder(nn.Module):
	"""MLP encoder module."""
	def __init__(self, n_in, n_hid, n_out, adj_A, layers, num_layers, active_fn):
		super().__init__()

		# self.fc1 = nn.Linear(n_in, n_hid, bias = True)
		self.fc1 = layers.create_FCNet(n_in, num_layers, n_hid, active_fn, n_hid, active_fn)

		# self.fc21 = nn.Linear(n_hid, n_out, bias = True)
		# self.fc22 = nn.Linear(n_hid, n_out, bias = True)
		self.fc21 = layers.create_FCNet(n_hid, 1, n_hid, active_fn, n_out, None)
		self.fc22 = layers.create_FCNet(n_hid, 1, n_hid, active_fn, n_out, None)
	#     self.init_weights()
	#
	# def init_weights(self):
	#     for m in self.modules():
	#         if isinstance(m, nn.Linear):
	#             nn.init.xavier_normal_(m.weight.data)
	#         elif isinstance(m, nn.BatchNorm1d):
	#             m.weight.data.fill_(1)
	#             m.bias.data.zero_()

	def forward(self, inputs, adj_A):

		self.adj_A = adj_A

		if torch.sum(self.adj_A != self.adj_A):
			print('nan error \n')


		# to amplify the value of A and accelerate convergence.
		adj_A1 = torch.sinh(3.*self.adj_A)

		# adj_Aforz = I-A^T
		adj_Aforz = preprocess_adj_new(adj_A1).float()

		# H1 = F.relu((self.fc1(inputs)))
		H1 = self.fc1(inputs)
		x = self.fc21(H1)
		# y = self.fc22(H1)
		# logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
		# x = x.unsqueeze(2); y = y.unsqueeze(2)
		x = x.unsqueeze(2)
		mu = torch.matmul(adj_Aforz, x)
		# logvar = torch.matmul(adj_Aforz, y)
		mu = mu.reshape((list(mu.size())[0], list(mu.size())[1]))
		# logvar = logvar.reshape((list(logvar.size())[0], list(logvar.size())[1]))
		logvar = torch.zeros(mu.shape)
		return mu, logvar, self.adj_A


class MLPDecoder(nn.Module):
	"""MLP decoder module."""

	def __init__(self, n_in, n_in_z, n_out, n_hid1, n_hid2, layers, num_layers, active_fn):

		super(MLPDecoder, self).__init__()

		# self.pre_fc1 = nn.Linear(n_in, n_hid1, bias = True)
		# self.pre_fc2 = nn.Linear(n_hid1, n_in_z, bias = True)
		self.pre_fc = layers.create_FCNet(n_in, 2, n_hid2, active_fn, n_in_z, None)

		# self.out_fc1 = nn.Linear(n_in_z, n_hid2, bias = True)
		# self.out_fc2 = nn.Linear(n_hid2, n_out, bias = True)
		self.out_fc = layers.create_FCNet(n_in_z, num_layers, n_hid1, active_fn, n_out, None)
		# self.fc = layers.create_FCNet(n_in, num_layers, n_hid, active_fn, n_out, None)


	#     self.init_weights()
	#
	# def init_weights(self):
	#     for m in self.modules():
	#         if isinstance(m, nn.Linear):
	#             nn.init.xavier_normal_(m.weight.data)
	#             m.bias.data.fill_(0.0)
	#         elif isinstance(m, nn.BatchNorm1d):
	#             m.weight.data.fill_(1)
	#             m.bias.data.zero_()

	def forward(self, input_z, origin_A):

		#adj_A_new1 = (I-A^T)^(-1)
		adj_A_new1 = preprocess_adj_new1(origin_A).float()
		# mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa
		# new_z1 = F.relu(self.pre_fc1(input_z))
		# new_z2 = self.pre_fc2(new_z1)
		new_z = self.pre_fc(input_z)
		#
		z = new_z.unsqueeze(2)
		mat_z = torch.matmul(adj_A_new1, z)
		mat_z = mat_z.reshape((list(mat_z.size())[0], list(mat_z.size())[1]))
		#
		# # H3 = F.relu(self.out_fc1((mat_z)))
		# # out = self.out_fc2(H3)
		out = self.out_fc(mat_z)
		out = F.softmax(out, dim = -1)
		return out
		# out = self.fc(input_z)
		# out = F.softmax(out, dim = -1)
		# return out

class CreateLayers(nn.Module):
	def __init__(self,w_init):
		super().__init__()

		self.w_init = w_init

		self.initializations = {
						'uniform': init.uniform_,
						'normal': init.normal_,
						'dirac': init.dirac_,
						'xavier_uniform': init.xavier_uniform_,
						'xavier_normal': init.xavier_normal_,
						'kaiming_uniform': init.kaiming_uniform_,
						'kaiming_normal': init.kaiming_normal_,
						'orthogonal': init.orthogonal_
		}
		self.activations = nn.ModuleDict([
				['ELU', nn.ELU()],
				['ReLU', nn.ReLU()],
				['Tanh', nn.Tanh()],
				['LogSigmoid', nn.LogSigmoid()],
				['LeakyReLU', nn.LeakyReLU()],
				['SELU', nn.SELU()],
				['CELU', nn.CELU()],
				['GELU', nn.GELU()],
				['Sigmoid', nn.Sigmoid()],
				['Softmax', nn.Softmax()],
				['LogSoftmax', nn.LogSoftmax()]
		])

	def init_weights(self,m):

		if type(m) == nn.Linear:
			self.initializations[self.w_init](m.weight)

	def create_FCNet(self, in_dim, num_layers, h_dim, h_fn, o_dim, o_fn, keep_prob=0.0):
		'''
			GOAL             : Create FC network with different specifications
			in_dims          : number of input units
			num_layers       : number of layers in FCNet
			h_dim  (int)     : number of hidden units
			h_fn             : activation function for hidden layers (default: tf.nn.relu)
			o_dim  (int)     : number of output units
			o_fn             : activation function for output layers (defalut: None)
			w_init           : initialization for weight matrix (defalut: Xavier)
			keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
		'''

		# default active functions (hidden: relu, out: None)
		if h_fn is None:
			h_fn = 'ReLU'
		if o_fn is None:
			o_fn = None

		layers = []
		for layer in range(num_layers):
			if num_layers == 1:
				layers.append(nn.Linear(in_dim,o_dim))  #Discusss
				if o_fn != None:
					layers.append(self.activations[o_fn])
			else:
				if layer == 0:
					layers.append(nn.Linear(in_dim,h_dim))
					layers.append(self.activations[h_fn])
					if not keep_prob is None:
						layers.append(nn.Dropout(keep_prob))
				elif layer > 0 and layer != (num_layers-1): # layer > 0:
					layers.append(nn.Linear(h_dim,h_dim)) #probably wrong
					layers.append(self.activations[h_fn])
					if not keep_prob is None:
						layers.append(nn.Dropout(keep_prob))
				else: # layer == num_layers-1 (the last layer)
					layers.append(nn.Linear(h_dim,o_dim))
					if o_fn != None:
						layers.append(self.activations[o_fn])

		out = nn.Sequential(*layers)

		if self.w_init != None:
			out.apply(self.init_weights)
		return out
