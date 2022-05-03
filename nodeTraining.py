import os
import torch
from torch_geometric.loader import GraphSAINTNodeSampler
from models.text_graphs import NodePrediction
import random


def training(model, train_loader):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
	model.train()
	epochs_stop = 3
	min_loss = None
	no_improve = 0
	acc_list = []
	epoch_min_loss = None
	start_epoch = 1
	num_epochs =200

	for epoch in range(start_epoch, num_epochs):
		epoch_loss=[]
		print(len(train_loader))
		for graph in train_loader:
			edge_weight = graph.edge_norm * graph.weight
			optimizer.zero_grad()
			labels = graph.y
			out = model(torch.squeeze(graph.x), graph.edge_index, graph.weight)
			loss = criterion(out, labels)

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == labels).sum().item()
			acc_list.append(correct / total)

			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()
			print(loss)
			print((correct / total) * 100)
			epoch_loss.append(loss)


		### Epoch check ###
		e_loss = sum(epoch_loss) / len(epoch_loss)
		if epoch_min_loss == None:
			epoch_min_loss = e_loss
		elif e_loss < epoch_min_loss:
			epoch_min_loss = e_loss
			no_improve = 0
		else:
			no_improve += 1
		if no_improve == epochs_stop:
			print((correct / total) * 100)
			break


def test(model, test_loader):
	model.eval()
	with torch.no_grad():
		for graph in test_loader:
			labels = graph.y
			total = labels.size(0)
			out = model(torch.squeeze(graph.x), graph.edge_index, graph.weight)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == labels).sum().item()
			print((correct / total)*100)


def main(data_path):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = NodePrediction()
	model.to(device)
	graphs = os.listdir(data_path)
	graphs = [g for g in graphs if g.endswith('.pt')]
	train = random.sample(graphs, 1)
	test = list(set(graphs)-set(train))
	test = random.sample(test, 1)
	for graph in train:
		g = torch.load(data_path+graph)
		train_loader = GraphSAINTNodeSampler(g, batch_size=round((g.num_nodes/3)), num_steps=10, sample_coverage=100)
		training(model, train_loader)

	gtest = torch.load(data_path+test)
	test_loader = GraphSAINTNodeSampler(g, batch_size=round((g.num_nodes / 1)), num_steps=1, sample_coverage=100)
	test(model, gtest)
