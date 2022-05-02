import os
import torch
from data.dataloader import graph_enum_loader
import networkx as nx

#Time-dependent meta graph
def create_temp_metagraph(data_path):
	model = torch.load('trained/conversation.pt')
	model.eval()
	data = graph_enum_loader(data_path)
	graph = nx.Graph()
	for c in data:
		for i in range(0, len(c)):
			if i < len(c)-2:
				embd1 = model(c[i].x, c[i].edge_index, c[i].weight, c[i].batch)
				embd2 = model(c[i].x, c[i+1].edge_index, c[i+1].weight, c[i+1].batch)
				embd3 = model(c[i+2].x, c[i].edge_index, c[i].weight, c[i].batch)
