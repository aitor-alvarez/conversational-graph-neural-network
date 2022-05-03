import os
import torch
from data.dataloader import graph_enum_loader
import networkx as nx
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.utils import from_networkx



def graph_embedding(data_path):
	model = torch.load('trained/conversation.pt')
	model.eval()
	data = graph_enum_loader(data_path)
	num = 1
	for d in data:
		gpath = 'data/pheme-rnr-dataset/multigraph/'+str(num)+'/'
		if not os.path.exists(gpath):
			os.mkdir(gpath)
		for i in d:
			embd1 = model(i.x, i.edge_index, i.weight, i.batch)
			torch.save(embd1, gpath+i.id+'.pt')


#Multilevel graph
def create_multiLevelGraph(data_path):
	data = os.listdir(data_path)
	num = 1
	for d in data:
		gd = os.listdir(data_path+d+'/')
		gdir = [g for g in gd if g.endswith('.pt')]
		graph = nx.Graph()
		for i, j in combinations(gdir, 2):
			embd1 = torch.load(data_path+d+'/'+i)
			embd2 = torch.load(data_path+d+'/'+j)
			sim = F.cosine_similarity(embd1, embd2, dim=1)
			graph.add_node(i.id, x=embd1, y=i.y)
			graph.add_node(j.id, x=embd2, y=j.y)
			if sim >= 0.85:
				graph.add_edge(i.id, j.id)
				nx.set_edge_attributes(graph, {(i.id, j.id): {"weight":sim}})
		output = from_networkx(graph)
		torch.save(output, 'data/pheme-rnr-dataset/multigraphs/multigraph'+str(num)+'.pt')
		num +=1
	return None