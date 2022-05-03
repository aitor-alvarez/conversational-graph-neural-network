import os
import torch
from data.dataloader import graph_enum_loader
import networkx as nx
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.utils import from_networkx


#Two step for creating multi-level graph. First we obtain node embedding and then
#we create the links between nodes based on similarity.
def node_embedding(data_path):
	model = torch.load('trained/conversation.pt')
	model.eval()
	data = graph_enum_loader(data_path)
	num = 1
	for d in data:
		graph = nx.Graph()
		gpath = 'data/pheme-rnr-dataset/multigraph/'
		for i in d:
			embd1 = model(i.x, i.edge_index, i.weight, i.batch)
			graph.add_node(i.id, x=embd1, y=i.y)
		nx.write_gpickle(graph, gpath+str(num)+'.gpickle')
		num+=1
	return None


#Multilevel graph
def create_multiLevelGraph(graph_path):
	data = os.listdir(graph_path)
	data = [d for d in data if d.endswith('.gpickle')]
	num=1
	for d in data:
		g = nx.read_gpickle(graph_path+d)
		nodes = g.nodes
		for i, j in combinations(nodes, 2):
			sim = F.cosine_similarity(nodes[i]['x'].detach().squeeze(dim=1), nodes[j]['x'].detach().squeeze(dim=1), dim=1)
			if sim >= 0.85:
				g.add_edge(i, j)
				nx.set_edge_attributes(g, {(i, j): {"weight":sim}})
		output = from_networkx(g)
		torch.save(output, 'data/pheme-rnr-dataset/multigraph/multigraph_'+str(num)+'.pt')
		num +=1
	return None