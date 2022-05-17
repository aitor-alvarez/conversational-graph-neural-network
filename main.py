from utils.preprocess import generate_conversation_graph, create_tree_pheme
import argparse
from data.dataset import process_pheme
from train import main as graph_training
from multiGraph import node_embedding, create_multiLevelGraph
from nodePrediction import main as GraphofGraphs


if __name__ == 'main':
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--train_graphs', type=str, default='data/pheme-rnr-dataset/',
	                    help='Path to dataset to generate graphs and train single models')

	parser.add_argument('-gog', '--train_graph_of_graphs', type=str,
	                    help='Data path to graph file.')



	args = parser.parse_args()

	if args.train_graphs:
		process_pheme(args.generate_graph)
		create_tree_pheme(args.generate_graph)
		generate_conversation_graph(args.generate_graph)
		graph_training(args.generate_graph)
		node_embedding(args.generate_graph)
		create_multiLevelGraph(args.generate_graph+'/multigraph/')
		GraphofGraphs(args.generate_graph+'/multigraph/')



