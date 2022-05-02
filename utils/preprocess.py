from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import os
import pandas as pd
import json
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
	DataCollatorWithPadding
import torch
from torch_geometric.data import TemporalData
from torch_geometric.utils import from_networkx
import networkx as nx


tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
tweet_model = AutoModel.from_pretrained("vinai/bertweet-base")

def create_tree_pheme(dir):
	for f in os.listdir(dir):
		tree = []
		if f.endswith('.csv'):
			data = pd.read_csv(dir + f, lineterminator='\n')
			roots = data[data['is_source_tweet'] == True]
			data = data[data['is_source_tweet'] == False]
			for root in roots.iterrows():
				root_txt = root[1][2]
				root_id = root[1][0]
				replies = data[data['thread'] == root_id]
				replies_out = []
				for reply in replies.iterrows():
					rep_dict = {'resp_id': int(reply[1][3]), 'resp_txt': reply[1][2],
					            'label': 1 if str(reply[1][6]) == 'True' else 0}
					replies_out.append(rep_dict)
				tree.append(
					{'root_id': int(root_id), 'root_txt': root_txt, 'root_label':1 if str(root[1][6]) == 'True' else 0,
					 'responses': replies_out})

			output = json.dumps(tree)
			with open('data/pheme-rnr-dataset/'+f.replace('.csv','')+'_tree.json', 'w') as outfile:
				outfile.write(output)


#Build and save conversation graph for each event data
def generate_conversation_graph(directory):
	for d in os.listdir(directory):
		dir = directory+'graphs/conversation/'+d.replace('.json','')+'/'
		if not os.path.exists(dir) and d.endswith('.json'):
			os.mkdir(dir)
		if d.endswith('.json'):
			with open(directory+d, 'r') as js:
				js_data = json.loads(js.read())
			for j in js_data:
				graph = nx.Graph()
				filename = str(j['root_id'])+'.pt'
				try:
					root_embedding = bert_tweet([j['root_txt']])
					graph.add_node(j['root_id'], x=root_embedding, y=j['root_label'])
				except:
					continue
				for node in j['responses']:
					try:
						node_embedding = bert_tweet([node['resp_txt']])
						graph.add_node(node['resp_id'], x= node_embedding, y=node['label'])
						graph.add_edge(j['root_id'], node['resp_id'])
						sim = embedding_similarity(root_embedding, node_embedding)
						nx.set_edge_attributes(graph, {(j['root_id'], node['resp_id']): {"weight": sim}})
					except:
						continue
				output = from_networkx(graph)
				torch.save(output, dir+filename)
	return None


# Create word embeddings for the tweets using shallow models
def word_embeddings(tweets):
	sentences = normalize_text(tweets)
	model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=5)
	embeddings = model.wv
	embeddings.save("pretrained/word2vec.wordvectors")
	return embeddings


#Cosine similarity of two embeddings with the same dimension
def embedding_similarity(emb1, emb2):
	similarity = util.cos_sim(emb1, emb2)
	return similarity


# Bertweet embedding
def bert_tweet(tweets):

	tokens = {'input_ids': [], 'attention_mask': []}

	for sen in tweets:
		sen_norm = tokenizer.normalizeTweet(sen)
		tkn = tokenizer.encode_plus(sen_norm, max_length=130,
		                            truncation=True, padding='max_length',
		                            return_tensors='pt')
		tokens['input_ids'].append(tkn['input_ids'][0])
		tokens['attention_mask'].append(tkn['attention_mask'][0])

	tokens['input_ids'] = torch.stack(tokens['input_ids'])
	tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

	with torch.no_grad():
		output = tweet_model(**tokens)
	tweet_embeddings = mean_pooling(output, tokens['attention_mask'])
	return tweet_embeddings


def transformer_sentences(tweets):
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	embeddings = model.encode(tweets)
	return embeddings


def normalize_text(sentences):
	word_tokenizer = NLTKWordTokenizer()
	stop_words = set(stopwords.words('english'))
	normalized_sentences = []
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if
		        t not in stop_words and t not in '@.,!#$%*:;"' and 'http' not in t and 'www' not in t and len(t)>1]
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences


# Mean Pooling using the attention mask of the tokens
def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0]
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Fine tuning for Transformer-based models for text classification
def transformer_fine_tuning(model_name, train_data, test_data, tokenizer, nlabels):
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=nlabels)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	training_args = TrainingArguments(output_dir="data/trained/model_" + model_name, learning_rate=2e-5,
	                                  per_device_train_batch_size=16, per_device_eval_batch_size=16,
	                                  num_train_epochs=5,
	                                  weight_decay=0.001, )
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_data,
		eval_dataset=test_data,
		tokenizer=tokenizer,
		data_collator=data_collator, )
	trainer.train()
	return model