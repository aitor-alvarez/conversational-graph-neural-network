import torch
from data.dataloader import graph_loader
from torch_geometric.loader import DataLoader
from models.text_graphs import ConversationalGraph
import random



def training(train, test, num_epochs=100, batch_size=32, num_classes=2):
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = ConversationalGraph(embedding_size=768, hidden_channels=128, num_classes=num_classes).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
	model.train()
	epochs_stop = 3
	min_loss = None
	no_improve = 0
	acc_list = []
	epoch_min_loss = None
	start_epoch = 1
	for epoch in range(start_epoch, num_epochs):
		epoch_loss = []
		for batch in train_loader:
			optimizer.zero_grad()
			out = model(batch.x, batch.edge_index, batch.weight, batch.batch)
			loss = criterion(out, batch.y)
			total = batch.y.size(0)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == batch.y).sum().item()
			acc_list.append(correct / total)
			loss.backward()
			optimizer.step()
			epoch_loss.append(loss)

		### Epoch check ###
		e_loss = sum(epoch_loss) / len(epoch_loss)
		print(e_loss)
		print(correct / total)
		if epoch_min_loss == None:
			epoch_min_loss = e_loss
		elif e_loss < epoch_min_loss:
			epoch_min_loss = e_loss
			no_improve = 0
		else:
			no_improve += 1
		if no_improve == epochs_stop:
			break

	###TEST####

	torch.save(model, 'trained/conversation.pt')
	model.eval()
	correct=0
	with torch.no_grad():
		for batch in test_loader:
			out = model(batch.x, batch.edge_index, batch.weight, batch.batch)
			loss = criterion(out, batch.y)
			total = batch.y.size(0)
			_, predicted = torch.max(out.data, 1)
			correct += (predicted == batch.y).sum().item()
		total_correct = correct/(len(test_loader)*batch_size)
		print(total_correct)
	return total_correct


def main(directory):
	data = graph_loader(directory)
	random.shuffle(data)
	split = round(len(data)*0.2)
	test = data[:split-1]
	train = data[split:]
	training(train, test)


