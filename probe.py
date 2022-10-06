import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProbeDataset(Dataset):

    def __init__(self, args, mode):
        self.novel = args.novel
        self.mode = mode
        self.model = args.model

        if self.novel:
            self.d_str = 'novel'
        else:
            self.d_str = 'geirhos'

        self.shape_classes = json.load(open('shape_classes/{}_shape_classes.json'.format(self.d_str)))
        self.embed_model = args.model

        self.blur = args.blur
        if self.blur == 0:
            self.blur_str = ''
        else:
            self.blur_str = '_{0}'.format(str(self.blur))

        self.bg = args.bg
        if self.bg:
            if '/' in self.bg:
                bg = self.bg.split('/')[-1]

            self.bg_str = 'background_{0}{1}/'.format(bg[:-4], self.blur_str)
        else:
            self.bg_str = ''

        self.alpha = int(args.alpha * 255)
        if args.alpha == 1:
            self.alpha_str = '1'
        else:
            self.alpha_str = str(args.alpha)

        self.percent = args.percent_size
        self.unaligned = args.unaligned
        if self.unaligned:
            self.a_str = 'unaligned'
        else:
            self.a_str = 'aligned'

        self.stimuli_dir = '{0}{1}-alpha{2}-size{3}-{4}'.format(self.bg_str, self.d_str,
                                                                self.alpha_str, self.percent,
                                                                self.a_str)

        self.embedding_dir = 'embeddings/{0}/{1}.json'.format(self.embed_model, self.stimuli_dir)
        self.embeddings = json.load(open(self.embedding_dir))

        for stim in self.embeddings.keys():
            self.embeddings[stim] = torch.tensor(self.embeddings[stim], device=device)

        self.stimuli = sorted(list(self.embeddings.keys()))
        self.embedding_size = len(self.embeddings[self.stimuli[0]])
        self.labels = {}

        try:
            os.mkdir('results/{0}/probe'.format(self.model))
        except FileExistsError:
            pass

        if self.bg:
            try:
                os.mkdir('results/{0}/probe/{1}'.format(self.model, self.bg_str))
            except FileExistsError:
                pass

        try:
            os.mkdir('results/{0}/probe/{1}'.format(self.model, self.stimuli_dir))
        except FileExistsError:
            pass

        try:
            os.mkdir('probe')
        except FileExistsError:
            pass

        self.create_labels()

    def create_labels(self):
        label_dir = 'probe/{}_{}_labels.json'.format(self.d_str, self.mode)

        if os.path.exists(label_dir):
            self.labels = json.load(open(label_dir))

            for stim in self.labels.keys():
                self.labels[stim] = torch.tensor(self.labels[stim], dtype=torch.int, device=device)

            return

        classes = []
        for stim in self.stimuli:
            if len(classes) == 16:
                break

            if self.shape_classes[stim][self.mode] not in classes:
                classes.append(self.shape_classes[stim][self.mode])

        classes = sorted(classes)

        for stim in self.stimuli:
            label = np.zeros(16)
            c = self.shape_classes[stim][self.mode]

            label[classes.index(c)] = 1
            self.labels[stim] = label.tolist()

        with open(label_dir, 'w') as file:
            json.dump(self.labels, file)

        for stim in self.labels.keys():
            self.labels[stim] = torch.tensor(self.labels[stim], dtype=torch.int, device=device)

    def __getitem__(self, index):
        stim = self.stimuli[index]
        return {'embeddings': self.embeddings[stim], 'labels': self.labels[stim]}

    def __len__(self):
        return len(self.stimuli)


class Probe(torch.nn.Module):
    def __init__(self, input_size, num_layers=1):
        super(Probe, self).__init__()
        self.num_layers = num_layers + 1
        layer_sizes = np.linspace(input_size, 16, num_layers + 1, dtype=int)

        self.layers = []
        for i in range(num_layers):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module('linear{}'.format(i + 1), layer)

        softmax = torch.nn.Softmax(dim=1)
        self.layers.append(softmax)
        self.add_module('softmax', softmax)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)

        return x


def run_probe(args, mode, num_epochs=100, num_probe_layers=1, i=1):
    learning_rate = 0.01
    batch_size = 64
    train_percentage = 0.8

    dataset = ProbeDataset(args, mode)
    train_size = int(train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

    name = 'probe{}_{}_{}'.format(i, num_probe_layers, mode)
    result_dir = 'results/{0}/probe/{1}/{2}.csv'.format(args.model, dataset.stimuli_dir, name)

    probe = Probe(dataset.embedding_size, num_layers=num_probe_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=learning_rate)

    columns = ['Probe', 'Number Probe Layers', 'Mode', 'Epoch', 'Train Loss', 'Train Acc',
               'Eval Loss', 'Eval Acc']
    probe_results = pd.DataFrame(index=range(num_epochs), columns=columns)
    probe_results['Probe'] = i
    probe_results['Number Probe Layers'] = num_probe_layers
    probe_results['Mode'] = mode

    for epoch in range(num_epochs):
        #print('Epoch {}'.format(epoch))
        probe_results.at[epoch, 'Epoch'] = epoch + 1
        epoch_loss = []
        epoch_acc = []

        probe.train()

        for idx, batch in enumerate(train_dataloader):
            inputs = batch['embeddings']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = probe(inputs)

            loss = criterion(outputs, labels.to(torch.float64))
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            labels = labels.numpy()
            predictions = np.zeros(outputs.shape, dtype=int)
            predictions[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1

            acc = np.sum(np.sum(np.multiply(predictions, labels), axis=1), axis=0) / batch_size
            epoch_acc.append(acc)

        #print('\tLoss: {}'.format(np.sum(epoch_loss) / len(epoch_loss)))
        #print('\tAcc: {}'.format(np.sum(epoch_acc) / len(epoch_acc)))
        probe_results.at[epoch, 'Train Loss'] = np.sum(epoch_loss) / len(epoch_loss)
        probe_results.at[epoch, 'Train Acc'] = np.sum(epoch_acc) / len(epoch_acc)

        probe.eval()

        with torch.no_grad():
            test_loss = []
            test_acc = []

            for idx, batch in enumerate(test_dataloader):
                inputs = batch['embeddings']
                labels = batch['labels']

                outputs = probe(inputs)

                loss = criterion(outputs, labels.to(torch.float64))
                test_loss.append(loss.item())

                labels = labels.numpy()
                predictions = np.zeros(outputs.shape, dtype=int)
                predictions[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1

                acc = np.sum(np.sum(np.multiply(predictions, labels), axis=1), axis=0) / test_size
                test_acc.append(acc)

            #print('\tEval Loss: {}'.format(test_loss[0]))
            #print('\tEval Acc: {}'.format(test_acc[0]))
            probe_results.at[epoch, 'Eval Loss'] = test_loss[0]
            probe_results.at[epoch, 'Eval Acc'] = test_acc[0]

    probe_results.to_csv(result_dir, index=False)
