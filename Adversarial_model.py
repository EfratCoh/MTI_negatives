import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
from utils.utilsfile import read_csv, to_csv
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader

def prepare_data():
    frames = []
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()
    train_dir = Path("/sise/home/efrco/efrco-master/data/train/underSampling/0")
    list_method = ["top", "Liver", "tarBase_human", "_nucleotides", "non_overlapping_sites_clip_data_darnell_human_ViennaDuplex_negative_"]
    for f_train in train_dir.glob('**/*.csv'):
        f_stem = f_train.stem
        train_dataset = f_stem.split(".csv")[0].split("features_train_underSampling_method_0")[0]
        if any(method in train_dataset for method in list_method):
            continue
        else:

            X_test, y_test =  feature_reader.file_reader(f_train)
            y_test = pd.DataFrame(y_test, columns=['Label'])
            df = pd.concat([X_test, y_test], axis=1)
            df = df[df['Label']==0]
            df['Label'] = train_dataset
            frames.append(df)
    print(df.shape)
    print(len(frames))
    results = pd.concat(frames)
    print(results.shape)



# prepare_data()




class AdversarialModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdversarialModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training data
train_data = prepare_data()
train_labels= ['tarBase_microArray', "mockMiRNA",
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3", "Non_overlapping_sites_clip_data_random"]

def creat_model():
    # Hyperparameters
    input_size = 500
    hidden_size = 200
    num_classes = 9  # Number of different methods

    # Create the adversarial model
    model = AdversarialModel(input_size, hidden_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = ...
    batch_size = ...

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / (len(train_data) / batch_size)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')