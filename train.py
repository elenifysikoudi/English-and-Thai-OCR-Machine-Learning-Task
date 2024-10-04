import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import tqdm
from OCR import transform_label,OCRDataset,OCRModel,unique_labels


parser = argparse.ArgumentParser(description='Train the model with arbitrary number of batches and epochs.')
parser.add_argument("batch_size", type=int, help="The batch size of the data that the model will be trained on.")
parser.add_argument("num_epochs", type=int, help="The number of epochs the model will train for.")
args = parser.parse_args()


label_to_idx,idx_to_label = transform_label('train_file.txt')
training_dataset = OCRDataset('train_file.txt',label_to_idx,target_size=(60,60))

train_dataloader = DataLoader(training_dataset,batch_size = args.batch_size, shuffle = True)
num_classes = unique_labels(training_dataset)


def train(num_epochs = 15, device="cuda:1"):
    model=OCRModel(num_classes)
    device = torch.device('cuda:1')
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_id ,(images,labels) in  enumerate(tqdm.tqdm(train_dataloader)):
            images = images.to(device)
            labels = torch.tensor([int(label) for label in labels]).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}%")

        torch.save(model.state_dict(),"ocr_model.pth")

train(args.num_epochs)