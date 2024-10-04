import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import tqdm
from OCR import transform_label,OCRDataset,OCRModel,unique_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser(description='Check how the model performs in the test data.')
parser.add_argument("batch_size", type=int, help="The batch size of the data that the model will be tested on (should be the same as in training).")
args = parser.parse_args()


label_to_idx,idx_to_label = transform_label('train_file.txt')
training_dataset = OCRDataset('train_file.txt',label_to_idx,target_size=(60,60))
test_dataset = OCRDataset('test_file.txt',label_to_idx,target_size=(60,60))

train_dataloader = DataLoader(training_dataset,batch_size = args.batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size, shuffle = True)
num_classes = unique_labels(training_dataset)

def test_model(device = 'cuda:1'):

    model = OCRModel(num_classes)  
    model.to(device)

    model.load_state_dict(torch.load('ocr_model.pth', map_location=device, weights_only=True))
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images,labels in test_dataloader:
            images = images.to(device)
            labels = torch.tensor([label for label in labels]).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.2f}%")
    print(f"Test Recall: {recall:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")

test_model()