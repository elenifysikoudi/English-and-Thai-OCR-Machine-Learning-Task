import argparse
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from PIL import Image


def transform_label(data_file):
    data = []
    label_to_idx = {}
    idx_to_label = []
    with open(data_file,'r') as file:
            for line in file:
                path,ocr_number = line.strip().split(',')
                data.append((path,ocr_number))
                if ocr_number not in label_to_idx:
                    label_to_idx[ocr_number] = len(label_to_idx)
                    idx_to_label.append(ocr_number)
    return label_to_idx , idx_to_label

class OCRDataset(Dataset):
    def __init__(self,data_file,label_to_idx,target_size):
        self.data = []
        self.label_to_idx = label_to_idx
        self.target_size = target_size
        
        with open(data_file,'r') as file:
            for line in file:
                path,ocr_number = line.strip().split(',')
                self.data.append((path,ocr_number))
                
        self.transform = transforms.Compose([
            transforms.ToTensor()
                ])
        
    def pad_image(self, image):
         
        """Pads the image to the target size while maintaining aspect ratio."""
        width, height = image.size  
        target_width, target_height = self.target_size

        pad_left = (max(0, target_width - width) // 2)
        pad_top = (max(0, target_height - height) // 2)
        pad_right = max(0, target_width - width) - pad_left
        pad_bottom = max(0, target_height - height) - pad_top

        transform = transforms.Compose([
        transforms.Pad((pad_left, pad_top, pad_right, pad_bottom),fill = 255),
        transforms.Resize(self.target_size) ])
    
        return transform(image)

        
    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):
        image_path,label = self.data[idx]
        label_idx = self.label_to_idx[label]
        #print(label_idx)

        image = Image.open(image_path) 
        image = self.pad_image(image)
        image = self.transform(image)

        return image,label_idx


class OCRModel(nn.Module):
    def __init__(self,num_classes):
        super(OCRModel,self).__init__()
        self.conv1= nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32) 
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(15 * 15 * 64, 300)
        self.tanh = nn.Tanh()
        #self.dropout2 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,image):
        output = self.conv1(image)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.flatten(output)
        output = self.dropout1(output)
        output = self.linear1(output)
        output = self.tanh(output)
        #output = self.dropout2(output)
        output = self.linear2(output)
        return self.softmax(output)
        
def unique_labels(dataset):
    unique_labels = set([label for _,label in dataset.data])
    return len(unique_labels)
    