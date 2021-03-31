
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils
import torchvision

def initiate(image_dir,lrn_rate,dropout,arch,hiddenlayer):

  data_dir = image_dir
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'



  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

  valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

  # TODO: Load the datasets with ImageFolder
  train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
  valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
  test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

  # TODO: Using the image datasets and the trainforms, define the dataloaders
  trainloaders =  torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
  validloaders =  torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
  testloaders =  torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)


  import json

  with open('cat_to_name.json', 'r') as f:
      cat_to_name = json.load(f)

  print(cat_to_name)
  print("\n Length:", len(cat_to_name))


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if arch == 'densenet':
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(512, hiddenlayer),
                                   nn.LogSoftmax(dim=1)) 
  elif arch == 'vgg':
    model = models.vgg19(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1024, hiddenlayer),
                                   nn.LogSoftmax(dim=1)) 
  
     
  
  #for param in model.parameters():
  #    param.requires_grad = False

  

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.classifier.parameters(), lr = lrn_rate)

  model.to(device);


  return trainloaders, testloaders ,device,optimizer,model,criterion,testloaders,train_datasets,validloaders




def load_checkpoint(filename):
    
    checkpoint = torch.load(filename)
   
    learning_rate = checkpoint['learning_rate']
    
    #model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    model_name = checkpoint['arch']
    if model_name == 'vgg':
        model = models.vgg19(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer