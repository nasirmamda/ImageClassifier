import numpy as np
import initiate
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os, random
from PIL import Image








def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


    
    
    
from torch.autograd import Variable

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    
    image = process_image(img_path)
    
    image = torch.from_numpy(np.array([image])).float()
    
    # The image becomes the input
    image = Variable(image)
    
    image = image.to(device)
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk    (=5) probabilites and indexes
 # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label


import argparse


parser = argparse.ArgumentParser(description='Flower Class+ification Predictor')
parser.add_argument('--image_path', type=str, help='path of image')
parser.add_argument('--saved_model' , type=str, default='checkpoint.pth', help='path of your saved model')
parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

args = parser.parse_args()


nn_filename = args.saved_model # 'checkpoint.pth'
model, optimizer = initiate.load_checkpoint(nn_filename)

#chkp_model = print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#img = random.choice(os.listdir('./flowers/test/6/'))
img_path = args.image_path#'./flowers/test/6/' + img


#with Image.open(img_path) as image:
   # plt.imshow(image)

model.to(device)
prob, classes = predict(img_path, model,args.topk)
print(prob)
print(classes)


import json

with open(args.mapper_json, 'r') as f:
    cat_to_name = json.load(f)
    

print([cat_to_name[x] for x in classes])
    






