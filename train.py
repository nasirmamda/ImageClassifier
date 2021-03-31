


import initiate
import get_input_args
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from torch.autograd import Variable



def TrainNow(arg,model,device,optimizer,epochs,allDL):
       
    model.to(device);
   
    steps = 0


    running_loss = 0
    accuracy = 0

    start = time.time()
    print('Training started')

    for e in range(epochs):

        trainNow = 0
        validateNow = 1

        for now in [trainNow, validateNow]:   
            if now == trainNow:
                model.train()
            else:
                model.eval()

            pass_count = 0


            for data in allDL[now]:
                pass_count += 1
                inputs, labels = data

                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))


                optimizer.zero_grad()
                # Forward
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward
                if now == trainNow:
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if now == trainNow:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0

    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    
    return model


def ValidateModel():

    
    model.eval()
    accuracy = 0

    model.to(device);

    pass_count = 0

    for data in testloaders:
        pass_count += 1
        images, labels = data


        images, labels = Variable(images.to(device)), Variable(labels.to(device))

        output = model.forward(images)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy: {:.4f}".format(accuracy/pass_count))

    

def SaveCheckPoint(model,optimizer,epochs,lrn_rate,arch,hiddenlayer):
    checkpoint = {'input_size': 25088,
                  'output_size': hiddenlayer,
                  'arch': arch,
                  'learning_rate': lrn_rate,
                  'batch_size': 64,
                  'classifier' : model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')



in_arg = get_input_args.get_input_args()

trainloaders,get_input_args,device,optimizer,model ,criterion,testloaders,train_datasets,validloaders = initiate.initiate(in_arg.data_dir,in_arg.learning_rate,in_arg.dropout,in_arg.arch,in_arg.hidden_units)

DL = [trainloaders,validloaders,testloaders]

model = TrainNow(in_arg,model,device,optimizer,in_arg.epochs,DL)

model.class_to_idx = train_datasets.class_to_idx

SaveCheckPoint(model,optimizer,in_arg.epochs,in_arg.learning_rate,in_arg.arch,in_arg.hidden_units)
