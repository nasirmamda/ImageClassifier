import argparse

def get_input_args():

 parser = argparse.ArgumentParser()
 
 parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
 parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
 parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
 parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.01)
 parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
 parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
 parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
 parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)                   
                    
 return parser.parse_args()