# ImageClassifier 
In this project we are takeing parameter from user training the model with given pretrained model and then predict given image 


#How to run project

#Train the Model

python train.py --epochs 4 --learning_rate 0.01 --arch 'vgg' --hidden_units 102

#Predict Image

 python predict.py --image_path 'flowers/test/10/image_07090.jpg' --mapper_json 'cat_to_name.json' --topk 5
