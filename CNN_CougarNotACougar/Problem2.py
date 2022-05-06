import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms as T
import torchvision
import cv2
import os
import torch.utils.data as data
import matplotlib.pyplot as plt

# Get base directory
BASE_PATH = os.getcwd()
print(BASE_PATH)
# Get path for teams' folders
TEAMS_PATH = os.path.join(BASE_PATH, 'CougarNotACougarData')
# Verify if path exists. If not, create folders to split the data into train (70%) and test (30%) data sets
# and Cougar and NotACougar folders
if os.path.exists(os.path.join(BASE_PATH, 'train')) is False:
    os.mkdir(os.path.join(BASE_PATH, 'train'))
if os.path.exists(os.path.join(BASE_PATH, 'test')) is False:
    os.mkdir(os.path.join(BASE_PATH, 'test'))
if os.path.exists(os.path.join(os.path.join(BASE_PATH, 'train'), 'Cougar')) is False:
    os.mkdir(os.path.join(os.path.join(BASE_PATH, 'train'), 'Cougar'))
if os.path.exists(os.path.join(os.path.join(BASE_PATH, 'test'), 'Cougar')) is False:
    os.mkdir(os.path.join(os.path.join(BASE_PATH, 'test'), 'Cougar'))
if os.path.exists(os.path.join(os.path.join(BASE_PATH, 'train'), 'NotACougar')) is False:
    os.mkdir(os.path.join(os.path.join(BASE_PATH, 'train'), 'NotACougar'))
if os.path.exists(os.path.join(os.path.join(BASE_PATH, 'test'), 'NotACougar')) is False:
    os.mkdir(os.path.join(os.path.join(BASE_PATH, 'test'), 'NotACougar'))


class CougarCnn:
    def __init__(self) -> None:
        self.train_data_path = os.path.join(os.path.join(BASE_PATH, 'train'))
        self.test_data_path = os.path.join(os.path.join(BASE_PATH, 'test'))
        self.batch_size = 40
        self.epochs = 3
        self.learning_rate = 0.001
        self.loss_list = []
        self.training_accuracy_list = []
    
    def organize_data(self):
        # For each team folder
        for team in os.listdir(TEAMS_PATH):
            # Get path to team folder
            team_path = os.path.join(TEAMS_PATH, team)
            i = 0
            for img in os.listdir(team_path):
                img_path = os.path.join(team_path, img)
                image = cv2.imread(img_path)
                # print(image)
                # Check if it still in the first 70% of the images inside the folder and which class it belongs to
                if i < 0.7 * len(os.listdir(team_path)) and 'WSU' in team:
                    dest_path = os.path.join(os.path.join(BASE_PATH, 'train'), 'Cougar') + '/' + team  + str(i) + '.jpeg'
                    # print(dest_path)
                    cv2.imwrite(dest_path, image) 
                elif i >= 0.7 * len(os.listdir(team_path)) and 'WSU' in team:
                    dest_path = os.path.join(os.path.join(BASE_PATH, 'test'), 'Cougar') + '/' + team + str(i) + '.jpeg'
                    # print(dest_path)
                    cv2.imwrite(dest_path, image)
                elif i < 0.7 * len(os.listdir(team_path)) and 'WSU' not in team:
                    dest_path = os.path.join(os.path.join(BASE_PATH, 'train'), 'NotACougar') + '/' + team + str(i) + '.jpeg'
                    # print(dest_path)
                    cv2.imwrite(dest_path, image)
                elif i >= 0.7 * len(os.listdir(team_path)) and 'WSU' not in team:
                    dest_path = os.path.join(os.path.join(BASE_PATH, 'test'), 'NotACougar') + '/' + team + str(i) + '.jpeg'
                    # print(dest_path)
                    cv2.imwrite(dest_path, image)
                i += 1  

    # Grab data to model
    def data_cnn(self):
        TRANSFORM = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )])

        train_data = torchvision.datasets.ImageFolder(root=self.train_data_path, transform=TRANSFORM)
        self.train_loader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,  num_workers=4)
        test_data = torchvision.datasets.ImageFolder(root=self.test_data_path, transform=TRANSFORM)
        self.test_loader = data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True,  num_workers=4)
        self.n_total_step = len(self.train_loader)
        # classes = ('Cougar', 'NotACougar')
    
    # Create CNN model from prebuilt and pretrained Resnet50
    def cnn(self):
        self.model = models.resnet50(pretrained=True)
        
        # Change the last fully connected layer to meet our number of classes
        numft=self.model.fc.in_features
        self.model.fc = nn.Linear(numft, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def train_cnn(self):
        for epoch in range(self.epochs):
            for i, (imgs , labels) in enumerate(self.train_loader):
                print(i)
                labels_hat = self.model(imgs)
                n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
                loss_value = self.criterion(labels_hat, labels)
                # Calculate accuracy in this batch
                accuracy = n_corrects/labels.size(0)
                # Store accuracy and loss values
                self.training_accuracy_list.append(accuracy)
                self.loss_list.append(float(loss_value))
                loss_value.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if (i+1) % 3 == 0:
                    print(f'epoch {epoch+1}/{self.epochs}, step: {i+1}/{self.n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%')
        print('Finished Training')
    
    # Calculate overall test accuracy
    def overall_test_accuracy(self):
        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set , test_labels_set) in enumerate(self.test_loader):
                test_images_set = test_images_set
                test_labels_set = test_labels_set
                y_predicted = self.model(test_images_set)
                labels_predicted = y_predicted.argmax(axis = 1)
                number_corrects += (labels_predicted==test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples)*100}%')
    
    # Plot training accuracy and loss values during training 
    def plots(self):
        plt.figure()
        plt.plot(np.arange(3*self.n_total_step), np.array(self.training_accuracy_list))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.figure()
        plt.plot(np.arange(3*self.n_total_step), np.array(self.loss_list))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

if __name__ == '__main__':
    cougar_cnn = CougarCnn()
    # Uncomment the next line if you do not have the data split yet (Before doing this, check first in your directory)
    # cougar_cnn.organize_data()
    cougar_cnn.data_cnn()
    cougar_cnn.cnn()
    cougar_cnn.train_cnn()
    cougar_cnn.overall_test_accuracy()
    cougar_cnn.plots()
