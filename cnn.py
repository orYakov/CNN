import os

import torch
import torch.utils.data as data
import torch.nn as nn
from gcommand_loader import GCommandLoader

# define use of gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # initiate three layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7680, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 30)

    # forward function
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# load files function
def load(directory):
    dir_name = "data/" + directory
    data_set = GCommandLoader(dir_name)
    dir_loader = torch.utils.data.DataLoader(
        data_set, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    return dir_loader


# the train function
def train(num_epochs, optimizer, train_loader, model, criterion):
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    model.train()  # set the model to train mode
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # execute Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy of the model
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            # print the status every 100 inputs
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


# try the model on the validation set
def predict_on_validation(model, valid_loader):
    model.eval()  # set the model to test mode
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))


# a function to tet the model and write the answers
def test_the_model(model, data_set_of_test, test_loader):
    # extract the inputs
    spects = data_set_of_test.spects
    file_names = []  # an array to keep the file names
    for spec in spects:
        temp = spec[0].split('/')  # split the line
        # the name that end with wav
        temp = temp[2]
        file_names.append(temp)

    # set the model to test mode
    model.eval()
    array_of_predictions = []
    with torch.no_grad():
        for the_inputs, the_answers in test_loader:
            the_inputs = the_inputs.to(device)
            the_outputs = model(the_inputs)
            _, predicted = torch.max(the_outputs.data, 1)
            array_of_predictions.extend(predicted.tolist())

        # insert the file names and their predictions
        pred_to_write = []
        for temp, prediction in zip(file_names, array_of_predictions):
            line_to_write = temp + ", " + str(prediction)
            pred_to_write.append(line_to_write)

        with open("test_y", "w") as f:  # write the answers to a file
            for i in pred_to_write:
                f.write("%s\n" % i)


def load_data_for_test(directory):
    dir_path = "data/" + directory
    data_set = GCommandLoader(dir_path)
    test_loader = torch.utils.data.DataLoader(
        data_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)
    return test_loader, data_set


def main():
    print("starting...")

    # get the files
    train_loader = load("train")
    valid_loader = load("valid")
    test_loader, dataset = load_data_for_test("test")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.001

    # initiate the model
    model = ConvNet().to(device)

    # define Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train and predict
    print("training...")
    train(num_epochs, optimizer, train_loader, model, criterion)
    print("predicting on validation set...")
    predict_on_validation(model, valid_loader)

    # test the model
    print("testing and writing the predictions...")
    test_the_model(model, dataset, test_loader)
    print("done.")


if __name__ == '__main__':
    main()
