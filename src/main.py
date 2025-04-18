import torch
from torch import nn, optim
from cnn_model import CNN
from data_loader import get_data_loaders
from trainer import Trainer
from evaluator import Evaluator
from visualizer import visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders()
cnn_model = CNN()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

trainer = Trainer(cnn_model, train_loader, loss_function, optimizer, device)
trainer.train(num_epochs=10)

evaluator = Evaluator(cnn_model, test_loader, device)
accuracy = evaluator.evaluate()
print(f'CNN Model Accuracy: {accuracy:.2f}%')

dataiter = iter(test_loader)
images, labels = next(dataiter)
visualize(images, cnn_model, device)
