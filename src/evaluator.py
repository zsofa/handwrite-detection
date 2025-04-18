import torch

class Evaluator:
    def __init__(self, cnn_model, test_loader, device):
        self.cnn_model = cnn_model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.cnn_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
