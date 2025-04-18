class Trainer:
    def __init__(self, cnn_model, train_loader, loss_function, optimizer, device):
        self.cnn_model = cnn_model.to(device)
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.cnn_model(images)

                loss = self.loss_function(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')