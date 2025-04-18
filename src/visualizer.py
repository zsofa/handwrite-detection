import matplotlib.pyplot as plt
import torch


def visualize(images, cnn_model, device):
    figure = plt.figure(figsize=(10, 4))
    with torch.no_grad():
        for i in range(6):
            axes = figure.add_subplot(2, 3, i + 1)
            axes.imshow(images[i].numpy().squeeze(), cmap='gray')
            axes.set_title(f'Prediction: {cnn_model(images[i].unsqueeze(0).to(device)).argmax().item()}')
    plt.show()
