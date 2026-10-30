# Import PyTorch to define and train the model.
import torch  # Tensor operations and automatic differentiation.
import torch.nn as nn  # Neural-network layers and loss functions.

# Import the MNIST dataset and image-to-tensor conversion.
from torchvision import datasets, transforms  # Download and prepare MNIST.

# Import matplotlib to display a test image.
import matplotlib.pyplot as plt  # Visualize the model example.

# Define a simple linear classifier; softmax is applied by the loss during training.
class LogisticRegression(nn.Module):

    # Set up one linear layer with 784 inputs and 10 class scores.
    def __init__(self):
        super().__init__()  # Initialize the parent PyTorch module.
        self.linear = nn.Linear(28 * 28, 10)  # Map image pixels to digit scores.

    # Define how an input image batch becomes class scores.
    def forward(self, images):
        images = images.view(images.size(0), -1)  # Flatten each 28x28 image.
        return self.linear(images)  # Return raw scores, called logits.

# Define the training and evaluation steps in one entry-point function.
def main():

    # Use a GPU when available; otherwise run on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert MNIST images into tensors with pixel values between 0 and 1.
    transform = transforms.ToTensor()

    # Download or load the training dataset.
    train_data = datasets.MNIST(root = "data", train = True, transform = transform, download = True)

    # Download or load the separate test dataset.
    test_data = datasets.MNIST(root = "data", train = False, transform = transform, download = True)

    # Create batches and shuffle only the training examples.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 100, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 100, shuffle = False)

    # Create the model and move it to the selected device.
    model = LogisticRegression().to(device)

    # CrossEntropyLoss expects raw logits and integer class labels.
    loss_function = nn.CrossEntropyLoss()

    # SGD updates model weights using the calculated gradients.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train for a small number of epochs to keep this project beginner-friendly.
    for epoch in range(5):
        model.train()  # Enable training behavior.
        total_loss = 0.0  # Track loss for this epoch.

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move batch to CPU/GPU.
            optimizer.zero_grad()  # Clear gradients from the previous batch.
            scores = model(images)  # Calculate class scores.
            loss = loss_function(scores, labels)  # Measure prediction error.
            loss.backward()  # Calculate gradients.
            optimizer.step()  # Update model weights.
            total_loss += loss.item()  # Add this batch's loss to the total.

        print(f"Epoch {epoch + 1}/5 - average loss: {total_loss / len(train_loader):.4f}")

    # Evaluate without calculating gradients to save memory and computation.
    model.eval()  # Switch to evaluation mode.
    correct, total = 0, 0  # Count correct predictions and all labels.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move batch to device.
            predicted = model(images).argmax(dim = 1)  # Select the class with the highest score.
            correct += (predicted == labels).sum().item()  # Count correct items.
            total += labels.size(0)  # Count items in this batch.

    print(f"Test accuracy: {correct / total:.2%}")

    # Save the learned model parameters for later loading.
    torch.save(model.state_dict(), "mnist_logistic_regression.pth")

    print("Saved model: mnist_logistic_regression.pth")

    # Show one test image and its predicted digit.
    image, label = test_data[0]
    model.eval()

    with torch.no_grad():
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim = 1).item())
    plt.imshow(image.squeeze(0), cmap = "gray")
    plt.title(f"True: {label} | Predicted: {prediction}")
    plt.axis("off")
    plt.show()

# Run the project when the file is executed directly.
if __name__ == "__main__":
    main()
