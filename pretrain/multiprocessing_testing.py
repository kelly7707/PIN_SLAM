import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to run in parallel
def train_model(rank, model, data, target):
    for i in range(5):  # Each process performs 5 iterations
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss (MSE)
        loss = torch.nn.functional.mse_loss(output, target)
        print(f"Process {rank}: Iteration {i}, Loss = {loss.item()}")

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

# Main function
def main():
    # Initialize model and optimizer
    input_size, hidden_size, output_size = 4, 10, 2
    model = MLP(input_size, hidden_size, output_size)

    # Share the model's parameters across processes
    model.share_memory()  # This allows model weights to be shared

    # Use SGD optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy data
    data = torch.randn(1, input_size)
    target = torch.randn(1, output_size)

    # Start multiprocessing
    processes = []
    for rank in range(4):  # Create 4 processes
        p = mp.Process(target=train_model, args=(rank, model, data, target))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set the start method for multiprocessing
    main()
