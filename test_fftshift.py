import torch

# Example dimensions
N = 5
dims = 3

# Create a batch tensor of shape (N, dims)
batch_tensor = torch.randn(N, dims)

# Create a single tensor of shape (dims)
single_tensor = torch.randn(dims)

# Multiply all N elements of the batch tensor with the single tensor
result_tensor = batch_tensor * single_tensor

# Sum along the batch dimension to get the final result
final_result = result_tensor.sum(dim=0)

# Print or use the final_result as needed
print(final_result)
