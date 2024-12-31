import torch

def top_k_accuracy(output, target, k=5):
    # Calculate top-k accuracy
    with torch.no_grad():  # Disable gradient calculation
        batch_size = target.size(0)  # Get batch size
        _, pred = output.topk(k, 1, True, True)  # Get top-k predictions
        pred = pred.t()  # Transpose predictions
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Check if predictions match targets
        return correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)  # Calculate accuracy
