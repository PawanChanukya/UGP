import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

# Data preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

# Model, Loss, Optimizer
model = torchvision.models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# Mixed Precision scaler
scaler = torch.cuda.amp.GradScaler()

# Train Function with tqdm
def train(epoch):
    model.train()
    total, correct, loss_total = 0, 0, 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", leave=False)

    for images, labels in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_total += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress.set_postfix({
            'Loss': f'{loss_total / (total / labels.size(0)):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    return 100. * correct / total

# Test Function
def test():
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total

# Training Loop
num_states = 100
total_epochs = 200
save_interval = max(1, total_epochs // num_states)

for epoch in range(1, total_epochs + 1):
    train_acc = train(epoch)

    if epoch % save_interval == 0 or epoch == total_epochs:
        state_id = epoch // save_interval
        # Move model to CPU for saving
        model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_cpu_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
        }, f'checkpoints/resnet_cifar10_state_{state_id:03d}.pth')
        print(f"[Checkpoint] Saved model state {state_id:03d} at epoch {epoch}")

    scheduler.step()

# Final Accuracy
final_test_acc = test()
print(f"\n Final Test Accuracy: {final_test_acc:.2f}%")
