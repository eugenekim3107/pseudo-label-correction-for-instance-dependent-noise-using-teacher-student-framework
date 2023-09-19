import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader

from utils import get_softmax_out, get_pseudo_labels
from ops import train, train_soft, test, generate_student_labels, train_siamese
from networks.cnn_mnist import MNIST_CNN
from networks.siamesenet import TripletNet, EmbeddingNet
from dataset import SiameseMNISTDataset, collate_fn, SiameseMNISTCorrectionDataset, CustomDataset
from loss import TripletLoss
import pickle

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs_student', type=int, default=20, help='number of epochs to train student model (default: 20)')
    parser.add_argument('--epochs_teacher', type=int, default=20, help='number of epochs to train teacher model (default: 20)')
    parser.add_argument('--epochs_retrain', type=int, default=50, help='number of epochs to retrain student model (default: 10)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_rate', type=float, default=0.22, help='Noise rate (default: 0.22)')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # datasets
    root = 'data'
    kwargs = {'num_workers':4} if torch.cuda.is_available() else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))]) #0.286, 0.353 are the mean and std of mnist
    train_dataset = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    train_dataset_noisy = datasets.FashionMNIST(root, train=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root, train=False, transform=transform)
    
    targets_noisy = torch.Tensor(pd.read_csv(os.path.join('./data/FashionMNIST/label_noisy', 'dependent' + str(args.noise_rate)+'.csv'))['label_noisy'].values.astype(int))
    train_dataset_noisy.targets = targets_noisy

    # stratify the sample for meta dataset
    class_indices = [[] for _ in range(10)]
    for idx, (image, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
        
    meta_indices = []
    base_indices = []
    for class_idx in range(10):
        class_data_indices = class_indices[class_idx]
        meta_indices_class, base_indices_class = train_test_split(class_data_indices, train_size=0.02, stratify=train_dataset.targets[class_data_indices], random_state=args.seed)
        meta_indices.extend(meta_indices_class)
        base_indices.extend(base_indices_class)
        
    # separate meta and base datasets
    meta_dataset = Subset(train_dataset, meta_indices)
    base_dataset = Subset(train_dataset_noisy, base_indices)
    meta_siamese_dataset = SiameseMNISTDataset(meta_dataset, args.seed, num_classes=10)
    
    y_noisy = torch.tensor([label for _,label in base_dataset])
    
    # create dataloaders
    meta_loader = DataLoader(meta_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    meta_siamese_loader = DataLoader(meta_siamese_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    train_loader = DataLoader(base_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # train student network on meta dataset and create initial pseudo labels
    student_model = MNIST_CNN().to(device)
    student_optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    student_loss_fn = nn.CrossEntropyLoss()
    print("Starting student model training...")
    for epoch in range(1, args.epochs_student+1):
        train(args.epochs_student, student_model, device, meta_loader, student_optimizer, student_loss_fn, epoch)
    
    print("Student model making initial predictions on noisy data...")
    student_pred, given_pred, images = generate_student_labels(args, student_model, device, train_loader)
    
    # train teacher network on meta dataset and correct pseudo labels
    embeddingnet = EmbeddingNet()
    teacher_model = TripletNet(embeddingnet).to(device)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=args.lr)
    teacher_loss_fn = TripletLoss(margin=1)
    
    print("Starting teacher model training...")
    for epoch in range(1, args.epochs_teacher+1):
        train_siamese(args.epochs_teacher, teacher_model, device, meta_siamese_loader, teacher_optimizer, teacher_loss_fn, epoch)

    # train student network on corrected pseudo labels
    corrected_dataset = SiameseMNISTCorrectionDataset(meta_dataset, base_dataset, student_model, teacher_model, args.seed, num_classes=10)
    corrected_loader = DataLoader(corrected_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # save the pseudo labels
    print("Creating corrected pseudo labels...")
    y_corrected = torch.Tensor(get_pseudo_labels(corrected_loader)).type(torch.int64)
    print(f"Estimated Noise Level: {torch.sum(y_noisy != y_corrected) / y_noisy.shape[0]}")
#     base_dataset.targets = torch.Tensor(y_corrected).type(torch.int)
    base_dataset = CustomDataset(base_dataset, y_corrected)
    train_loader = DataLoader(base_dataset, batch_size = args.batch_size, shuffle=False, **kwargs)
    torch.save(base_dataset, f"final_dataset{args.noise_rate}.pth")
    torch.save(torch.sum(y_noisy != y_corrected) / y_noisy.shape[0], f"result/fmnist_noise_estimate_{args.noise_rate}_{args.seed}.pth")
                      
    print("Retraining student model on corrected labels...")
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(1, args.epochs_retrain+1):
        #train(args.epochs_retrain, student_model, device, corrected_loader, student_optimizer, student_loss_fn, epoch)
        train_loss, train_acc = train(args.epochs_retrain, student_model, device, train_loader, student_optimizer, student_loss_fn, epoch)
        train_losses.append(round(train_loss,2))
        train_accs.append(round(train_acc,2))
        test_loss, test_acc = test(args, student_model, device, test_loader, top5=False)
        test_losses.append(round(test_loss,2))
        test_accs.append(round(test_acc,2))
    print("Student model final accuracy...")
    test(args, student_model, device, test_loader, top5=False)
    print(train_losses)
    print(train_accs)
    print(test_losses)
    print(test_accs)
    torch.save(torch.Tensor(test_accs), f"result/fmnist_test_acc_{args.noise_rate}_{args.seed}.pth")
        
if __name__ == "__main__":
    main()
    
    
    
    