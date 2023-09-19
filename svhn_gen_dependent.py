import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn

from utils import get_softmax_out
from ops import train, test
from torchvision.models import resnet34
 
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 20)') # On clean data, 20 is sufficiently large to achiece 100% training accuracy.
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate (default: 0.0)')
    parser.add_argument('--load', action='store_true', default=False, help='Load existing averaged softmax')
    parser.add_argument('--gen', action='store_true', default=False, help='Generate noisy labels')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = 'data/SVHN'
    kwargs = {} if torch.cuda.is_available() else {}
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    (0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))
                                   ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                    (0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])
    train_dataset = datasets.SVHN(root, split="train", download=True, transform=transform_train)
    test_dataset = datasets.SVHN(root, split="test", download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.load:
        softmax_out_avg = np.load('data/SVHN/label_noisy/softmax_out_avg.npy')
        print('softmax_out_avg loaded, shape: ', softmax_out_avg.shape)

    else:
        # Building model
        model = resnet34(pretrained=False)
        num_classes = 10  # CIFAR-10 has 10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training
        softmax_out_avg = np.zeros([len(train_dataset), 10])
        softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        for epoch in range(1, args.epochs + 1):
            train(args.epochs, model, device, train_loader, optimizer, nn.CrossEntropyLoss(), epoch)
            test(args, model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= args.epochs
        np.save('data/SVHN/label_noisy/softmax_out_avg.npy', softmax_out_avg)

    if args.gen:
        print('Generating noisy labels according to softmax_out_avg...')
        label = np.array(train_dataset.labels)
        label_noisy_cand, label_noisy_prob = [], []
        for i in range(len(label)):
            pred = softmax_out_avg[i,:].copy()
            pred[label[i]] = -1
            label_noisy_cand.append(np.argmax(pred))
            label_noisy_prob.append(np.max(pred))
            
        label_noisy = label.copy()
        index = np.argsort(label_noisy_prob)[-int(args.noise_rate*len(label)):]
        label_noisy[index] = np.array(label_noisy_cand)[index]

        save_pth = os.path.join('./data/SVHN/label_noisy', 'dependent'+str(args.noise_rate)+'.csv')
        pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(save_pth, index=False)
        print('Noisy label data saved to ',save_pth)
        
if __name__ == '__main__':
    main()