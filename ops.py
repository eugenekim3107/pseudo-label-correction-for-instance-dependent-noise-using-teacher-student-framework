import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn

from loss import nll_loss_soft, forward_loss
from metrics import SiameseAccuracy, RegularAccuracy

""" Training/testing models """
# normal training
def train(total_epochs, model, device, loader, optimizer, loss_fn, epoch):
    model.train()
    loop = tqdm(loader, leave=True)
    loss_list = []
    acc_list = []
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        acc = RegularAccuracy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        
        loop.set_postfix(loss=loss.item())
    train_loss = sum(loss_list) / len(loss_list)
    train_acc = 100.*(sum(acc_list) / len(acc_list))
    print('Epoch: {}/{}\nStudent Training Loss: {:.4f}, Student Training Accuracy: {:.2f}%'.format(
        epoch, total_epochs, train_loss, train_acc))
    return train_loss, train_acc

# Siamese training
def train_siamese(total_epochs, model, device, loader, optimizer, loss_fn, epoch):
    model.train()
    loop = tqdm(loader, leave=True)
    loss_list = []
    acc_list = []
    for anc, pos, neg in loop:
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        anc_out, pos_out, neg_out = model(anc, pos, neg)
        loss = loss_fn(anc_out, pos_out, neg_out)
        loss_list.append(loss.item())
        acc = SiameseAccuracy(anc_out, pos_out, neg_out)
        acc_list.append(acc.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
    train_loss = sum(loss_list) / len(loss_list)
    train_acc = 100.*(sum(acc_list) / len(acc_list))
    print('Epoch: {}/{}\nTeacher Training Loss: {:.4f}, Teacher Training Accuracy: {:.2f}%'.format(
        epoch, total_epochs, train_loss, train_acc))

# SEAL training
def train_soft(args, model, device, loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for data, target_soft, target in loader:
        data, target_soft, target = data.to(device), target_soft.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss_soft(output, target_soft)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# dac
def train_dac(args, model, device, loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target, epoch)
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # If the maximum is the last entry, then it means abstained
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(loader.dataset)
    print('Epoch: {}/{}\nTraining loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs, train_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

# Co-teaching
def train_ct(args, model1, model2, device, loader, optimizer1, optimizer2, epoch, p_keep):
    model1.train(), model2.train()
    train_loss1, train_loss2 = 0, 0
    correct1, correct2 = 0, 0
    for data, target in loader:
        n_keep = round(p_keep*data.size(0))
        data, target = data.to(device), target.to(device)
        output1, output2 = model1(data), model2(data)
        loss1, loss2 = F.nll_loss(output1, target, reduction='none'), F.nll_loss(output2, target, reduction='none')

        # selecting #n_keep small loss instances
        _, index1 = torch.sort(loss1.detach())
        _, index2 = torch.sort(loss2.detach())
        index1, index2 = index1[:n_keep], index2[:n_keep]

        # taking a optimization step
        optimizer1.zero_grad()
        loss1[index2].mean().backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2[index1].mean().backward()
        optimizer2.step()

        train_loss1, train_loss2 = train_loss1+loss1.sum().item(), train_loss2+loss2.sum().item()
        pred1, pred2 = output1.argmax(dim=1, keepdim=True), output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct1, correct2 = correct1+pred1.eq(target.view_as(pred1)).sum().item(), correct2+pred2.eq(target.view_as(pred2)).sum().item()
    train_loss1, train_loss2 = train_loss1/len(loader.dataset), train_loss2/len(loader.dataset)
    print('Epoch: {}/{}\nModel1 Training. Training loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)\nModel2 Training. Training loss: {:.4f}, Training accuracy: {}/{} ({:.2f}%)'.format(
        epoch, args.epochs,
        train_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset),
        train_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset)))

# normal testing
def test(args, model, device, loader, top5=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()

    test_loss /= len(loader.dataset)
    if top5:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset), correct_k, len(loader.dataset), 100. * correct_k / len(loader.dataset)))
    else:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        return test_loss, (100.* correct / len(loader.dataset))

# with a validation set
def val_test(args, model, device, val_loader, test_loader, best_val_acc, save_path):
    model.eval()

    # val
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss, val_acc = loss/len(val_loader.dataset), 100.*correct/len(val_loader.dataset)

    # test
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss, test_acc = loss/len(test_loader.dataset), 100.*correct/len(test_loader.dataset)

    if val_acc>best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)

    print('Val loss: {:.4f}, Testing loss: {:.4f}, Val accuracy: {:.2f}%, Best Val accuracy: {:.2f}%, Testing accuracy: {:.2f}%\n'.format(
        val_loss, test_loss, val_acc, best_val_acc, test_acc))

    return best_val_acc


# dac
def test_dac(args, model, device, loader, epoch, criterion, top5=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, epoch).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # If the maximum is the last entry, then it means abstained
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()
    test_loss /= len(loader.dataset)
    if top5:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset), correct_k, len(loader.dataset), 100. * correct_k / len(loader.dataset)))
    else:
        print('Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

def test_ct(args, model1, model2, device, loader, top5=False):
    model1.eval(), model2.eval()
    test_loss1, test_loss2 = 0, 0
    correct1, correct2 = 0, 0
    correct1_k, correct2_k = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model1(data), model2(data)
            test_loss1, test_loss2 = test_loss1+F.nll_loss(output1, target, reduction='sum').item(), test_loss2+F.nll_loss(output2, target, reduction='sum').item() # sum up batch loss
            pred1, pred2 = output1.argmax(dim=1, keepdim=True), output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct1, correct2 = correct1+pred1.eq(target.view_as(pred1)).sum().item(), correct2+pred2.eq(target.view_as(pred2)).sum().item()
            if top5:
                _, pred1 = output1.topk(5, 1, True, True)
                correct1_k += pred1.eq(target.view(-1,1)).sum().item()
                _, pred2 = output2.topk(5, 1, True, True)
                correct2_k += pred2.eq(target.view(-1,1)).sum().item()

    test_loss1, test_loss2 = test_loss1/len(loader.dataset), test_loss2/len(loader.dataset)
    if top5:
        print('Model1 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\nModel2 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%), Top5 Testing accuracy: {}/{} [{:.2f}%]\n'.format(
            test_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset), correct1_k, len(loader.dataset), 100. * correct1_k / len(loader.dataset),
            test_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset), correct2_k, len(loader.dataset), 100. * correct2_k / len(loader.dataset)))
    else:
        print('Model1 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\nModel2 Testing. Testing loss: {:.4f}, Testing accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss1, correct1, len(loader.dataset), 100. * correct1 / len(loader.dataset),
            test_loss2, correct2, len(loader.dataset), 100. * correct2 / len(loader.dataset)))

def generate_student_labels(args, model, device, train_loader):
    model.eval()
    with torch.no_grad():
        student_pred = []
        given_pred = []
        images = []
        for (x,y) in train_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            out = torch.argmax(out, axis=1).to(torch.int64)
            student_pred.append(out)
            given_pred.append(y)
            images.append(x)
    student_pred = torch.hstack(student_pred)
    given_pred = torch.hstack(given_pred)
    images = torch.vstack(images)
    print(f"Proportion of same labels: {torch.sum(student_pred == given_pred) / student_pred.shape[0]}")
    return student_pred, given_pred, images

