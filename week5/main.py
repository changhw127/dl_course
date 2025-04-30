# -*- utf-8 -*-
import torch
import datetime
from deal_data import MyData
from utils import Logger, get_device, plt_result
from models.lenet import MyLeNet
from models.alexnet import MyAlexNet
from models.vgg import MyVGG
from models.nin import MyNIN
from models.googlenet import MyGoogLeNet
from models.resnet import MyResNet


def train_model(model, device, data_loader, epoch, logger=None):
    model.train()
    loss = 0
    correct = 0

    ix = 0
    total = len(data_loader)
    for batch, (X, Y) in enumerate(data_loader):
        ix += 1
        X, Y = X.to(device), Y.to(device)
        # 4步
        y = model.forward(X)
        model.calc_loss(y, Y, save_loss=True)
        model.backward()
        model.update_params()

        loss += model.loss.item()
        pred = y.argmax(dim=1, keepdim=True)
        correct += pred.eq(Y.view_as(pred)).sum().item()
        if logger:
            logger.progress_bar(ix, total)

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    msg = f"Epoch {epoch + 1}, Train Loss: {loss:.4f}, Accuracy: {acc:.4f} %"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    return loss, acc


def test_model(model, device, data_loader, logger=None):
    model.eval()
    loss = 0
    correct = 0
    # 关闭梯度计算，用于加速
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            y = model.forward(X)
            loss += model.calc_loss(y, Y, save_loss=False)
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(Y.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    msg = f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f} %"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    return loss, acc


def load_model(name, path='results'):
    model = torch.load('{}/{}.pth'.format(path, name))
    model.eval()
    return model


def save_model(model, name, path='results'):
    torch.save(model, '{}/{}.pth'.format(path, name))
    return


def retry_model(name):
    device = get_device(use_gpu=True, loger=None)
    model = load_model(name=name).to(device)
    my_data = MyData('../data/course/faces', batch_size=batch_size, test_ratio=0.2, random_state=2, is_subset=False, subset_percent=0.1)
    train_data, test_data, train_loader, test_loader = my_data.train_dataset, my_data.test_dataset, my_data.train_loader, my_data.test_loader
    test_model(model, device, test_loader)
    return


def get_model(name):
    dic = {
        'lenet': MyLeNet,
        'alexnet': MyAlexNet,
        'vgg': MyVGG,
        'nin': MyNIN,
        'googlenet': MyGoogLeNet,
        'resnet': MyResNet,
    }
    return dic.get(name)

def run(model_name, batch_size, opt, lr):
    # model_name = 'resnet'
    name_ = f'{model_name}_{opt}_{lr}_{batch_size}'
    loger = Logger(path='results', filename=name_)
    train_loss_ls, train_acc_ls = [], []
    test_loss_ls, test_acc_ls = [], []
    epochs = 30
    st = datetime.datetime.now()
    loger.log(f'start ==========================\nepoch {epochs}', clear=False)
    device = get_device(use_gpu=True, loger=loger)
    model = get_model(name=model_name)(lr=lr, opt=opt).to(device)
    loger.log(f'model:{model_name}')
    loger.log(f'lr:{model.lr}')
    loger.log(f'opt:{model.opt}')
    loger.log(model.get_desc())

    my_data = MyData('../data/course/faces', batch_size=batch_size, resize=model.raw_size,
                     test_ratio=0.2, random_state=6,
                     is_subset=False, subset_percent=0.1, logger=loger)
    train_data, test_data, train_loader, test_loader = my_data.train_dataset, my_data.test_dataset, my_data.train_loader, my_data.test_loader
    loger.log(f'sample size:{train_data.dataset[0][0].shape}')
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, device, train_loader, epoch, logger=loger)
        test_loss, test_acc = test_model(model, device, test_loader, logger=loger)

        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)

    ret_data = [train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls]
    plt_result(ret_data, path='results', name=name_, logger=loger)

    save_model(model, name=name_, path='results')

    ed = datetime.datetime.now()
    loger.log(f'cost {(ed - st).total_seconds()} s')
    loger.log(f'最终结果: 测试损失: {test_loss_ls[-1]:.4f}, 准确率: {test_acc_ls[-1]:.4f}')
    loger.log('finished.')
    return True

if __name__ == '__main__':
    batches = [64, 32]
    model_names = ['lenet', 'alexnet', 'vgg', 'nin', 'googlenet', 'resnet']
    opts = ['SGD', 'Adam']
    lr_ls = [0.01, 0.001]
    for batch_size in batches:
        for lr in lr_ls:
            for opt in opts:
                for model_name in model_names:
                    run(model_name, batch_size, opt, lr)

    # model_name, batch_size, opt, lr = 'nin', 64, 'SGD', 0.01,
    # run(model_name, batch_size, opt, lr)

