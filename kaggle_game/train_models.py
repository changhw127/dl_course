# -*- utf-8 -*-
import torch
import datetime
from deal_data import SmallPicData
from models.lenet import MyLeNet
from models.vgg import MyVGG
from utlis.globals import Logger, get_device, plt_result, EarlyStopping, save_model, load_model
"""
这里用来进行模型的训练和本身切分测试集合看看
模型保存，仅保存参数，根据模型是小图模型还是大图模型，在predict_result下进行数据组装
训练模型时，设定5组随机数种子，用来评测模型的稳定性
"""

def get_model(name):
    dic = {
        'lenet': MyLeNet,
        'vgg': MyVGG,
        # 'alexnet': MyAlexNet,
        # 'vgg': MyVGG,
        # 'nin': MyNIN,
        # 'googlenet': MyGoogLeNet,
        # 'resnet': MyResNet,
    }
    return dic.get(name)


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


def run():
    # 所有的参数
    model_name = 'vgg'
    batch_size = 64
    opt = 'Adam' # Adam, SGD
    lr = 0.001
    epochs = 50
    random_seed = 8
    first_clear_log = True
    use_early = True
    early_stop = False
    # data_path='../data/a-i-2025-01/yanzhengma_train/sub_train'
    # data_path='../data/a-i-2025-01/yanzhengma_train/sub_train_middle'
    data_path = '../data/a-i-2025-01/yanzhengma_train/train'

    name_ = f'{model_name}_{opt}_{lr}_{batch_size}'
    loger = Logger(path='results', filename=name_)
    train_loss_ls, train_acc_ls = [], []
    test_loss_ls, test_acc_ls = [], []

    st = datetime.datetime.now()
    loger.log(f'start ==========================\nepoch {epochs}', clear=first_clear_log)
    device = get_device(use_gpu=True, loger=loger)
    model = get_model(name=model_name)(lr=lr, opt=opt).to(device)
    loger.log(f'model:{model_name}')
    loger.log(f'lr:{model.lr}')
    loger.log(f'opt:{model.opt}')
    loger.log(model.get_desc())

    my_data = SmallPicData(
        data_path,
        batch_size=batch_size, test_ratio=0.2, random_state=random_seed, logger=loger
    )
    train_data, test_data = my_data.train_dataset, my_data.test_dataset
    train_loader, test_loader = my_data.train_loader, my_data.test_loader
    loger.log(f'trian sample size:{train_data.dataset[0][0].shape}')

    # 定义一个早停策略
    early_stopping = EarlyStopping(patience=5, verbose=True, path=f'results/{name_}_best_model.pth')
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, device, train_loader, epoch, logger=loger)
        test_loss, test_acc = test_model(model, device, test_loader, logger=loger)

        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)

        if use_early:
            early_stopping(test_loss, model)

            if early_stopping.early_stop:
                loger.log("早停触发...")
                if early_stop:
                    # 直接退出
                    loger.log("早停退出")
                    break
                else:
                    # 关闭早停，继续run
                    use_early = False


    ret_data = [train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls]
    plt_result(ret_data, path='results', name=name_, logger=loger)

    save_model(model, name=name_, path='results')

    ed = datetime.datetime.now()
    loger.log(f'cost {(ed - st).total_seconds()} s')
    loger.log(f'最终结果: 测试损失: {test_loss_ls[-1]:.4f}, 准确率: {test_acc_ls[-1]:.4f}')
    loger.log('finished.')
    return True


if __name__ == '__main__':
    run()