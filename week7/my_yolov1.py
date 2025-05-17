#  -*- utf-8 -*-
from PIL import Image
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utlis.globals import Logger, plt_loss_result, get_device, save_model, load_model


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, lr=0.000001, opt='SGD'):
        """
        YOLOv1 网络实现

        参数:
            S: 网格划分数量 (S x S)
            B: 每个网格预测的边界框数量
            C: 类别数量 (PASCAL VOC有20类)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        # 网络架构
        self.conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 输入3通道,输出64通道
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层4 (重复4次)
            *self._make_conv_block(512, 256, 512, 4),

            # 卷积层5
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层6 (重复2次)
            *self._make_conv_block(1024, 512, 1024, 2),

            # 卷积层7
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # 卷积层8
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # self.fc_layers = nn.Sequential(
        #     # 全连接层
        #     nn.Flatten(),
        #     nn.Linear(1024 * S * S, 4096),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, S * S * (C + B * 5))  # 输出维度: S*S*(C + B*5)
        # )

        self.final_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, B * 5 + C, kernel_size=1)  # 输出通道数=B*5+C
        )
        # 初始化权重
        # self._initialize_weights()
        self.init_criterion()
        self.init_optimizer()

    def _make_conv_block(self, in_channels, mid_channels, out_channels, repeat):
        """创建重复的卷积块"""
        layers = []
        for _ in range(repeat):
            layers += [
                nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                # nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(0.1),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ]
            in_channels = out_channels
        return layers

    def _initialize_weights_old(self):
        """
        这里问了下大模型，然后有说用参数初始化的时候会比较好，比如使用Kaiming初始化LeakyReLU，正态分布全链接的初始化。
        但是发现初始化后有时候运行到一般就会出现空张量的问题，导致模型预测异常。所以后来取消掉了。
        不用的化，损失有时候一开始会很多，用了的话，基本都是100开始的。
        :return:
        """

        for m in self.modules():  # 遍历所有网络层
            if isinstance(m, nn.Conv2d):
                # 卷积层：使用Kaiming初始化（针对LeakyReLU优化）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):
                # 全连接层：正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0

    def _initialize_weights(self):
        # 确保 卷积层、全连接层 的权重没有被初始化为 0 或极端值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_criterion(self):
        self.criterion = YOLOv1Loss(S=self.S, B=self.B, C=self.C, lambda_coord=5, lambda_noobj=0.5)
        return True

    def init_optimizer(self):
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=5e-4)
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        self.optimizer.zero_grad()
        return True

    def get_desc(self):
        msg = '''
        全连接层计算量过大，会产生约2亿参数, 使用全卷积
        在卷积层后直接使用Flatten+FC会破坏空间相关性
        部分使用BN，
        取消权重初始化，因为可能导致空张量问题
        使用L2正则化
        '''.strip().replace(' ', '')
        return msg

    def forward(self, x):
        x = self.conv_layers(x)
        # x = self.fc_layers(x)
        x = self.final_conv(x)
        # x = x.permute(0, 2, 3, 1)  # 调整为[batch, S, S, B*5+C]
        # 将输出reshape为 (batch_size, S, S, C + B*5)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x

    def calc_loss(self, outputs, targets, save_loss=True):
        loss = self.criterion(outputs, targets)
        if save_loss:
            self.loss = loss
        return loss

    def backward(self):
        self.loss.backward()
        return True

    def update_params(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        return True


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        S: 网格划分数量 (S x S)
        B: 每个网格预测的边界框数量
        C: 类别数量
        lambda_coord: 坐标损失的权重
        lambda_noobj: 无目标置信度损失的权重
        """
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # 预测的形状： (batch_size, S*S*(B*5 + C))
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        targets = targets.view(-1, self.S, self.S, self.B * 5 + self.C)

        # 分离预测各个部分
        pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B*5:]
        target_boxes = targets[..., :5].view(-1, self.S, self.S, 5)
        target_boxes = target_boxes.unsqueeze(3)
        target_boxes = target_boxes.expand(-1, -1, -1, 2, -1) # 最终形状为 batch, 7,7,2,5
        target_classes = targets[..., self.B*5:]

        # 创建掩码
        obj_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0

        obj_mask = obj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B)
        noobj_mask = noobj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B)

        # 计算IOU并找到负责预测的边界框
        ious = YOLOv1Tools.compute_iou(pred_boxes[..., :4], target_boxes[..., :4])
        _, best_box = ious.max(dim=-1, keepdim=True)
        best_box = best_box.long() # 确保数据类型

        resp_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
        resp_mask.scatter_(-1, best_box, True)

        coord_loss = self.lambda_coord * self.coordinate_loss(pred_boxes, target_boxes, resp_mask)
        conf_loss = self.confidence_loss(pred_boxes[..., 4], target_boxes[..., 4], obj_mask, noobj_mask, resp_mask)
        class_loss = self.class_loss(pred_classes, target_classes, obj_mask)
        total_loss = coord_loss + conf_loss + class_loss
        return total_loss

    def coordinate_loss(self, pred_boxes, target_boxes, resp_mask):
        # 只计算预测物体的边界框的坐标损失
        pred_xy = pred_boxes[..., :2][resp_mask]
        pred_wh = pred_boxes[..., 2:4][resp_mask].clamp(min=0)
        # pred_wh = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)

        target_xy = target_boxes[..., :2][resp_mask]
        target_wh = target_boxes[..., 2:4][resp_mask]

        # 转换到平方根空间（更稳定的梯度）
        pred_wh = torch.sqrt(pred_wh + 1e-6)
        target_wh = torch.sqrt(target_wh + 1e-6)  # 目标也添加保护

        # 计算损失
        xy_loss = nn.functional.mse_loss(pred_xy, target_xy, reduction='sum')
        wh_loss = nn.functional.mse_loss(pred_wh, target_wh, reduction='sum')
        coord_loss = xy_loss + wh_loss
        return coord_loss

    def confidence_loss(self, pred_conf, target_conf, obj_mask, noobj_mask, resp_mask):
        # obj_mask?
        conf_loss_obj = nn.functional.mse_loss(pred_conf[resp_mask], target_conf[resp_mask], reduction='sum')

        conf_loss_noobj = nn.functional.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask], reduction='sum')
        conf_loss_noobj = self.lambda_noobj * conf_loss_noobj

        conf_loss = conf_loss_obj + conf_loss_noobj
        return conf_loss

    def class_loss(self, pred_classes, target_classes, obj_mask):
        obj_mask = obj_mask.any(dim=-1)
        pred_classes = pred_classes[obj_mask]
        target_classes = target_classes[obj_mask]
        class_loss = nn.functional.mse_loss(pred_classes, target_classes, reduction='sum')
        return class_loss


class YOLOv1Tools(object):
    @classmethod
    def _cxcywh_to_xyxy(cls, boxes):
        """
        将cxcywh格式转换为xyxy格式
        输入: Tensor(n,4) -> (x_center, y_center, width, height)
        输出: Tensor(n,4) -> (x1, y1, x2, y2)
        """
        x_center, y_center = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    @classmethod
    def compute_iou(cls, box1, box2):
        """
        计算交并比
        :param box1: Tensor(1, 4) -> (x_center, y_center, width, height)
        :param box2: Tensor(n, 4) -> (x_center, y_center, width, height)
        :return:
        """
        # 确保输入为torch.Tensor
        if not isinstance(box1, torch.Tensor):
            box1 = torch.tensor(box1)
        if not isinstance(box2, torch.Tensor):
            box2 = torch.tensor(box2)
        # 转换cxcywh到xyxy格式
        box1_xyxy = cls._cxcywh_to_xyxy(box1)
        box2_xyxy = cls._cxcywh_to_xyxy(box2)

        # 计算交集区域坐标 逐元素比较
        lt = torch.max(box1_xyxy[:, :2], box2_xyxy[:, :2])  # 左上角 [n,2]
        rb = torch.min(box1_xyxy[:, 2:], box2_xyxy[:, 2:])  # 右下角 [n,2]

        # 计算交集面积 [n]
        # rb - lt = [x_max - x_min, y_max - y_min]  # 得到宽度和高度
        # clamp将宽度和高度限制在≥0的范围内， prod 最终的宽高相乘 [width, height].prod() = width * height
        inter = (rb - lt).clamp(min=0).prod(dim=1)  # 宽*高

        # 计算各自面积
        area1 = box1_xyxy[:, 2] * box1_xyxy[:, 3]  # box1面积 [1]
        area2 = box2_xyxy[:, 2] * box2_xyxy[:, 3]  # box2面积 [n]

        # 计算并集面积 [n]
        union = area1 + area2 - inter

        # 计算IoU [n]
        iou = inter / (union + 1e-6)  # 加小常数避免除零
        return iou

    @classmethod
    def postprocess(cls, output, S=7, B=2, C=20, conf_threshold=0.2, num_threshold=0.5):
        batch_size = output.size(0)

        # 对边界框的预测部分进行重塑 (batch_size, S, S, B, 5)
        bbox_output = output[..., :B*5].view(batch_size, S, S, B, 5)
        # 类别预测结果保持不变 (batch_size, S, S, B, C) 后面的20个
        class_output = output[..., B*5:]

        #将bbox_output切成相应的部分
        box_xy = bbox_output[..., :2] # (batch_size, S, S, B, 2), 中心坐标xy
        box_wh = bbox_output[..., 2:4] # (batch_size, S, S, B, 2), 框的宽高wh
        confidence = bbox_output[..., 4] # (batch_size, S, S, B), 置信度

        # 计算类别特定的置信度分数 (得分=类别概率 * 置信度) (batch_size, S, S, B, C)
        # (batch_size, S, S, 1, C) * (batch_size, S, S, B, 1) 用到广播  torch.Size([1, 7, 7, 2, 20])
        class_scores = class_output.unsqueeze(3) * confidence.unsqueeze(-1)

        # 重新调整类别的得分和边界框的形状
        class_scores = class_scores.view(batch_size, -1, C) # (batch_size, S*S*B, C) [1, 98, 20]
        box_conf = confidence.reshape(batch_size, -1)  # (batch_size, S*S*B) [1, 98]
        boxes = torch.cat([box_xy, box_wh], dim=-1).view(batch_size, -1, 4) # # (batch_size, S*S*B, 4) [1, 98, 4]

        # 应用置信度阈值，过滤
        mask = box_conf > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_class_scores = class_scores[mask]

        final_boxes = []
        final_scores = []
        final_labels = []

        # 开始进行便利，每个类别执行NMS
        for i in range(C):
            class_mask = filtered_class_scores[:, i] > conf_threshold
            if class_mask.sum() == 0:
                # 空了
                continue
            # 取出当前分类的过滤后的分数已经对应的box
            scores = filtered_class_scores[class_mask, i]
            selected_boxes = filtered_boxes[class_mask]
            # 排序
            sorted_indices = torch.argsort(scores, descending=True)
            selected_boxes = selected_boxes[sorted_indices]
            scores = scores[sorted_indices]

            while len(scores) > 0:
                best_box = selected_boxes[0].unsqueeze(0)
                best_score = scores[0].unsqueeze(0)
                if len(scores) == 1:
                    # 只有一个，保留
                    final_boxes.append(best_box)
                    final_scores.append(best_score)
                    final_labels.append(i)
                    break

                # 计算IOU， 并移除与当前最佳框重叠度过大的框
                ious = cls.compute_iou(best_box, selected_boxes[1:])
                keep_indices = ious < num_threshold

                selected_boxes = selected_boxes[1:][keep_indices]
                scores = scores[1:][keep_indices]

                final_boxes.append(best_box)
                final_scores.append(best_score)
                final_labels.append(i)

        if len(final_boxes) > 0:
            return torch.cat(final_boxes), torch.cat(final_scores), final_labels
        else:
            return torch.empty(0), torch.empty(0), []


class VOCDataset(Dataset):
    def __init__(self, root='../data', year='2007',  image_set='train', S=7, B=2, C=20, download=False):
        self.S = S
        self.B = B
        self.C = C
        self.transform = self._transform()
        self.dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)

        self.VOC_CLASSES = [
            'aeroplane',  # 飞机
            'bicycle',    # 自行车
            'bird',       # 鸟
            'boat',       # 船
            'bottle',     # 瓶子
            'bus',        # 公交车
            'car',        # 汽车
            'cat',        # 猫
            'chair',      # 椅子
            'cow',        # 牛
            'diningtable',# 餐桌
            'dog',        # 狗
            'horse',      # 马
            'motorbike',  # 摩托车
            'person',     # 人
            'pottedplant',# 盆栽植物
            'sheep',      # 羊
            'sofa',       # 沙发
            'train',      # 火车
            'tvmonitor'   # 电视/显示器
        ]

    def _transform(self):
        return transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        target = self.parse_voc_annotation(target)
        if self.transform:
            image = self.transform(image)
        return image, target

    def parse_voc_annotation(self, target):
        width = int(target['annotation']['size']['width'])
        height = int(target['annotation']['size']['height'])

        yolo_target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for obj in target['annotation']['object']:
            class_name = obj['name']
            if class_name not in self.VOC_CLASSES:
                continue
            class_idx = self.VOC_CLASSES.index(class_name)

            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin']) / width
            ymin = float(bndbox['ymin']) / height
            xmax = float(bndbox['xmax']) / width
            ymax = float(bndbox['ymax']) / height

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)

            x_offset = x_center * self.S - grid_x
            y_offset = y_center * self.S - grid_y

            yolo_target[grid_y, grid_x, :5] = torch.tensor([x_offset, y_offset, w, h, 1])
            yolo_target[grid_y, grid_x, 10+ class_idx] = 1

        return yolo_target


def train_model(model, device, data_loader, epoch, logger=None):
    model.train()
    loss = 0

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
        if logger:
            logger.progress_bar(ix, total)

    loss /= len(data_loader)
    msg = f"Epoch {epoch + 1}, Train Loss: {loss:.4f}"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    return loss


def run():
    # 所有的参数
    model_name = 'yolo_v1'
    batch_size = 16
    opt = 'SGD'
    lr = 0.000001
    epochs = 10
    first_clear_log = True

    name_ = f'{model_name}_{opt}_{lr}_{batch_size}_o2'
    loger = Logger(path='results', filename=name_)
    train_loss_ls = []

    st = datetime.datetime.now()
    loger.log(f'start ==========================\nepoch {epochs}', clear=first_clear_log)
    device = get_device(use_gpu=True, loger=loger)
    model = YOLOv1(lr=lr, opt=opt).to(device)
    loger.log(f'model:{model_name}')
    loger.log(f'lr:{model.lr}')
    loger.log(f'opt:{model.opt}')
    loger.log(model.get_desc())

    train_data = VOCDataset()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    loger.log(f'train sample size:{train_data.dataset[0][0]}')

    for epoch in range(epochs):
        train_loss = train_model(model, device, train_loader, epoch, logger=loger)
        train_loss_ls.append(train_loss)

    plt_loss_result(train_loss_ls, path='results', name=name_, logger=loger)

    save_model(model, name=name_, path='results')

    ed = datetime.datetime.now()
    loger.log(f'cost {(ed - st).total_seconds()} s')
    loger.log(f'最终结果: 训练损失: {train_loss_ls[-1]:.4f}')
    loger.log('finished.')
    return True


def test_script():
    if False:
        # test model
        model = YOLOv1(S=7, B=2, C=20)
        input_tensor = torch.Tensor(1, 3, 448, 448)
        output = model(input_tensor)
        print(output.shape)
        # torch.Size([1, 7, 7, 30])

    if False:
        # test 后处理方法
        # 模拟假数据看看
        output = torch.zeros((1, 7, 7, 30))
        output[0, 1, 1, 4] = 0.9
        output[0, 3, 3, 4] = 0.85

        output[0, 1, 1, -20:] = torch.nn.functional.softmax(torch.tensor([2.0] + [0.1] * 19), dim=0)
        output[0, 3, 3, -20:] = torch.nn.functional.softmax(torch.tensor([0.1] * 19 + [2.0]), dim=0)

        output[0, 1, 1, 0:2] = torch.tensor([0.5, 0.5])
        output[0, 1, 1, 2:4] = torch.tensor([0.2, 0.3])

        output[0, 3, 3, 0:2] = torch.tensor([0.6, 0.6])
        output[0, 3, 3, 2:4] = torch.tensor([0.25, 0.4])

        print(output)
        result = YOLOv1Tools.postprocess(output, S=7, B=2, C=20, conf_threshold=0.2, num_threshold=0.5)
        print(result)

    if True:
        model = YOLOv1(S=7, B=2, C=20)
        input_tensor = torch.Tensor(1, 3, 448, 448)
        output = model(input_tensor)
        print(output.shape)


if __name__ == '__main__':
    # 用于测试 忽略
    # test_script()

    run()

    # 结果记录
    """
    yolo_v1_SGD_1e-05_16 OK
    
    yolo_v1_SGD_1e-06_16_o1
    发现不加权重初始化的话，损失初始值很大
    """