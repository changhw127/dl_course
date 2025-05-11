#  -*- utf-8 -*-
from torchvision.datasets import VOCDetection


def prepare_data():
    dataset = VOCDetection(root='../data', year='2007', image_set='train', download=False)
    return



if __name__ == '__main__':
    # prepare_data()

    ls = [106.7671,
105.4415,
104.8311,
103.9622,
103.1421,
102.7806,
102.2485,
101.7940,
101.6762,
100.8262,
100.7501,
101.6948,
101.0217,
100.8358,
100.5765,
100.8072,
100.3860,
100.4830,
100.6068,
100.4909,
100.2471,
100.3487,
100.5224,
100.7111,
101.0352,
100.5069,
100.6232,
100.5040,]
    from utlis.globals import plt_loss_result
    plt_loss_result(ls, path='results', name='yolo_v1_SGD_1e-06_16_o1_half', logger=None)
    plt_loss_result(ls, path='results', name='fcn_Adam_0.0001_8_o1', logger=None)
    pass

