# -*- utf-8 -*-


def iou(box1, box2):
    """
    计算两个box的IOU
    输入:
        box1: (a1, a2, c1, c2)
        box2: (b1, b2, d1, d2)
    输出:
        iou值，范围[0,1]
    """
    e1 = max(box1[0], box2[0])
    f2 = max(box1[1], box2[1])
    f1 = min(box1[2], box2[2])
    e2 = min(box1[3], box2[3])

    width = max(0, f1 - e1)
    height = max(0, e2 - f2)
    inter_area = width * height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    iou = (inter_area / union_area) if union_area > 0 else 0
    return iou


def test():
    iou((1, 1, 5, 3), (4, 2, 6, 4))
    # 1 / 11 = 0.09090909090909091