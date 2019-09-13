import torch
import torch.nn.functional as F

def get_miou(xmin, ymin, xbias, ybias, xmin_true, ymin_true, xbias_true, ybias_true):
    # calculates miou
    xA, _ = torch.max(torch.stack([xmin, xmin_true]), dim=0)
    yA, _ = torch.max(torch.stack([ymin, ymin_true]), dim=0)
    xB, _ = torch.min(torch.stack([xmin + xbias, xmin_true + xbias_true]), dim=0)
    yB, _ = torch.min(torch.stack([ymin + ybias, ymin_true + ybias_true]), dim=0)
    
    interArea = F.relu(xB - xA) * F.relu(yB - yA)
    predArea = xbias * ybias
    gtArea = xbias_true * ybias_true
    
    loss = -(interArea / (predArea + gtArea - interArea)).mean() # try log
    return loss

def calculate_miou(preds, targets, l2=0.):
    loss = get_miou(
        preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3],
        targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
    )

    return loss + (0. if l2 == 0. else l2 * ((preds - targets)**2).sum(dim=1).mean(dim=0) / targets.var(dim=0).sum())

def area(box):
    return box[2] * box[3]

def get_iou(boxes, values=True):
    assert(len(boxes) == 8)
    boxA = boxes[:4].values if values else boxes[:4]
    boxB = boxes[4:].values if values else boxes[4:]
    
    boxAArea = area(boxA)
    boxBArea = area(boxB)
    
    if (boxAArea == 0 or boxBArea == 0):
        return 0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

GT_coords = ['Xmin_true', 'Ymin_true', 'Xbias_true', 'Ybias_true']
coords = ['Xmin', 'Ymin', 'Xbias', 'Ybias']

get_df_miou = lambda train, targets: train.merge(targets, on='item')[coords + GT_coords].apply(get_iou, axis=1).mean()

def calculate_minmax_loss(df, targets):
    df = df.copy()
    df['Xmax'] = df['Xmin'] + df['Xbias']
    df['Ymax'] = df['Ymin'] + df['Ybias']

    tmp = df.groupby('item').agg({
        'Xmin': 'min',
        'Ymin': 'min',
        'Xmax': 'max',
        'Ymax': 'max'
    })

    tmp['Xbias'] = tmp['Xmax'] - tmp['Xmin']
    tmp['Ybias'] = tmp['Ymax'] - tmp['Ymin']
    tmp.drop(['Xmax', 'Ymax'], axis=1, inplace=True)
    return get_df_miou(tmp, targets)
