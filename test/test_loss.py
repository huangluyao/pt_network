import torch
import numpy as np
from networks.seg.models.criterions import CrossEntropyLoss, GeneralizedWassersteinDiceLoss

def test_gwd_loss():
    dist_mat = np.array([
        [0., 1., 1.],
        [1., 0., 0.5],
        [1., 0.5, 0.]
    ])
    wass_loss = GeneralizedWassersteinDiceLoss()

    pred = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.float32).cuda()
    grnd = torch.tensor([0, 1, 2], dtype=torch.int64).cuda()
    print(wass_loss(pred, grnd))

if __name__=="__main__":
    test_gwd_loss()