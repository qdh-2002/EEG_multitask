import torch.nn.functional as F

def multitask_loss(out_epi, y_epi, out_emot, y_emot, out_sleep, y_sleep, w1=1.0, w2=1.0, w3=1.0):
    loss1 = F.cross_entropy(out_epi, y_epi)
    loss2 = F.cross_entropy(out_emot, y_emot)
    loss3 = F.cross_entropy(out_sleep, y_sleep)
    return w1 * loss1 + w2 * loss2 + w3 * loss3

