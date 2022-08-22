

def get_negative_loss(loss_class):
    class NegativeLoss(loss_class):
        def forward(self, *args, **kwargs):
            return -super().forward(*args, **kwargs)
    return NegativeLoss