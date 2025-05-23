import torch
from torch.nn.functional import softmax
from torch import nn
import numpy as np
from warnings import warn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        return

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, return_components=False):
        """ return_components is not relevant here, but is included for consistency with other loss functions"""
        prediction = softmax(prediction, 1)
        tp = prediction * target
        fp = prediction * (1 - target)
        fn = (1 - prediction) * target
        # axes = [0, 2, 3, 4]
        axes = [2, 3, 4]
        tp = tp.sum(dim=axes)
        fp = fp.sum(dim=axes)
        fn = fn.sum(dim=axes)
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + max(self.smooth, 1e-8))
        # dice = dice[1:] # Remove the background class
        dice = dice[:, 1:]
        return 1 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, class_weights_ce=None):
        super(CrossEntropyDiceLoss, self).__init__()
        class_weights_ce = torch.Tensor(class_weights_ce) if class_weights_ce is not None else None
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights_ce)
        self.dice_loss = SoftDiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def get_component_names(self):
        return ['CE', 'Dice Loss']

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, return_components=False):
        ce_loss = self.cross_entropy_loss(prediction, torch.argmax(target, dim=1).long())
        weighted_ce_loss = self.weight_ce * ce_loss if self.weight_ce > 0 else 0
        dice_loss = self.dice_loss(prediction, target)
        weighted_dice_loss = self.weight_dice * dice_loss if self.weight_dice > 0 else 0
        if return_components:
            return weighted_ce_loss + weighted_dice_loss, {'CE': ce_loss, 'Dice Loss': dice_loss}
        else:
            return weighted_ce_loss + weighted_dice_loss


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss, unet, n_levels=3):
        super().__init__()
        n = len(unet.decoder)  # does not take into account the lowest resolution output (bottleneck)

        self.loss = loss

        if not unet.deep_supervision:
            self.weights = [1]
            self.deep_supervision = False
            return

        self.deep_supervision = True

        if n_levels > n:
            raise ValueError(f'Number of levels for deep supervision loss must be less than or equal to the number of '
                             f'decoder blocks. Expected at most {n} levels, got {n_levels}')
        elif n_levels == 1:
            # Warn the user that deep supervision is not being used
            warn('DeepSupervisionLoss is being used, but only one level is being supervised. This is equivalent to '
                 'using a standard loss function, and deep supervision will not have any effect.')
        elif n_levels < 1:
            raise ValueError(f'Number of levels for deep supervision loss should be at least 2. Got {n_levels}')

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(n)])
        # Set the weight to 0 for the lowest resolution outputs if a max number of supervision levels is specified
        if n_levels < n:
            # Only keep the weights for the highest resolution outputs
            weights[n_levels:] = 0

        # normalize weights so that they sum to 1
        weights = weights / weights.sum()
        self.weights = weights[::-1]

    def get_component_names(self):
        parent_names = self.loss.get_component_names()
        return [f'{name} - Level {i+1}' for i in range(len(self.weights)) for name in parent_names]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, return_components=False):
        if not self.deep_supervision:
            if return_components:
                return self.loss(predictions, targets, return_components=return_components)
            else:
                return self.loss(predictions, targets)

        if len(self.weights) != len(predictions):
            raise ValueError(f'Number of weights for deep supervision loss must be the same as the number of deep '
                             f'supervision outputs. Expected {len(predictions)} weights, got {len(self.weights)}')

        # Get the loss each level
        total_losses = [self.loss(pred, target, return_components=return_components)
                        for pred, target in zip(predictions, targets)]
        if return_components:
            # For each list entry, we'll have a 2-tuple of the total loss and the components - unpack these
            total_losses, loss_components = zip(*total_losses)
        # Calculate a weighted sum of the losses
        total_loss = sum([w * l for w, l in zip(self.weights, total_losses)])

        if return_components:
            # Parse and rename the components at all levels
            loss_components = {f'{k} - Level {i+1}': v for i, comp in enumerate(loss_components)
                               for k, v in comp.items()}
            return total_loss, loss_components
        else:
            return total_loss


