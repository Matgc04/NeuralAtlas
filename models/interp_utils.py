from torch import nn


def disable_inplace_relu(model: nn.Module) -> nn.Module:
    """Recursively set inplace=False on all ReLU modules in the model.
    This is necessary for layer attribution methods (e.g. LayerIntegratedGradients)
    that require gradient computation through
    activations.
    
    Functionally identical to the original model — only affects memory usage.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model