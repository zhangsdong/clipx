from clipx.models.base import BaseModel
from clipx.models.cascadepsp import CascadePSPModel


# Will need to import U2NetModel once implemented
# from clipx.models.u2net import U2NetModel


def get_model(model_name):
    """
    Factory method to get model by name

    Args:
        model_name: Name of the model (u2net, cascadepsp, auto)

    Returns:
        Instance of BaseModel subclass
    """
    model_name = model_name.lower()

    if model_name == 'cascadepsp':
        return CascadePSPModel()
    # elif model_name == 'u2net':
    #     return U2NetModel()
    # elif model_name == 'auto':
    #     # Auto model implementation will be added later
    #     pass
    else:
        raise ValueError(f"Unknown model: {model_name}, supported models: cascadepsp, u2net, auto")