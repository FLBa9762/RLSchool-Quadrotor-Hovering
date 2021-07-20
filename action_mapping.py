import numpy as np

# class Source():
#     def __init__(self):
#         pass
# def my_action_mapping(model_output_act, low_bound, high_bound):
#         """ mapping action space [-1, 1] of model output
#             to new action space [low_bound, high_bound].
#
#         Args:
#             model_output_act: np.array, which value is in [-1, 1]
#             low_bound: float, low bound of env action space
#             high_bound: float, high bound of env action space
#
#         Returns:
#             action: np.array, which value is in [low_bound, high_bound]
#         """
#         assert np.all(((model_output_act<=1.0 + 1e-3), (model_output_act>=-1.0 - 1e-3))), \
#             'the action should be in range [-1.0, 1.0]'
#         assert high_bound > low_bound
#         action = low_bound + (model_output_act - (-1.0)) * (
#             (high_bound - low_bound) / 2.0)
#         action = np.clip(action, low_bound, high_bound)
#         return action

def my_action_mapping(model_output_act, low_bound, high_bound):
    """ mapping action space [-1, 1] of model output
        to new action space [low_bound, high_bound].

    Args:
        model_output_act: np.array, which value is in [-1, 1]
        low_bound: float, low bound of env action space
        high_bound: float, high bound of env action space

    Returns:
        action: np.array, which value is in [low_bound, high_bound]
    """
    assert np.all(((model_output_act<=1.0), (model_output_act>=-1.0))), \
        'the action should be in range [-1.0, 1.0]'
    assert high_bound > low_bound
    action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    action = np.clip(action, low_bound, high_bound)
    return action
