from omegaconf import ListConfig


def parse_scale(scale_dict: dict):
    """Parse scale dictionary to get all possible scales.

    Args:
        scale_dict (dict): Dictionary containing scale information.

    Returns:
        dict: Dictionary with output scales for each dataset.
    """
    scale = {}
    for scale_type in scale_dict.keys():
        if not scale_type.endswith("_scales"):
            continue
        if (isinstance(scale_dict[scale_type], list) or isinstance(scale_dict[scale_type], ListConfig)) and not isinstance(scale_dict[scale_type][0], str):
            scale[scale_type] = scale_dict[scale_type]
        else:
            assert isinstance(scale_dict[scale_type][0], str), f"Scale value must either be a list or a string. got {type(scale_dict[scale_type])}"
            key = f"{scale_dict[scale_type][0]}_scales"
            assert key in scale_dict, f"Scale key '{key}' not found in scale_dict"
            scale[scale_type] = scale_dict[key]

    return scale
