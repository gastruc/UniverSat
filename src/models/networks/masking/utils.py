from typing import Dict, List, Optional, Sequence

from .mask import Mask, MaskSpatial

ModalityMaskDict = Dict[str, Mask]
ModalityMaskCollection = List[ModalityMaskDict]


def extract_non_empty_spatial_masks(
    modality_masks_per_sample: ModalityMaskCollection,
    modalities: Optional[Sequence[str]] = None,
) -> List[MaskSpatial]:
    """
    Build one spatial mask per sample from the first non-empty modality mask.

    For each sample in ``modality_masks_per_sample``, the first modality with a
    non-empty spatial index tensor is used to build the corresponding
    ``MaskSpatial``.
    """
    if len(modality_masks_per_sample) == 0:
        return []

    if modalities is None:
        modalities = list(modality_masks_per_sample[0].keys())

    spatial_masks = []
    for i, modality_masks in enumerate(modality_masks_per_sample):
        selected_mask = None
        for modality in modalities:
            if modality in modality_masks and modality_masks[modality].S.numel() > 0:
                selected_mask = modality_masks[modality]
                break

        if selected_mask is None:
            raise ValueError(f"All modality masks are empty for modality_masks_per_sample[{i}]. Cannot build spatial mask.")

        spatial_masks.append(MaskSpatial(selected_mask.S, S_length=selected_mask.S_length))

    return spatial_masks
