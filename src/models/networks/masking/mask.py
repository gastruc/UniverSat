import math
from typing import List, Union

import torch


class Mask(object):
    """Mask for spatial, temporal and channel
    """
    def __init__(self,
                 S:torch.Tensor=None,
                 T:torch.Tensor=None,
                 C:torch.Tensor=None,
                 S_length:int=None,
                 T_length:int=None,
                 C_length:int=None,
                ):
        self.S = S
        self.T = T
        self.C = C

        assert S is None or S_length is not None, f"Invalid S_length {S_length}. Must be provided if S is not None"
        assert T is None or T_length is not None, f"Invalid T_length {T_length}. Must be provided if T is not None"
        assert C is None or C_length is not None, f"Invalid C_length {C_length}. Must be provided if C is not None"

        assert S is None or S.numel() == 0 or S.max() < S_length, f"Invalid S_length {S_length}. The mask is bigger ({S.max()})"
        assert T is None or T.numel() == 0 or T.max() < T_length, f"Invalid T_length {T_length}. The mask is bigger ({T.max()})"
        assert C is None or C.numel() == 0 or C.max() < C_length, f"Invalid C_length {C_length}. The mask is bigger ({C.max()})"

        self.S_length = S_length if S is not None else 0
        self.T_length = T_length if T is not None else 0
        self.C_length = C_length if C is not None else 0

        self.B = S.shape[0] if S is not None else T.shape[0] if T is not None else C.shape[0] if C is not None else None

    def _unsqueeze_mask(self, mask, mask_type, current_shape):
        """Unsqueeze the mask to match the target shape
        :param mask: mask to unsqueeze
        :param mask_type: type of the mask (S, T, C)
        :param current_shape: string: current shape of the mask
        :return: unsqueezed mask
        """
        assert current_shape[0] == 'B', f"Invalid current shape {current_shape}. Must start with B"
        position = 1 #1 before the mask axis, -1 after
        for ax in current_shape[1:]:
            if ax != mask_type:
                mask = mask.unsqueeze(position)
            elif ax == mask_type:
                position = -1

        assert position == -1, f"mask type {mask_type} not found in current shape {current_shape}"
        return mask


    def apply(self, x:torch.Tensor, axis="STC", current_shape="BTCSD", upsample_S_to=None):
        """Apply the modality mask to the input tensor
        :param x: input tensor to apply the mask on
        :param axis: axis to apply the mask on
        :param current_shape: current shape of the input tensor
        :return: masked input tensor
        """
        for ax in axis:
            assert ax in ['S', 'T', 'C'], f"Invalid axis {ax}. Must be one of ['S', 'T', 'C']"
            assert ax in current_shape, f"Invalid axis {ax}. Must be among the current axis : {current_shape}"
            if ax == 'S' and upsample_S_to is not None and upsample_S_to != self.S_length:
                mask, _ = self._upsample_mask(upsample_S_to)
                assert mask is None or x.shape[current_shape.index(ax)] == upsample_S_to, f"Invalid upsample_S_to {upsample_S_to}. The upsampled mask length does not match the input shape {x.shape[current_shape.index(ax)]}"
            else:
                mask = getattr(self, ax)
            if mask is None:
                continue # No masking


            if 'T' in current_shape and len(x.shape) == len(current_shape)-1:
                # This particular modality doesn't have a T dimension
                if ax == 'T':
                    raise ValueError(f"Invalid axis T. The current data does not support a T dimension.")
                mask = self._unsqueeze_mask(mask, ax, current_shape.replace('T', ''))
            else:
                # Unsqueeze the mask for other cases
                mask = self._unsqueeze_mask(mask, ax, current_shape)
            # expand mask to match x shape (view, no memory allocation unlike repeat)
            expand_shape = list(x.shape)
            expand_shape[current_shape.index(ax)] = mask.shape[current_shape.index(ax)]
            x = torch.gather(x, dim=current_shape.index(ax), index=mask.expand(expand_shape))
        return x

    def _upsample_mask(self, new_S_length):
        """Upsample the S mask to a new scale
        :param new_S_length: new scale to upsample the mask to
        :return: upsampled mask
        """
        assert new_S_length >= self.S_length, f"Invalid new length {new_S_length}. Must be bigger than current length {self.S_length}"
        if self.S is None:
            return None, 0

        old_side = math.isqrt(self.S_length)
        new_side = math.isqrt(new_S_length)
        if old_side * old_side != self.S_length:
            raise ValueError(
                f"_upsample_mask: S_length={self.S_length} is not a perfect square "
                f"(expected patch grid side**2; isqrt gives {old_side} -> {old_side**2}). "
                f"Fix the collator / mask so S_length matches a square HxW patch layout."
            )
        if new_side * new_side != new_S_length:
            raise ValueError(
                f"_upsample_mask: new_S_length={new_S_length} is not a perfect square "
                f"(isqrt -> {new_side}**2={new_side * new_side})."
            )
        if new_side % old_side != 0:
            raise ValueError(
                f"_upsample_mask: fine grid side {new_side} not divisible by coarse side {old_side} "
                f"(S_length={self.S_length}, new_S_length={new_S_length})."
            )
        factor = new_side // old_side
        if factor * factor * self.S_length != new_S_length:
            raise ValueError(
                f"_upsample_mask: scale mismatch: need new_S_length == (new_side/old_side)**2 * S_length, "
                f"got S_length={self.S_length}, new_S_length={new_S_length}, factor={factor}."
            )

        # Deconstruct S into row, col indices on the coarse square grid
        r = self.S // old_side
        c = self.S % old_side

        # Calculate base index for top-left of the fine block
        base_index = (r * factor) * new_side + (c * factor)
        base_index = base_index.unsqueeze(-1).unsqueeze(-1) # B x S x 1 x 1

        # Add offsets
        # row offset (dr * new_side)
        row_offsets = torch.arange(factor, device=self.S.device)[None,None,:,None] * new_side
        # col offset (dc)
        col_offsets = torch.arange(factor, device=self.S.device)[None,None,None,:]

        S_upsampled = base_index + row_offsets + col_offsets # B x S x factor x factor
        S_upsampled = S_upsampled.reshape(self.S.shape[0], -1)
        if S_upsampled.numel() > 0:
            mx = int(S_upsampled.max().item())
            if mx >= new_S_length:
                raise RuntimeError(
                    f"_upsample_mask: internal error — max index {mx} >= new_S_length {new_S_length}. "
                    f"S_length={self.S_length}, old_side={old_side}, new_side={new_side}, factor={factor}."
                )
        return S_upsampled, new_S_length

    def upsample_mask(self, new_S_length):
        """Upsample the S mask to a new scale
        :param new_S_length: new scale to upsample the mask to
        :return: upsampled mask
        """
        S_upsampled, S_length_upsampled = self._upsample_mask(new_S_length)
        return Mask(S=S_upsampled, T=self.T, C=self.C,
                    S_length=S_length_upsampled, T_length=self.T_length, C_length=self.C_length)

    def revert(self, x: torch.Tensor, axis: str = "S", current_shape="BTCSD", target_S_length=-1, fill_value=0.0):
        """Revert the mask for a single axis to the original shape.
        :param x: input tensor to revert the mask on
        :param axis: axis to revert the mask on ('S', 'T', or 'C')
        :param current_shape: current shape of the input tensor
        :param target_S_length: spatial length to revert to (-1 uses the mask's
            own ``S_length``); when larger, the spatial mask is upsampled to match
        :param fill_value: value to fill the masked positions with
        :return: reverted input tensor
        """
        assert axis in current_shape, f"Invalid axis {axis}. Must be among the current axis: {current_shape}"
        idx_ax = current_shape.index(axis)
        shape = list(x.shape)

        if axis in ['T', 'C']:
            target_axis_length = self.T_length if axis == 'T' else self.C_length
            mask = getattr(self, axis)
        else:
            if target_S_length == -1:
                target_axis_length = self.S_length
                mask = getattr(self, axis)
            else:
                assert target_S_length >= self.S_length, f"Invalid target S length {target_S_length}. Must be bigger than current length {self.S_length}"
                mask = self._upsample_mask(target_S_length)[0]
                target_axis_length = target_S_length

        shape[idx_ax] = target_axis_length

        if mask is None or mask.numel() == 0:
            return torch.full(shape, fill_value=fill_value, device=x.device, dtype=x.dtype)

        x_reverted = torch.full(shape, fill_value=fill_value, device=x.device, dtype=x.dtype)
        mask_unsq = self._unsqueeze_mask(mask, axis, current_shape).expand_as(x)
        x_reverted.scatter_(dim=idx_ax, index=mask_unsq, src=x)
        return x_reverted


def merge_Mask(masks: List[Mask]):
    """Merge the masks into a single mask
    :param masks: list of masks to merge
    :return: merged mask
    """
    S, T, C = [], [], []
    S_length = masks[0].S_length
    T_length = masks[0].T_length
    C_length = masks[0].C_length
    for mask in masks:
        if masks[0].S is not None:
            S.append(mask.S)
        if masks[0].T is not None:
            T.append(mask.T)
        if masks[0].C is not None:
            C.append(mask.C)
        assert mask.S_length == S_length, f"Invalid S_length {mask.S_length}. Must be the same for all masks"
        assert mask.T_length == T_length, f"Invalid T_length {mask.T_length}. Must be the same for all masks"
        assert mask.C_length == C_length, f"Invalid C_length {mask.C_length}. Must be the same for all masks"
    S = torch.cat(S, dim=0) if len(S) > 0 else None
    T = torch.cat(T, dim=0) if len(T) > 0 else None
    C = torch.cat(C, dim=0) if len(C) > 0 else None
    return Mask(S=S, T=T, C=C,
                S_length=S_length, T_length=T_length, C_length=C_length)

class MaskSpatial(Mask):
    """Mask for spatial
    """
    def __init__(self,
                 S:torch.Tensor,
                 S_length:int=None,
                ):
        super(MaskSpatial, self).__init__(S=S, S_length=S_length)

    def upsample_mask(self, new_S_length: int) -> "MaskSpatial":
        """
        Upsample the spatial mask to a finer grid.

        `new_S_length` must be >= current `S_length` and correspond to a square
        grid that is an integer factor finer than the current grid.
        """
        S_upsampled, S_length_upsampled = self._upsample_mask(new_S_length)
        return MaskSpatial(S=S_upsampled, S_length=S_length_upsampled)


class EmptyMask(Mask):
    """Empty mask for spatial
    """
    def __init__(self,
                ):
        pass

    def apply(self, x, axis='S', current_shape="BTCSD"):
        return x

def apply_spatial_masks(x, masks:Union[Mask,List[Mask]], upsample_S_to=None):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    :param current_S_length: current spatial length of the input tensor (e.g. 36 for 6x6 patches). If provided, the masks will be upsampled to match this length if needed.
    """
    all_x = []
    if not isinstance(masks, list):
        masks = [masks]
    for m in masks:
        all_x.append(m.apply(x, axis='S', current_shape='BSD', upsample_S_to=upsample_S_to))
    return torch.cat(all_x, dim=0)
