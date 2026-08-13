"""Compatibility shim for the JEPAMEMORY HWM framework source tree."""

from __future__ import annotations

from pathlib import Path

_SOURCE = Path("/home/d013/桌面/JEPAMEMORY/starVLA/model/framework/VLA_JEPA_HWM.py")
_SOURCE_TEXT = _SOURCE.read_text(encoding="utf-8").replace(
    '        return {"normalized_actions": actions.detach().cpu().numpy()}',
    '        return {"normalized_actions": actions.detach().to(torch.float32).cpu().numpy()}',
)
_SOURCE_TEXT = _SOURCE_TEXT.replace(
    '        z_subgoal = self._resolve_subgoal(z_cur, c_vla, z_subgoal)\n',
    '        if kwargs.get("hwm_subgoal_mode", "predict") == "current":\n'
    '            z_subgoal = z_cur\n'
    '        else:\n'
    '            z_subgoal = self._resolve_subgoal(z_cur, c_vla, z_subgoal)\n',
)
exec(compile(_SOURCE_TEXT, str(_SOURCE), "exec"), globals(), globals())


if "VLA_JEPA_HWM" in globals():
    _orig_preprocess = VLA_JEPA_HWM._preprocess_vjepa_videos_batch

    def _preprocess_vjepa_videos_batch(self, flat_videos):
        videos = _orig_preprocess(self, flat_videos)
        target_device = next(self.vj_encoder.parameters()).device
        target_dtype = next(self.vj_encoder.parameters()).dtype
        return videos.to(device=target_device, dtype=target_dtype)

    VLA_JEPA_HWM._preprocess_vjepa_videos_batch = _preprocess_vjepa_videos_batch
