"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from simple.core.scene import Scene, TabletopScene
from simple.scenes.scene_manager import SceneManager 
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simple.core import Scene
    from simple.core import Asset

import yaml
import random
import numpy as np
from simple.utils import resolve_res_path, resolve_data_path
from dotenv import load_dotenv
import os

load_dotenv()

class HssdSuite(TabletopScene):
    name: str
    center_offset: list[float]
    center_orientation: list[float]
    
    def __init__(self, conf) -> None:
        self.uid = f"hssd:{conf['uid']}"
        self.conf = conf
        self.name = conf["name"]
        # self.table = table
        self.data_dir = f"{conf['data_dir']}/{conf['name']}"
        
        self.middle()

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table2: Asset) -> None:
        self.table2 = table2

    def dr(self) -> HssdSuite: # MOVE TO scene DR
        """ randomize with reasonable range """
        self.center_offset = np.random.uniform(self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"])
        self.center_orientation = np.random.uniform(self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"])
        return self
    
    def middle(self) -> HssdSuite:
        self.center_offset = [(u + d) / 2.0 for u, d in zip(self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"])]
        self.center_orientation = [(u + d) / 2.0 for u, d in zip(self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"])]
        return self

@SceneManager.register("hssd")
class HssdSceneManager(SceneManager):

    def __init__(self) -> None:
        # load hssd scenes config
        config_file_path = resolve_res_path("hssd-scenes/config.yaml")
        with open(config_file_path) as f:
            try:
                self.hssd_scenes = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    

    def sample(self, exclude: list[str] | None = None) -> Scene: 
        sampled = HssdSuite(random.choice(self.hssd_scenes))
        return sampled # some dr ranges must be valid for this specific scene

    def load(self, scene_uid: str) -> Scene: 
        """Load an asset by its ID."""
        if ":" in scene_uid:
            scene_uid = scene_uid.split(":")[1]

        hssd_scenes_dict = {s["uid"]:s for s in self.hssd_scenes}
        scene_name = hssd_scenes_dict[scene_uid]["name"]

        scene_dir = resolve_data_path(f"scenes/hssd/{scene_name}",auto_download=True)
        usd_path = os.path.join(scene_dir, f"{scene_name}.usd")
        self._hack_fix_tmp_paths(usd_path)
        
        return HssdSuite(hssd_scenes_dict[scene_uid])

    def _hack_fix_tmp_paths(self, usd_path: str) -> None:
        import subprocess
        import shutil
        import re
        import errno
        
        if not os.path.exists(usd_path):
            return

        try:
            # Inspect embedded asset paths inside the binary USD scene file.
            result = subprocess.run(
                f"strings {usd_path}",
                shell=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout
            
            # Find patterns like tmp/e64068067e09dc45/
            matches = re.findall(r"tmp/([a-zA-Z0-9]+)/", output)
            unique_hashes = set(matches)
            
            scene_dir = os.path.dirname(usd_path)

            def _ensure_root_hash_alias(abs_dir: str) -> None:
                match = re.fullmatch(r"/([A-Za-z0-9]+)/(props|textures)", abs_dir)
                if not match:
                    return

                hash_id = match.group(1)
                tmp_root = f"/tmp/{hash_id}"
                root_alias = f"/{hash_id}"

                if os.path.exists(root_alias):
                    return

                try:
                    os.symlink(tmp_root, root_alias, target_is_directory=True)
                    print(f"Hack: Symlinked {root_alias} -> {tmp_root}")
                except PermissionError as e:
                    raise RuntimeError(
                        "Failed to prepare HSSD root alias for an absolute USD asset path. "
                        f"Scene references {abs_dir}, which requires a root-level alias. "
                        f"Create it once with: sudo ln -sfn {tmp_root} {root_alias}"
                    ) from e
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to prepare HSSD root alias {root_alias} -> {tmp_root}."
                    ) from e

            def _mirror_tree(src: str, dst: str) -> None:
                if not os.path.exists(src):
                    return

                if os.path.islink(dst):
                    return

                if not os.path.exists(dst):
                    parent_dir = os.path.dirname(dst)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                    try:
                        os.symlink(src, dst, target_is_directory=True)
                        print(f"Hack: Symlinked {dst} -> {src}")
                        return
                    except OSError:
                        os.makedirs(dst, exist_ok=True)

                for root, dirs, files in os.walk(src):
                    rel_root = os.path.relpath(root, src)
                    dst_root = dst if rel_root == "." else os.path.join(dst, rel_root)
                    os.makedirs(dst_root, exist_ok=True)

                    for d in dirs:
                        os.makedirs(os.path.join(dst_root, d), exist_ok=True)

                    for file_name in files:
                        src_file = os.path.join(root, file_name)
                        dst_file = os.path.join(dst_root, file_name)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)

            def _ensure_casefold_aliases(root_dir: str) -> None:
                if not os.path.isdir(root_dir):
                    return

                for root, _, files in os.walk(root_dir):
                    for file_name in files:
                        lower_name = file_name.lower()
                        if lower_name == file_name:
                            continue

                        src_file = os.path.join(root, file_name)
                        alias_file = os.path.join(root, lower_name)
                        if os.path.exists(alias_file):
                            continue

                        try:
                            os.symlink(src_file, alias_file)
                        except OSError:
                            shutil.copy2(src_file, alias_file)
            
            for h in unique_hashes:
                target_dir = f"/tmp/{h}"
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                for folder in ["props", "textures"]:
                    src = os.path.join(scene_dir, folder)
                    dst = os.path.join(target_dir, folder)

                    try:
                        _mirror_tree(src, dst)
                        _ensure_casefold_aliases(dst)
                    except (OSError, shutil.Error) as e:
                        tmp_usage = shutil.disk_usage("/tmp")
                        no_space = (
                            isinstance(e, OSError) and e.errno == errno.ENOSPC
                        ) or "No space left on device" in str(e)
                        if no_space:
                            raise RuntimeError(
                                "Failed to prepare HSSD scene assets because /tmp is out of space. "
                                f"Copy failed from {src} to {dst}. "
                                f"/tmp free space: {tmp_usage.free / (1024 ** 3):.2f} GiB. "
                                "Free space in /tmp and rerun."
                            ) from e
                        raise RuntimeError(
                            f"Failed to prepare HSSD scene assets by mirroring {src} to {dst}."
                        ) from e

            # Some released HSSD scenes contain absolute references like
            # /home/<user>/props/*.usd instead of scene-local paths.
            #
            # Keep this outside the tmp-hash loop so it still runs for scenes
            # that only contain broken absolute references.
            abs_matches = re.findall(r"(/[^\s\"']+/(?:props|textures))/", output)
            abs_dirs = set(abs_matches)

            # The broken references we have seen most often point directly to
            # the current user's home directory, even when `strings` does not
            # surface those paths from the USD binary.
            home_dir = os.path.expanduser("~")
            abs_dirs.update(
                os.path.join(home_dir, folder) for folder in ("props", "textures")
            )

            for abs_dir in abs_dirs:
                folder = os.path.basename(abs_dir)
                src = os.path.join(scene_dir, folder)
                try:
                    _ensure_root_hash_alias(abs_dir)
                    _mirror_tree(src, abs_dir)
                    _ensure_casefold_aliases(abs_dir)
                except (OSError, shutil.Error) as e:
                    raise RuntimeError(
                        f"Failed to prepare HSSD scene assets by mirroring {src} to {abs_dir}."
                    ) from e
                            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to prepare HSSD temp asset paths for {usd_path}.") from e
    
