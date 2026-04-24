"""
./src/seer_app.py

python -m src.seer_app

Main Seer app maker. Amalgamates everything into a single runnable file.
"""

from typing import Any, cast

import numpy as np
from direct.gui.DirectGui import DirectLabel, DirectSlider
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath, Point3, TransparencyAttrib

from src.render_molecules.arrange_molecules import build_templates_from_object
from src.render_molecules.arrangement.geometry import compute_bounding_sphere_radius
from src.render_molecules.arrangement.placement import PlacementConfig, place_molecules
from src.render_molecules.arrangement.renderer import (
    render_object_state,
    set_atom_scale_factor,
)
from src.render_molecules.arrangement.scene_state import MoleculeTemplate, ObjectState
from src.utils.constants import (
    CHUNK_MOL_COUNT_PER_TEMPLATE,
    CHUNK_SIZE_A,
    FADE_FOV_START,
    FINAL_AGGREGATED,
    LOAD_RADIUS_CHUNKS,
    MAX_CHUNKS_PER_FRAME,
    MOL_CAM_SPEED_A,
    MOL_VIEW_SCALE,
    UNLOAD_RADIUS_CHUNKS,
    WORLD_CHUNKS,
)
from src.utils.json_io import load_json
from src.utils.type_annotations import Aggregations, Bounds
from src.video_processing.environment import (
    AGGREGATION_PATH,
    RoomState,
    decrease_fov,
    env_setup,
    increase_fov,
    mouse_look,
    move,
)
from src.zoom.raycast_picker import RaycastPicker


class SeerApp(ShowBase):
    def __init__(
        self,
        room_state: RoomState | None = None,
        aggregation_path: str | None = AGGREGATION_PATH,
        debug: bool = False,
    ) -> None:
        """
        Initialization of the SeerApp entrypoint class.

        Args:
            room_state (RoomState | None, optional): The current room state with all necessary default values. Defaults to None.
            aggregation_path (str | None, optional): The path to the aggregations JSON. Defaults to AGGREGATION_PATH.
            debug (bool, optional): Whether to print debug statements or not. Defaults to False.

        Raises:
            ValueError: If the loader does not exist.
        """
        super().__init__()

        self.disableMouse()

        self.room_root = self.render.attachNewNode("room_root")
        self.room_root.show()
        self.room_root.setTransparency(TransparencyAttrib.MAlpha)

        self.mol_root: NodePath = self.render.attachNewNode("mol_root")
        self.mol_root.hide()

        bg = self.getBackgroundColor()
        self._natural_bg: tuple[float, float, float] = (
            float(bg[0]),
            float(bg[1]),
            float(bg[2]),
        )
        self._saved_camera_state: dict | None = None
        self._mol_instance_roots: dict[int, NodePath] = {}
        self._in_molecular_scene: bool = False

        self._loaded_chunks: dict[tuple[int, int, int], NodePath] = {}
        self._mol_templates: dict[int, MoleculeTemplate] | None = None
        self._mol_origin: Point3 | None = None
        self._atom_slider: DirectSlider | None = None
        self._atom_label: DirectLabel | None = None
        self._sim_threads: dict[tuple[int, int, int], Any] = {}
        self._chunk_object_states: dict[tuple[int, int, int], ObjectState] = {}
        self._chunk_instance_roots: dict[tuple[int, int, int], dict[int, NodePath]] = {}
        self._cloud_rendering: bool = False
        self._sim_running: bool = False
        # Per-chunk interpolation state for smooth animation between MACE steps
        self._chunk_interp: dict[tuple[int, int, int], dict] = {}

        if room_state is None:
            room_state = RoomState(window=self.win, camera=self.camLens)

        self.room_state = room_state
        self._room_move_speed: float = self.room_state.current_move_speed
        self.keys = self.room_state.mvt_key_states
        self.move_speed = self.room_state.current_move_speed
        self.mouse_locked = self.room_state.mouse_locked
        self.sensitivity = self.room_state.default_sensitivity
        self.heading = 0.0
        self.pitch = 0.0

        if aggregation_path:
            self.room_state.aggregation_path = aggregation_path
        self.room_state.debug = debug
        self.room_state.camera = self.camLens
        self.room_state.window = self.win

        if self.loader:
            self.room_geo = env_setup(
                loader=self.loader, parent=self.room_root, room_state=self.room_state
            )
        else:
            raise ValueError("Missing loader.")

        self.room_data = load_json(FINAL_AGGREGATED)
        self.room_picker = RaycastPicker(
            camera_node=self.camera,
            cam_node=self.camNode,
            target_root=self.room_geo,
        )
        self.room_picker.mark_pickable(self.room_geo)
        self._debug_ray_node: NodePath | None = None
        self._debug_hit_dot: NodePath | None = None
        self._debug_lock_box: NodePath | None = None

        self.taskMgr.add(self._mouse_look_task, "mouse-look")
        self.taskMgr.add(self._move_task, "move")
        self.taskMgr.add(self._bg_fade_task, "bg-fade")
        self._build_atom_slider()
        self._build_md_controls()
        self.taskMgr.add(self._chunk_stream_task, "chunk-stream")
        self.taskMgr.add(self._md_update_task, "md-update")
        if self.room_state.debug:
            self.taskMgr.add(self._debug_preview_task, "debug-preview")

        for key, func, args in self.room_state.movement_commands:
            if args is not None:
                self.accept(key, func, [self.room_state, *args])
            else:
                self.accept(key, func, [self.room_state])

        self.accept("wheel_up", self._on_wheel_up)
        self.accept("wheel_down", self._on_wheel_down)

    def _build_atom_slider(self) -> None:
        """
        Creates the atom scale slider and label, which are hidden until molecular mode is entered.
        """
        self._atom_slider = DirectSlider(
            range=(0.1, 1.0),
            value=0.33,
            pageSize=0.1,
            command=self._on_atom_scale_changed,
            pos=(-0.85, 0, -0.85),
            scale=0.35,
        )
        set_atom_scale_factor(0.33)
        self._atom_label = DirectLabel(
            text="Atom Scale: 0.33x",
            pos=(-0.85, 0, -0.78),
            scale=0.05,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self._atom_slider.hide()
        self._atom_label.hide()

    def _build_md_controls(self) -> None:
        """
        Creates the time toggle, temperature slider, and cloud toggle, hidden until molecular mode.
        """
        from direct.gui.DirectGui import DirectCheckButton

        self._time_toggle: DirectCheckButton = DirectCheckButton(
            text="Time",
            scale=0.05,
            pos=(1.1, 0, 0.7),
            command=self._on_time_toggle,
        )
        self._time_toggle.hide()

        def _update_temperature() -> None:
            temp = float(self._temp_slider["value"])
            self._temp_label["text"] = f"Temp: {temp:.0f} K"
            for sim in self._sim_threads.values():
                sim.set_temperature(temp)

        self._temp_slider: DirectSlider = DirectSlider(
            range=(100, 1000),
            value=298,
            pageSize=25,
            command=_update_temperature,
            pos=(1.1, 0, 0.55),
            scale=0.25,
        )
        self._temp_slider.hide()

        self._temp_label: DirectLabel = DirectLabel(
            text="Temp: 298 K",
            pos=(1.1, 0, 0.6),
            scale=0.04,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self._temp_label.hide()

        self._cloud_toggle: DirectCheckButton = DirectCheckButton(
            text="Electron Clouds",
            scale=0.05,
            pos=(1.1, 0, 0.4),
            command=self._on_cloud_toggle,
        )
        self._cloud_toggle.hide()

    def _on_time_toggle(self, status: bool) -> None:
        """Handle the time toggle checkbox."""
        from src.render_molecules.arrangement.renderer import (
            replace_clouds_with_sticks,
            restore_clouds_from_sticks,
        )

        self._sim_running = status
        if status:
            # Switch all loaded chunks to cheap stick bonds before simulation starts
            for coords, inst_roots in self._chunk_instance_roots.items():
                obj_state = self._chunk_object_states.get(coords)
                if obj_state:
                    replace_clouds_with_sticks(inst_roots, obj_state)
            self._start_chunk_simulations()
        else:
            for sim in self._sim_threads.values():
                sim.pause()
            self._chunk_interp.clear()
            # Restore full bond clouds now that dynamics is paused
            for coords, inst_roots in self._chunk_instance_roots.items():
                obj_state = self._chunk_object_states.get(coords)
                if obj_state:
                    restore_clouds_from_sticks(inst_roots, obj_state, self)

    def _on_cloud_toggle(self, status: bool) -> None:
        """Handle the cloud rendering toggle checkbox. Only applies during dynamics."""
        self._cloud_rendering = status
        if not self._sim_running:
            return
        for inst_roots in self._chunk_instance_roots.values():
            for root in inst_roots.values():
                if root is None or root.isEmpty():
                    continue
                for child in root.getChildren():
                    if child.getName() == "stick_bonds":
                        if status:
                            child.hide()
                        else:
                            child.show()

    def _get_camera_chunk(self) -> tuple[int, int, int] | None:
        """Return the chunk coordinate the camera is currently inside."""
        if self._mol_origin is None or self.camera is None:
            return None
        cam_pos = self.camera.getPos()
        origin = self._mol_origin
        mol_cam = np.array([
            (cam_pos.x - origin.x) / MOL_VIEW_SCALE,
            (cam_pos.y - origin.y) / MOL_VIEW_SCALE,
            (cam_pos.z - origin.z) / MOL_VIEW_SCALE,
        ])
        return (
            int(np.floor(mol_cam[0] / CHUNK_SIZE_A)),
            int(np.floor(mol_cam[1] / CHUNK_SIZE_A)),
            int(np.floor(mol_cam[2] / CHUNK_SIZE_A)),
        )

    def _start_chunk_simulations(self) -> None:
        """Start a SimulationThread for all loaded chunks."""
        from src.dynamics.engine import MDEngine
        from src.dynamics.sim_thread import SimulationThread

        # Stop threads for chunks that are no longer loaded
        for coords in list(self._sim_threads):
            if coords not in self._chunk_object_states:
                self._sim_threads.pop(coords).stop()

        # Load model once, share across all chunk threads
        engine = MDEngine(is_metallic=False)
        engine.load_model()
        temperature = float(self._temp_slider["value"])

        # Start/resume simulations for all loaded chunks
        for coords, obj_state in self._chunk_object_states.items():
            if obj_state is None or not obj_state.instances:
                continue
            if coords in self._sim_threads:
                self._sim_threads[coords].resume()
                continue
            sim = SimulationThread(
                engine=engine,
                object_state=obj_state,
                active_instance_ids=list(obj_state.instances.keys()),
                temperature=temperature,
            )
            sim.start()
            sim.resume()
            self._sim_threads[coords] = sim

    def _md_update_task(self, task) -> int:
        """
        Read shared position buffers each frame and move atom NodePaths.
        Interpolates between the previous and current simulation step so that
        atoms move smoothly rather than snapping every MACE inference cycle.

        Args:
            task: Panda3D task object.

        Returns:
            int: Task continuation token.
        """
        if not self._in_molecular_scene or not self._sim_running:
            return task.cont

        import time as _time

        from src.render_molecules.arrangement.renderer import (
            rebuild_bond_clouds,
            update_atom_positions,
            update_stick_bonds,
        )

        now = _time.monotonic()

        for coords, sim in list(self._sim_threads.items()):
            if not sim.is_running():
                continue
            obj_state = self._chunk_object_states.get(coords)
            inst_roots = self._chunk_instance_roots.get(coords)
            if obj_state is None or inst_roots is None:
                continue

            new_buf = sim.buffer.read()

            # Initialise interpolation state on first encounter
            if coords not in self._chunk_interp:
                self._chunk_interp[coords] = {
                    "prev": new_buf.copy(),
                    "curr": new_buf.copy(),
                    "last_step_time": now,
                    "step_duration": 0.5,  # initial guess; adapts via EMA
                }

            state = self._chunk_interp[coords]

            # Detect when the background thread has written a new frame
            if not np.array_equal(new_buf, state["curr"]):
                elapsed = now - state["last_step_time"]
                # Exponential moving average keeps step_duration estimate stable
                state["step_duration"] = elapsed * 0.4 + state["step_duration"] * 0.6
                state["prev"] = state["curr"]
                state["curr"] = new_buf.copy()
                state["last_step_time"] = now

            # t in [0, 1]: how far through the current step we are
            t = min(
                (now - state["last_step_time"]) / max(state["step_duration"], 1e-4),
                1.0,
            )
            positions = state["prev"] * (1.0 - t) + state["curr"] * t

            update_atom_positions(inst_roots, sim.mapping, positions, obj_state)
            update_stick_bonds(inst_roots, obj_state)
            if self._cloud_rendering:
                rebuild_bond_clouds(inst_roots, obj_state, self)

        return task.cont

    def _on_atom_scale_changed(self) -> None:
        """
        Reads the slider value and applies it to all rendered atom spheres.
        """
        if self._atom_slider is None:
            return
        scale = float(self._atom_slider["value"])
        set_atom_scale_factor(scale)
        if self._atom_label is not None:
            self._atom_label["text"] = f"Atom Scale: {scale:.2f}x"

    def _chunk_stream_task(self, task) -> int:
        """
        Per frame task that loads chunks near the camera and unloads distant ones.

        Args:
            task: Panda3D task object

        Returns:
            int: Task continuation token
        """
        if not self._in_molecular_scene or self._mol_origin is None:
            return task.cont

        cam_pos = self.camera.getPos()
        origin = self._mol_origin
        mol_cam = np.array(
            [
                (cam_pos.x - origin.x) / MOL_VIEW_SCALE,
                (cam_pos.y - origin.y) / MOL_VIEW_SCALE,
                (cam_pos.z - origin.z) / MOL_VIEW_SCALE,
            ]
        )

        # Wrap camera when it crosses the world boundary (toroidal topology)
        world_size_a = WORLD_CHUNKS * CHUNK_SIZE_A
        half_world = world_size_a * 0.5
        teleport = np.zeros(3)
        for dim in range(3):
            if mol_cam[dim] < -half_world:
                mol_cam[dim] += world_size_a
                teleport[dim] = world_size_a * MOL_VIEW_SCALE
            elif mol_cam[dim] >= half_world:
                mol_cam[dim] -= world_size_a
                teleport[dim] = -world_size_a * MOL_VIEW_SCALE
        if np.any(teleport != 0) and self.camera is not None:
            self.camera.setPos(
                cam_pos.x + teleport[0],
                cam_pos.y + teleport[1],
                cam_pos.z + teleport[2],
            )

        cx = int(np.floor(mol_cam[0] / CHUNK_SIZE_A))
        cy = int(np.floor(mol_cam[1] / CHUNK_SIZE_A))
        cz = int(np.floor(mol_cam[2] / CHUNK_SIZE_A))

        r = LOAD_RADIUS_CHUNKS
        candidates: list[tuple[int, int, int]] = [
            (ix, iy, iz)
            for ix in range(cx - r, cx + r + 1)
            for iy in range(cy - r, cy + r + 1)
            for iz in range(cz - r, cz + r + 1)
        ]
        candidates.sort(
            key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2
        )

        ur = UNLOAD_RADIUS_CHUNKS
        for coords in list(self._loaded_chunks):
            if (
                abs(coords[0] - cx) > ur
                or abs(coords[1] - cy) > ur
                or abs(coords[2] - cz) > ur
            ):
                self._loaded_chunks.pop(coords).detachNode()
                self._chunk_object_states.pop(coords, None)
                self._chunk_instance_roots.pop(coords, None)
                sim = self._sim_threads.pop(coords, None)
                if sim is not None:
                    sim.stop()

        generated = 0
        for coords in candidates:
            if generated >= MAX_CHUNKS_PER_FRAME:
                break
            if coords not in self._loaded_chunks:
                self._loaded_chunks[coords] = self._generate_chunk(coords)
                generated += 1

        return task.cont

    def _generate_chunk(self, chunk_coords: tuple[int, int, int]) -> NodePath:
        """
        Places and renders molecules for one chunk. Attaches result to mol_root.

        Args:
            chunk_coords (tuple[int, int, int]): Integer (ix, iy, iz) grid coords

        Returns:
            NodePath: The chunk node attache to mol_root
        """
        ix, iy, iz = chunk_coords
        wc = WORLD_CHUNKS
        seed_key = (ix % wc, iy % wc, iz % wc)
        seed = abs(hash(seed_key)) % (2**31)
        s = CHUNK_SIZE_A
        cmin = np.array([ix, iy, iz], dtype=float) * s
        cmax = cmin + s

        box_bottom = np.array(
            [
                [cmin[0], cmax[0], cmax[0], cmin[0]],
                [cmin[1], cmin[1], cmax[1], cmax[1]],
                [cmin[2], cmin[2], cmin[2], cmin[2]],
            ],
            dtype=float,
        )
        box_top = np.array(
            [
                [cmin[0], cmax[0], cmax[0], cmin[0]],
                [cmin[1], cmin[1], cmax[1], cmax[1]],
                [cmax[2], cmax[2], cmax[2], cmax[2]],
            ],
            dtype=float,
        )

        templates = self._mol_templates or {}
        target_counts = {tid: CHUNK_MOL_COUNT_PER_TEMPLATE for tid in templates}
        total = sum(target_counts.values())

        object_state = ObjectState(
            object_key=f"chunk_{ix}_{iy}_{iz}",
            object_name=f"chunk_{ix}_{iy}_{iz}",
            instance_id=f"chunk_{ix}_{iy}_{iz}",
            display_name=f"chunk_{ix}_{iy}_{iz}",
            templates=templates,
            instances={},
            box_bottom=box_bottom,
            box_top=box_top,
            rng_seed=seed,
        )

        if templates:
            max_r = max(compute_bounding_sphere_radius(t) for t in templates.values())
            # Single-atom templates have radius 0, which would make min_center_distance
            # 0 and allow all instances to pile at the same point. Floor at 2.0Å so
            # even bare metal atoms are spaced apart.
            effective_r = max(max_r, 2.0)
            config = PlacementConfig(
                seed=seed,
                frontier_radius=CHUNK_SIZE_A * 0.5,
                min_center_distance=effective_r * 3.5,
                max_total_attempts=total * 100,
                target_instance_count=total,
                stop_when_target_met=True,
                require_in_bounds=True,
                require_no_overlap=True,
            )
            object_state = place_molecules(
                object_state=object_state, config=config, target_counts=target_counts
            )

        chunk_np = self.mol_root.attachNewNode(f"chunk_{ix}_{iy}_{iz}")
        instance_roots = render_object_state(
            base=self, parent=chunk_np, object_state=object_state
        )
        self._chunk_object_states[chunk_coords] = object_state
        self._chunk_instance_roots[chunk_coords] = instance_roots
        return chunk_np

    def _bg_fade_task(self, task) -> int:
        """
        Gradients background colour from natural to black as FOV decreases below fade threshold.

        Args:
            task: Panda3D task object

        Returns:
            int: Task continuation token
        """
        if self._in_molecular_scene:
            return task.cont

        fov = self.room_state.current_fov
        fade_end = self.room_state.min_room_fov

        if fov >= FADE_FOV_START:
            self.setBackgroundColor(*self._natural_bg, 1.0)
            return task.cont

        t = max(0.0, (fov - fade_end) / (FADE_FOV_START - fade_end))
        r, g, b = self._natural_bg
        self.setBackgroundColor(r * t, g * t, b * t, 1.0)
        return task.cont

    def _enter_molecular_mode(self) -> None:
        """
        Hides the room, loads molecules for target, positions camera to face cluster, shows molecule scene.
        """
        if self._in_molecular_scene:
            return
        self._in_molecular_scene = True

        self._saved_camera_state = {
            "pos": self.camera.getPos(),
            "hpr": self.camera.getHpr(),
            "fov": self.room_state.current_fov,
        }
        self._room_move_speed = self.move_speed

        self.room_root.hide()
        self.setBackgroundColor(0, 0, 0, 1)

        obj_key = self.room_state.target_object_key
        if obj_key is not None:
            self._mol_templates = build_templates_from_object(
                self.room_data.get(obj_key, {})
            )

        tp = self.room_state.target_point
        self._mol_origin = tp

        self.mol_root.setScale(MOL_VIEW_SCALE)
        if tp is not None:
            self.mol_root.setPos(tp.x, tp.y, tp.z)
            self.camera.setPos(tp.x, tp.y, tp.z)
        self.camera.setHpr(0, 0, 0)
        self.room_state.camera.setFov(90.0)
        self.room_state.current_fov = 90.0
        self.move_speed = MOL_CAM_SPEED_A * MOL_VIEW_SCALE

        if self._atom_slider is not None:
            self._atom_slider.show()
        if self._atom_label is not None:
            self._atom_label.show()

        self.mol_root.show()
        self._time_toggle.show()
        self._temp_slider.show()
        self._temp_label.show()
        self._cloud_toggle.show()

    def _exit_molecular_mode(self) -> None:
        if not self._in_molecular_scene:
            return
        self._in_molecular_scene = False

        for chunk_np in self._loaded_chunks.values():
            chunk_np.detachNode()
        self._loaded_chunks.clear()
        self._mol_templates = None
        self._mol_origin = None

        for child in self.mol_root.getChildren():
            child.detachNode()
        self.mol_root.hide()

        self.room_root.show()
        self.move_speed = self._room_move_speed

        if self._atom_slider is not None:
            self._atom_slider.hide()
        if self._atom_label is not None:
            self._atom_label.hide()

        self._time_toggle.hide()
        self._temp_slider.hide()
        self._temp_label.hide()
        self._cloud_toggle.hide()
        for sim in self._sim_threads.values():
            sim.stop()
        self._sim_threads.clear()
        self._chunk_object_states.clear()
        self._chunk_instance_roots.clear()
        self._chunk_interp.clear()
        self._sim_running = False

        if self._saved_camera_state is not None:
            self.camera.setPos(self._saved_camera_state["pos"])
            self.camera.setHpr(self._saved_camera_state["hpr"])
            fov = self._saved_camera_state["fov"]
            self.room_state.camera.setFov(fov)
            self.room_state.current_fov = fov
            self._saved_camera_state = None

    def _mouse_look_task(self, task):
        """
        Proxy task that keeps local mouse-lock state in sync and delegates look updates.

        Args:
            task: Panda3D task object.

        Returns:
            Any: Task continuation token from `mouse_look`.
        """
        if (
            self.room_state.molecular_mode
            and self.room_state.target_locked
            and not self._in_molecular_scene
        ):
            mouse_watcher = self.mouseWatcherNode
            if (
                self.win is not None
                and mouse_watcher is not None
                and mouse_watcher.hasMouse()
            ):
                cx = self.win.getXSize() // 2
                cy = self.win.getYSize() // 2
                self.win.movePointer(0, cx, cy)
            return task.cont

        self.mouse_locked = self.room_state.mouse_locked
        return mouse_look(self, task)

    def _move_task(self, task):
        """
        Proxy task that applies per-frame movement updates.

        Args:
            task: Panda3D task object.

        Returns:
            Any: Task continuation token from `move`.
        """
        return move(self, task)

    def _debug_preview_task(self, task):
        """
        Continuously renders center-ray and lock-target preview while in room state.

        Args:
            task: Panda3D task object.

        Returns:
            Any: Task continuation token.
        """
        if not self.room_state.debug:
            return task.cont

        if self.room_state.molecular_mode:
            self._clear_debug_visuals()
            return task.cont

        hit = self.room_picker.pick_center()
        if hit is None:
            self._clear_debug_visuals()
            return task.cont

        _hit_node, hit_point = hit
        self._draw_debug_raycast(hit_point)
        object_key = self._find_object_key_for_point(hit_point)
        self._draw_debug_lock_box(object_key)

        return task.cont

    def _on_wheel_up(self) -> None:
        """
        On zoom in.
        """
        was_molecular = self.room_state.molecular_mode
        increase_fov(self.room_state)
        if was_molecular and not self.room_state.molecular_mode:
            self._clear_target_lock()
            self._exit_molecular_mode()

    def _on_wheel_down(self) -> None:
        """
        On zoom out.
        """
        decrease_fov(self.room_state)
        if self.room_state.molecular_mode and not self.room_state.target_locked:
            self._lock_target_from_center()

    def _find_object_key_for_point(self, point: Point3) -> str:
        """
        Resolves a world-space point to the nearest aggregated object key.

        Args:
            point (Point3): Hit point in world/object space.

        Returns:
            str: Matching object key from aggregated room data.

        Raises:
            RuntimeError: If no object candidates exist in loaded room data.
        """
        candidate_key: str | None = None
        candidate_distance: float | None = None

        for object_key, object_data in self.room_data.items():
            corners = object_data["corners"]["bottom"] + object_data["corners"]["top"]
            bounds: Bounds = {
                "mins": cast(
                    tuple[float, float, float],
                    tuple(
                        min(float(corner[index]) for corner in corners)
                        for index in range(3)
                    ),
                ),
                "maxs": cast(
                    tuple[float, float, float],
                    tuple(
                        max(float(corner[index]) for corner in corners)
                        for index in range(3)
                    ),
                ),
            }
            mins = bounds["mins"]
            maxs = bounds["maxs"]

            if all(
                mins[index] <= coord <= maxs[index]
                for index, coord in enumerate((point.x, point.y, point.z))
            ):
                return object_key

            center = tuple((mins[index] + maxs[index]) * 0.5 for index in range(3))
            distance = sum(
                (coord - center[index]) ** 2
                for index, coord in enumerate((point.x, point.y, point.z))
            )
            if candidate_distance is None or distance < candidate_distance:
                candidate_distance = distance
                candidate_key = object_key

        if candidate_key is None:
            raise RuntimeError("No object bounds were available for raycast locking.")

        return candidate_key

    def _lock_target_from_center(self) -> None:
        """
        Raycasts from screen center and locks room-state target to the selected object.
        """
        hit = self.room_picker.pick_center()
        if hit is None:
            if self.room_state.debug:
                self._clear_debug_visuals()
            return

        _hit_node, hit_point = hit
        if self.room_state.debug:
            self._draw_debug_raycast(hit_point)

        object_key = self._find_object_key_for_point(hit_point)
        self.room_state.target_point = hit_point
        self.room_state.target_object_key = object_key
        self.room_state.target_locked = True

        if self.room_state.debug:
            self._draw_debug_lock_box(object_key)

        self._enter_molecular_mode()

    def _clear_target_lock(self) -> None:
        """
        Clears any currently locked zoom target from room state.
        """
        self.room_state.target_point = None
        self.room_state.target_object_key = None
        self.room_state.target_locked = False
        if self.room_state.debug:
            self._clear_debug_visuals()

    def _draw_debug_raycast(self, hit_point: Point3) -> None:
        """
        Draws a debug line from camera origin to raycast hit, plus a hit marker dot.

        Args:
            hit_point (Point3): Raycast hit point in `room_geo` local coordinates.
        """
        if self._debug_ray_node is not None:
            self._debug_ray_node.removeNode()
            self._debug_ray_node = None

        if self._debug_hit_dot is not None:
            self._debug_hit_dot.setPos(hit_point)

        camera_origin = self.room_geo.getRelativePoint(self.camera, Point3(0, 0, 0))

        ray_line = LineSegs("debug_raycast")
        ray_line.setThickness(2.0)
        ray_line.setColor(0.2, 1.0, 0.2, 1.0)
        ray_line.moveTo(camera_origin)
        ray_line.drawTo(hit_point)
        self._debug_ray_node = self.room_geo.attachNewNode(ray_line.create())

        if self.loader is None or self._debug_hit_dot is not None:
            return

        hit_dot = cast(NodePath, self.loader.loadModel("models/misc/sphere"))
        hit_dot.reparentTo(self.room_geo)
        hit_dot.setPos(hit_point)
        hit_dot.setScale(0.01)
        hit_dot.setColor(1.0, 0.2, 0.2, 1.0)
        self._debug_hit_dot = hit_dot

    def _draw_debug_lock_box(self, object_key: str) -> None:
        """
        Draws a wireframe box around the object that will be locked.

        Args:
            object_key (str): Object key selected from aggregated room data.
        """
        if self._debug_lock_box is not None:
            self._debug_lock_box.removeNode()
            self._debug_lock_box = None

        object_data = self.room_data[object_key]
        corners = object_data["corners"]["bottom"] + object_data["corners"]["top"]
        points = [Point3(*corner) for corner in corners]

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        box_lines = LineSegs("debug_lock_box")
        box_lines.setThickness(2.0)
        box_lines.setColor(1.0, 0.75, 0.1, 1.0)

        for start_idx, end_idx in edges:
            box_lines.moveTo(points[start_idx])
            box_lines.drawTo(points[end_idx])

        self._debug_lock_box = self.room_geo.attachNewNode(box_lines.create())

    def _clear_debug_visuals(self) -> None:
        """
        Removes debug ray and target-box visuals from the scene.
        """
        if self._debug_ray_node is not None:
            self._debug_ray_node.removeNode()
            self._debug_ray_node = None

        if self._debug_hit_dot is not None:
            self._debug_hit_dot.removeNode()
            self._debug_hit_dot = None

        if self._debug_lock_box is not None:
            self._debug_lock_box.removeNode()
            self._debug_lock_box = None


if __name__ == "__main__":
    app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=True)
    app.run()
