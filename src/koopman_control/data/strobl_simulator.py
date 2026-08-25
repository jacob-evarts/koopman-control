"""Python process adapter for the controlled Strobl Java simulator.

The Java side is intentionally treated as a versioned external plant. The
protocol is a small tab-delimited command stream and exposes only a scalar,
spatially homogeneous drug action. Grid states are transferred as categorical
values, never through rendered images.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

UPSTREAM_COMMIT = "aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2"
PROTOCOL_VERSION = "1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_ROOT = (
    _REPOSITORY_ROOT / "vendor" / "strobl2021_space_modulates_competition_AT"
)


class StroblProtocolError(RuntimeError):
    """Raised when the controlled Java runner violates its protocol."""


@dataclass(frozen=True)
class StroblLauncherConfig:
    """Java launch and protocol-validation settings.

    ``command`` may include the line-protocol flag.  Otherwise the launcher
    tries known controlled-jar layouts and finally a classpath invocation.
    ``STROBL_RUNNER_COMMAND`` and ``STROBL_VENDOR_ROOT`` provide deployment
    overrides without requiring changes to project path helpers.
    """

    command: tuple[str, ...] | None = None
    batch_command: tuple[str, ...] | None = None
    vendor_root: Path = DEFAULT_VENDOR_ROOT
    java_executable: str = "java"
    runner_class: str = "strobl.control.ControlledCli"
    line_protocol_args: tuple[str, ...] = ("--mode", "serve")
    batch_args: tuple[str, ...] = ("--mode", "batch")
    model_args: tuple[str, ...] = ()
    expected_commit: str = UPSTREAM_COMMIT
    expected_protocol: str = PROTOCOL_VERSION
    startup_timeout: float = 15.0
    response_timeout: float = 30.0
    environment: Mapping[str, str] = field(default_factory=dict)

    def resolved_vendor_root(self) -> Path:
        """Return the configured vendor root, honoring its environment override."""
        override = os.environ.get("STROBL_VENDOR_ROOT")
        return (
            Path(override).expanduser().resolve()
            if override
            else Path(self.vendor_root)
        )


@dataclass(frozen=True)
class StroblState:
    """One exact Java simulator observation converted to NumPy."""

    grid: np.ndarray
    counts: np.ndarray
    diagnostics: Mapping[str, np.ndarray | float | int | str | bool | None]
    time: float
    done: bool = False
    terminal_reason: str | None = None

    @property
    def sensitive(self) -> int:
        """Number of sensitive cells."""
        return int(self.counts[0])

    @property
    def resistant(self) -> int:
        """Number of resistant cells."""
        return int(self.counts[1])

    @property
    def empty(self) -> int:
        """Number of empty lattice sites."""
        return int(self.counts[2])

    @property
    def occupancy(self) -> float:
        """Fraction of lattice sites occupied by tumour cells."""
        return float((self.sensitive + self.resistant) / self.grid.size)


def _jsonable_mapping(
    config: Mapping[str, Any] | object | None,
) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("reset config must be a mapping or dataclass instance")


def _decode_diagnostic(value: Any) -> Any:
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, dict):
        return {key: _decode_diagnostic(item) for key, item in value.items()}
    return value


def _state_from_response(response: Mapping[str, Any]) -> StroblState:
    payload = response.get("state", response)
    if not isinstance(payload, Mapping):
        raise StroblProtocolError("runner response has no state object")
    try:
        grid = np.asarray(payload["grid"], dtype=np.uint8)
    except KeyError as exc:
        raise StroblProtocolError("runner state is missing grid") from exc
    if grid.ndim != 2 or grid.size == 0:
        raise StroblProtocolError(f"expected a non-empty 2-D grid, got {grid.shape}")
    if np.any(grid > 2):
        raise StroblProtocolError(
            "grid must contain only 0=empty, 1=sensitive, 2=resistant"
        )

    derived = np.bincount(grid.reshape(-1), minlength=3).astype(np.int64)
    raw_counts = payload.get("counts")
    if isinstance(raw_counts, Mapping):
        try:
            counts = np.asarray(
                [
                    raw_counts["sensitive"],
                    raw_counts["resistant"],
                    raw_counts["empty"],
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise StroblProtocolError(
                "count objects require sensitive, resistant, and empty keys"
            ) from exc
    else:
        counts = (
            derived[[1, 2, 0]]
            if raw_counts is None
            else np.asarray(raw_counts, dtype=np.int64)
        )
    if counts.shape != (3,):
        raise StroblProtocolError(
            "counts must have shape (3,) ordered sensitive/resistant/empty"
        )
    # Grid labels are empty/sensitive/resistant, while the public counts order
    # is sensitive/resistant/empty.
    expected_counts = derived[[1, 2, 0]]
    if not np.array_equal(counts, expected_counts):
        raise StroblProtocolError(
            f"runner counts {counts.tolist()} disagree with grid {expected_counts.tolist()}"
        )

    diagnostics = {
        str(key): _decode_diagnostic(value)
        for key, value in dict(payload.get("diagnostics", {})).items()
    }
    return StroblState(
        grid=grid,
        counts=counts,
        diagnostics=diagnostics,
        time=float(payload.get("time", payload.get("t", 0.0))),
        done=bool(payload.get("done", False)),
        terminal_reason=payload.get("terminal_reason"),
    )


class StroblSimulator:
    """Persistent reset/step client for the controlled Java runner."""

    def __init__(
        self,
        launcher: StroblLauncherConfig | None = None,
        *,
        d_max: float = 1.0,
    ) -> None:
        self.launcher = launcher or StroblLauncherConfig()
        self.d_max = float(d_max)
        if not np.isfinite(self.d_max) or self.d_max < 0:
            raise ValueError("d_max must be finite and non-negative")
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file: Any = None
        self.version_info: dict[str, Any] | None = None
        self.last_state: StroblState | None = None

    def _candidate_commands(self, *, batch: bool = False) -> list[list[str]]:
        cfg = self.launcher
        explicit = cfg.batch_command if batch else cfg.command
        if explicit is not None:
            return [list(explicit)]
        env_command = os.environ.get(
            "STROBL_BATCH_COMMAND" if batch else "STROBL_RUNNER_COMMAND"
        )
        if env_command:
            return [shlex.split(env_command)]

        root = cfg.resolved_vendor_root()
        jars = (
            root / "controlled-model.jar",
            root / "controlled" / "build" / "strobl-controlled.jar",
            root / "build" / "strobl-controlled.jar",
        )
        mode_args = cfg.batch_args if batch else cfg.line_protocol_args
        commands = [
            [cfg.java_executable, "-jar", str(jar), *mode_args, *cfg.model_args]
            for jar in jars
            if jar.is_file()
        ]
        class_dirs = (
            root / "controlled" / "build" / "classes",
            root / "build" / "classes",
        )
        commands.extend(
            [
                cfg.java_executable,
                "-cp",
                str(class_dir),
                cfg.runner_class,
                *mode_args,
                *cfg.model_args,
            ]
            for class_dir in class_dirs
            if class_dir.is_dir()
        )
        return commands

    def start(self) -> "StroblSimulator":
        """Start a runner and validate protocol, simulator version, and commit."""
        if self._process is not None and self._process.poll() is None:
            return self
        commands = self._candidate_commands()
        if not commands:
            raise FileNotFoundError(
                "no controlled Strobl runner found; set STROBL_RUNNER_COMMAND "
                "or configure StroblLauncherConfig.command"
            )

        errors: list[str] = []
        for command in commands:
            try:
                self._start_command(command)
                ready = self._read_line(timeout=self.launcher.startup_timeout)
                expected = (
                    f"READY\tstrobl-controlled-v{self.launcher.expected_protocol}"
                    f"\t{self.launcher.expected_commit}"
                )
                if ready != expected:
                    raise StroblProtocolError(
                        f"runner greeting {ready!r} does not match {expected!r}"
                    )
                self.version_info = {
                    "protocol": self.launcher.expected_protocol,
                    "version": "strobl-controlled-v1",
                    "commit": self.launcher.expected_commit,
                }
                return self
            except (OSError, StroblProtocolError) as exc:
                errors.append(f"{shlex.join(command)}: {exc}")
                self.close()
        raise StroblProtocolError(
            "unable to start a compatible runner:\n" + "\n".join(errors)
        )

    def _read_line(self, *, timeout: float | None = None) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise StroblProtocolError("runner is not started")
        wait = self.launcher.response_timeout if timeout is None else float(timeout)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(wait):
                raise StroblProtocolError(
                    f"runner response timed out after {wait:g}s: {self._stderr()}"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise StroblProtocolError(
                f"runner closed stdout with code {process.poll()}: {self._stderr()}"
            )
        line = line.rstrip("\r\n")
        if line.startswith("ERROR\t"):
            raise StroblProtocolError(line.partition("\t")[2])
        return line

    def _send_line(self, line: str) -> str:
        process = self._process
        if process is None or process.stdin is None:
            raise StroblProtocolError("runner is not started")
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise StroblProtocolError(
                f"failed to write to runner: {self._stderr()}"
            ) from exc
        return self._read_line()

    @staticmethod
    def _parse_fields(line: str, prefix: str) -> dict[str, str]:
        fields = line.split("\t")
        if not fields or fields[0] != prefix:
            raise StroblProtocolError(f"expected {prefix}, received {line!r}")
        parsed: dict[str, str] = {}
        for field in fields[1:]:
            key, separator, value = field.partition("=")
            if not separator:
                raise StroblProtocolError(f"malformed runner field {field!r}")
            parsed[key] = value
        return parsed

    def _read_grid(self) -> np.ndarray:
        fields = self._parse_fields(self._send_line("GRID"), "GRID")
        width = int(fields["width"])
        height = int(fields["height"])
        values = np.fromstring(fields["values"], sep=",", dtype=np.uint8)
        if values.size != width * height or np.any(values > 2):
            raise StroblProtocolError("runner returned an invalid categorical grid")
        return values.reshape(height, width)

    def _parse_state(self, line: str) -> StroblState:
        fields = self._parse_fields(line, "STATE")
        grid = self._read_grid()
        counts = np.asarray(
            [
                int(fields["sensitive"]),
                int(fields["resistant"]),
                int(np.sum(grid == 0)),
            ],
            dtype=np.int64,
        )
        expected = np.asarray(
            [np.sum(grid == 1), np.sum(grid == 2), np.sum(grid == 0)], dtype=np.int64
        )
        if not np.array_equal(counts, expected):
            raise StroblProtocolError("runner counts disagree with categorical grid")
        diagnostic_names = (
            "attempted_sensitive",
            "attempted_resistant",
            "blocked_sensitive",
            "blocked_resistant",
            "natural_deaths_sensitive",
            "natural_deaths_resistant",
            "drug_deaths_sensitive",
            "drug_deaths_resistant",
        )
        return StroblState(
            grid=grid,
            counts=counts,
            diagnostics={name: int(fields[name]) for name in diagnostic_names},
            time=float(fields["time"]),
        )

    def _start_command(self, command: Sequence[str]) -> None:
        environment = os.environ.copy()
        environment.update(
            {str(k): str(v) for k, v in self.launcher.environment.items()}
        )
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._process = subprocess.Popen(
            list(command),
            cwd=self.launcher.resolved_vendor_root(),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )

    def _stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        return self._stderr_file.read().strip()

    def _validate_version(self, response: Mapping[str, Any]) -> None:
        if response.get("ok") is False:
            raise StroblProtocolError(
                str(response.get("error", "version request failed"))
            )
        protocol = str(response.get("protocol", response.get("protocol_version", "")))
        commit = str(response.get("commit", response.get("upstream_commit", "")))
        version = response.get("version", response.get("simulator_version"))
        if protocol != self.launcher.expected_protocol:
            raise StroblProtocolError(
                f"protocol {protocol!r} does not match {self.launcher.expected_protocol!r}"
            )
        if commit != self.launcher.expected_commit:
            raise StroblProtocolError(
                f"commit {commit!r} does not match pinned {self.launcher.expected_commit!r}"
            )
        if not version:
            raise StroblProtocolError("runner did not report a simulator version")

    def _request(
        self, message: Mapping[str, Any], *, timeout: float | None = None
    ) -> Mapping[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise StroblProtocolError("runner is not started")
        if process.poll() is not None:
            raise StroblProtocolError(
                f"runner exited with code {process.returncode}: {self._stderr()}"
            )
        try:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise StroblProtocolError(
                f"failed to write to runner: {self._stderr()}"
            ) from exc

        wait = self.launcher.response_timeout if timeout is None else float(timeout)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(wait):
                raise StroblProtocolError(
                    f"runner response timed out after {wait:g}s: {self._stderr()}"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise StroblProtocolError(
                f"runner closed stdout with code {process.poll()}: {self._stderr()}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StroblProtocolError(
                f"invalid JSON response: {line.rstrip()!r}"
            ) from exc
        if not isinstance(response, Mapping):
            raise StroblProtocolError("runner response must be a JSON object")
        if response.get("ok") is False:
            raise StroblProtocolError(
                str(response.get("error", "runner request failed"))
            )
        return response

    def reset(
        self,
        config: Mapping[str, Any] | object | None = None,
        **overrides: Any,
    ) -> StroblState:
        """Reset one Java episode and return its exact initial state."""
        self.start()
        parameters = _jsonable_mapping(config)
        parameters.update(overrides)
        family = str(
            parameters.get("family", parameters.get("architecture", "random_mixed"))
        )
        sensitive = int(
            parameters.get("sensitive", parameters.get("sensitive_count", 0))
        )
        resistant = int(
            parameters.get("resistant", parameters.get("resistant_count", 0))
        )
        simulation_seed = int(
            parameters.get("simulation_seed", parameters.get("seed", 0))
        )
        ic_seed = int(
            parameters.get(
                "ic_seed", parameters.get("initial_condition_seed", simulation_seed)
            )
        )
        shared_occupied_mask = bool(parameters.get("shared_occupied_mask", False))
        if any(character.isspace() for character in family):
            raise ValueError("initial-condition family cannot contain whitespace")
        line = self._send_line(
            f"RESET {family} {sensitive} {resistant} {simulation_seed} {ic_seed} "
            f"{str(shared_occupied_mask).lower()}"
        )
        state = self._parse_state(line)
        self.last_state = state
        return state

    def step(self, dose: float) -> StroblState:
        """Advance one control interval under one bounded scalar global dose."""
        dose = float(dose)
        if not np.isfinite(dose) or not 0.0 <= dose <= self.d_max:
            raise ValueError(f"dose must lie in [0, {self.d_max:g}], got {dose}")
        state = self._parse_state(self._send_line(f"STEP {dose:.17g}"))
        self.last_state = state
        return state

    def close(self) -> None:
        """Stop the child process and release all process resources."""
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write("QUIT\n")
                        process.stdin.flush()
                    process.wait(timeout=2.0)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def __enter__(self) -> "StroblSimulator":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _spatial_diagnostics(grid: np.ndarray) -> dict[str, float | int]:
    """Compute inexpensive morphology diagnostics from one categorical grid."""
    resistant = grid == 2
    sensitive = grid == 1
    occupied = resistant | sensitive
    sr_edges = int(
        np.sum(
            (resistant[:, :-1] & sensitive[:, 1:])
            | (sensitive[:, :-1] & resistant[:, 1:])
        )
        + np.sum(
            (resistant[:-1, :] & sensitive[1:, :])
            | (sensitive[:-1, :] & resistant[1:, :])
        )
    )
    exposed_edges = int(
        np.sum(resistant[:, :-1] & ~occupied[:, 1:])
        + np.sum(resistant[:, 1:] & ~occupied[:, :-1])
        + np.sum(resistant[:-1, :] & ~occupied[1:, :])
        + np.sum(resistant[1:, :] & ~occupied[:-1, :])
    )
    seen = np.zeros_like(resistant, dtype=bool)
    components = 0
    height, width = resistant.shape
    for y, x in np.argwhere(resistant):
        if seen[y, x]:
            continue
        components += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and resistant[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return {
        "sensitive_resistant_edges": sr_edges,
        "resistant_boundary_exposure": exposed_edges,
        "resistant_components": components,
    }


def external_terminal_reason(
    *, initial_total: int, total: int, day: float
) -> str | None:
    """Apply benchmark cure/progression rules outside the Java simulator."""
    if total == 0:
        return "cure"
    if day >= 150.0 and total > 1.2 * initial_total:
        return "progression"
    return None


def simulate_episode(
    *,
    architecture: str,
    actions: Sequence[float] | np.ndarray,
    initial_counts: Sequence[int] | np.ndarray,
    parameters: Mapping[str, float],
    width: int,
    height: int,
    seed: int,
    ic_seed: int | None = None,
    policy_name: str | None = None,
    stop_on_terminal: bool = True,
    shared_occupied_mask: bool = False,
) -> dict[str, Any]:
    """Run one controlled episode and return the canonical dataset fields.

    ``initial_counts`` is ordered ``[S, R, N]``. Progression is evaluated
    externally as ``N > 1.2 N0`` after day 150; the Java plant never stops
    internally.
    """
    initial = np.asarray(initial_counts, dtype=np.int64)
    if initial.shape != (3,) or initial[2] != initial[0] + initial[1]:
        raise ValueError("initial_counts must be [S, R, N] with N=S+R")
    action_plan = np.asarray(actions, dtype=np.float32)
    if action_plan.ndim != 1:
        raise ValueError("actions must be a one-dimensional scalar schedule")
    r_s = float(parameters.get("r_s", 0.027))
    r_r = float(parameters.get("r_r", r_s))
    delta_t = float(parameters.get("delta_t", 0.0))
    d_d = float(parameters.get("d_d", 0.75))
    dt = float(parameters.get("dt", 1.0))
    d_max = float(parameters.get("d_max", 1.0))
    if d_max <= 0:
        raise ValueError("d_max must be positive")
    model_args = (
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--dt",
        f"{dt:.17g}",
        "--division-sensitive",
        f"{r_s:.17g}",
        "--division-resistant",
        f"{r_r:.17g}",
        "--death-sensitive",
        f"{delta_t:.17g}",
        "--death-resistant",
        f"{delta_t:.17g}",
        "--drug-kill",
        f"{d_d:.17g}",
    )
    launcher = StroblLauncherConfig(model_args=model_args)
    grids: list[np.ndarray] = []
    counts: list[list[int]] = []
    spatial: list[dict[str, float | int]] = []
    event_rows: list[Mapping[str, Any]] = []
    realized: list[float] = []
    terminal_reason = "max_time"
    terminal_time = float(len(action_plan) * dt)
    current_dose = d_max
    with StroblSimulator(launcher, d_max=d_max) as simulator:
        state = simulator.reset(
            family=architecture,
            sensitive=int(initial[0]),
            resistant=int(initial[1]),
            simulation_seed=int(seed),
            ic_seed=int(seed if ic_seed is None else ic_seed),
            shared_occupied_mask=shared_occupied_mask,
        )
        grids.append(state.grid.copy())
        counts.append(
            [state.sensitive, state.resistant, state.sensitive + state.resistant]
        )
        spatial.append(_spatial_diagnostics(state.grid))
        for index, planned in enumerate(action_plan):
            if policy_name == "paper_adaptive":
                total = counts[-1][2]
                if total > initial[2]:
                    current_dose = d_max
                elif total < 0.5 * initial[2]:
                    current_dose = 0.0
                dose = current_dose
            elif policy_name == "paper_text_adaptive":
                total = counts[-1][2]
                if total >= initial[2]:
                    current_dose = d_max
                elif total < 0.5 * initial[2]:
                    current_dose = 0.0
                dose = current_dose
            else:
                dose = float(planned)
            state = simulator.step(dose)
            realized.append(dose)
            grids.append(state.grid.copy())
            counts.append(
                [state.sensitive, state.resistant, state.sensitive + state.resistant]
            )
            event_rows.append(state.diagnostics)
            spatial.append(_spatial_diagnostics(state.grid))
            total = counts[-1][2]
            day = (index + 1) * dt
            reason = external_terminal_reason(
                initial_total=int(initial[2]), total=total, day=day
            )
            if reason is not None and terminal_reason == "max_time":
                terminal_reason = reason
                terminal_time = day
                if stop_on_terminal:
                    break
    event_names = sorted({name for row in event_rows for name in row})
    diagnostics: dict[str, np.ndarray] = {
        name: np.asarray([row.get(name, 0) for row in event_rows], dtype=np.int32)
        for name in event_names
    }
    for name in spatial[0]:
        diagnostics[name] = np.asarray([row[name] for row in spatial], dtype=np.float32)
    return {
        "grid": np.stack(grids).astype(np.uint8),
        "action": np.asarray(realized, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.int64),
        "occupancy": np.asarray(counts, dtype=np.float32)[:, 2] / float(width * height),
        "diagnostics": diagnostics,
        "terminal_reason": terminal_reason,
        "terminal_time": terminal_time,
    }


def run_batch(
    reset_config: Mapping[str, Any] | object,
    actions: Sequence[float] | np.ndarray,
    *,
    launcher: StroblLauncherConfig | None = None,
    d_max: float = 1.0,
    timeout: float | None = None,
) -> list[StroblState]:
    """Run an episode with the batch interface, falling back to reset/step."""
    cfg = launcher or StroblLauncherConfig()
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 1 or not np.all(np.isfinite(action_array)):
        raise ValueError("actions must be a finite one-dimensional scalar schedule")
    if np.any((action_array < 0) | (action_array > float(d_max))):
        raise ValueError(f"all doses must lie in [0, {float(d_max):g}]")

    commands = StroblSimulator(cfg, d_max=d_max)._candidate_commands(batch=True)
    request = {
        "command": "episode",
        "config": _jsonable_mapping(reset_config),
        "actions": action_array.tolist(),
    }
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=cfg.resolved_vendor_root(),
                env={
                    **os.environ,
                    **{str(k): str(v) for k, v in cfg.environment.items()},
                },
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            response = json.loads(completed.stdout)
            if not isinstance(response, Mapping) or response.get("ok") is False:
                raise StroblProtocolError(str(response))
            version = response.get("version_info", response)
            StroblSimulator(cfg, d_max=d_max)._validate_version(version)
            states = response.get("states")
            if not isinstance(states, list):
                raise StroblProtocolError("batch response is missing states")
            return [_state_from_response(state) for state in states]
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            StroblProtocolError,
        ):
            continue

    with StroblSimulator(cfg, d_max=d_max) as simulator:
        states = [simulator.reset(reset_config)]
        states.extend(simulator.step(float(dose)) for dose in action_array)
        return states


def main(argv: list[str] | None = None) -> None:
    """Run one episode from the command line using a named scalar-dose policy."""
    from koopman_control.data.strobl_policies import (
        constant,
        pulses,
        random_piecewise_constant,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=(
            "open_loop",
            "constant",
            "paper_adaptive",
            "paper_text_adaptive",
            "random_piecewise_constant",
            "pulses",
        ),
        default="open_loop",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--sensitive", type=int, default=4950)
    parser.add_argument("--resistant", type=int, default=50)
    parser.add_argument(
        "--architecture",
        choices=(
            "random_mixed",
            "resistant_core",
            "resistant_edge",
            "resistant_dispersed",
            "two_resistant_nests",
        ),
        default="random_mixed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-s", type=float, default=0.027)
    parser.add_argument("--resistance-cost", type=float, default=0.0)
    parser.add_argument("--turnover", type=float, default=0.0)
    parser.add_argument("--d-d", type=float, default=0.75)
    parser.add_argument("--d-max", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.policy == "open_loop":
        actions = np.zeros(args.steps, dtype=np.float32)
    elif args.policy == "constant":
        actions = constant(args.steps, args.dose, d_max=args.d_max)
    elif args.policy == "random_piecewise_constant":
        actions = random_piecewise_constant(
            args.steps, d_max=args.d_max, seed=args.seed
        )
    elif args.policy == "pulses":
        actions = pulses(
            args.steps,
            d_max=args.d_max,
            pulse_dose=args.dose,
            width=max(1, args.steps // 40),
            period=max(4, args.steps // 8),
        )
    else:
        actions = np.full(args.steps, args.d_max, dtype=np.float32)
    result = simulate_episode(
        architecture=args.architecture,
        actions=actions,
        initial_counts=(
            args.sensitive,
            args.resistant,
            args.sensitive + args.resistant,
        ),
        parameters={
            "r_s": args.r_s,
            "r_r": args.r_s * (1.0 - args.resistance_cost),
            "delta_t": args.r_s * args.turnover,
            "d_d": args.d_d,
            "d_max": args.d_max,
            "dt": 1.0,
        },
        width=args.width,
        height=args.height,
        seed=args.seed,
        policy_name=(
            args.policy
            if args.policy in {"paper_adaptive", "paper_text_adaptive"}
            else None
        ),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            grid=result["grid"],
            action=result["action"],
            counts=result["counts"],
            occupancy=result["occupancy"],
        )
    print(
        json.dumps(
            {
                "policy": args.policy,
                "steps": len(result["action"]),
                "initial_counts": result["counts"][0].tolist(),
                "final_counts": result["counts"][-1].tolist(),
                "terminal_reason": result["terminal_reason"],
                "terminal_time": result["terminal_time"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
        )
    )


__all__ = [
    "DEFAULT_VENDOR_ROOT",
    "PROTOCOL_VERSION",
    "StroblLauncherConfig",
    "StroblProtocolError",
    "StroblSimulator",
    "StroblState",
    "UPSTREAM_COMMIT",
    "external_terminal_reason",
    "simulate_episode",
    "run_batch",
]


if __name__ == "__main__":
    main()
