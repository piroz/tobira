"""High-availability management for tobira API server.

Provides instance registration, leader election via file locks, and
readiness state tracking for multi-instance deployments.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = "/var/lib/tobira/ha"
LOCK_FILE = "leader.lock"
INSTANCES_FILE = "instances.json"


@dataclass
class InstanceInfo:
    """Metadata for a running tobira instance."""

    instance_id: str
    host: str
    port: int
    pid: int = field(default_factory=os.getpid)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceInfo:
        return cls(
            instance_id=data["instance_id"],
            host=data["host"],
            port=data["port"],
            pid=data.get("pid", 0),
            started_at=data.get("started_at", 0.0),
        )


class ReadinessState:
    """Tracks whether the server is ready to accept requests.

    The server is not ready until the backend model has been loaded.
    During shutdown, readiness is set to ``False`` so that load balancers
    stop routing new requests before the process terminates.
    """

    def __init__(self) -> None:
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    def set_ready(self) -> None:
        self._ready = True
        logger.info("Instance is now ready to accept requests")

    def set_not_ready(self) -> None:
        self._ready = False
        logger.info("Instance is no longer accepting requests")


class LeaderElection:
    """File-lock based leader election for multi-instance deployments.

    Only one instance holds the lock at a time.  When the leader process
    exits or crashes, the OS releases the lock and another instance can
    acquire it.
    """

    def __init__(self, state_dir: str = DEFAULT_STATE_DIR) -> None:
        self._state_dir = Path(state_dir)
        self._lock_path = self._state_dir / LOCK_FILE
        self._lock_fd: int | None = None
        self._is_leader: bool = False

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def try_acquire(self) -> bool:
        """Attempt to acquire the leader lock (non-blocking).

        Returns:
            ``True`` if this instance is now the leader.
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self._lock_path), os.O_CREAT | os.O_WRONLY)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd = fd
            self._is_leader = True
            # Write our PID so operators can identify the leader
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            logger.info("Acquired leader lock")
            return True
        except OSError:
            logger.debug("Leader lock held by another instance")
            return False

    def release(self) -> None:
        """Release the leader lock if held."""
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            self._lock_fd = None
            self._is_leader = False
            logger.info("Released leader lock")


class InstanceRegistry:
    """JSON-file based instance registry for service discovery.

    Each running instance registers itself so that peers and operators
    can enumerate running instances.  Stale entries (whose PID no
    longer exists) are pruned on each read.
    """

    def __init__(self, state_dir: str = DEFAULT_STATE_DIR) -> None:
        self._state_dir = Path(state_dir)
        self._instances_path = self._state_dir / INSTANCES_FILE

    def register(self, info: InstanceInfo) -> None:
        """Register or update an instance entry."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        instances = self._load()
        instances[info.instance_id] = info.to_dict()
        self._save(instances)
        logger.info("Registered instance %s", info.instance_id)

    def deregister(self, instance_id: str) -> None:
        """Remove an instance entry."""
        instances = self._load()
        instances.pop(instance_id, None)
        self._save(instances)
        logger.info("Deregistered instance %s", instance_id)

    def list_instances(self) -> list[InstanceInfo]:
        """Return all live instances, pruning stale entries."""
        instances = self._load()
        live: dict[str, dict[str, Any]] = {}
        for iid, data in instances.items():
            pid = data.get("pid", 0)
            if pid and _pid_exists(pid):
                live[iid] = data
        if len(live) != len(instances):
            self._save(live)
        return [InstanceInfo.from_dict(d) for d in live.values()]

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._instances_path.exists():
            return {}
        try:
            text = self._instances_path.read_text()
            result: dict[str, dict[str, Any]] = json.loads(text)
            return result
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, instances: dict[str, dict[str, Any]]) -> None:
        self._instances_path.write_text(json.dumps(instances, indent=2))


def _pid_exists(pid: int) -> bool:
    """Check whether a process with the given PID exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it
        return True
    return True


class GracefulShutdown:
    """Manages graceful shutdown for the API server.

    Installs signal handlers for SIGTERM and SIGINT that:
    1. Set readiness to ``False`` (stop receiving new requests from LB).
    2. Wait for a configurable drain period.
    3. Trigger uvicorn shutdown.
    """

    def __init__(
        self,
        readiness: ReadinessState,
        drain_seconds: float = 5.0,
    ) -> None:
        self._readiness = readiness
        self._drain_seconds = drain_seconds
        self._shutting_down = False
        self._original_handlers: dict[int, Any] = {}

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def install(self) -> None:
        """Install signal handlers for SIGTERM and SIGINT."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        logger.info(
            "Graceful shutdown handlers installed (drain=%ss)",
            self._drain_seconds,
        )

    def _handle_signal(self, signum: int, frame: Any) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, starting graceful shutdown", sig_name)
        self._readiness.set_not_ready()

        # Re-raise with original handler after drain period
        original = self._original_handlers.get(signum)
        if callable(original):
            signal.signal(signum, original)
            os.kill(os.getpid(), signum)
