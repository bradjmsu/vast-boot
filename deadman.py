#!/usr/bin/env python3
"""Dead-man watchdog + supervisor for a vast.ai vLLM rental.

Money-safety purpose: a forgotten or wedged GPU rental burns cash forever if
nothing ever turns it off. This process runs under vast.ai's onstart launcher
(NOT as container PID 1) and SUPERVISES
vLLM: it spawns the server as a child (the serve command is passed as argv),
monitors it, and destroys ITS OWN vast.ai instance the moment any of these is
true:

  - vLLM dies unexpectedly and cannot be restarted (crash/OOM after health would
    otherwise leave a stopped-but-billing box); one restart is attempted first
  - accepted requests make no prompt or generation token progress; one forced
    restart is attempted first, then a repeated stall destroys the rental
  - the server never becomes healthy within a boot grace window (pure waste)
  - the server has been idle (no new inference activity) past IDLE_MINUTES
  - the instance has been alive past TTL_HOURS, regardless of activity, as a
    hard backstop against a runaway "always busy" rental nobody asked for

Backgrounding the watchdog behind vLLM would be unsafe: a vLLM crash would
kill the watchdog before it could destroy the instance. Supervising vLLM as a
child process
fixes that.

Standard library only (subprocess, urllib, os, sys, time, json, logging) so this
file has no extra pip install and cannot be broken by a dependency drifting
under it.

If VAST_INSTANCE_ID is missing, non-numeric, or VAST_DESTROY_KEY is missing,
self-destruct is disabled but the watchdog still runs and logs exactly what it
WOULD have done. VAST_INSTANCE_ID must be the numeric vast contract id; if the
provision env passes a non-numeric value (e.g. $CONTAINER_ID, issue #3712), the
watchdog attempts to discover the numeric id from the vast API at startup and
logs CRITICAL if it cannot. NOTE: because this process is not container PID 1,
exiting does NOT stop the box or its billing; with self-destruct disabled the
only backstops are the production compute steward and the operator.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

LOG = logging.getLogger("deadman")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("deadman: %(message)s"))
LOG.addHandler(_handler)
LOG.propagate = False

# The provision env used to set VAST_INSTANCE_ID=$CONTAINER_ID, but
# CONTAINER_ID is a docker/container identifier, NOT the numeric vast contract
# id the destroy API requires (issue #3712). The watchdog now validates its id
# at startup and, if the env var is missing or non-numeric, resolves it by
# asking the vast API which instance this box is.
VAST_INSTANCE_ID_RAW = os.environ.get("VAST_INSTANCE_ID", "").strip()
CONTAINER_ID = os.environ.get("CONTAINER_ID", "").strip()
VAST_DESTROY_KEY = os.environ.get("VAST_DESTROY_KEY", "").strip()
IDLE_MINUTES = float(os.environ.get("IDLE_MINUTES", "10"))
STALL_MINUTES = float(os.environ.get("STALL_MINUTES", "5"))
TTL_HOURS = float(os.environ.get("TTL_HOURS", "6"))
BOOT_GRACE_MINUTES = float(os.environ.get("BOOT_GRACE_MINUTES", "15"))
VLLM_HEALTH_URL = os.environ.get("VLLM_HEALTH_URL", "http://127.0.0.1:8000/health")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "30"))
# How many times to restart vLLM after an unexpected exit before giving up and
# destroying the box. Default 1 (one restart, then destroy).
RESTART_ATTEMPTS = int(os.environ.get("VLLM_RESTART_ATTEMPTS", "1"))

HTTP_TIMEOUT_SECONDS = 5
DESTROY_MAX_ATTEMPTS = 5
DESTROY_INITIAL_BACKOFF_SECONDS = 5

# Populated lazily by _get_instance_id(). The raw env var is no longer trusted
# to be the numeric contract id.
_RESOLVED_INSTANCE_ID: str | None = None


def _is_numeric_instance_id(value: str) -> bool:
    """The vast API destroy path accepts only the bare numeric contract id."""
    return bool(value) and value.isdigit()


def _my_public_ip(timeout: float = 5.0) -> str | None:
    """Best-effort public IPv4 of this box. Used to identify ourselves in the
    vast fleet list when CONTAINER_ID is not exposed by the API.

    Uses a no-proxy opener so inherited proxy settings cannot return a proxy's
    IP instead of this box's public IP.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in (
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://ifconfig.me/ip",
    ):
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with opener.open(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace").strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001 - best-effort metadata fetch
            continue
    return None


def _discover_instance_id() -> str | None:
    """Ask the vast API which instance this box is.

    We have the account key (VAST_DESTROY_KEY) but the env did not give us a
    numeric contract id. Identify ourselves by CONTAINER_ID if the API exposes
    it, otherwise by public IP address.
    """
    if not VAST_DESTROY_KEY:
        LOG.critical(
            "VAST_DESTROY_KEY is not set; cannot discover our instance id "
            "from the vast API"
        )
        return None

    try:
        req = urllib.request.Request(
            "https://console.vast.ai/api/v0/instances/",
            method="GET",
            headers={"Authorization": f"Bearer {VAST_DESTROY_KEY}"},
        )
        with _VAST_API_OPENER.open(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        LOG.critical(
            "could not list vast instances to discover our instance id: %s", exc
        )
        return None

    instances = payload.get("instances") or []
    if not isinstance(instances, list):
        LOG.critical(
            "vast instances list returned unexpected shape: %r", type(payload)
        )
        return None

    # Strategy 1: match by the docker/container id vast injected as CONTAINER_ID.
    if CONTAINER_ID:
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            if str(inst.get("container_id") or "").strip() == CONTAINER_ID:
                candidate = str(inst.get("id") or "").strip()
                if _is_numeric_instance_id(candidate):
                    return candidate

    # Strategy 2: match by public IP. This needs outbound network, but the box
    # already needed outbound network to pull weights.
    my_ip = _my_public_ip()
    if my_ip:
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            if str(inst.get("public_ipaddr") or "").strip() == my_ip:
                candidate = str(inst.get("id") or "").strip()
                if _is_numeric_instance_id(candidate):
                    return candidate

    LOG.critical(
        "could not identify this box in the vast fleet list "
        "(tried CONTAINER_ID=%r, public_ip=%r)",
        CONTAINER_ID or "<unset>",
        my_ip or "<unresolved>",
    )
    return None


def _resolve_instance_id() -> str | None:
    """Return the numeric vast contract id for this box, or None.

    A kill path that silently cannot fire is worse than no kill path, so any
    mismatch is logged at CRITICAL.
    """
    if _is_numeric_instance_id(VAST_INSTANCE_ID_RAW):
        return VAST_INSTANCE_ID_RAW

    if VAST_INSTANCE_ID_RAW:
        LOG.critical(
            "VAST_INSTANCE_ID=%r is not a numeric vast instance id; "
            "attempting to discover the correct id from the vast API",
            VAST_INSTANCE_ID_RAW,
        )
    else:
        LOG.info(
            "VAST_INSTANCE_ID is not set; attempting to discover the correct "
            "instance id from the vast API"
        )

    discovered = _discover_instance_id()
    if _is_numeric_instance_id(discovered):
        LOG.info("discovered vast instance id: %s", discovered)
        return discovered

    LOG.critical(
        "could not resolve a numeric vast instance id for this box. "
        "Self-destruct is disabled; the compute steward remains the only "
        "automatic backstop."
    )
    return None


def _get_instance_id() -> str | None:
    """Lazily resolve and cache the numeric instance id."""
    global _RESOLVED_INSTANCE_ID
    if _RESOLVED_INSTANCE_ID is None:
        _RESOLVED_INSTANCE_ID = _resolve_instance_id()
    return _RESOLVED_INSTANCE_ID


def check_health() -> bool:
    """Return True if VLLM_HEALTH_URL answers HTTP 200 right now."""
    try:
        req = urllib.request.Request(VLLM_HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            return resp.status == 200
    except Exception as exc:  # noqa: BLE001 - health poll must never crash the loop
        LOG.debug("health check failed: %s", exc)
        return False


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Sum Prometheus exposition text into base metric name to total value.

    Lines that share a metric name but differ only by labels (for example the
    same counter split by finished_reason) are summed together, since the
    watchdog only cares about the totals.
    """
    sums: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        brace_idx = line.find("{")
        space_idx = line.find(" ")
        if brace_idx != -1 and (space_idx == -1 or brace_idx < space_idx):
            name = line[:brace_idx]
        elif space_idx != -1:
            name = line[:space_idx]
        else:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        sums[name] = sums.get(name, 0.0) + value
    return sums


def read_activity() -> tuple[float | None, float]:
    """Read vLLM metrics and return (activity_counter, currently_running).

    activity_counter prefers the sum of prompt and generation tokens because
    either can move during one long request before request_success_total does.
    It falls back to completed requests, then to a level proxy of running plus
    waiting requests when no monotonic counter is exposed. Returns (None, 0.0)
    on any read/parse failure so a transient scrape error never trips a false
    idle/destroy decision.
    """
    try:
        req = urllib.request.Request(VLLM_METRICS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - metrics poll must never crash the loop
        LOG.warning("metrics read failed, treating as no signal this poll: %s", exc)
        return None, 0.0

    metrics = parse_prometheus_text(text)
    running = metrics.get("vllm:num_requests_running", 0.0)
    waiting = metrics.get("vllm:num_requests_waiting", 0.0)

    token_counters = [
        metrics[name]
        for name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")
        if name in metrics
    ]
    if token_counters:
        return sum(token_counters), running
    if "vllm:request_success_total" in metrics:
        return metrics["vllm:request_success_total"], running
    return running + waiting, running


def requests_stalled(*, running: float, seconds_since_progress: float) -> bool:
    """Return True when accepted requests have made no token progress.

    HTTP 200 health and num_requests_running only prove that vLLM accepted the
    requests. They do not prove useful inference. A wedged H100 held 32 running
    requests for hours with fixed token counters, so running must never reset
    the useful-activity clock by itself.
    """
    return running > 0 and seconds_since_progress >= STALL_MINUTES * 60


def update_stall_started_at(
    current: float | None,
    *,
    running: float,
    progressed: bool,
    now: float,
) -> float | None:
    """Track when the current no-progress running interval began."""
    if running <= 0:
        return None
    if progressed or current is None:
        return now
    return current


def select_watchdog_action(
    *,
    running: float,
    seconds_since_progress: float,
    idle_minutes: float,
    uptime_hours: float,
) -> str | None:
    """Select the next safety action with hard TTL first."""
    if uptime_hours >= TTL_HOURS:
        return "ttl"
    if requests_stalled(
        running=running,
        seconds_since_progress=seconds_since_progress,
    ):
        return "stall"
    if running <= 0 and idle_minutes >= IDLE_MINUTES:
        return "idle"
    return None


# The destroy path is the money-safety backstop and ignores inherited proxy
# settings. Every Vast API call goes direct.
_VAST_API_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def destroy_once(instance_id: str) -> bool:
    """Call the vast.ai destroy API once. Returns True on a 2xx response."""
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
    req = urllib.request.Request(
        url,
        data=json.dumps({}).encode("utf-8"),
        method="DELETE",
        headers={
            "Authorization": f"Bearer {VAST_DESTROY_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _VAST_API_OPENER.open(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            LOG.info("destroy call responded status=%s body=%s", resp.status, body)
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - body read is best effort
            body = ""
        LOG.error("destroy call HTTP error status=%s body=%s", exc.code, body)
        return False
    except Exception as exc:  # noqa: BLE001 - destroy must never crash the loop
        LOG.error("destroy call failed: %s", exc)
        return False


def request_destroy(reason: str) -> bool:
    """Attempt to destroy this instance for the given reason.

    Returns True if the instance was actually destroyed (in which case the
    caller should exit). Returns False if self-destruct is disabled, or if
    every retry attempt failed, in which case the caller keeps looping and
    will try again on the next idle/TTL check. This function never gives up
    silently: a failed destroy is always logged loudly.
    """
    LOG.warning("destroy triggered: %s", reason)

    instance_id = _get_instance_id()
    if not instance_id or not VAST_DESTROY_KEY:
        LOG.warning(
            "self-destruct is DISABLED (numeric VAST_INSTANCE_ID and/or VAST_DESTROY_KEY "
            "not available); this is what I WOULD have destroyed for, continuing to "
            "monitor only"
        )
        return False

    backoff = DESTROY_INITIAL_BACKOFF_SECONDS
    for attempt in range(1, DESTROY_MAX_ATTEMPTS + 1):
        LOG.warning("destroy attempt %s/%s", attempt, DESTROY_MAX_ATTEMPTS)
        if destroy_once(instance_id):
            LOG.warning("instance destroyed successfully, exiting")
            return True
        if attempt < DESTROY_MAX_ATTEMPTS:
            LOG.error("destroy attempt %s failed, retrying in %ss", attempt, backoff)
            time.sleep(backoff)
            backoff *= 2

    LOG.error(
        "all %s destroy attempts failed; vast.ai API may be unreachable. "
        "Will retry again on the next idle/TTL check instead of giving up.",
        DESTROY_MAX_ATTEMPTS,
    )
    return False


class VllmSupervisor:
    """Owns the vLLM child process and its restart budget."""

    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self.restarts_used = 0

    def start(self) -> bool:
        # The command includes the vLLM bearer as an argv value. Never copy the
        # full command into logs or incident transcripts.
        LOG.info("starting vLLM")
        try:
            self.proc = subprocess.Popen(self.cmd)  # noqa: S603 - trusted argv from entrypoint
            return True
        except Exception as exc:  # noqa: BLE001 - watchdog must reach destroy
            self.proc = None
            LOG.error("starting vLLM failed: %s", exc)
            return False

    def check(self) -> str:
        """One liveness check: 'alive', 'restarted' (was dead, respawned), or
        'dead' (exited and restart budget exhausted)."""
        if self.proc is None:
            return "dead"
        rc = self.proc.poll()
        if rc is None:
            return "alive"
        LOG.error("vLLM exited unexpectedly (rc=%s)", rc)
        if self.restarts_used < RESTART_ATTEMPTS:
            self.restarts_used += 1
            LOG.warning("restarting vLLM (attempt %s/%s)", self.restarts_used, RESTART_ATTEMPTS)
            return "restarted" if self.start() else "dead"
        LOG.error("vLLM exhausted its restart budget (%s)", RESTART_ATTEMPTS)
        return "dead"

    def terminate(self) -> bool:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                LOG.warning("vLLM ignored SIGTERM for 10s; force killing it")
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                except Exception as exc:  # noqa: BLE001 - watchdog must reach destroy
                    LOG.error("force killing vLLM failed: %s", exc)
                    return False
            except Exception as exc:  # noqa: BLE001 - best-effort shutdown
                LOG.warning("terminating vLLM failed: %s", exc)
                return False
        return True

    def restart_stalled(self) -> bool:
        """Spend one restart attempt on a live process making no progress."""
        if self.restarts_used >= RESTART_ATTEMPTS:
            LOG.error("vLLM exhausted its restart budget (%s)", RESTART_ATTEMPTS)
            return False
        self.restarts_used += 1
        LOG.warning(
            "restarting stalled vLLM (attempt %s/%s)",
            self.restarts_used,
            RESTART_ATTEMPTS,
        )
        if not self.terminate():
            return False
        return self.start()


def destroy_and_exit(sup: VllmSupervisor, reason: str) -> None:
    """Tear the vLLM child down, destroy this instance (retrying until it dies),
    then exit. If self-destruct is disabled, exit anyway after logging loudly;
    exiting does not stop the box (not PID 1), so the production compute
    steward is the remaining backstop in that mode."""
    sup.terminate()
    while not request_destroy(reason):
        instance_id = _get_instance_id()
        if not instance_id or not VAST_DESTROY_KEY:
            LOG.error(
                "self-destruct is DISABLED; exiting. NOTE this does not stop "
                "the box (not PID 1); the compute steward must reap it"
            )
            sys.exit(1)
        time.sleep(POLL_SECONDS)
    sys.exit(0)


def wait_for_boot_health(sup: VllmSupervisor) -> bool:
    """Poll VLLM_HEALTH_URL until the first HTTP 200, up to BOOT_GRACE_MINUTES,
    restarting vLLM if it dies during boot. Returns True on health, False if the
    grace window expires or vLLM dies past its restart budget during boot."""
    deadline = time.monotonic() + BOOT_GRACE_MINUTES * 60
    attempt = 0
    while time.monotonic() < deadline:
        if sup.check() == "dead":
            LOG.error("vLLM died during boot and exhausted its restart budget")
            return False
        attempt += 1
        if check_health():
            LOG.info("health confirmed on attempt %s", attempt)
            return True
        if attempt == 1 or attempt % 4 == 0:
            LOG.info(
                "still waiting for health at %s (attempt %s)", VLLM_HEALTH_URL, attempt
            )
        time.sleep(POLL_SECONDS)
    return False


def main() -> None:
    vllm_cmd = sys.argv[1:]
    if not vllm_cmd:
        LOG.error(
            "no vLLM command passed as argv; the watchdog is the supervisor and "
            "must be launched as 'deadman.py <vllm serve ...>'"
        )
        sys.exit(2)

    LOG.info(
        "watchdog armed, supervising vLLM, entering boot grace window (%s minutes)",
        BOOT_GRACE_MINUTES,
    )
    # Resolve the numeric instance id NOW, before we need it in an emergency.
    # A kill path that silently cannot fire is worse than no kill path.
    resolved_id = _get_instance_id()
    if not resolved_id:
        LOG.warning(
            "numeric VAST_INSTANCE_ID could not be resolved at startup; "
            "self-destruct is DISABLED for this run, watchdog will only log "
            "what it would do"
        )
    elif not VAST_DESTROY_KEY:
        LOG.warning(
            "VAST_DESTROY_KEY is not set at startup; self-destruct is DISABLED "
            "for this run, watchdog will only log what it would do"
        )
    else:
        LOG.info("self-destruct armed for instance %s", resolved_id)

    sup = VllmSupervisor(vllm_cmd)
    if not sup.start():
        destroy_and_exit(sup, "vLLM could not start; a box that cannot serve is pure waste")

    if not wait_for_boot_health(sup):
        destroy_and_exit(
            sup,
            "vLLM never became healthy within boot grace (or died during boot); a "
            "box that never serves is pure waste",
        )

    now = time.time()
    boot_time = now
    last_activity_time = now
    last_activity_value: float | None = None
    stall_started_at: float | None = None
    LOG.info("idle clock and TTL clock armed at first successful health check")

    while True:
        time.sleep(POLL_SECONDS)

        # 1) vLLM liveness first: a crash/OOM after health must trigger teardown,
        #    not leave a stopped-but-billing box.
        state = sup.check()
        if state == "dead":
            destroy_and_exit(
                sup, "vLLM died after health and exhausted its restart budget"
            )
        if state == "restarted":
            # A restart re-enters warmup; reset the idle clock so we do not
            # idle-kill during the reload, and skip the activity read this tick.
            last_activity_time = time.time()
            last_activity_value = None
            stall_started_at = None
            continue

        now = time.time()
        try:
            activity_value, running = read_activity()
        except Exception as exc:  # noqa: BLE001 - poll loop must never crash
            LOG.error("unexpected error reading activity, skipping this poll: %s", exc)
            continue

        progressed = False
        if activity_value is not None:
            if last_activity_value is None or activity_value > last_activity_value:
                last_activity_time = now
                progressed = True
                LOG.info("activity seen, idle clock reset")
            last_activity_value = activity_value

        stall_started_at = update_stall_started_at(
            stall_started_at,
            running=running,
            progressed=progressed,
            now=now,
        )

        idle_minutes = (now - last_activity_time) / 60.0
        uptime_hours = (now - boot_time) / 3600.0
        seconds_since_progress = (
            now - stall_started_at if stall_started_at is not None else 0.0
        )
        action = select_watchdog_action(
            running=running,
            seconds_since_progress=seconds_since_progress,
            idle_minutes=idle_minutes,
            uptime_hours=uptime_hours,
        )

        if action == "ttl":
            destroy_and_exit(
                sup, f"TTL reached: {uptime_hours:.1f} hours alive (limit {TTL_HOURS})"
            )

        if action == "stall":
            stall_minutes = (now - stall_started_at) / 60.0
            reason = (
                f"{int(running)} running requests made no token progress for "
                f"{stall_minutes:.1f} minutes"
            )
            LOG.error("inference stall detected: %s", reason)
            if not sup.restart_stalled() or not wait_for_boot_health(sup):
                destroy_and_exit(
                    sup,
                    f"{reason}; restart failed or restart budget exhausted",
                )
            last_activity_time = time.time()
            last_activity_value = None
            stall_started_at = None
            LOG.info("stalled vLLM recovered; useful-activity clock reset")
            continue

        if action == "idle":
            destroy_and_exit(
                sup, f"idle for {idle_minutes:.1f} minutes (limit {IDLE_MINUTES})"
            )


if __name__ == "__main__":
    main()
