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

If VAST_INSTANCE_ID or VAST_DESTROY_KEY is missing, self-destruct is disabled
but the watchdog still runs and logs exactly what it WOULD have done. NOTE:
because this process is not container PID 1, exiting does NOT stop the box or
its billing; with self-destruct disabled the only backstops are the hermes-side
compute steward (never-healthy / TTL / unproductive destroys) and the operator.
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

# vast.ai injects the instance id as CONTAINER_ID inside every rented box, so
# the provision env can set VAST_INSTANCE_ID=$CONTAINER_ID (resolved on the
# box). Fall back to CONTAINER_ID directly if VAST_INSTANCE_ID did not resolve.
VAST_INSTANCE_ID = (
    os.environ.get("VAST_INSTANCE_ID", "").strip() or os.environ.get("CONTAINER_ID", "").strip()
)
VAST_DESTROY_KEY = os.environ.get("VAST_DESTROY_KEY", "").strip()
IDLE_MINUTES = float(os.environ.get("IDLE_MINUTES", "10"))
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

DESTROY_ENABLED = bool(VAST_INSTANCE_ID and VAST_DESTROY_KEY)


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

    activity_counter prefers vllm:request_success_total, falls back to
    vllm:generation_tokens_total, and finally falls back to a level proxy of
    running plus waiting requests when neither counter is exposed. Returns
    (None, 0.0) on any read/parse failure so a transient scrape error never
    trips a false idle/destroy decision.
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

    if "vllm:request_success_total" in metrics:
        return metrics["vllm:request_success_total"], running
    if "vllm:generation_tokens_total" in metrics:
        return metrics["vllm:generation_tokens_total"], running
    return running + waiting, running


def destroy_once() -> bool:
    """Call the vast.ai destroy API once. Returns True on a 2xx response."""
    url = f"https://console.vast.ai/api/v0/instances/{VAST_INSTANCE_ID}/"
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
        with urllib.request.urlopen(req, timeout=10) as resp:
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

    if not DESTROY_ENABLED:
        LOG.warning(
            "self-destruct is DISABLED (VAST_INSTANCE_ID and/or VAST_DESTROY_KEY "
            "not set); this is what I WOULD have destroyed for, continuing to "
            "monitor only"
        )
        return False

    backoff = DESTROY_INITIAL_BACKOFF_SECONDS
    for attempt in range(1, DESTROY_MAX_ATTEMPTS + 1):
        LOG.warning("destroy attempt %s/%s", attempt, DESTROY_MAX_ATTEMPTS)
        if destroy_once():
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

    def start(self) -> None:
        LOG.info("starting vLLM: %s", " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd)  # noqa: S603 - trusted argv from entrypoint

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
            self.start()
            return "restarted"
        LOG.error("vLLM exhausted its restart budget (%s)", RESTART_ATTEMPTS)
        return "dead"

    def terminate(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception as exc:  # noqa: BLE001 - best-effort shutdown
                LOG.warning("terminating vLLM failed: %s", exc)


def destroy_and_exit(sup: VllmSupervisor, reason: str) -> None:
    """Tear the vLLM child down, destroy this instance (retrying until it dies),
    then exit. If self-destruct is disabled, exit anyway after logging loudly;
    exiting does not stop the box (not PID 1), so the hermes-side compute
    steward is the remaining backstop in that mode."""
    sup.terminate()
    while not request_destroy(reason):
        if not DESTROY_ENABLED:
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
    if not DESTROY_ENABLED:
        LOG.warning(
            "VAST_INSTANCE_ID and/or VAST_DESTROY_KEY are not set at startup; "
            "self-destruct is DISABLED for this run, watchdog will only log "
            "what it would do"
        )

    sup = VllmSupervisor(vllm_cmd)
    sup.start()

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
            continue

        now = time.time()
        try:
            activity_value, running = read_activity()
        except Exception as exc:  # noqa: BLE001 - poll loop must never crash
            LOG.error("unexpected error reading activity, skipping this poll: %s", exc)
            continue

        if activity_value is not None:
            if last_activity_value is not None and activity_value > last_activity_value:
                last_activity_time = now
                LOG.info("activity seen, idle clock reset")
            last_activity_value = activity_value

        if running and running > 0:
            last_activity_time = now
            LOG.info("activity seen, idle clock reset")

        idle_minutes = (now - last_activity_time) / 60.0
        uptime_hours = (now - boot_time) / 3600.0

        if idle_minutes >= IDLE_MINUTES:
            destroy_and_exit(
                sup, f"idle for {idle_minutes:.1f} minutes (limit {IDLE_MINUTES})"
            )

        if uptime_hours >= TTL_HOURS:
            destroy_and_exit(
                sup, f"TTL reached: {uptime_hours:.1f} hours alive (limit {TTL_HOURS})"
            )


if __name__ == "__main__":
    main()
