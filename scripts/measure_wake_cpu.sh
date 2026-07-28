#!/usr/bin/env bash
#
# Measures what the always-on wake word listener actually costs in CPU time.
#
# Takes two samples of utime+stime from /proc/<pid>/stat over a fixed window —
# once with always-listening ON, once with it OFF — and reports the delta as a
# percentage of one core. That number decides whether the audio-path
# optimisations in backlog.md are worth doing.
#
# Usage: ./measure_wake_cpu.sh [window_seconds]   (default 600)

set -euo pipefail

PACKAGE=dev.heyari.ari
WINDOW=${1:-600}

die() { echo "error: $*" >&2; exit 1; }

# No internal pipe: piping this into `grep -q` under `set -o pipefail` lets grep's
# early exit SIGPIPE the helper and fail the whole pipeline despite a match.
adbsh() { local out; out=$(adb shell "$@"); printf '%s\n' "${out//$'\r'/}"; }

device_count=$(adb devices | grep -cw device || true)
[[ $device_count -eq 1 ]] || die "need exactly one device on adb, found $device_count"

resolve_pid() { adbsh pidof "$PACKAGE" | awk '{print $1}'; }

# utime and stime are fields 14 and 15 of /proc/<pid>/stat. The comm field can
# contain spaces and parens, so slice from the last ')' rather than splitting
# the whole line.
read_cpu_ticks() {
    local pid=$1 line rest
    line=$(adbsh cat "/proc/$pid/stat" 2>/dev/null) || return 1
    [[ -n $line ]] || return 1
    rest=${line##*') '}
    local -a f
    read -r -a f <<< "$rest"
    echo $(( f[11] + f[12] ))
}

sample() {
    local label=$1 pid_before pid_after ticks_before ticks_after start end

    pid_before=$(resolve_pid)
    [[ -n $pid_before ]] || die "$PACKAGE is not running — open the app first, then background it"

    ticks_before=$(read_cpu_ticks "$pid_before") \
        || die "cannot read /proc/$pid_before/stat (adb shell lacks readproc?)"
    start=$(date +%s)

    echo "  sampling $label for ${WINDOW}s (pid $pid_before)..."
    local waited=0
    while (( waited < WINDOW )); do
        local step=$(( WINDOW - waited ))
        (( step > 60 )) && step=60
        sleep "$step"
        waited=$(( waited + step ))
        printf '    %ds / %ds\n' "$waited" "$WINDOW"
    done

    end=$(date +%s)
    pid_after=$(resolve_pid)
    [[ $pid_after == "$pid_before" ]] \
        || die "process restarted mid-window (pid $pid_before -> ${pid_after:-gone}) — sample is invalid"

    ticks_after=$(read_cpu_ticks "$pid_after") || die "cannot re-read /proc/$pid_after/stat"

    SAMPLE_TICKS=$(( ticks_after - ticks_before ))
    SAMPLE_ELAPSED=$(( end - start ))
}

model=$(adbsh getprop ro.product.model)
build=$(adbsh getprop ro.build.type)
clk_tck=$(adbsh getconf CLK_TCK)
[[ $clk_tck =~ ^[0-9]+$ ]] || clk_tck=100

echo "device:  $model ($build), CLK_TCK=$clk_tck"

wake_service_running() {
    local dump
    dump=$(adbsh dumpsys activity services "$PACKAGE")
    grep -q 'ServiceRecord.*WakeWordService' <<< "$dump"
}

# Toggling the switch and the service actually appearing in dumpsys are not the
# same instant, so give it a few seconds either way before believing the answer.
await_wake_service() {
    local want=$1
    for _ in 1 2 3 4 5; do
        if wake_service_running; then
            [[ $want == running ]] && return 0
        else
            [[ $want == stopped ]] && return 0
        fi
        sleep 2
    done
    return 1
}

if grep -q 'DEBUGGABLE' <<< "$(adbsh dumpsys package "$PACKAGE")"; then
    echo "note: installed build is debuggable. The native side is -O3 either way (see"
    echo "      cpp/CMakeLists.txt), so only the Kotlin read loop is affected — this"
    echo "      overstates the real cost slightly. Fine for a go/no-go number."
fi

echo
echo "Window 1 of 2 — always-listening ON."
echo "Turn the wake word ON in Ari, background the app, screen off, then press Enter."
read -r

if ! await_wake_service running; then
    adbsh dumpsys activity services "$PACKAGE" | head -20 >&2
    die "WakeWordService is not running — always-listening does not appear to be on"
fi

sample "ON"
on_ticks=$SAMPLE_TICKS
on_elapsed=$SAMPLE_ELAPSED

echo
echo "Window 2 of 2 — always-listening OFF."
echo "Turn the wake word OFF in Ari, background the app again, screen off, then press Enter."
read -r

if ! await_wake_service stopped; then
    die "WakeWordService is still running — always-listening is still on"
fi

sample "OFF"
off_ticks=$SAMPLE_TICKS
off_elapsed=$SAMPLE_ELAPSED

pct() { awk -v t="$1" -v e="$2" -v c="$clk_tck" 'BEGIN { printf "%.2f", (t / c) / e * 100 }'; }

on_pct=$(pct "$on_ticks" "$on_elapsed")
off_pct=$(pct "$off_ticks" "$off_elapsed")
delta_pct=$(awk -v a="$on_pct" -v b="$off_pct" 'BEGIN { printf "%.2f", a - b }')

echo
echo "----------------------------------------"
echo "listening ON   ${on_ticks} ticks over ${on_elapsed}s  = ${on_pct}% of one core"
echo "listening OFF  ${off_ticks} ticks over ${off_elapsed}s  = ${off_pct}% of one core"
echo "cost of always-on listening            = ${delta_pct}% of one core"
echo "----------------------------------------"
echo
echo "Under ~1%: skip the audio micro-optimisations, ship listening modes."
echo "Nearer ~8%: the audio path and the LiteRT swap are worth real work."
