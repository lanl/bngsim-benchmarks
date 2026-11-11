#!/usr/bin/env bash
# survey_cluster.sh -- describe a cluster (or any Linux machine) for running
# bngsim-benchmarks there.
#
# Read-only: it runs only commands that report on the system, each one under a
# time limit, and it never needs root. A tool that is missing is reported as
# missing; the survey does not stop.
#
# Run it twice:
#
#   1. on a login node:
#        bash tools/survey_cluster.sh
#   2. on a compute node, inside an allocation (salloc, or however you check a
#      node out):
#        bash tools/survey_cluster.sh
#
# Each run writes one report, by default
#
#   $HOME/bngsim-cluster-survey-<host>-<UTC timestamp>.txt
#
# and prints the same text to the terminal. -o PATH writes the report somewhere
# else. The reports describe local infrastructure (host names, partitions,
# accounts), so they are written outside the repository on purpose: do not
# commit them.
#
# What the answers decide (docs/DESIGN.md):
#   CPU model, sockets, SMT, NUMA     -> the node type timing runs declare
#   frequency governor, turbo         -> how stable a timing ratio can be
#   CPUs this process may use         -> how many workers a node can run
#   glibc                             -> whether the pinned wheels install
#                                        (libroadrunner needs glibc >= 2.28)
#   node-local scratch                -> where compile caches go
#   partitions, QOS, time limits      -> sbatch settings, job packing
#   containers, modules, toolchains   -> how the environment gets onto a node
#   network reachability              -> whether installs work on compute nodes

set -u

OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="${2:?-o needs a path}"; shift 2 ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (try -h)" >&2; exit 2 ;;
  esac
done

HOST_SHORT=$(hostname -s 2>/dev/null || hostname)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -n "$OUT" ] || OUT="$HOME/bngsim-cluster-survey-$HOST_SHORT-$STAMP.txt"

# Seconds any one probe may take. A hung filesystem or an unreachable host
# must not hang the survey.
LIMIT=30
if command -v timeout >/dev/null 2>&1; then HAVE_TIMEOUT=1; else HAVE_TIMEOUT=0; fi

section() { printf '\n===== %s =====\n' "$1"; }

# try 'shell command line': echo it, run it under the time limit, report a
# non-zero exit instead of stopping.
try() {
  printf '$ %s\n' "$1"
  if [ "$HAVE_TIMEOUT" = 1 ]; then
    timeout "$LIMIT" bash -c "$1" 2>&1
  else
    bash -c "$1" 2>&1
  fi
  rc=$?
  if [ "$rc" -eq 124 ] && [ "$HAVE_TIMEOUT" = 1 ]; then
    echo "[timed out after ${LIMIT}s]"
  elif [ "$rc" -ne 0 ]; then
    echo "[exit $rc]"
  fi
  return 0
}

# have PROG: print the version line of PROG, or say it is missing.
have() {
  if command -v "$1" >/dev/null 2>&1; then
    printf '%-12s %s\n' "$1" "$(command -v "$1")"
  else
    printf '%-12s (not found)\n' "$1"
  fi
}

# Print only WHETHER a variable is set, never its value: proxy settings can
# carry credentials.
is_set() {
  for v in "$@"; do
    if [ -n "$(printenv "$v" 2>/dev/null)" ]; then
      printf '%s=<set>\n' "$v"
    else
      printf '%s=<unset>\n' "$v"
    fi
  done
}

show_file() {
  for f in "$@"; do
    [ -r "$f" ] && printf '%s: %s\n' "$f" "$(head -c 200 "$f" | tr '\n' ' ')"
  done
  return 0
}

survey() {
  section "survey"
  echo "script:  bngsim-benchmarks tools/survey_cluster.sh"
  echo "when:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "host:    $(hostname 2>/dev/null)"
  echo "user:    ${USER:-$(id -un 2>/dev/null)}"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "context: inside Slurm job ${SLURM_JOB_ID}"
  else
    echo "context: not inside a Slurm job (a login node, or a node checked out some other way)"
  fi

  section "operating system"
  try 'uname -a'
  try 'grep -E "^(NAME|VERSION|ID|ID_LIKE|PRETTY_NAME)=" /etc/os-release'
  try 'getconf GNU_LIBC_VERSION'
  try 'ldd --version | head -1'
  try 'uptime'

  section "CPU"
  try 'nproc; nproc --all'
  try 'lscpu'
  try 'grep -m1 -E "^microcode" /proc/cpuinfo'
  try 'grep -m1 -o -w -E "avx512f|avx2" /proc/cpuinfo | sort -u'
  echo "-- CPUs this process may run on (cgroup / affinity):"
  try 'taskset -cp $$'
  try 'grep -E "^Cpus_allowed_list" /proc/self/status'
  echo "-- frequency scaling and turbo:"
  show_file /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver \
            /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
            /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq \
            /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq \
            /sys/devices/system/cpu/intel_pstate/no_turbo \
            /sys/devices/system/cpu/cpufreq/boost \
            /sys/devices/system/cpu/smt/active
  try 'numactl --hardware'

  section "memory and limits"
  try 'free -g'
  try 'grep -E "^(MemTotal|MemAvailable|HugePages_Total)" /proc/meminfo'
  try 'ulimit -a'
  echo "-- cgroup limits of this process:"
  try 'cat /proc/self/cgroup'
  try 'cg=$(awk -F: '"'"'$1=="0"{print $3}'"'"' /proc/self/cgroup); for f in cpu.max cpuset.cpus.effective memory.max; do [ -r "/sys/fs/cgroup$cg/$f" ] && echo "$f: $(cat "/sys/fs/cgroup$cg/$f")"; done; true'

  section "filesystems (scratch, home, here)"
  echo "TMPDIR=${TMPDIR:-<unset>}  SCRATCH=${SCRATCH:-<unset>}"
  for d in "${TMPDIR:-}" /tmp /var/tmp /dev/shm /scratch /lscratch /local /tmp/local \
           "${SCRATCH:-}" "$HOME" "$PWD"; do
    [ -n "$d" ] && [ -d "$d" ] && try "df -hT '$d' | tail -n +2"
  done
  try 'quota -s'

  section "Slurm: this job"
  is_set SLURM_JOB_ID
  for v in SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_JOB_ACCOUNT SLURM_JOB_QOS \
           SLURM_JOB_NODELIST SLURM_CPUS_ON_NODE SLURM_JOB_CPUS_PER_NODE \
           SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU; do
    printf '%s=%s\n' "$v" "$(printenv "$v" 2>/dev/null || echo '<unset>')"
  done
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    try 'squeue -h -j "$SLURM_JOB_ID" -o "time limit %l, time left %L, nodes %D, cpus %C"'
    try 'scontrol show job "$SLURM_JOB_ID"'
  fi
  try 'scontrol show node "$(hostname -s)"'

  section "Slurm: the cluster"
  have sbatch
  have salloc
  try 'sinfo --version'
  try 'sinfo -s'
  try 'sinfo -o "%20P %5a %12l %12L %6D %8c %10m %30f %N"'
  try 'scontrol show partition'
  try 'scontrol show config | grep -E -i "^(ClusterName|MaxArraySize|MaxJobCount|MaxStepCount|OverTimeLimit|PreemptMode|PreemptType|SchedulerType|SelectType|SelectTypeParameters|DefMemPerCPU|MaxMemPerCPU|DefMemPerNode|MaxMemPerNode|KillWait|PriorityType|TaskPlugin|ProctrackType) "'
  try 'sacctmgr -nP show assoc user="${USER:-$(id -un)}" format=cluster,account,partition,qos,maxwall,maxjobs,maxsubmit,grptres'
  try 'sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu,maxsubmitpu,priority'

  section "containers and modules"
  for p in apptainer singularity ch-run ch-image podman docker; do have "$p"; done
  try 'apptainer --version || singularity --version'
  try 'ch-run --version'
  if [ -n "${LMOD_CMD:-}" ]; then
    try '"$LMOD_CMD" bash --terse avail 2>&1 | head -400'
  else
    try 'bash -lc "module -t avail" 2>&1 | head -400'
  fi

  section "toolchains and package managers"
  for p in gcc g++ cc c++ clang cmake make git perl python3 pixi uv conda mamba micromamba \
           pip3 curl wget; do have "$p"; done
  try 'gcc --version | head -1'
  try 'cc --version | head -1'
  try 'cmake --version | head -1'
  try 'git --version'
  try 'perl -e "print \"perl \$]\n\""'
  try 'python3 --version'
  try 'pixi --version'
  try 'uv --version'

  section "network"
  is_set http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
  # Header-only requests: any HTTP status (200, 301, 404, ...) means the host is
  # reachable; 000 means it is not. A full GET of a package index downloads tens
  # of megabytes.
  for url in https://github.com \
             https://raw.githubusercontent.com \
             https://pypi.org/simple/bngsim/ \
             https://files.pythonhosted.org \
             https://conda.anaconda.org/conda-forge/noarch/repodata.json \
             https://prefix.dev \
             https://repo.prefix.dev/conda-forge/noarch/repodata.json \
             https://www.ebi.ac.uk/biomodels/; do
    try "curl -sS -I -o /dev/null -m 15 -w '%{http_code} in %{time_total}s' '$url'; echo '  $url'"
  done
  try 'git ls-remote https://github.com/lanl/bngsim-benchmarks HEAD'

  section "done"
  echo "report: $OUT"
}

survey 2>&1 | tee "$OUT"
