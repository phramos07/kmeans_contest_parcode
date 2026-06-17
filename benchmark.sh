#!/usr/bin/env bash
set -euo pipefail

BINDIR="${BINDIR:-bin}"
RUNS="${RUNS:-10}"
EPSILON="${EPSILON:-1e-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [build|run|check] <sequential|openmp|cuda|all> [--runs N]

Examples:
  $(basename "$0") sequential              # build + run (default 10 runs)
  $(basename "$0") build all               # compile all targets
  $(basename "$0") run openmp --runs 20    # run only, 20 iterations
  $(basename "$0") check openmp            # compare openmp inertia vs sequential
  EPSILON=1e-6 $(basename "$0") check openmp
  RUNS=15 $(basename "$0") cuda            # build + run with 15 iterations
EOF
}

is_mode() {
    case "$1" in
        sequential | openmp | cuda | all) return 0 ;;
        *) return 1 ;;
    esac
}

build_target() {
    local mode="$1"
    make -C "$SCRIPT_DIR" "$mode"
}

run_binary() {
    local mode="$1"
    local binary="$SCRIPT_DIR/$BINDIR/$mode"
    local i
    local elapsed
    local total="0"
    local count=0

    if [[ ! -x "$binary" ]]; then
        echo "error: executable not found: $binary" >&2
        echo "hint: run '$(basename "$0") build $mode' first" >&2
        return 1
    fi

    echo "=== $mode ($RUNS runs) ==="

    for ((i = 1; i <= RUNS; i++)); do
        elapsed="$(/usr/bin/time -p "$binary" 2>&1 >/dev/null | awk '/^real/ {print $2}')"
        if [[ -z "$elapsed" ]]; then
            echo "error: failed to measure elapsed time for run $i" >&2
            return 1
        fi
        printf "run %2d: %ss\n" "$i" "$elapsed"
        total="$(awk -v a="$total" -v b="$elapsed" 'BEGIN { printf "%.6f", a + b }')"
        count=$((count + 1))
    done

    local average
    average="$(awk -v total="$total" -v count="$count" 'BEGIN { printf "%.3f", total / count }')"
    echo "average real time: ${average}s"
    echo
}

run_mode() {
    local mode="$1"
    if [[ "$mode" == "all" ]]; then
        run_binary sequential
        run_binary openmp
        run_binary cuda
    else
        run_binary "$mode"
    fi
}

build_mode() {
    local mode="$1"
    if [[ "$mode" == "all" ]]; then
        build_target all
    else
        build_target "$mode"
    fi
}

get_inertia() {
    local mode="$1"
    local binary="$SCRIPT_DIR/$BINDIR/$mode"
    local inertia

    if [[ ! -x "$binary" ]]; then
        echo "error: executable not found: $binary" >&2
        echo "hint: build '$mode' before running check" >&2
        return 1
    fi

    inertia="$("$binary" 2>/dev/null | awk -F'= ' '/^inertia = / { value=$2 } END { print value }')"
    if [[ -z "$inertia" ]]; then
        echo "error: could not read inertia from '$mode' output" >&2
        echo "hint: program should print a line like 'inertia = 123.456789'" >&2
        return 1
    fi

    echo "$inertia"
}

check_correctness() {
    local parallel_mode="$1"
    local reference_inertia
    local parallel_inertia
    local diff
    local ok

    if [[ "$parallel_mode" == "sequential" ]]; then
        echo "error: check requires a parallel target (openmp or cuda), not sequential" >&2
        return 1
    fi

    echo "=== correctness check ($parallel_mode vs sequential, epsilon=$EPSILON) ==="
    build_mode sequential
    build_mode "$parallel_mode"

    reference_inertia="$(get_inertia sequential)"
    parallel_inertia="$(get_inertia "$parallel_mode")"

    echo "reference (sequential) inertia = $reference_inertia"
    echo "$parallel_mode inertia          = $parallel_inertia"

    ok="$(awk -v ref="$reference_inertia" -v par="$parallel_inertia" -v eps="$EPSILON" \
        'BEGIN { diff = (ref - par); if (diff < 0) diff = -diff; print (diff <= eps) ? 1 : 0 }')"
    diff="$(awk -v ref="$reference_inertia" -v par="$parallel_inertia" \
        'BEGIN { diff = (ref - par); if (diff < 0) diff = -diff; printf "%.6f", diff }')"

    echo "absolute difference            = $diff"

    if [[ "$ok" == "1" ]]; then
        echo "result: parallel version is correct"
        return 0
    fi

    echo "result: parallel version is incorrect"
    return 1
}

ACTION="both"
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        build | run | check)
            ACTION="$1"
            shift
            ;;
        sequential | openmp | cuda | all)
            MODE="$1"
            shift
            ;;
        --runs)
            if [[ $# -lt 2 ]]; then
                echo "error: --runs requires a value" >&2
                usage
                exit 1
            fi
            RUNS="$2"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    usage
    exit 1
fi

if ! is_mode "$MODE"; then
    echo "error: invalid mode '$MODE'" >&2
    usage
    exit 1
fi

if [[ "$RUNS" -lt 10 && "$ACTION" != "check" ]]; then
    echo "warning: RUNS=$RUNS is below the recommended minimum of 10" >&2
fi

case "$ACTION" in
    build)
        build_mode "$MODE"
        ;;
    run)
        run_mode "$MODE"
        ;;
    check)
        check_correctness "$MODE"
        ;;
    both)
        build_mode "$MODE"
        run_mode "$MODE"
        ;;
esac
