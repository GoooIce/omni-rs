# Local cubecl-common 0.10.0 patch

Source: crates.io `cubecl-common` 0.10.0. Original MIT and Apache-2.0 licenses are
retained alongside this file.

Purpose: keep the device dispatch thread alive when a task panics. Kernel-expansion
failures inside autotune candidates (cubek-matmul 0.2 on block-quantized operands)
panic on the dispatch thread; upstream lets the thread die, which surfaces as
spurious `CallError` panics for unrelated later work. The local change catches the
panic per task: dropping the task closes its response channel and the caller
receives `CallError`, which the autotuner already treats as "candidate failed".

See VALIDATION.md ("MiniCPM-V-4.7 35B" sections) for the numerical evidence. This
patch is local; it has not been submitted to or accepted upstream.
