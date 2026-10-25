# Local cubecl-runtime 0.10.0 patch

Source: crates.io `cubecl-runtime` 0.10.0. Original MIT and Apache-2.0 licenses are
retained alongside this file.

Changed file: `src/client.rs` (`ComputeClient::profile`) only.

Purpose: a panicked autotune candidate is already turned into `CallError` by the
`cubecl-common` dispatch patch. `profile` then unwrapped that error, which killed
the dispatch thread again. The unwraps now return `ProfileError`, so the autotuner
can discard the candidate. This patch is local; it has not been submitted upstream.
