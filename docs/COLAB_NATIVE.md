# Native search windows through Colab CLI

`tools/colab_native.py` runs a CPU collector for each source owner, using the
normal interactive terminal Drive mount. It copies and verifies one original
RGB TIFF and its capture TIFF at a time, reads only bounded windows, and returns
lossless PNG crops with pixel digests and original native coordinates. It does
not register, recolor, normalize, sharpen, score focus, or accept training pairs.

The workorder has `groups`, `candidates`, `trainingAccepted: false` and
`pixel_ground_truth: false`. Each group records native dimensions, patient and
split, original mosaic/capture SHA256, and a contained Drive-relative directory
under `source_reference`. Each candidate records a full 3×3 moving-native to
fixed-native transform and `ROI_search_windows`. A window records its integer
fixed box and the inverse-mapped moving support, with the same 2/3-pixel halo as
`inspect_native_pair`. The collector recomputes support before any source I/O.
Only train/validation sources of the same patient and split can be combined.

Local settings JSON (keep real scientific metadata outside Git):

```json
{
  "workorder_path": "/absolute/private/native_search_workorder.json",
  "output_dir": "/absolute/private/native_transfers",
  "drive_destination": "project/native_search_v1",
  "accounts": [
    {"profile": "example-profile", "email": "example@example.test"}
  ]
}
```

Run in the user's terminal:

```bash
python3 tools/colab_native.py --config /absolute/private/settings.json
```

For each owner the tool verifies the CLI email, allocates a unique CPU session,
probes the real kernel and required libraries, and invokes `drivemount`
interactively. Keep the terminal open: the user grants consent on the matching
Google account, then the checked file/module child starts immediately. No OAuth
URL or credentials are captured by this tool. The CLI handles interactive auth.

The child SHA-verifies original TIFFs before decoding. PNGs preserve native
integer pixels and capture values without interpolation. Repeated group/box
requests are deduplicated. The source TIFFs are streamed through disk cache, not
decoded as whole arrays. A window is at most 1024² fixed pixels; the combined
ROI/warp working estimate must be below 32 MiB. The reader also checks actual
tile/ROI allocations. Interpreter/library RAM is additional to the array budget.

The archive and completion record are persisted in the owner's Drive under a
configuration/code digest. Completion is written last. Transfer parts and the
assembled local archive are checked before exclusive local publication. A
verified local owner is reused without another VM/consent on a later invocation.
Each runtime created by this invocation is stopped in `finally`; no unrelated
session is stopped. `NATIVE_COLLECTION_COMPLETE` requires all requested owners
and successful runtime cleanup. A failed transfer or cleanup has its own
`transfer.json`; inspect it before retrying an uncommitted local archive.

The source owner and output directory are trusted job inputs, while filenames,
patient/split consistency, geometry and sizes are validated. Capture masks show
where the camera observed pixels; they are not anatomical correspondence or
supervision masks. Search windows may overlap and are not independent landmarks.
Archive extraction and scientific review are separate checked operations.

Tests run real tiled TIFF reads, projective support checks, lossless pixel
comparison, file/module children without notebook `__file__`, archive integrity,
corrupt transfer rejection and owned-runtime cleanup. CLI transport is simulated;
those tests do not confirm Google consent or a real Colab/Drive execution.
