# Fixed-owner photometry pilot mathematics

Experimental helpers for controlled illumination ablations, separate from Studio
and the dataset exporter. They do not fit fields, perform registration, change
ownership, or mark data accepted for training. The production pipeline is not
modified.

`render_fixed_crop` replays the original 3×3 transforms and unsigned ownership map
into three bounded float32 RGB crops: raw observation, shared-RGB spatial gain
before warp, and the same corrected observation with scalar exposure gain. No
sharpening, deblurring, defringing, or feathering is performed. Float crops retain
intensities outside the uint8 display range; display clipping must be reported
separately. `sensor_gain_roi` samples a positive sensor-coordinate gain map using
pixel-center scaling. `native_box` adapts NumPy integers at the caller boundary
without weakening strict backend ROI validation. `make_preview` is only a proxy
overview, never a native clarity measurement.

Native crops/field ROIs are capped at 1,048,576 pixels; overview ownership is
capped at 2,000,000 pixels. Real camera sources must already satisfy the bounded
FOV/validated-manifest contracts. These helpers are for trusted experiment
manifests, not HTTP payloads or arbitrary gigapixel source files.

Validation: synthetic identity replay, fixed-owner separation without blending,
float preservation over 255, strict coordinate adaptation, and rejection before
large ROI allocation. Real-data P0 replay must match the original mosaic crop
before interpreting P1/P2. Geometry acceptance and histology validation remain
independent requirements.

Scientific references: Peng et al. 2017, https://doi.org/10.1038/ncomms14836;
implementation https://github.com/peng-lab/BaSiCPy. BaSiCPy 2.0 uses a Torch
backend; do not assume the older JAX API or equate a median/Gaussian field with
BaSiC. Sensor-mode/experiment stratification, held-out evaluation, clipping,
convergence and color drift must be recorded by the experiment driver. Applying
a luminance-derived shared RGB gain is a conservative experimental adaptation,
not physical calibration of gamma or a measured dark field.
