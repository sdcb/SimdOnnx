"""ONNX Runtime reference for multi-input models (bench `--micro multi`).

usage: python -I ort_ref_multi.py <model.onnx> <out_dir> <shape0;shape1;..> [image]

Writes in<k>.f32 per graph input and ref.f32 (ORT CPU, graph optimizations off). With an image
(any format Pillow reads) the first rank-4 3-channel input gets it as RGB in [0,1] resized to its
shape (bilinear), and every other 3-channel input a box-filtered thumbnail (DocGainNet's image /
thumb pair);
otherwise inputs are smooth random values in [0,1].
"""
import os
import sys

import numpy as np
import onnxruntime as ort

model, out, shapes = sys.argv[1:4]
dims = [[int(d) for d in s.split("x")] for s in shapes.split(";")]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
src = None
if len(sys.argv) > 4:
    from PIL import Image
    src = Image.open(sys.argv[4]).convert("RGB")
xs = []
for k, d in enumerate(dims):
    if src is not None and len(d) == 4 and d[1] == 3:
        im = np.asarray(src.resize((d[3], d[2]), Image.BILINEAR if k == 0 else Image.BOX), dtype=np.float32) / 255.0
        x = np.ascontiguousarray(im.transpose(2, 0, 1)[None])
    else:
        x = np.clip(rng.normal(0.5, 0.2, d), 0, 1).astype(np.float32)
    x.tofile(os.path.join(out, f"in{k}.f32"))
    xs.append(x)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
y = s.run(None, {i.name: x for i, x in zip(s.get_inputs(), xs)})[0]
print("ort output", y.shape, y.dtype)
np.ascontiguousarray(y, dtype=np.float32).tofile(os.path.join(out, "ref.f32"))
