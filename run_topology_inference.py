#!/usr/bin/env python3
"""
Nexus-Forge Topology Segmenter — Standalone inference CLI.

Segments binary topology (material/void) from physical field images using
the fine-tuned HoVerNet ONNX model. Can run on CPU or GPU.

Usage:
  python run_topology_inference.py input.npz -o output.png
  python run_topology_inference.py input.png -o output.png --resize 256
  python run_topology_inference.py --demo --dice 0.7

Input:
  * .npz containing 'image' key (256,256,3) uint8 [0,255] HWC
  * Any image file (resized to 256x256, converted to 3-channel)

Output:
  * Binary segmentation overlay saved as PNG
  * .npz with 'np_probs' (164,164) float32 foreground probability
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).parent / "models" / "hovernet_topology_seg.onnx"
META_PATH = Path(__file__).parent / "models" / "hovernet_topology_meta.json"


def load_model(path=None):
    path = Path(path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return sess


def preprocess_image(img):
    """Convert HWC uint8 [0,255] (256,256,3) to NCHW float32 (1,3,256,256)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[0] == 3 and img.ndim == 3 and img.shape[1] == 256:
        # Already CHW
        x = img.astype(np.float32)
    else:
        img = img.astype(np.float32)
        x = np.transpose(img, (2, 0, 1))
    if x.shape != (1, 3, 256, 256):
        x = x.reshape(1, 3, 256, 256)
    return np.ascontiguousarray(x)


def postprocess(np_logits):
    """Raw logits (1,2,164,164) → foreground probability (164,164)."""
    probs = np.exp(np_logits - np_logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[0, 1]  # foreground class


def predict(sess, x):
    inputs = sess.get_inputs()
    name = inputs[0].name
    outs = sess.run(None, {name: x})
    return outs


def load_input(path):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        for key in ["image", "images"]:
            if key in data:
                img = data[key]
                if img.ndim == 4:
                    img = img[0]
                return img
        raise ValueError("No 'image' key in npz")
    else:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB").resize((256, 256)))
            return img
        except ImportError:
            import cv2
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            return img


def save_output(np_probs, out_path, img_input=None):
    out_path = Path(out_path)
    # Save npz
    np.savez(out_path.with_suffix(".npz"), np_probs=np_probs.astype(np.float32))
    # Save PNG overlay
    mask = (np_probs > 0.5).astype(np.uint8) * 255
    try:
        from PIL import Image
        Image.fromarray(mask).resize((256, 256), Image.NEAREST).save(out_path.with_suffix(".png"))
    except ImportError:
        import cv2
        cv2.imwrite(str(out_path.with_suffix(".png")), mask)
    return out_path.with_suffix(".npz"), out_path.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description="Nexus-Forge Topology Segmentation")
    parser.add_argument("input", nargs="?", help="Input npz or image")
    parser.add_argument("-o", "--output", default="topology_output", help="Output base name")
    parser.add_argument("--model", default=None, help="Alternative ONNX model path")
    parser.add_argument("--info", action="store_true", help="Show model info and exit")
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic demo image")
    args = parser.parse_args()

    sess = load_model(args.model)

    if args.info:
        meta = {}
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
        print(json.dumps(meta, indent=2))
        for i in sess.get_inputs():
            print(f"Input:  {i.name} {i.shape} {i.type}")
        for o in sess.get_outputs():
            print(f"Output: {o.name} {o.shape} {o.type}")
        return

    if args.demo:
        # Synthetic sinusoid pattern (void/material)
        yy, xx = np.mgrid[0:256, 0:256]
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        pattern = ((np.sin(xx / 20) * np.cos(yy / 25)) > 0.5).astype(np.uint8)
        img[..., 0] = pattern * 200 + 30
        img[..., 1] = pattern * 100 + 50
        img[..., 2] = (1 - pattern) * 180 + 40
        print("Running demo on synthetic pattern...")
    elif args.input:
        img = load_input(args.input)
    else:
        parser.print_help()
        return

    x = preprocess_image(img)
    t0 = time.time()
    outs = predict(sess, x)
    dt = time.time() - t0
    np_logits = outs[0]
    np_probs = postprocess(np_logits)

    fg = (np_probs > 0.5).mean()
    print(f"Inference: {dt*1000:.0f}ms | foreground={fg*100:.1f}% | prob range=[{np_probs.min():.3f}, {np_probs.max():.3f}]")

    npz_path, png_path = save_output(np_probs, args.output)
    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()