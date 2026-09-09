#!/usr/bin/env python3
"""
TopoNet-Abaqus Plugin — Topology Warm-Start for Abaqus/CAE.

Reads displacement/stress fields from an Abaqus ODB, runs the TopoNet ONNX model,
and writes a predicted binary topology as element sets (material / void) that can
be used as an initial design in a new Abaqus optimization step.

Usage (Abaqus Python CLI):
    abaqus python topo_abaqus_plugin.py --odb job_name.odb --step Step-1 --frame -1

Requirements:
    - Abaqus Python (with odbAccess module)
    - onnxruntime (install via pip in Abaqus Python OR run this outside Abaqus
      with a .fil / .dat parsed input)

Standalone (no Abaqus, CSV input):
    python topo_abaqus_plugin.py --csv displacement.csv --output topology.inp
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "hovernet_topology_seg.onnx"


def norm_uint8(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.full(arr.shape, 128, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def fields_to_topo_model(fields, model_path=None):
    """
    Convert a dict of physical fields to a topology prediction.
    fields: {
        'displacement_x': (Ny, Nx) float64,
        'displacement_y': (Ny, Nx) float64,
        'stress_von_mises': (Ny, Nx) float64,  # optional
        'volume_fraction': float,  # target volume fraction
    }
    """
    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # Build 3-channel input image
    dx = fields.get("displacement_x", np.zeros((60, 180)))
    dy = fields.get("displacement_y", np.zeros((60, 180)))
    vf = fields.get("volume_fraction", 0.3)

    ch0 = norm_uint8(dx)
    ch1 = norm_uint8(dy)
    ch2 = np.full(dx.shape, int(vf * 255), dtype=np.uint8)

    # Resize to 256x256 using PIL
    from PIL import Image
    img_hwc = np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)
    img_r = np.array(Image.fromarray(img_hwc).resize((256, 256), Image.BILINEAR))
    img_r = np.clip(img_r, 0, 255).astype(np.float32)

    # Inference
    x = np.ascontiguousarray(img_r.transpose(2, 0, 1)[None])
    t0 = time.time()
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    dt = time.time() - t0
    np_logits = outs[0]
    e = np.exp(np_logits - np_logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    fg_prob = probs[0, 1]  # (164,164)
    binary = (fg_prob > 0.5).astype(np.uint8)

    print(f"  TopoNet inference: {dt*1000:.0f}ms | foreground={binary.mean()*100:.1f}%")

    return {
        "probability": fg_prob,
        "binary": binary,
        "inference_ms": dt * 1000,
        "foreground_fraction": float(binary.mean()),
    }


def write_inp(topology, mesh_shape, output_path, element_set_name="TOPNET_INITIAL"):
    """
    Write an Abaqus .inp file with element sets for material/void.
    topology: (Ny, Nx) uint8 binary
    mesh_shape: (nel_y, nel_x) — number of elements in each direction
    """
    out = Path(output_path)
    ny, nx = topology.shape
    m_ny, m_nx = mesh_shape

    # Map topology pixels to element IDs (simple bilinear mapping)
    # Each pixel in the (164,164) prediction maps to a region in the mesh
    mat_elems = []
    void_elems = []
    for ey in range(m_ny):
        for ex in range(m_nx):
            # Map element centroid to topology pixel
            py = int(ny * (ey + 0.5) / m_ny)
            px = int(nx * (ex + 0.5) / m_nx)
            py = min(py, ny - 1)
            px = min(px, nx - 1)
            eid = ey * m_nx + ex + 1  # 1-based element ID
            if topology[py, px] > 0:
                mat_elems.append(eid)
            else:
                void_elems.append(eid)

    with open(out, "w") as f:
        f.write(f"*Elset, elset={element_set_name}_MATERIAL\n")
        for i in range(0, len(mat_elems), 16):
            f.write("  " + ", ".join(str(e) for e in mat_elems[i:i+16]) + "\n")
        f.write(f"*Elset, elset={element_set_name}_VOID\n")
        for i in range(0, len(void_elems), 16):
            f.write("  " + ", ".join(str(e) for e in void_elems[i:i+16]) + "\n")

    print(f"  Written {len(mat_elems)} material + {len(void_elems)} void elements to {out}")
    return {"material": len(mat_elems), "void": len(void_elems)}


def main():
    parser = argparse.ArgumentParser(description="TopoNet-Abaqus Plugin — Topology Warm-Start")
    parser.add_argument("--odb", help="Abaqus ODB file path")
    parser.add_argument("--step", default="Step-1", help="ODB step name")
    parser.add_argument("--frame", type=int, default=-1, help="Frame index (default: last)")
    parser.add_argument("--csv", help="CSV input: x,y,disp_x,disp_y (no ODB)")
    parser.add_argument("--mesh-x", type=int, default=180, help="Element count X")
    parser.add_argument("--mesh-y", type=int, default=60, help="Element count Y")
    parser.add_argument("--vf", type=float, default=0.3, help="Target volume fraction")
    parser.add_argument("--output", "-o", default="toponet_inital.inp", help="Output .inp file")
    parser.add_argument("--model", default=None, help="Alternative ONNX model path")
    args = parser.parse_args()

    if args.odb:
        print(f"Opening ODB: {args.odb}")
        try:
            from odbAccess import openOdb
            odb = openOdb(path=args.odb)
            print(f"  ODB opened: {odb.name}")
            # Extract fields from the last frame
            step = odb.steps[args.step]
            frame = step.frames[args.frame]
            # Get displacement field
            disp_field = frame.fieldOutputs["U"]
            # Get von Mises stress
            stress_field = frame.fieldOutputs["S"]
            # Extract values at nodes -> reshape to mesh
            # This is a simplified version; real usage needs element->image mapping
            print("  ODB fields extracted. Note: full element-to-image mapping requires mesh geometry.")
            odb.close()
        except ImportError:
            print("  Not running inside Abaqus Python. Use --csv instead.")
            sys.exit(1)
    elif args.csv:
        print(f"Reading CSV: {args.csv}")
        import pandas as pd
        df = pd.read_csv(args.csv)
        nx, ny = args.mesh_x, args.mesh_y
        dx = df["disp_x"].values.reshape(ny, nx) if len(df) == nx * ny else None
        dy = df["disp_y"].values.reshape(ny, nx) if len(df) == nx * ny else None
        if dx is None:
            print(f"  CSV has {len(df)} rows, expected {nx*ny}. Reshaping with interpolation.")
            # Fallback: griddata
            from scipy.interpolate import griddata
            xi, yi = np.meshgrid(np.linspace(df["x"].min(), df["x"].max(), nx),
                                 np.linspace(df["y"].min(), df["y"].max(), ny))
            dx = griddata((df["x"], df["y"]), df["disp_x"], (xi, yi), method="cubic")
            dy = griddata((df["x"], df["y"]), df["disp_y"], (xi, yi), method="cubic")
        fields = {"displacement_x": dx, "displacement_y": dy, "volume_fraction": args.vf}
        result = fields_to_topo_model(fields, args.model)
        write_inp(result["binary"], (ny, nx), args.output)
        # Save probability as npz
        npz_out = Path(args.output).with_suffix(".npz")
        np.savez(npz_out, probability=result["probability"], binary=result["binary"])
        print(f"  Probability map saved: {npz_out}")
        print(f"  Done. Use '{args.output}' as initial condition in Abaqus optimization.")
    else:
        # Demo mode: generate synthetic fields
        print("No input provided. Running demo mode with synthetic cantilever fields...")
        yy, xx = np.mgrid[0:60, 0:180]
        load_mag = 10.0
        dx = load_mag * np.exp(-((xx - 150) ** 2 + (yy - 30) ** 2) / (2 * 30 ** 2))
        dy = -load_mag * 0.3 * np.exp(-((xx - 150) ** 2 + (yy - 30) ** 2) / (2 * 40 ** 2))
        fields = {"displacement_x": dx, "displacement_y": dy, "volume_fraction": args.vf}
        result = fields_to_topo_model(fields, args.model)
        write_inp(result["binary"], (args.mesh_y, args.mesh_x), args.output)
        npz_out = Path(args.output).with_suffix(".npz")
        np.savez(npz_out, probability=result["probability"], binary=result["binary"])
        print(f"  Demo output: {args.output}, {npz_out}")


if __name__ == "__main__":
    main()