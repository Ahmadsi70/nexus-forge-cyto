# Notice

## Copyright
Copyright (c) 2024 ClinicalGuard. All Rights Reserved.

## Third-Party Software
This project incorporates code from the following open-source projects,
each under its own license:

| Component | Project | License |
|-----------|---------|---------|
| Nuclear segmentation architecture | **HoVerNet** (Graham et al.) | MIT |
| ONNX inference runtime | **ONNX Runtime** (Microsoft) | MIT |
| Deep learning framework | **PyTorch** (Linux Foundation) | BSD |
| Rust ONNX inference | **tract** (Tract project) | Apache 2.0 / MIT |
| HTTP API server | **Axum** (Tokio project) | MIT |
| Web framework | **FastAPI** (Sebastián Ramírez) | MIT |
| Interactive dashboard | **Streamlit** (Snowflake) | Apache 2.0 |
| Image processing | **OpenCV** | Apache 2.0 |
| Scientific computing | **NumPy / SciPy** | BSD |
| Gradient boosting | **scikit-learn** | BSD |

## Dataset Acknowledgments

| Dataset | License | Use |
|---------|---------|-----|
| **PanNuke** (Gamper et al.) | CC BY-NC-SA 4.0 | Base model training (non-commercial) |
| **TopoDiff** (MIT) | Synthetic — no restrictions | Topology segmentation fine-tuning |
| **Cantilever** | Synthetic — no restrictions | Topology segmentation validation |
| **Pereira** | Public dataset | Topology segmentation validation |

## Model Weights Notice

The topology segmentation model (`models/hovernet_topology_seg.onnx`) is
trained on synthetic data (TopoDiff) and carries no third-party restrictions.
Users are responsible for complying with any third-party license terms
when using other model weights.

## Contact
ClinicalGuard — https://github.com/Ahmadsi70