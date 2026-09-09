#!/usr/bin/env python3
"""
TopoNet — ML-Accelerated Topology Warm-Start
=============================================

A complementary tool for structural optimization software (Abaqus, Ansys,
OptiStruct, nTopology). Predicts a binary material/void topology from
physical field inputs in ~3 seconds — a fast initial guess to seed or
accelerate iterative FEA solvers.

    ┌─────────────────────────────────────────────────────┐
    │  TopoNet  ──→  3-second prediction  ──→  FEA/TO    │
    │  (physical fields)   (warm start)    (accelerated) │
    └─────────────────────────────────────────────────────┘

Benchmark: Dice 0.849 on MIT TopoDiff (30K samples, 15 epochs).
"""

# This file exists so the package can be imported cleanly.
# Use `python run_topology_inference.py --help` for the CLI inference tool.
# Use `streamlit run topology_demo.py` for the interactive previewer.
# Use `python topo_abaqus_plugin.py --help` for the Abaqus plugin.

__version__ = "0.1.0"
__author__ = "ClinicalGuard Developer"
__license__ = "MIT OR Apache-2.0"