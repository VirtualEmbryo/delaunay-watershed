"""Benchmark harness for dw3d mesh reconstruction.

Measurement-only: every function here reads state already produced by
`MeshReconstructionAlgorithm`, or recomputes an existing pure stage to expose
intermediate counts the public API does not store. No dw3d algorithm behaviour
is changed by this package.
"""
