"""FEM data factory: parametric plate-with-hole elasticity solver + Kirsch check."""
from fieldnet.fem.generate import generate_dataset, sample_theta
from fieldnet.fem.interpolate import compute_sdf_mask, interpolate_to_grid, reference_grid
from fieldnet.fem.kirsch import kirsch_stress_cartesian, kirsch_stress_polar, kirsch_von_mises
from fieldnet.fem.mesh import plate_with_hole_mesh
from fieldnet.fem.solver import ElasticitySolution, solve_plate

__all__ = [
    "generate_dataset",
    "sample_theta",
    "compute_sdf_mask",
    "interpolate_to_grid",
    "reference_grid",
    "kirsch_stress_cartesian",
    "kirsch_stress_polar",
    "kirsch_von_mises",
    "plate_with_hole_mesh",
    "ElasticitySolution",
    "solve_plate",
]
