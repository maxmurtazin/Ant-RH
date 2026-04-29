# Operator PDE Discovery Report

## Inputs

```json
{
  "operator_path_used": "/Users/machome/Documents/GitHub/Ant-RH/runs/artin_operator.npy",
  "operator_requested": "/Users/machome/Documents/GitHub/Ant-RH/runs/artin_operator_structured.npy",
  "operator_fallback": "/Users/machome/Documents/GitHub/Ant-RH/runs/artin_operator.npy",
  "n": 64,
  "seed_used": 42,
  "optional_files": {
    "runs/artin_structured_spectrum.csv": false,
    "runs/artin_structured_report.json": false,
    "runs/operator_symbolic.json": false,
    "runs/operator_stability_report.json": true,
    "runs/v12_config_used.yaml": true
  }
}
```

## Eigenpair quality

```json
{
  "eigh_report": {
    "input_shape": [
      64,
      64
    ],
    "dtype": "float64",
    "stabilize_input_shape": [
      64,
      64
    ],
    "stabilize_dtype": "float64",
    "stabilize_nan_count": 0,
    "stabilize_inf_count": 0,
    "stabilize_symmetry_error_before": 0.0,
    "stabilize_fro_norm_before": 298.7368748278515,
    "stabilize_symmetry_error_after": 0.0,
    "stabilize_normalization": "frobenius",
    "stabilize_normalization_scale": 298.7368680983233,
    "stabilize_fro_norm_after": 0.9999999886750988,
    "stabilize_gershgorin_min": -0.3017920891585181,
    "stabilize_gershgorin_max": 9.536472385190953e-06,
    "stabilize_diagonal_shift": 1e-06,
    "stabilize_jitter": 1e-08,
    "stabilize_condition_proxy_before": 3190867.5863290234,
    "stabilize_condition_proxy_after": 3190867.5863290234,
    "stabilize_diagonal_shift_effective": 1e-06,
    "stabilize_stabilization_time_s": 0.00242329200000313,
    "stabilize_backend": "numpy",
    "n": 64,
    "requested_k": null,
    "eigh_backend_used": "numpy",
    "eigh_success": true,
    "num_repeated_eigenvalues_estimate": 0,
    "eig_time_s": 0.0038552920000256563
  },
  "eig_min": -0.15403723886611298,
  "eig_max": -0.1324971479733483,
  "eig_std": 0.0059468703811283805
}
```

## Candidate term library

- psi
- laplacian_psi
- rbf_kernel_psi
- inv_distance_kernel_psi
- potential_psi
- density_psi
- mean_length_psi
- std_length_psi
- psi_cubed
- abs_psi_times_psi
- inv_im_psi
- log_im_psi

## Selected equation

```tex
H\psi \approx +15.1708\,V(z)\psi -0.0923697\,\bar{\ell}\,\psi +0.293411\,y^{-1}\psi +0.179639\,\log(y)\psi
```

## Coefficients

```json
[
  {
    "term": "psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "laplacian_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "rbf_kernel_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "inv_distance_kernel_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "potential_psi",
    "coefficient": 15.170815904399689,
    "abs_coefficient": 15.170815904399689,
    "active": true
  },
  {
    "term": "density_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "mean_length_psi",
    "coefficient": -0.09236974905410877,
    "abs_coefficient": 0.09236974905410877,
    "active": true
  },
  {
    "term": "std_length_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "psi_cubed",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "abs_psi_times_psi",
    "coefficient": 0.0,
    "abs_coefficient": 0.0,
    "active": false
  },
  {
    "term": "inv_im_psi",
    "coefficient": 0.2934110900747837,
    "abs_coefficient": 0.2934110900747837,
    "active": true
  },
  {
    "term": "log_im_psi",
    "coefficient": 0.17963895410646252,
    "abs_coefficient": 0.17963895410646252,
    "active": true
  }
]
```

## Fit quality

```json
{
  "mse": 3.453145757331714e-07,
  "normalized_error": 0.0010713967177783843,
  "active_terms": 4,
  "score": 0.005071396717778384,
  "k_requested": 32,
  "k_used": 32
}
```

## Interpretation

LLM interpretation unavailable.

## Limitations

- Regression is performed on reconstructed geometry, not exact stored z-points.
- Feature library is finite and may miss true operator components.
- Shared coefficients across eigenstates impose a strong model bias.

## Next steps

- Compare fits across alternative kernels and Laplacian constructions.
- Condition on spectral windows rather than global first-k stacking.
- Validate discovered terms on held-out eigenpairs.
