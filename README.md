# sqpnp_simple
[![crate](https://img.shields.io/crates/v/sqpnp_simple.svg)](https://crates.io/crates/sqpnp_simple)

A pure rust implementation of SQPnP. It **was** a simplified version. Here's the original [c++ implementation](https://github.com/terzakig/sqpnp).

## Usage
```rust
use sqpnp_simple::sqpnp_solve;

// p3ds: &[(f64, f64, f64)] 3D points (x, y, z)
// p2ds: &[(f64, f64)] 2D undistorted image points on z=1 plane
let (rvec, tvec) = sqpnp_solve(&p3ds, &p2ds).unwrap();
```

By default, `sqpnp_simple` re-exports `glam` 0.30 and `nalgebra` 0.33 as
`sqpnp_simple::glam` and `sqpnp_simple::nalgebra`. To choose other versions,
disable the default features and select exactly one version of each:

```toml
sqpnp_simple = { version = "0.2", default-features = false, features = ["glam-0-34", "nalgebra-0-35"] }
```

Supported features are `glam-0-30` through `glam-0-34` and `nalgebra-0-33`
through `nalgebra-0-35`. Use the re-exported crates for types passed to or
returned from `sqpnp_simple`.

## Example
```sh
cargo run -r --example random_points
```

## Reference
```
@inproceedings{terzakis2020SQPnP,
  title={A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem},
  author={George Terzakis and Manolis Lourakis},
  booktitle={European Conference on Computer Vision},
  pages={478--494},
  year={2020},
  publisher={Springer International Publishing}
}
```