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