//! Benchmarks for EfficientAD operations
//!
//! Run with: cargo bench --bench efficientad_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array2, Array3};
use rand::Rng;

/// Generate random f32 array for testing
fn random_f32_array(h: usize, w: usize, c: usize) -> Array3<f32> {
    let mut rng = rand::thread_rng();
    Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0..1.0))
}

/// Generate random 2D f32 array
fn random_f32_2d(h: usize, w: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((h, w), |_| rng.gen_range(0.0..1.0))
}

/// Generate random binary mask
fn random_mask(h: usize, w: usize, density: f32) -> Array2<u8> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((h, w), |_| {
        if rng.gen::<f32>() < density { 255 } else { 0 }
    })
}

fn bench_percentile_quickselect(c: &mut Criterion) {
    let mut group = c.benchmark_group("percentile");

    for size in [256, 512, 1024].iter() {
        let data = random_f32_2d(*size, *size);
        let values: Vec<f32> = data.iter().cloned().collect();

        group.bench_with_input(
            BenchmarkId::new("quickselect", format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let mut v = values.clone();
                    let idx = (0.95 * v.len() as f32) as usize;
                    v.select_nth_unstable_by(idx, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    black_box(v[idx])
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_sort", format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let mut v = values.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let idx = (0.95 * v.len() as f32) as usize;
                    black_box(v[idx])
                })
            },
        );
    }

    group.finish();
}

fn bench_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");

    for size in [256, 512].iter() {
        let mask = random_mask(*size, *size, 0.3);

        group.bench_with_input(
            BenchmarkId::new("union_find", format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate Union-Find with path compression
                    let height = mask.dim().0;
                    let width = mask.dim().1;
                    let mut parent: Vec<i32> = (0..(height * width) as i32).collect();

                    fn find(parent: &mut [i32], x: i32) -> i32 {
                        if parent[x as usize] != x {
                            parent[x as usize] = find(parent, parent[x as usize]);
                        }
                        parent[x as usize]
                    }

                    let mut union_count = 0usize;
                    for y in 0..height {
                        for x in 0..width {
                            if mask[[y, x]] > 0 {
                                let idx = (y * width + x) as i32;
                                if y > 0 && mask[[y-1, x]] > 0 {
                                    let top = ((y-1) * width + x) as i32;
                                    let r1 = find(&mut parent, idx);
                                    let r2 = find(&mut parent, top);
                                    if r1 != r2 {
                                        parent[r1 as usize] = r2;
                                        union_count += 1;
                                    }
                                }
                            }
                        }
                    }
                    black_box(union_count)
                })
            },
        );
    }

    group.finish();
}

fn bench_jet_colormap(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet_colormap");

    for size in [256, 512, 1024].iter() {
        let heatmap = random_f32_2d(*size, *size);

        group.bench_with_input(
            BenchmarkId::new("apply_colormap", format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let (h, w) = heatmap.dim();
                    let mut result = Array3::<u8>::zeros((h, w, 3));

                    for y in 0..h {
                        for x in 0..w {
                            let v = heatmap[[y, x]].clamp(0.0, 1.0);
                            let r = (1.5 - (4.0 * v - 3.0).abs()).clamp(0.0, 1.0);
                            let g = (1.5 - (4.0 * v - 2.0).abs()).clamp(0.0, 1.0);
                            let b_val = (1.5 - (4.0 * v - 1.0).abs()).clamp(0.0, 1.0);
                            result[[y, x, 0]] = (r * 255.0) as u8;
                            result[[y, x, 1]] = (g * 255.0) as u8;
                            result[[y, x, 2]] = (b_val * 255.0) as u8;
                        }
                    }
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_anomaly_map_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("anomaly_maps");

    // Typical EfficientAD sizes
    for (h, w, c) in [(56, 56, 384), (112, 112, 256)].iter() {
        let teacher = random_f32_array(*h, *w, *c);
        let student = random_f32_array(*h, *w, *c);

        group.bench_with_input(
            BenchmarkId::new("compute_distance", format!("{}x{}x{}", h, w, c)),
            &(*h, *w, *c),
            |b, _| {
                b.iter(|| {
                    let mut st_map = Array2::<f32>::zeros((*h, *w));
                    let c_f32 = *c as f32;

                    for y in 0..*h {
                        for x in 0..*w {
                            let mut sum = 0.0f32;
                            for ch in 0..*c {
                                let diff = teacher[[y, x, ch]] - student[[y, x, ch]];
                                sum += diff * diff;
                            }
                            st_map[[y, x]] = (sum / c_f32).sqrt();
                        }
                    }
                    black_box(st_map)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_percentile_quickselect,
    bench_connected_components,
    bench_jet_colormap,
    bench_anomaly_map_computation,
);

criterion_main!(benches);
