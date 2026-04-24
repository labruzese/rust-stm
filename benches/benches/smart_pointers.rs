use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use stm::{TArc, TVar, atomically, init_transaction};

/// Compare `TVar<Vec<u64>>` (deep-clone every read/write) against
/// `TArc<Vec<u64>>` (Arc refcount bump every read; clone inner on
/// mutation only).
///
/// For small element counts the two are comparable; the `TArc` win
/// grows linearly with `Vec` length.
fn smart_pointers(c: &mut Criterion) {
    const SMALL: usize = 64;
    const LARGE: usize = 4096;

    let small_vec: Vec<u64> = (0..SMALL as u64).collect();
    let large_vec: Vec<u64> = (0..LARGE as u64).collect();

    // Read times ------------------------------------------------------

    let mut g1 = c.benchmark_group("smart-pointer-read");

    {
        let tvar = TVar::new(small_vec.clone());
        g1.bench_function("TVar<Vec<u64; 64>>::read", |b| {
            let mut tx = init_transaction();
            b.iter(|| black_box(tx.read(&tvar)))
        });
    }
    {
        let tarc = TArc::new(small_vec.clone());
        g1.bench_function("TArc<Vec<u64; 64>>::read", |b| {
            b.iter(|| black_box(atomically(|tx| tarc.read(tx))))
        });
    }
    {
        let tvar = TVar::new(large_vec.clone());
        g1.bench_function("TVar<Vec<u64; 4096>>::read", |b| {
            let mut tx = init_transaction();
            b.iter(|| black_box(tx.read(&tvar)))
        });
    }
    {
        let tarc = TArc::new(large_vec.clone());
        g1.bench_function("TArc<Vec<u64; 4096>>::read", |b| {
            b.iter(|| black_box(atomically(|tx| tarc.read(tx))))
        });
    }
    g1.finish();

    // Write times -----------------------------------------------------

    let mut g2 = c.benchmark_group("smart-pointer-write");

    {
        let tvar = TVar::new(small_vec.clone());
        let src = small_vec.clone();
        g2.bench_function("TVar<Vec<u64; 64>>::write", |b| {
            let mut tx = init_transaction();
            b.iter(|| black_box(tx.write(&tvar, src.clone())))
        });
    }
    {
        let tarc = TArc::new(small_vec.clone());
        let src = Arc::new(small_vec.clone());
        g2.bench_function("TArc<Vec<u64; 64>>::write(Arc)", |b| {
            b.iter(|| black_box(atomically(|tx| tarc.write(tx, src.clone()))))
        });
    }
    {
        let tvar = TVar::new(large_vec.clone());
        let src = large_vec.clone();
        g2.bench_function("TVar<Vec<u64; 4096>>::write", |b| {
            let mut tx = init_transaction();
            b.iter(|| black_box(tx.write(&tvar, src.clone())))
        });
    }
    {
        let tarc = TArc::new(large_vec.clone());
        let src = Arc::new(large_vec.clone());
        g2.bench_function("TArc<Vec<u64; 4096>>::write(Arc)", |b| {
            b.iter(|| black_box(atomically(|tx| tarc.write(tx, src.clone()))))
        });
    }
    g2.finish();

    // Modify-in-place vs copy-on-write -------------------------------

    let mut g3 = c.benchmark_group("smart-pointer-modify");

    {
        let tvar = TVar::new(small_vec.clone());
        g3.bench_function("TVar<Vec<u64; 64>>::modify push", |b| {
            b.iter(|| {
                atomically(|tx| {
                    let mut v = tvar.read(tx)?;
                    if v.len() > SMALL {
                        v.truncate(SMALL);
                    }
                    v.push(0);
                    tvar.write(tx, v)
                })
            })
        });
    }
    {
        let tarc = TArc::new(small_vec.clone());
        g3.bench_function("TArc<Vec<u64; 64>>::modify_cow push", |b| {
            b.iter(|| {
                atomically(|tx| {
                    tarc.modify_cow(tx, |v| {
                        if v.len() > SMALL {
                            v.truncate(SMALL);
                        }
                        v.push(0);
                    })
                })
            })
        });
    }
    g3.finish();
}

criterion_group!(benches, smart_pointers);
criterion_main!(benches);
