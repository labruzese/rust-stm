# Smart pointers in rust-stm

`TVar<T>` requires `T: Any + Sync + Send + Clone` and deep-clones `T`
on every transactional read. For types that are expensive to clone
(long strings, large vecs, heavy structs), this guide describes the
idiomatic way to avoid those clones using smart pointers.

## TArc&lt;T&gt; — shared transactional values

`TArc<T>` wraps `TVar<Arc<T>>`. Reads return `Arc<T>`, which is a
refcount bump; the inner `T` is never cloned by a read.

```rust
use stm::{TArc, atomically};
use std::sync::Arc;

let cfg = TArc::new(String::from("port=80"));

atomically(|tx| {
    let current: Arc<String> = cfg.read(tx)?;
    // Use current cheaply; no String clone was performed.
    assert!(current.starts_with("port="));
    Ok(())
});
```

`T` only needs to be `Send + Sync + 'static`; `Clone` is not required
unless you want to use the copy-on-write helpers described below.

### Writing

`write` takes an `Arc<T>`:

```rust
# use stm::{TArc, atomically};
# use std::sync::Arc;
# let cfg = TArc::new(String::from("port=80"));
let new_cfg = Arc::new(String::from("port=8080"));
atomically(|tx| cfg.write(tx, new_cfg.clone()));
```

If you don't already have an `Arc`, `write_val` boxes it for you:

```rust
# use stm::{TArc, atomically};
# let cfg = TArc::new(String::from("port=80"));
atomically(|tx| cfg.write_val(tx, String::from("port=8080")));
```

### Copy-on-write

For `T: Clone`, `modify_cow` uses [`Arc::make_mut`] to obtain a mutable
reference to the inner value, cloning `T` only when the current `Arc`
is not uniquely owned. Inside a transaction the log always holds a
parallel `Arc`, so `modify_cow` always clones once per mutation — but
it never clones on the read path.

```rust
use stm::{TArc, atomically};

let v = TArc::new(vec![1, 2, 3]);
atomically(|tx| v.modify_cow(tx, |xs| xs.push(4)));
assert_eq!(&*v.read_atomic(), &[1, 2, 3, 4]);
```

The equivalent purely-functional form is `update`:

```rust
# use stm::{TArc, atomically};
let n = TArc::new(21_i32);
atomically(|tx| n.update(tx, |x| x * 2));
assert_eq!(*n.read_atomic(), 42);
```

## TWeak&lt;T&gt; — weak transactional references

`TWeak<T>` wraps `TVar<Weak<T>>` and pairs with `TArc<T>` to break
cycles in transactional graph and cache structures.

```rust
use stm::{TArc, TWeak, atomically};

let strong = TArc::new(String::from("hello"));
let weak = atomically(|tx| TWeak::downgrade(&strong, tx));

// While the strong reference exists, upgrade succeeds.
let alive = atomically(|tx| weak.upgrade(tx));
assert!(alive.is_some());

drop(strong);

// Once the last strong reference drops, upgrade returns None.
let gone = atomically(|tx| weak.upgrade(tx));
assert!(gone.is_none());
```

## Transaction abort and drop semantics

Smart pointer contents interact safely with the STM machinery:

- **Abort**: when a transaction returns `Err(Failure)` its log is
  dropped. Every `Arc<T>` or `Weak<T>` clone in the log decrements
  refcounts exactly once. The canonical `TVar` value is unchanged.
- **Explicit `retry`**: the log is cleared and the closure re-runs.
  Any smart pointer clones from the aborted run drop normally.
- **`or(first, second)`**: the log is cloned before `first` runs. If
  `first` retries, the log is restored, and any smart pointer clones
  produced by `first` drop with the discarded log.
- **Commit**: only committed writes mutate the `TVar`. Tentative
  writes inside aborted transactions never affect the refcounts seen
  by other threads.

Integration tests in `stm-core/tests/drop_safety.rs` exercise each of
these paths with drop-counting instrumented payloads.

## When to use which

| Use case                                    | Recommendation          |
|---------------------------------------------|-------------------------|
| Small `Copy` types (`u32`, `bool`, enums)   | `TVar<T>`               |
| Large `T: Clone` read often, written rarely | `TArc<T>`               |
| Shared structure with back-edges / caches   | `TArc<T>` + `TWeak<T>`  |
| Structural mutation of a large value        | `TArc<T>` + `modify_cow`|

## Not supported

- **`Rc<T>`**: `Rc` is `!Sync`, so it cannot live in a `TVar`. A
  single-threaded variant of STM would be required to support it;
  this is deferred (see `docs/proposal-smart-pointers.md`, phase 6).
- **`Box<dyn Trait>`**: `Box` is `Clone` only for sized `T`. For
  trait-object payloads, store an `Arc<dyn Trait + Send + Sync>` in
  a `TVar` directly. `TArc` cannot express this because of the
  `Sized` bound on the inner type parameter.

[`Arc::make_mut`]: https://doc.rust-lang.org/std/sync/struct.Arc.html#method.make_mut
