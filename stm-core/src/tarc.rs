//! `TArc<T>` is an ergonomic wrapper around `TVar<Arc<T>>`.
//!
//! A plain `TVar<T>` deep-clones `T` on every transactional read. For large
//! values this is expensive. Wrapping the value in an [`Arc`] reduces the
//! per-read cost to a single refcount bump, at the price of paying for a
//! clone of the inner `T` when the value is mutated.
//!
//! `TArc<T>` does not add any new concurrency primitives. It is a thin
//! type-safe facade over `TVar<Arc<T>>` that also provides a
//! copy-on-write [`modify_cow`](TArc::modify_cow) helper.
//!
//! # Example
//!
//! ```
//! # use stm_core::*;
//! # use std::sync::Arc;
//! #[derive(Clone)]
//! struct Config { port: u16, host: String }
//!
//! let cfg = TArc::new(Config { port: 80, host: "localhost".into() });
//!
//! atomically(|tx| {
//!     let current: Arc<Config> = cfg.read(tx)?; // cheap Arc clone
//!     assert_eq!(current.port, 80);
//!     cfg.modify_cow(tx, |c| c.port = 8080)
//! });
//!
//! assert_eq!(cfg.read_atomic().port, 8080);
//! ```

use std::fmt::{self, Debug};
use std::sync::Arc;

use super::Transaction;
use super::result::StmResult;
use super::tvar::TVar;

/// A transactional reference-counted shared value.
///
/// `TArc<T>` is a wrapper around `TVar<Arc<T>>`. Reads return an
/// `Arc<T>` and therefore do not clone the inner `T`. Writes accept an
/// `Arc<T>` (or, via [`write_val`](TArc::write_val), a bare `T`).
///
/// `T` only needs to be `Send + Sync + 'static`; `Clone` is only required
/// for the copy-on-write helpers [`modify_cow`](TArc::modify_cow) and
/// [`update`](TArc::update).
pub struct TArc<T: Send + Sync + 'static> {
    inner: TVar<Arc<T>>,
}

impl<T: Send + Sync + 'static> TArc<T> {
    /// Create a new `TArc` holding `val`, wrapped in a fresh `Arc`.
    pub fn new(val: T) -> TArc<T> {
        TArc {
            inner: TVar::new(Arc::new(val)),
        }
    }

    /// Create a new `TArc` from an existing `Arc`. Useful when the
    /// caller already owns the `Arc`, for example to share a value
    /// between a `TArc` and plain reference counting.
    pub fn from_arc(a: Arc<T>) -> TArc<T> {
        TArc {
            inner: TVar::new(a),
        }
    }

    /// Read the current value through a transaction.
    ///
    /// Returns a clone of the inner `Arc<T>`, i.e. a refcount bump.
    /// The inner `T` is not cloned.
    pub fn read(&self, transaction: &mut Transaction) -> StmResult<Arc<T>> {
        self.inner.read(transaction)
    }

    /// Read the current value without starting a transaction.
    ///
    /// Returns a clone of the inner `Arc<T>`.
    pub fn read_atomic(&self) -> Arc<T> {
        self.inner.read_atomic()
    }

    /// Write a new value through a transaction.
    pub fn write(&self, transaction: &mut Transaction, val: Arc<T>) -> StmResult<()> {
        self.inner.write(transaction, val)
    }

    /// Write a new bare `T` through a transaction, wrapping it in a
    /// fresh `Arc`.
    pub fn write_val(&self, transaction: &mut Transaction, val: T) -> StmResult<()> {
        self.inner.write(transaction, Arc::new(val))
    }

    /// Write a new value without starting a transaction.
    pub fn write_atomic(&self, val: Arc<T>) {
        self.inner.write_atomic(val)
    }

    /// Replace the stored value and return the previous one.
    pub fn replace(
        &self,
        transaction: &mut Transaction,
        val: Arc<T>,
    ) -> StmResult<Arc<T>> {
        self.inner.replace(transaction, val)
    }

    /// Check if two `TArc`s refer to the same location.
    ///
    /// This is pointer equality on the underlying control block, not
    /// structural equality of the contained values.
    pub fn ref_eq(this: &TArc<T>, other: &TArc<T>) -> bool {
        TVar::ref_eq(&this.inner, &other.inner)
    }

    /// Access the underlying `TVar<Arc<T>>`.
    ///
    /// Escape hatch for the rare case where `TVar` API is needed
    /// (e.g. passing the var to generic STM code).
    pub fn as_tvar(&self) -> &TVar<Arc<T>> {
        &self.inner
    }
}

impl<T: Clone + Send + Sync + 'static> TArc<T> {
    /// Copy-on-write mutation.
    ///
    /// Reads the current `Arc<T>` through the transaction, then uses
    /// [`Arc::make_mut`] to obtain a mutable reference to the inner
    /// `T`. If the `Arc` is not uniquely owned (which, inside a
    /// transaction, is always the case because the transaction log
    /// keeps its own clone), `T` is cloned once. The transaction then
    /// writes back the mutated `Arc`.
    ///
    /// The *saving* relative to `TVar<T>::modify` comes from the
    /// read path: readers that do not mutate do not pay for a clone.
    ///
    /// ```
    /// # use stm_core::*;
    /// let v = TArc::new(vec![1, 2, 3]);
    /// atomically(|tx| v.modify_cow(tx, |xs| xs.push(4)));
    /// assert_eq!(&*v.read_atomic(), &[1, 2, 3, 4]);
    /// ```
    pub fn modify_cow<F>(&self, transaction: &mut Transaction, f: F) -> StmResult<()>
    where
        F: FnOnce(&mut T),
    {
        let current = self.read(transaction)?;
        let mut next = current;
        f(Arc::make_mut(&mut next));
        self.write(transaction, next)
    }

    /// Purely-functional update.
    ///
    /// Reads the current value, computes a new one from a reference,
    /// and writes the new value. Equivalent to
    /// `modify_cow` for callers who prefer to return a fresh `T`
    /// rather than mutate in place.
    ///
    /// ```
    /// # use stm_core::*;
    /// let v = TArc::new(21_i32);
    /// atomically(|tx| v.update(tx, |n| n * 2));
    /// assert_eq!(*v.read_atomic(), 42);
    /// ```
    pub fn update<F>(&self, transaction: &mut Transaction, f: F) -> StmResult<()>
    where
        F: FnOnce(&T) -> T,
    {
        let current = self.read(transaction)?;
        let next = f(&current);
        self.write(transaction, Arc::new(next))
    }
}

impl<T: Send + Sync + 'static> Clone for TArc<T> {
    fn clone(&self) -> TArc<T> {
        TArc {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + Sync + 'static> Debug for TArc<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let x = self.read_atomic();
        f.debug_struct("TArc").field("value", &*x).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{atomically, retry};

    #[test]
    fn read_atomic_round_trip() {
        let v = TArc::new(42_u32);
        assert_eq!(*v.read_atomic(), 42);
    }

    #[test]
    fn read_inside_transaction() {
        let v = TArc::new(String::from("hi"));
        let s = atomically(|tx| v.read(tx));
        assert_eq!(&*s, "hi");
    }

    #[test]
    fn write_val_and_read() {
        let v = TArc::new(0_u32);
        atomically(|tx| v.write_val(tx, 7));
        assert_eq!(*v.read_atomic(), 7);
    }

    #[test]
    fn write_arc_and_read() {
        let v = TArc::new(0_u32);
        let new_arc = Arc::new(99_u32);
        atomically(|tx| v.write(tx, new_arc.clone()));
        assert_eq!(*v.read_atomic(), 99);
    }

    #[test]
    fn replace_returns_old() {
        let v = TArc::new(1_u32);
        let old = atomically(|tx| v.replace(tx, Arc::new(2)));
        assert_eq!(*old, 1);
        assert_eq!(*v.read_atomic(), 2);
    }

    #[test]
    fn modify_cow_mutates() {
        let v = TArc::new(vec![1, 2, 3]);
        atomically(|tx| v.modify_cow(tx, |xs| xs.push(4)));
        assert_eq!(&*v.read_atomic(), &[1, 2, 3, 4]);
    }

    #[test]
    fn update_builds_new_value() {
        let v = TArc::new(21_i32);
        atomically(|tx| v.update(tx, |n| n * 2));
        assert_eq!(*v.read_atomic(), 42);
    }

    #[test]
    fn cheap_read_does_not_deep_clone() {
        // A type whose Clone panics. If TArc::read cloned the inner T
        // this test would panic; instead only the Arc is cloned.
        struct NoClone(u32);
        impl Clone for NoClone {
            fn clone(&self) -> Self {
                panic!("inner T cloned on read");
            }
        }

        let v = TArc::new(NoClone(1));
        let got = atomically(|tx| v.read(tx));
        assert_eq!(got.0, 1);
    }

    #[test]
    fn or_rollback_on_first_retry() {
        let v = TArc::new(0_u32);
        let x = atomically(|tx| {
            tx.or(
                |tx| {
                    v.write_val(tx, 123)?;
                    retry()
                },
                |tx| v.read(tx),
            )
        });
        assert_eq!(*x, 0);
        assert_eq!(*v.read_atomic(), 0);
    }

    #[test]
    fn ref_eq_identifies_shared_handle() {
        let a = TArc::new(1_u32);
        let b = a.clone();
        let c = TArc::new(1_u32);
        assert!(TArc::ref_eq(&a, &b));
        assert!(!TArc::ref_eq(&a, &c));
    }

    #[test]
    fn threaded_visibility() {
        use std::thread;
        use std::time::Duration;

        let v = TArc::new(0_u32);
        let vc = v.clone();
        let t = thread::spawn(move || {
            atomically(|tx| {
                let x = *vc.read(tx)?;
                if x == 0 { retry() } else { Ok(x) }
            })
        });
        thread::sleep(Duration::from_millis(50));
        atomically(|tx| v.write_val(tx, 77));
        assert_eq!(t.join().unwrap(), 77);
    }
}
