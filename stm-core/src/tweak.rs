//! `TWeak<T>` is a transactional weak reference.
//!
//! It wraps `TVar<Weak<T>>` and pairs with [`TArc`](crate::TArc) for
//! building graph structures with back-edges, caches, and other
//! patterns where cycle-breaking is needed.
//!
//! # Example
//!
//! ```
//! # use stm_core::*;
//! # use std::sync::Arc;
//! let strong = TArc::new(String::from("hello"));
//! let weak: TWeak<String> = atomically(|tx| TWeak::downgrade(&strong, tx));
//!
//! // While the strong reference is alive, upgrade succeeds.
//! let up: Option<Arc<String>> = atomically(|tx| weak.upgrade(tx));
//! assert_eq!(up.map(|a| (*a).clone()), Some("hello".into()));
//! ```

use std::fmt::{self, Debug};
use std::sync::{Arc, Weak};

use super::Transaction;
use super::result::StmResult;
use super::tarc::TArc;
use super::tvar::TVar;

/// A transactional `Weak<T>` reference.
///
/// Conceptually equivalent to `TVar<Weak<T>>` with ergonomic helpers
/// for the common `downgrade`/`upgrade` patterns.
pub struct TWeak<T: Send + Sync + 'static> {
    inner: TVar<Weak<T>>,
}

impl<T: Send + Sync + 'static> TWeak<T> {
    /// Create a new, empty `TWeak` that does not reference any value.
    ///
    /// [`upgrade`](TWeak::upgrade) on this `TWeak` will return `None`
    /// until a strong reference is written into it via
    /// [`write_arc`](TWeak::write_arc) or [`write`](TWeak::write).
    pub fn new() -> TWeak<T> {
        TWeak {
            inner: TVar::new(Weak::new()),
        }
    }

    /// Create a `TWeak` from an existing `Weak`.
    pub fn from_weak(w: Weak<T>) -> TWeak<T> {
        TWeak { inner: TVar::new(w) }
    }

    /// Downgrade a [`TArc`] to a `TWeak`, reading the current strong
    /// reference through `transaction`.
    pub fn downgrade(src: &TArc<T>, transaction: &mut Transaction) -> StmResult<TWeak<T>> {
        let arc = src.read(transaction)?;
        Ok(TWeak::from_weak(Arc::downgrade(&arc)))
    }

    /// Read the raw `Weak<T>` through a transaction.
    pub fn read(&self, transaction: &mut Transaction) -> StmResult<Weak<T>> {
        self.inner.read(transaction)
    }

    /// Read the raw `Weak<T>` without starting a transaction.
    pub fn read_atomic(&self) -> Weak<T> {
        self.inner.read_atomic()
    }

    /// Try to obtain a strong reference through a transaction.
    ///
    /// Returns `None` if the value has already been dropped.
    ///
    /// Callers who want to block until the value is alive again can
    /// combine this with [`unwrap_or_retry`](crate::unwrap_or_retry),
    /// but note that `TWeak` alone has no way to wake such waiters —
    /// a write to the backing `TWeak` or to a related `TArc` is
    /// required.
    pub fn upgrade(&self, transaction: &mut Transaction) -> StmResult<Option<Arc<T>>> {
        let w = self.read(transaction)?;
        Ok(w.upgrade())
    }

    /// Write a new `Weak` reference through a transaction.
    pub fn write(&self, transaction: &mut Transaction, w: Weak<T>) -> StmResult<()> {
        self.inner.write(transaction, w)
    }

    /// Store a fresh `Weak` obtained by downgrading the given `Arc`.
    pub fn write_arc(&self, transaction: &mut Transaction, a: &Arc<T>) -> StmResult<()> {
        self.inner.write(transaction, Arc::downgrade(a))
    }

    /// Clear the `TWeak`, making subsequent `upgrade` calls return
    /// `None`.
    pub fn clear(&self, transaction: &mut Transaction) -> StmResult<()> {
        self.inner.write(transaction, Weak::new())
    }

    /// Check if two `TWeak`s refer to the same location.
    pub fn ref_eq(this: &TWeak<T>, other: &TWeak<T>) -> bool {
        TVar::ref_eq(&this.inner, &other.inner)
    }

    /// Access the underlying `TVar<Weak<T>>`.
    pub fn as_tvar(&self) -> &TVar<Weak<T>> {
        &self.inner
    }
}

impl<T: Send + Sync + 'static> Default for TWeak<T> {
    fn default() -> Self {
        TWeak::new()
    }
}

impl<T: Send + Sync + 'static> Clone for TWeak<T> {
    fn clone(&self) -> TWeak<T> {
        TWeak {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + Sync + 'static> Debug for TWeak<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let w = self.read_atomic();
        f.debug_struct("TWeak")
            .field("alive", &(w.strong_count() > 0))
            .field("strong_count", &w.strong_count())
            .field("weak_count", &w.weak_count())
            .finish()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TArc, atomically};

    #[test]
    fn empty_upgrade_is_none() {
        let w: TWeak<u32> = TWeak::new();
        let got = atomically(|tx| w.upgrade(tx));
        assert!(got.is_none());
    }

    #[test]
    fn downgrade_and_upgrade() {
        let strong = TArc::new(42_u32);
        let weak = atomically(|tx| TWeak::downgrade(&strong, tx));
        let got = atomically(|tx| weak.upgrade(tx));
        assert_eq!(got.map(|a| *a), Some(42));
    }

    #[test]
    fn upgrade_returns_none_after_all_strong_drop() {
        // Create a local Arc, downgrade, then drop the Arc. The Weak
        // must no longer upgrade.
        let a = Arc::new(String::from("transient"));
        let weak = TWeak::from_weak(Arc::downgrade(&a));
        drop(a);

        let got = atomically(|tx| weak.upgrade(tx));
        assert!(got.is_none());
    }

    #[test]
    fn write_arc_updates_weak() {
        let w: TWeak<u32> = TWeak::new();
        let a = Arc::new(7_u32);
        atomically(|tx| w.write_arc(tx, &a));
        let got = atomically(|tx| w.upgrade(tx));
        assert_eq!(got.map(|v| *v), Some(7));
    }

    #[test]
    fn clear_releases_weak() {
        let a = Arc::new(7_u32);
        let w = TWeak::from_weak(Arc::downgrade(&a));
        atomically(|tx| w.clear(tx));
        let got = atomically(|tx| w.upgrade(tx));
        assert!(got.is_none());
    }

    /// Storing a `Weak` in a `TWeak` does not keep the value alive.
    /// Once the last strong reference is dropped, the inner value
    /// is reclaimed and later `upgrade` calls return `None`.
    #[test]
    fn weak_does_not_keep_alive() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountOnDrop(Arc<AtomicUsize>);
        impl Drop for CountOnDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let strong = TArc::new(CountOnDrop(drops.clone()));
        let weak = atomically(|tx| TWeak::downgrade(&strong, tx));

        drop(strong);
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let got = atomically(|tx| weak.upgrade(tx));
        assert!(got.is_none());
    }

    #[test]
    fn ref_eq_identifies_shared_handle() {
        let a: TWeak<u32> = TWeak::new();
        let b = a.clone();
        let c: TWeak<u32> = TWeak::new();
        assert!(TWeak::ref_eq(&a, &b));
        assert!(!TWeak::ref_eq(&a, &c));
    }
}
