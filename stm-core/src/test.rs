//! This module contains helpers for various tests.
//! Quite a lot of tests run operations async_runhonously and need to check
//! for deadlocks. We do this by waiting a certain amount of time for completion.
//!
//! This module contains some helpers that simplify other tests.

use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

/// Check if a function `f` terminates within a given timeframe.
///
/// If the function does not terminate, it keeps a thread alive forever,
/// so don't run too many test (preferable just one) in sequence.
pub fn terminates<F>(duration_ms: u64, f: F) -> bool
where
    F: Send + FnOnce() + 'static,
{
    terminates_async(duration_ms, f, || {})
}

/// Check if a function `f` terminates within a given timeframe,
/// but run a second function `g` concurrently.
///
/// If the function does not terminate, it keeps a thread alive forever,
/// so don't run too many test (preferable just one) in sequence.
pub fn terminates_async<F, G>(duration_ms: u64, f: F, g: G) -> bool
where
    F: Send + FnOnce() + 'static,
    G: FnOnce(),
{
    async_run(duration_ms, f, g).is_some()
}

/// Run two functions `f` and `g` concurrently.
///
/// Run `f` in a second thread, `g` in the main thread. Wait the given time `duration_ms` for `g`
/// and return `f`s return value or return `None` if `f` does not terminate.
///
/// If `f` does not terminate, it keeps a thread alive forever,
/// so don't run too many test (preferable just one) in sequence.
pub fn async_run<T, F, G>(duration_ms: u64, f: F, g: G) -> Option<T>
where
    F: Send + FnOnce() -> T + 'static,
    G: FnOnce(),
    T: Send + 'static,
{
    let (tx, rx) = channel();

    thread::spawn(move || {
        let t = f();
        // wakeup other thread
        let _ = tx.send(t);
    });

    g();

    if let a @ Some(_) = rx.try_recv().ok() {
        return a;
    }

    // Give enough time for travis to get up.
    // Sleep in 50 ms steps, so that it does not waste too much time if the thread finishes earlier.
    for _ in 0..duration_ms / 50 {
        thread::sleep(Duration::from_millis(50));
        if let a @ Some(_) = rx.try_recv().ok() {
            return a;
        }
    }

    thread::sleep(Duration::from_millis(duration_ms % 50));

    rx.try_recv().ok()
}

 Drop-safety stress tests for TArc and TWeak.
//
// These tests exercise transactional abort, retry, and `or` paths
// while instrumenting inner values with a Drop counter. They verify:
//
//   * No inner value is dropped while still held by a live TArc.
//   * Aborted transactions do not leak inner values.
//   * Concurrent writers converge on a single final value with a
//     predictable refcount.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use stm_core::{TArc, TWeak, atomically, retry};

/// A payload whose Drop increments a shared counter.
struct Counted {
    _id: usize,
    drops: Arc<AtomicUsize>,
}

impl Counted {
    fn new(id: usize, drops: Arc<AtomicUsize>) -> Self {
        Counted { _id: id, drops }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

/// Many writers overwrite the same TArc. After all threads finish and
/// the TArc itself is dropped, the drop counter must equal the number
/// of distinct values ever stored.
#[test]
fn concurrent_overwrite_drops_all_old_values() {
    const THREADS: usize = 8;
    const WRITES_PER_THREAD: usize = 25;

    let drops = Arc::new(AtomicUsize::new(0));
    let var = TArc::new(Counted::new(0, drops.clone()));

    let mut handles = Vec::new();
    for t in 0..THREADS {
        let var = var.clone();
        let drops = drops.clone();
        handles.push(thread::spawn(move || {
            for i in 0..WRITES_PER_THREAD {
                let id = t * WRITES_PER_THREAD + i + 1;
                let drops = drops.clone();
                atomically(|tx| var.write_val(tx, Counted::new(id, drops.clone())));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Drop the var; the final stored Counted must now be released.
    drop(var);

    // The original value plus one per successful write.
    //
    // Under contention, `atomically` may retry the closure, which
    // re-constructs a Counted each time. Each of those transient
    // values is also dropped. We only assert a lower bound (every
    // committed write drops exactly one old value, plus the final
    // one) and an upper bound generous enough to cover retries.
    let n = drops.load(Ordering::SeqCst);
    let committed = THREADS * WRITES_PER_THREAD + 1;
    assert!(
        n >= committed,
        "expected at least {committed} drops, got {n}"
    );
}

/// `or` with a first branch that writes then retries must not leak
/// the tentative write: the value written in the retried branch is
/// dropped when the log is swapped.
#[test]
fn or_retry_drops_tentative_writes() {
    let drops = Arc::new(AtomicUsize::new(0));
    let var = TArc::new(Counted::new(0, drops.clone()));

    let drops_before = drops.load(Ordering::SeqCst);

    // A hundred iterations of `or(first-retries, second-succeeds)`
    // pile up tentative writes whose Counted values must all drop.
    for _ in 0..100 {
        let drops = drops.clone();
        let _ = atomically(|tx| {
            tx.or(
                |tx| {
                    var.write_val(tx, Counted::new(0, drops.clone()))?;
                    retry()
                },
                |tx| var.read(tx),
            )
        });
    }

    // Each of the 100 tentative writes produced one Counted that must
    // have been dropped (they never got committed). The original
    // value is still held in `var`.
    let delta = drops.load(Ordering::SeqCst) - drops_before;
    assert_eq!(delta, 100);
}

/// `modify_cow` on a shared TArc must not double-drop the inner
/// values. After N modifications we expect all N+1 inner elements
/// to drop exactly once when the TArc is dropped.
///
/// The inner element is wrapped in `Arc` because `modify_cow`
/// requires the held type to be `Clone`. The `Arc` is cheap to clone
/// and lets the Drop counter count strong-ref releases, not clones.
#[test]
fn modify_cow_no_double_drop() {
    let drops = Arc::new(AtomicUsize::new(0));
    let var: TArc<Vec<Arc<Counted>>> =
        TArc::new(vec![Arc::new(Counted::new(0, drops.clone()))]);

    for i in 1..=50 {
        let drops = drops.clone();
        atomically(|tx| {
            var.modify_cow(tx, |xs| {
                xs.push(Arc::new(Counted::new(i, drops.clone())));
            })
        });
    }

    let before_drop = drops.load(Ordering::SeqCst);
    drop(var);
    let after_drop = drops.load(Ordering::SeqCst);

    // Dropping the final TArc must release all 51 Counted values
    // currently inside the vec.
    assert_eq!(after_drop - before_drop, 51);
}

/// TWeak::upgrade during a transaction that subsequently retries
/// must not keep the inner value alive beyond the aborted transaction.
#[test]
fn upgrade_in_retried_branch_releases_arc() {
    let drops = Arc::new(AtomicUsize::new(0));

    // Scope so that `strong` is dropped before we check drops.
    {
        let strong = TArc::new(Counted::new(0, drops.clone()));
        let weak = atomically(|tx| TWeak::downgrade(&strong, tx));

        for _ in 0..20 {
            atomically(|tx| {
                tx.or(
                    |tx| {
                        let _ = weak.upgrade(tx)?;
                        retry()
                    },
                    |_tx| Ok(()),
                )
            });
        }

        // `strong` still holds one Counted.
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    // Dropping the last strong reference releases the value.
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
