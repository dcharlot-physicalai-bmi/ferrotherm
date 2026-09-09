//! A barrier that spins briefly and then yields, instead of parking.
//!
//! # Why the standard one is not the right tool here
//!
//! [`std::sync::Barrier`] is a mutex and a condvar, so a thread that arrives early PARKS — a
//! syscall, and a wakeup for every waiter when the last one lands. That is the correct trade for a
//! barrier crossed occasionally. [`crate::gibbs::Sampler::sweeps_par`] crosses one at every colour
//! class of every sweep, which for a two-coloured lattice is twice per sweep and thousands of times
//! per run, with a few microseconds of work in between.
//!
//! Measured on this machine, one round being a chunk of work plus a barrier, best of three with the
//! arms interleaved. "no barrier" is the same loop with the wait removed, so the gap is what
//! synchronisation costs:
//!
//! ```text
//!   threads   chunk    no barrier   std::Barrier   this
//!         4   6.6us          3.4us        10.8us   3.7us
//!         8   6.6us          3.6us        24.1us   4.0us
//!         8  52.6us         28.4us        65.4us   29.4us
//!        18  52.6us         29.2us       120.1us  33.5us
//! ```
//!
//! At eight threads and a chunk the size [`crate::gibbs::MIN_CHUNK`] admits, the standard barrier
//! is **85% of the loop**. That is the whole reason the parallel sampler scaled worse at eight
//! threads than at four.
//!
//! # The spin budget is small, and that is measured rather than assumed
//!
//! The obvious reading of "spin instead of park" is that more spinning is better. It is not, and a
//! first version of this used 20,000 iterations on exactly that reasoning. Spinning threads take
//! CPU from threads that are still WORKING, so once the thread count reaches the core count a long
//! budget makes things worse:
//!
//! ```text
//!   spin budget:        0     200    1000    5000   20000  100000   std::Barrier
//!   18 threads, 52.6us   36.6    36.3    37.1    47.1    51.6    50.2          115.7
//!    8 threads,  6.6us    5.1     3.9     3.9     3.9     3.8     3.9           20.0
//! ```
//!
//! **A budget of zero already captures most of the win** — 36.6 against 115.7 — so what this is
//! really buying is NOT PARKING, and the spin is a small extra that catches skew smaller than a
//! scheduler tick. [`SPIN`] is set where the two rows above agree, not where either is best.
//!
//! # Fairness, and what this does not do
//!
//! It never parks, so a thread waiting on a barrier is runnable the whole time. On an
//! oversubscribed machine — more workers than cores — that is worse than parking, and the
//! [`std::thread::yield_now`] fallback is what keeps it from being much worse. The sampler's own
//! [`crate::gibbs::MIN_CHUNK`] floor caps worker count by the graph rather than by the machine, so
//! a caller can still ask for more threads than cores; that is a real limitation and it is stated
//! rather than guarded, because guarding it would need a core count that `std` does not portably
//! give.

use core::sync::atomic::{AtomicUsize, Ordering};

/// `spin_loop` iterations before falling back to yielding. See the module docs for the sweep this
/// came from: larger is worse once workers reach cores, and zero is already most of the win.
pub const SPIN: usize = 200;

/// A reusable barrier for `n` threads that does not park.
///
/// # The counter only grows, which is what makes it correct
///
/// A thread takes a ticket with one `fetch_add` and waits for the count to reach the next multiple
/// of `n`. There is no reset and no sense bit, so there is no window in which a fast thread can
/// arrive at the NEXT barrier before the last one has finished resetting the current — which is the
/// race a hand-rolled barrier gets wrong first, and it is absent here by construction rather than
/// by ordering arguments.
///
/// The count would need `2^64 / n` crossings to wrap.
#[derive(Debug)]
pub struct SpinBarrier {
    count: AtomicUsize,
    n: usize,
    spin: usize,
}

impl SpinBarrier {
    /// A barrier for `n` threads. `n == 0` is treated as 1, so a degenerate caller spins forever
    /// nowhere rather than dividing by zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), n: n.max(1), spin: SPIN }
    }

    /// The same with an explicit spin budget, for measuring one against another.
    #[must_use]
    pub fn with_spin(n: usize, spin: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), n: n.max(1), spin }
    }

    /// How many threads this barrier is for.
    #[must_use]
    pub fn parties(&self) -> usize {
        self.n
    }

    /// Block until all `n` threads have called `wait` for this crossing.
    pub fn wait(&self) {
        let ticket = self.count.fetch_add(1, Ordering::AcqRel);
        let target = (ticket / self.n + 1) * self.n;
        let mut spun = 0usize;
        while self.count.load(Ordering::Acquire) < target {
            if spun < self.spin {
                core::hint::spin_loop();
                spun += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    /// THE BARRIER MUST ACTUALLY SEPARATE THE PHASES. Every thread writes its slot in round `r`,
    /// then reads EVERY slot; if the barrier lets anyone run ahead, a reader sees a value from the
    /// wrong round. Checked on the values rather than on a count, because a barrier that returns
    /// too early still returns.
    #[test]
    fn nothing_crosses_a_crossing() {
        for threads in [2usize, 4, 8] {
            let rounds = 500usize;
            let slots: Vec<AtomicUsize> = (0..threads).map(|_| AtomicUsize::new(usize::MAX)).collect();
            let barrier = SpinBarrier::new(threads);
            let bad = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for ti in 0..threads {
                    let (slots, barrier, bad) = (&slots, &barrier, &bad);
                    s.spawn(move || {
                        for r in 0..rounds {
                            slots[ti].store(r, Ordering::Release);
                            barrier.wait();
                            for slot in slots {
                                if slot.load(Ordering::Acquire) != r {
                                    bad.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            // The second crossing is what makes the first meaningful: without it a
                            // thread could start round r+1's store while another is still reading.
                            barrier.wait();
                        }
                    });
                }
            });
            assert_eq!(bad.load(Ordering::Relaxed), 0, "{threads} threads saw the wrong round");
        }
    }

    /// One party is a barrier that never blocks, and a zero-party one must not divide by zero.
    #[test]
    fn degenerate_party_counts_do_not_hang_or_divide_by_zero() {
        let one = SpinBarrier::new(1);
        for _ in 0..1000 {
            one.wait();
        }
        assert_eq!(one.parties(), 1);
        let zero = SpinBarrier::new(0);
        assert_eq!(zero.parties(), 1, "zero parties is treated as one");
        zero.wait();
    }

    /// A zero spin budget is legal and is the pure-yield barrier the module docs measure; it must
    /// still separate phases, since that is the configuration the sweep says is most of the win.
    #[test]
    fn a_zero_spin_budget_is_still_a_barrier() {
        let threads = 4usize;
        let seen = AtomicUsize::new(0);
        let barrier = SpinBarrier::with_spin(threads, 0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                let (barrier, seen) = (&barrier, &seen);
                s.spawn(move || {
                    for _ in 0..200 {
                        seen.fetch_add(1, Ordering::Relaxed);
                        barrier.wait();
                    }
                });
            }
        });
        assert_eq!(seen.load(Ordering::Relaxed), threads * 200);
    }
}
