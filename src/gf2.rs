//! Linear algebra over GF(2): bit-packed matrices, rank, reduced row echelon form, nullspace and
//! solving -- the arithmetic under parity-check codes, syndromes and every constraint that reads
//! "an even number of these".
//!
//! # Why this exists
//!
//! A parity check is a linear equation over the field with two elements, and a code is the
//! nullspace of its check matrix. Decoding a code on a thermodynamic sampler ([`crate::sourlas`]
//! and its successors) needs the code first: its rank, a generator for its nullspace, a
//! systematic form to encode with, and a syndrome to decode from. None of that is floating
//! point, none of it was in the crate, and the temptation to do it with `f64` and `% 2` is how
//! a check matrix silently loses rank. Rows are packed sixty-four bits to a word, elimination is
//! word-parallel, and every answer is checked against brute force on matrices small enough to
//! enumerate.
//!
//! # What is checked
//!
//! Rank equals the rank of the transpose; the nullspace has `cols - rank` vectors, each of which
//! the matrix annihilates, and on a matrix of six columns those are exactly the `2^(cols - rank)`
//! vectors brute force finds; [`Matrix::solve`] returns a vector the matrix maps to the target
//! whenever brute force finds one and `None` exactly when it does not; and the reduced row
//! echelon form is idempotent and row-equivalent to its input.

/// A dense matrix over GF(2), each row packed little-endian into `u64` words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
    /// Words per row, `ceil(cols / 64)`.
    pub words: usize,
    /// The bits, `rows * words` words, row-major.
    pub data: Vec<u64>,
}

impl Matrix {
    /// The zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let words = cols.div_ceil(64).max(1);
        Matrix {
            rows,
            cols,
            words,
            data: vec![0; rows * words],
        }
    }

    /// The identity.
    #[must_use]
    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// From rows of `0`/`1` bytes (anything non-zero is one).
    ///
    /// # Panics
    ///
    /// If the rows have different lengths.
    #[must_use]
    pub fn from_rows(rows: &[Vec<u8>]) -> Matrix {
        let cols = rows.first().map_or(0, Vec::len);
        let mut m = Matrix::zeros(rows.len(), cols);
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), cols, "every row has the same length");
            for (c, &b) in row.iter().enumerate() {
                if b != 0 {
                    m.set(r, c, true);
                }
            }
        }
        m
    }

    /// A random matrix with each bit one with probability `density`, from a seed.
    #[must_use]
    pub fn random(rows: usize, cols: usize, density: f64, seed: u64) -> Matrix {
        let mut m = Matrix::zeros(rows, cols);
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        for r in 0..rows {
            for c in 0..cols {
                state = splitmix(state);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                if u < density {
                    m.set(r, c, true);
                }
            }
        }
        m
    }

    /// The bit at `(r, c)`.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> bool {
        (self.data[r * self.words + c / 64] >> (c % 64)) & 1 == 1
    }

    /// Set the bit at `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, one: bool) {
        let w = &mut self.data[r * self.words + c / 64];
        if one {
            *w |= 1u64 << (c % 64);
        } else {
            *w &= !(1u64 << (c % 64));
        }
    }

    /// Row `r` as a slice of words.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }

    /// Row `r` as bytes, one per column.
    #[must_use]
    pub fn row_bits(&self, r: usize) -> Vec<u8> {
        (0..self.cols).map(|c| u8::from(self.get(r, c))).collect()
    }

    /// `row dst ^= row src`.
    fn xor_rows(&mut self, dst: usize, src: usize) {
        let w = self.words;
        let source: Vec<u64> = self.row(src).to_vec();
        for (d, x) in self.data[dst * w..(dst + 1) * w].iter_mut().zip(&source) {
            *d ^= *x;
        }
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let w = self.words;
        for k in 0..w {
            self.data.swap(a * w + k, b * w + k);
        }
    }

    /// The transpose.
    #[must_use]
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    t.set(c, r, true);
                }
            }
        }
        t
    }

    /// The reduced row echelon form and its pivot columns, by Gauss--Jordan elimination.
    #[must_use]
    pub fn rref(&self) -> (Matrix, Vec<usize>) {
        let mut m = self.clone();
        let mut pivots = Vec::new();
        let mut row = 0;
        for c in 0..m.cols {
            if row >= m.rows {
                break;
            }
            let Some(p) = (row..m.rows).find(|&r| m.get(r, c)) else {
                continue;
            };
            m.swap_rows(row, p);
            for r in 0..m.rows {
                if r != row && m.get(r, c) {
                    m.xor_rows(r, row);
                }
            }
            pivots.push(c);
            row += 1;
        }
        (m, pivots)
    }

    /// The rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rref().1.len()
    }

    /// A basis of the nullspace `{x : M x = 0}`, one vector per free column, each as bytes.
    #[must_use]
    pub fn nullspace(&self) -> Vec<Vec<u8>> {
        let (r, pivots) = self.rref();
        let free: Vec<usize> = (0..self.cols).filter(|c| !pivots.contains(c)).collect();
        free.iter()
            .map(|&f| {
                let mut x = vec![0u8; self.cols];
                x[f] = 1;
                for (i, &p) in pivots.iter().enumerate() {
                    if r.get(i, f) {
                        x[p] = 1;
                    }
                }
                x
            })
            .collect()
    }

    /// `M x` over GF(2), `x` as bytes.
    ///
    /// # Panics
    ///
    /// If `x` is not `cols` long.
    #[must_use]
    pub fn mul_vec(&self, x: &[u8]) -> Vec<u8> {
        assert_eq!(x.len(), self.cols, "a vector of {} bits", self.cols);
        let mut packed = vec![0u64; self.words];
        for (c, &b) in x.iter().enumerate() {
            if b != 0 {
                packed[c / 64] |= 1u64 << (c % 64);
            }
        }
        (0..self.rows)
            .map(|r| {
                let parity: u32 = self
                    .row(r)
                    .iter()
                    .zip(&packed)
                    .map(|(a, b)| (a & b).count_ones())
                    .sum();
                (parity & 1) as u8
            })
            .collect()
    }

    /// One solution of `M x = b`, or `None` if there is none.
    ///
    /// # Panics
    ///
    /// If `b` is not `rows` long.
    #[must_use]
    pub fn solve(&self, b: &[u8]) -> Option<Vec<u8>> {
        assert_eq!(b.len(), self.rows, "a target of {} bits", self.rows);
        // Augment with b as an extra column, reduce, read off.
        let mut aug = Matrix::zeros(self.rows, self.cols + 1);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    aug.set(r, c, true);
                }
            }
            if b[r] != 0 {
                aug.set(r, self.cols, true);
            }
        }
        let (red, pivots) = aug.rref();
        if pivots.contains(&self.cols) {
            return None;
        }
        let mut x = vec![0u8; self.cols];
        for (i, &p) in pivots.iter().enumerate() {
            if red.get(i, self.cols) {
                x[p] = 1;
            }
        }
        Some(x)
    }

    /// Whether every bit is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&w| w == 0)
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_vectors(n: usize) -> impl Iterator<Item = Vec<u8>> {
        (0..1usize << n).map(move |x| (0..n).map(|i| ((x >> i) & 1) as u8).collect())
    }

    /// Rank is invariant under transposition, the identity has full rank, and the reduced form
    /// is idempotent.
    #[test]
    fn rank_survives_transposition_and_the_reduced_form_is_idempotent() {
        assert_eq!(Matrix::identity(9).rank(), 9);
        for seed in 0..6u64 {
            let m = Matrix::random(7, 11, 0.4, seed);
            assert_eq!(m.rank(), m.transpose().rank(), "seed {seed}");
            let (r, pivots) = m.rref();
            let (rr, again) = r.rref();
            assert_eq!(r, rr);
            assert_eq!(pivots, again);
            assert!(m.rank() <= 7);
        }
        let wide = Matrix::random(3, 130, 0.5, 9);
        assert_eq!(wide.words, 3);
        assert_eq!(wide.rank(), wide.transpose().rank());
    }

    /// The nullspace basis has `cols - rank` vectors the matrix annihilates, and its span is
    /// exactly the set brute force finds on six columns.
    #[test]
    fn the_nullspace_is_exactly_what_brute_force_finds() {
        for seed in 0..8u64 {
            let m = Matrix::random(4, 6, 0.5, seed);
            let basis = m.nullspace();
            assert_eq!(basis.len(), 6 - m.rank(), "seed {seed}");
            for x in &basis {
                assert!(
                    m.mul_vec(x).iter().all(|&b| b == 0),
                    "seed {seed}: {x:?} not annihilated"
                );
            }
            let brute: Vec<Vec<u8>> = all_vectors(6)
                .filter(|x| m.mul_vec(x).iter().all(|&b| b == 0))
                .collect();
            assert_eq!(brute.len(), 1usize << basis.len(), "seed {seed}");
            // Every span element is in the brute-force set: check by counting distinct spans.
            let mut span = std::collections::BTreeSet::new();
            for coeff in all_vectors(basis.len()) {
                let mut x = vec![0u8; 6];
                for (k, &c) in coeff.iter().enumerate() {
                    if c == 1 {
                        for (xi, bi) in x.iter_mut().zip(&basis[k]) {
                            *xi ^= bi;
                        }
                    }
                }
                span.insert(x);
            }
            assert_eq!(span.len(), brute.len(), "seed {seed}");
            for x in &brute {
                assert!(span.contains(x), "seed {seed}: {x:?} missing from the span");
            }
        }
    }

    /// `solve` finds a solution exactly when brute force does, and the solution solves.
    #[test]
    fn solve_agrees_with_brute_force_on_consistency_and_solves_when_it_can() {
        let mut consistent = 0;
        let mut inconsistent = 0;
        for seed in 0..6u64 {
            let m = Matrix::random(5, 6, 0.45, seed);
            for b in all_vectors(5) {
                let brute = all_vectors(6).any(|x| m.mul_vec(&x) == b);
                match m.solve(&b) {
                    Some(x) => {
                        assert!(brute, "seed {seed}: solved an inconsistent system");
                        assert_eq!(m.mul_vec(&x), b, "seed {seed}: the solution does not solve");
                        consistent += 1;
                    }
                    None => {
                        assert!(!brute, "seed {seed}: missed a solvable system {b:?}");
                        inconsistent += 1;
                    }
                }
            }
        }
        assert!(
            consistent > 0 && inconsistent > 0,
            "both branches must be exercised: {consistent} / {inconsistent}"
        );
    }
}
