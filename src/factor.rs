//! Factors, and the mistakes a factor must not be able to express.
//!
//! A factor over spins contributes `-w * prod(s_i)` to the energy. At arity two that is the
//! ordinary Ising coupling `-w s_i s_j`; the type exists so higher-order terms have somewhere to
//! live and so one particular error is impossible to write down.
//!
//! **The repeated variable.** If a variable appears twice among a factor's arguments, the factor is
//! not what was written: `s_i * s_i = 1` always, so an even multiplicity collapses the term to a
//! constant and an odd one silently reduces its order. THRML's documentation is candid that this
//! "condition has not been enforced in the code", which means a model can be quietly not the model
//! the user described. Here it is a `Err(FactorError::RepeatedVariable)` at construction, so it
//! cannot reach a sampler.

/// Why a factor could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum FactorError {
    /// No variables. A factor over nothing is a constant, which belongs in the energy offset.
    Empty,
    /// A variable appears more than once, which silently changes the factor's order.
    RepeatedVariable { var: usize, times: usize },
    /// A variable index is not in the model.
    OutOfRange { var: usize, n: usize },
    /// A weight that is NaN or infinite poisons every energy it touches.
    NonFinite,
}

impl core::fmt::Display for FactorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FactorError::Empty => write!(f, "a factor needs at least one variable"),
            FactorError::RepeatedVariable { var, times } => write!(
                f,
                "variable {var} appears {times} times in one factor; s*s = 1, so this is not the \
                 factor you wrote -- an even count collapses it to a constant and an odd count \
                 lowers its order"
            ),
            FactorError::OutOfRange { var, n } => {
                write!(f, "variable {var} is out of range for a model of {n} variables")
            }
            FactorError::NonFinite => write!(f, "a factor's weight must be finite"),
        }
    }
}

/// A term contributing `-weight * prod(s_v for v in vars)` to the energy.
#[derive(Clone, Debug, PartialEq)]
pub struct Factor {
    vars: Vec<u32>,
    weight: f64,
}

impl Factor {
    /// Build a factor, rejecting everything that would make it silently not what was written.
    pub fn new(vars: &[usize], weight: f64, n: usize) -> Result<Factor, FactorError> {
        if vars.is_empty() {
            return Err(FactorError::Empty);
        }
        if !weight.is_finite() {
            return Err(FactorError::NonFinite);
        }
        for &v in vars {
            if v >= n {
                return Err(FactorError::OutOfRange { var: v, n });
            }
        }
        // Report *which* variable repeated and how often; "duplicate found" is not actionable when
        // a factor has eight arguments.
        let mut sorted: Vec<usize> = vars.to_vec();
        sorted.sort_unstable();
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i + 1;
            while j < sorted.len() && sorted[j] == sorted[i] {
                j += 1;
            }
            if j - i > 1 {
                return Err(FactorError::RepeatedVariable { var: sorted[i], times: j - i });
            }
            i = j;
        }
        Ok(Factor { vars: vars.iter().map(|&v| v as u32).collect(), weight })
    }

    pub fn vars(&self) -> impl Iterator<Item = usize> + '_ {
        self.vars.iter().map(|&v| v as usize)
    }

    pub fn arity(&self) -> usize {
        self.vars.len()
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// This factor's contribution to the energy: `-w * prod(s_v)`.
    ///
    /// The sign is the same convention as the rest of the crate: energy is
    /// `-sum_ij J_ij s_i s_j - sum_i h_i s_i`, so a positive weight prefers the product to be +1.
    pub fn energy(&self, s: &[i8]) -> f64 {
        let mut p = 1.0;
        for &v in &self.vars {
            p *= s[v as usize] as f64;
        }
        -self.weight * p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_variable_cannot_be_built() {
        // The footgun THRML documents as unenforced.
        let e = Factor::new(&[0, 1, 0], 1.0, 4).unwrap_err();
        assert_eq!(e, FactorError::RepeatedVariable { var: 0, times: 2 });
        // and the message names the variable, not merely the fact
        assert!(e.to_string().contains("variable 0 appears 2 times"));

        assert!(matches!(
            Factor::new(&[2, 2, 2], 1.0, 4),
            Err(FactorError::RepeatedVariable { var: 2, times: 3 })
        ));
        // the pairwise case, which is the ordinary self-coupling mistake
        assert!(Factor::new(&[1, 1], 1.0, 4).is_err());
    }

    #[test]
    fn repetition_really_would_have_changed_the_factor() {
        // Why the guard exists, demonstrated rather than asserted: had [0,1,0] been accepted, it
        // would have behaved as the order-1 factor [1], not the order-3 factor written.
        let honest = Factor::new(&[1], 1.0, 4).unwrap();
        for mask in 0..16usize {
            let s: Vec<i8> = (0..4).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            // s0 * s1 * s0 == s1
            let would_have_been = -(s[0] as f64 * s[1] as f64 * s[0] as f64);
            assert_eq!(would_have_been, honest.energy(&s));
        }
    }

    #[test]
    fn other_malformed_factors_are_rejected() {
        assert_eq!(Factor::new(&[], 1.0, 4), Err(FactorError::Empty));
        assert_eq!(Factor::new(&[9], 1.0, 4), Err(FactorError::OutOfRange { var: 9, n: 4 }));
        assert_eq!(Factor::new(&[0], f64::NAN, 4), Err(FactorError::NonFinite));
        assert_eq!(Factor::new(&[0], f64::INFINITY, 4), Err(FactorError::NonFinite));
    }

    #[test]
    fn arity_two_matches_the_ising_convention() {
        // A factor must agree with the coupling it replaces, or the IR has two energies.
        let f = Factor::new(&[0, 1], 0.75, 2).unwrap();
        for (a, b) in [(1i8, 1i8), (1, -1), (-1, 1), (-1, -1)] {
            let s = [a, b];
            assert_eq!(f.energy(&s), -0.75 * a as f64 * b as f64);
        }
    }

    #[test]
    fn a_positive_weight_prefers_the_product_to_be_plus_one() {
        // Sign-convention pin. Weight-sign inversion is a documented incumbent footgun and the kind
        // of error that produces a plausible-looking wrong answer rather than a crash.
        let f = Factor::new(&[0, 1, 2], 1.0, 3).unwrap();
        assert_eq!(f.energy(&[1, 1, 1]), -1.0, "aligned should be LOW energy");
        assert_eq!(f.energy(&[1, 1, -1]), 1.0, "product -1 should be HIGH energy");
        let anti = Factor::new(&[0, 1, 2], -1.0, 3).unwrap();
        assert_eq!(anti.energy(&[1, 1, 1]), 1.0);
        assert_eq!(anti.energy(&[1, 1, -1]), -1.0);
    }

    #[test]
    fn higher_order_factors_are_expressible() {
        let f = Factor::new(&[0, 1, 2, 3, 4], 2.0, 8).unwrap();
        assert_eq!(f.arity(), 5);
        assert_eq!(f.vars().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!(f.energy(&[1, 1, 1, 1, 1, -1, -1, -1]), -2.0);
        assert_eq!(f.energy(&[-1, 1, 1, 1, 1, -1, -1, -1]), 2.0);
    }
}
