use super::utils::{accumulate_left_round, fold_dense_left_round, fold_factor_in_place};
#[cfg(test)]
use super::utils::{accumulate_right_round, fold_left_round, fold_right_round, product_claim};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::uni_poly::UniPoly;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBaseUnreduced, Zero};

/// One dense factored setup-product term
/// `sum_{left,right} table[left,right] * left_factor[left] * right_factor[right]`.
#[cfg(test)]
pub(super) struct FactoredProductTerm<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// Two-pass setup-product term over the canonical flat setup layout.
///
/// The setup source stays in the base field with row-major layout
/// `setup[setup_index * coefficient_len + coefficient]`. Akita's committed
/// setup MLE binds coefficient variables first, so this term first proves the
/// common-coefficient rounds and then the setup-index rounds. This preserves
/// the Stage-3 suffix-opening projection while storing only one contracted
/// coefficient vector and one contracted setup-index vector in the extension
/// field.
pub(super) struct RectangularSetupProductTerm<'a, F: FieldCore, E: FieldCore> {
    setup: &'a [F],
    required_rows: usize,
    row_capacity: usize,
    coefficient_len: usize,
    coefficient_rounds: usize,
    total_rounds: usize,
    coefficient_challenges: Vec<E>,
    coefficient_tables: Vec<Vec<E>>,
    coefficient_factors: Vec<Vec<E>>,
    index_table: Option<Vec<E>>,
    index_factors: Vec<Vec<E>>,
    index_factor: Vec<E>,
    input_claim: E,
}

impl<'a, F, E> RectangularSetupProductTerm<'a, F, E>
where
    F: FieldCore,
    E: FieldCore + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    #[cfg(test)]
    pub(super) fn new(
        setup: &'a [F],
        required_rows: usize,
        index_factor: Vec<E>,
        coefficient_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        Self::new_ranked(
            setup,
            required_rows,
            vec![index_factor],
            vec![coefficient_factor],
        )
    }

    pub(super) fn new_ranked(
        setup: &'a [F],
        required_rows: usize,
        index_factors: Vec<Vec<E>>,
        coefficient_factors: Vec<Vec<E>>,
    ) -> Result<Self, AkitaError> {
        let row_capacity = index_factors.first().map_or(0, Vec::len);
        let coefficient_len = coefficient_factors.first().map_or(0, Vec::len);
        if required_rows == 0
            || index_factors.is_empty()
            || index_factors.len() != coefficient_factors.len()
            || row_capacity == 0
            || coefficient_len == 0
            || !row_capacity.is_power_of_two()
            || !coefficient_len.is_power_of_two()
            || required_rows > row_capacity
            || index_factors
                .iter()
                .any(|factor| factor.len() != row_capacity)
            || coefficient_factors
                .iter()
                .any(|factor| factor.len() != coefficient_len)
        {
            return Err(AkitaError::InvalidInput(
                "rectangular setup-product dimensions are invalid".into(),
            ));
        }
        let source_len = row_capacity
            .checked_mul(coefficient_len)
            .ok_or_else(|| AkitaError::InvalidSetup("setup source length overflow".into()))?;
        if setup.len() < source_len {
            return Err(AkitaError::InvalidSize {
                expected: source_len,
                actual: setup.len(),
            });
        }
        let term_count = index_factors.len();
        let coefficient_tables = {
            let _span = tracing::info_span!(
                "stage3_setup_coefficient_pass",
                kernel = "rectangular_base_field",
                source_pass = 1u64,
                source_rows = required_rows as u64,
                coefficient_len = coefficient_len as u64,
                term_count = term_count as u64,
                base_to_extension_lifts = 0u64,
                setup_table_state_elements = (term_count * (coefficient_len + row_capacity)) as u64,
            )
            .entered();
            let accumulators = cfg_fold_reduce!(
                0..required_rows,
                || (0..term_count)
                    .map(|_| {
                        (0..coefficient_len)
                            .map(|_| {
                                <E as akita_field::unreduced::HasUnreducedOps>::ProductAccum::zero()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
                |mut accumulators, setup_index| {
                    let row_start = setup_index * coefficient_len;
                    let row = &setup[row_start..row_start + coefficient_len];
                    for ((term_accumulators, index_factor), _) in accumulators
                        .iter_mut()
                        .zip(&index_factors)
                        .zip(&coefficient_factors)
                    {
                        let factor = index_factor[setup_index];
                        if factor.is_zero() {
                            continue;
                        }
                        for (accumulator, &coefficient) in term_accumulators.iter_mut().zip(row) {
                            *accumulator += factor.mul_base_to_product_accum(coefficient);
                        }
                    }
                    accumulators
                },
                |mut left, right| {
                    for (left_term, right_term) in left.iter_mut().zip(right) {
                        for (left, right) in left_term.iter_mut().zip(right_term) {
                            *left += right;
                        }
                    }
                    left
                }
            );
            accumulators
                .into_iter()
                .map(|term| {
                    term.into_iter()
                        .map(E::reduce_product_accum)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let input_claim = coefficient_tables.iter().zip(&coefficient_factors).fold(
            E::zero(),
            |claim, (table, factor)| {
                claim
                    + table
                        .iter()
                        .zip(factor)
                        .fold(E::zero(), |sum, (&value, &weight)| sum + value * weight)
            },
        );
        let coefficient_rounds = coefficient_len.trailing_zeros() as usize;
        let total_rounds = coefficient_rounds + row_capacity.trailing_zeros() as usize;

        let mut term = Self {
            setup,
            required_rows,
            row_capacity,
            coefficient_len,
            coefficient_rounds,
            total_rounds,
            coefficient_challenges: Vec::with_capacity(coefficient_rounds),
            coefficient_tables,
            coefficient_factors,
            index_table: None,
            index_factors,
            index_factor: Vec::new(),
            input_claim,
        };
        // A one-coefficient setup view has no coefficient-round challenge to
        // trigger the normal phase transition. Materialize its index state at
        // construction so the first index round (or a zero-round final value)
        // consumes the same canonical table as every other geometry.
        if coefficient_rounds == 0 {
            term.materialize_index_state();
        }
        Ok(term)
    }

    pub(super) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(&self, round: usize) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.coefficient_rounds {
            self.coefficient_tables
                .iter()
                .zip(&self.coefficient_factors)
                .map(|(table, factor)| accumulate_left_round(table, factor, E::one()))
                .fold(
                    (E::zero(), E::zero(), E::zero()),
                    |(c0, c1, c2), (t0, t1, t2)| (c0 + t0, c1 + t1, c2 + t2),
                )
        } else {
            accumulate_left_round(
                self.index_table
                    .as_deref()
                    .expect("setup index table exists after coefficient rounds"),
                &self.index_factor,
                E::one(),
            )
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(&mut self, round: usize, challenge: E) {
        if round < self.coefficient_rounds {
            self.coefficient_challenges.push(challenge);
            for table in &mut self.coefficient_tables {
                fold_dense_left_round(table, challenge);
            }
            for factor in &mut self.coefficient_factors {
                fold_factor_in_place(factor, challenge);
            }
            if round + 1 == self.coefficient_rounds {
                self.materialize_index_state();
            }
        } else {
            fold_dense_left_round(
                self.index_table
                    .as_mut()
                    .expect("setup index table exists after coefficient rounds"),
                challenge,
            );
            fold_factor_in_place(&mut self.index_factor, challenge);
        }
    }

    fn materialize_index_state(&mut self) {
        let mut index_factor = vec![E::zero(); self.row_capacity];
        for (term_factor, coefficient_factor) in
            self.index_factors.iter().zip(&self.coefficient_factors)
        {
            let scalar = coefficient_factor[0];
            for (combined, &factor) in index_factor.iter_mut().zip(term_factor) {
                *combined += scalar * factor;
            }
        }
        self.index_factor = index_factor;
        self.index_factors.clear();
        let coefficient_eq = EqPolynomial::evals(&self.coefficient_challenges)
            .expect("validated power-of-two setup coefficient domain");
        debug_assert_eq!(coefficient_eq.len(), self.coefficient_len);
        let _span = tracing::info_span!(
            "stage3_setup_index_pass",
            kernel = "rectangular_base_field",
            source_pass = 2u64,
            source_rows = self.row_capacity as u64,
            active_weight_rows = self.required_rows as u64,
            coefficient_len = self.coefficient_len as u64,
            base_to_extension_lifts = 0u64,
            setup_table_state_elements = (self.row_capacity + self.coefficient_len) as u64,
        )
        .entered();
        let index_table = cfg_into_iter!(0..self.row_capacity)
            .map(|setup_index| {
                let start = setup_index * self.coefficient_len;
                eval_flat_ring_at_pows_fast(
                    &self.setup[start..start + self.coefficient_len],
                    &coefficient_eq,
                )
            })
            .collect::<Vec<_>>();
        self.index_table = Some(index_table);
    }

    pub(super) fn folded_table_value(&self) -> Result<E, AkitaError> {
        let table = self
            .index_table
            .as_deref()
            .ok_or(AkitaError::InvalidProof)?;
        if table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: table.len(),
            });
        }
        Ok(table[0])
    }
}

#[cfg(test)]
impl<E: FieldCore + FromPrimitiveInt> FactoredProductTerm<E> {
    /// Construct a dense factored product-sumcheck term.
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    pub(super) fn new_dense(
        table: Vec<E>,
        left_factor: Vec<E>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if left_factor.is_empty()
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".into(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = product_claim(&table, &left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    pub(super) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(&self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(&mut self, round: usize, challenge: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, challenge);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, challenge);
        }
    }

    pub(super) fn folded_table_value(&self) -> Result<E, AkitaError> {
        if self.table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: self.table.len(),
            });
        }
        Ok(self.table[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::scalar_powers;
    use akita_field::Prime128Offset275 as F;

    fn scalar(value: u64) -> F {
        F::from_u64(value)
    }

    fn setup_source(rows: usize, coefficient_len: usize) -> Vec<F> {
        (0..rows * coefficient_len)
            .map(|index| scalar(((index * 17 + index / coefficient_len * 5) % 251 + 1) as u64))
            .collect()
    }

    fn dense_term(
        setup: &[F],
        _required_rows: usize,
        row_capacity: usize,
        coefficient_len: usize,
        index_factor: Vec<F>,
        coefficient_factor: Vec<F>,
    ) -> FactoredProductTerm<F> {
        let mut table = vec![F::zero(); row_capacity * coefficient_len];
        table.copy_from_slice(&setup[..row_capacity * coefficient_len]);
        FactoredProductTerm::new_dense(table, index_factor, coefficient_factor)
            .expect("dense setup product")
    }

    fn assert_round_parity(required_rows: usize, row_capacity: usize, coefficient_len: usize) {
        let setup = setup_source(row_capacity, coefficient_len);
        let mut index_factor = (0..row_capacity)
            .map(|index| scalar((index * 13 + 3) as u64))
            .collect::<Vec<_>>();
        index_factor[required_rows..].fill(F::zero());
        let coefficient_factor = scalar_powers(scalar(7), coefficient_len).to_vec();
        let mut dense = dense_term(
            &setup,
            required_rows,
            row_capacity,
            coefficient_len,
            index_factor.clone(),
            coefficient_factor.clone(),
        );
        let mut rectangular = RectangularSetupProductTerm::new(
            &setup,
            required_rows,
            index_factor,
            coefficient_factor,
        )
        .expect("rectangular setup product");
        assert_eq!(dense.input_claim(), rectangular.input_claim());
        assert_eq!(dense.num_rounds(), rectangular.num_rounds());

        for round in 0..dense.num_rounds() {
            let dense_poly = dense.compute_round_univariate(round, dense.input_claim());
            let rectangular_poly = rectangular.compute_round_univariate(round);
            assert_eq!(dense_poly, rectangular_poly, "round {round}");
            let challenge = scalar((round * 19 + 11) as u64);
            dense.ingest_challenge(round, challenge);
            rectangular.ingest_challenge(round, challenge);
        }
        assert_eq!(
            dense.folded_table_value().expect("dense folded setup"),
            rectangular
                .folded_table_value()
                .expect("rectangular folded setup")
        );
    }

    #[test]
    fn rectangular_setup_product_matches_dense_rounds_with_padding() {
        assert_round_parity(5, 8, 8);
    }

    #[test]
    fn rectangular_setup_product_matches_dense_rounds_without_padding() {
        assert_round_parity(16, 16, 64);
    }

    #[test]
    fn rectangular_setup_product_materializes_index_state_without_coefficient_rounds() {
        assert_round_parity(3, 4, 1);
        assert_round_parity(1, 1, 1);
    }
}
