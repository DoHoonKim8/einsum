use std::{collections::HashMap, iter::once, marker::PhantomData};

use ff::FromUniformBytes;
use halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    circuit::{AssignedCell, Layouter, Region, Value, floor_planner::V1},
    dev::{FailureLocation, MockProver, VerifyFailure, metadata::Constraint},
    plonk::{
        Advice, Challenge, Column, ConstraintSystem, Error, Expression, FirstPhase, SecondPhase,
        Selector,
    },
    poly::{
        VerificationStrategy,
        commitment::{Params, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::AccumulatorStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{EpAffine, Fq};
use itertools::{Itertools, izip};
use rand::{RngCore, rngs::OsRng};

use super::freivalds::*;

struct FlattenedTensor<F: Field> {
    dims: Vec<usize>,
    values: Vec<Value<F>>,
}

pub enum Tensor<F: Field> {
    Scalar(Value<F>),
    Vector {
        /// dimensions of tensor
        dims: Vec<usize>,
        inner: Vec<Tensor<F>>,
    },
}

impl<F: Field> Tensor<F> {
    fn dims(&self) -> Vec<usize> {
        match self {
            Self::Scalar(_) => vec![],
            Self::Vector { dims, .. } => dims.clone(),
        }
    }

    fn flatten_ordered(&self, order: Vec<usize>) -> Vec<Value<F>> {
        // Flatten along axes in default order
        let flattened = self.flatten();
        let dims = self.dims();
        let mut stride = once(1)
            .chain(dims.iter().skip(1).rev().scan(1, |state, dim| {
                *state *= dim;
                Some(*state)
            }))
            .collect_vec();
        stride.reverse();

        let mut permuted_dims = vec![];
        for dims_idx in order.iter() {
            permuted_dims.push(dims[*dims_idx]);
        }

        let mut permuted_stride = once(1)
            .chain(permuted_dims.iter().skip(1).rev().scan(1, |state, dim| {
                *state *= dim;
                Some(*state)
            }))
            .collect_vec();
        permuted_stride.reverse();

        // Given a coordinate in the original tensor,
        // returns:
        // - its coordinate in the unordered flattened tensor, and
        // - its permuted coordinate in the ordered flattened tensor
        let permuted_coord = |coord: Vec<usize>| -> (usize, usize) {
            let mut permuted_coord = vec![0; coord.len()];
            for (permuted_idx, coord_idx) in order.iter().enumerate() {
                permuted_coord[permuted_idx] = coord[*coord_idx];
            }

            // Do something with dims
            let flattened_coord = coord
                .into_iter()
                .zip(stride.iter())
                .fold(0, |acc, (coord, stride)| acc + coord * *stride);

            let permuted_flattened_coord = permuted_coord
                .into_iter()
                .zip(permuted_stride.iter())
                .fold(0, |acc, (coord, permuted_stride)| {
                    acc + coord * *permuted_stride
                });

            (flattened_coord, permuted_flattened_coord)
        };

        let mut ordered_flattened = vec![Value::unknown(); flattened.len()];
        // Get all coords in original tensor
        let coords: Vec<_> = dims
            .iter()
            .map(|d| (0..*d))
            .multi_cartesian_product()
            .collect();

        for coord in coords.into_iter() {
            let (orig_idx, permuted_idx) = permuted_coord(coord);
            ordered_flattened[permuted_idx] = flattened[orig_idx];
        }

        ordered_flattened
    }

    /// Flattens a tensor along its dimensions into a single vector
    /// along a default ordering of its axes
    fn flatten(&self) -> Vec<Value<F>> {
        match self {
            Tensor::Scalar(v) => vec![v.clone()],
            Tensor::Vector { inner, .. } => inner.iter().flat_map(|t| t.flatten()).collect_vec(),
        }
    }
}

struct Config<F: Field> {
    /// `path` holds the vector of column indices which expresses
    /// the order of tensor contractions on input equation side.
    /// Not sure this should be here
    /// (\sum_{b} r_b W_{bn} Y_{bm})
    path: Vec<Vec<usize>>,
    input_summations: Vec<SummationConfig<F>>,
    /// Witness output
    /// TODO: reuse advice column?
    output: Column<Advice>,
    output_summations: Vec<SummationConfig<F>>,
    // One challenge per output axis
    challenges: Vec<Challenge>,
    _marker: PhantomData<F>,
}

struct MyCircuit<F: Field> {
    inputs: Vec<FlattenedTensor<F>>,
    output: FlattenedTensor<F>,
    /// einsum expression
    equation: String,
}

impl<F: Field> Config<F> {
    fn assign_input(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<FlattenedTensor<F>>,
    ) -> Result<(), Error> {
        // I think we need some function to slice the tensors while fixing some axes
        // For example, \sum_{n,m} (\sum_{a} q_a X_{anm}) (\sum_{b} r_b W_{bn} Y_{bm})
        // when we allocate `SummationConfig` for (\sum_{b} r_b W_{bn} Y_{bm}),
        // MN dot product configs
        // for a fixed `n`, slice W_{bn} ranges: [0..b, n..n+1]
        // the input tensors for this config should be [W_{bn}]_{b} [Y_{bm}]_{b} for fixed `n, m`.

        // bind uncommon indices of input tensors first and pair each of them into tuples
        // for each tuple, expand to the full range of common indices of input tensors
        // and group them into blocks
        // for each block, layout the elements of input tensors
        // n = 2, m = 3
        // (n,m) coordinates:
        // (1,1), (1,2), (1,3)
        // (2,1), (2,2), (2,3)
        //
        // Consider the (1,1) running sum.
        // \sum_{b} r_b W_{b1} Y_{b1}

        // We need a method that:
        //   - Takes in einsum expression, e.g. "bn,anm,bm->ba"
        //   - outputs a series of summations
        //   - each summation involves slicing and then flattening
        //
        // We will always contract the inputs along the output axes.
        // We do four summations on the input eq:
        // 1. a' = \sum_{a} q_a X_{anm}   dim(a') = nm
        //    - flatten X on `a` axis
        //    - in other words, slice X_{anm} ranges: [0..a, 0..n, 0..m]
        // 2. wy = \sum_{b} r_b W_{bn} Y_{bm}    dim(wy) = nm
        //    e.g. n = 2, m = 3
        //    (n,m) coordinates:
        //    (1,1), (1,2), (1,3)
        //    (2,1), (2,2), (2,3)
        //
        //    Consider the (1,1) running sum.
        //    \sum_{b} r_b W_{b1} Y_{b1}
        //      - Slice W_{bn} ranges: [0..b, 1..=1]
        //      - Slice Y_{bm} ranges: [0..b, 1..=1]
        // 3. wy' = \sum_{n} a' wy   dim(wy') = m
        //    - flatten a' on `n` axis, flatten wy' on `n` axis
        //    - do `m` length-`n` running sums
        // 4. wy'' = \sum_{m} wy'    dim(wy'') = 0
        //    - do a single length-`m` running sum

        todo!()
    }

    fn assign_output(
        &self,
        mut layouter: impl Layouter<F>,
        output: FlattenedTensor<F>,
    ) -> Result<(), Error> {
        let mut challenges = vec![];
        for challenge in self.challenges.iter() {
            let challenge = layouter.get_challenge(*challenge);
            challenges.push(challenge);
        }

        // Witness flattened output
        let mut intermediate_values: Vec<AssignedCell<F, F>> = vec![];
        layouter.assign_region(
            || "witness output",
            |mut region| {
                for (offset, value) in output.values.iter().enumerate() {
                    let value = region.assign_advice(|| "", self.output, offset, || *value)?;
                    intermediate_values.push(value);
                }
                Ok(())
            },
        )?;

        // Intermediate values output from the previous summation
        for (i, (summation_config, challenge)) in
            self.output_summations.iter().zip(challenges).enumerate()
        {
            // Powers of the challenge up to dim
            let powers_of_challenge = (0..output.dims[i])
                .scan(Value::known(F::ONE), |state, _| {
                    *state = *state * challenge;
                    Some(*state)
                })
                .collect_vec();
            intermediate_values = summation_config.assign_output(
                layouter.namespace(|| ""),
                &intermediate_values,
                powers_of_challenge,
            )?;
        }

        todo!()
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        equation: &str,
        path: Vec<Vec<usize>>,
        index_to_size: HashMap<char, usize>,
    ) -> Self {
        // based on the contraction path given, for each vector along the path,
        // should generate the sequence of summation_configs
        let (input_eqs, output_eq) = equation.split_once("->").unwrap();
        let input_eqs = input_eqs.split(",").collect_vec();
        let inputs = input_eqs
            .iter()
            .map(|_| meta.advice_column_in(FirstPhase))
            .collect_vec();
        let output = meta.advice_column_in(FirstPhase);

        let num_output_axes = output_eq.chars().count();
        // TODO: Let's first squeeze out all the challenges after first phase
        // and reason about the security later
        let challenges = (0..num_output_axes)
            .map(|_| meta.challenge_usable_after(FirstPhase))
            .collect_vec();

        // TODO: should decide how to compute output tensor squashing
        // 1. | output tensor | ⊗ (challenge vectors) | output running sum |
        //    | ------------- | --------------------- | ------------------ |
        //
        // 2. squash axes one-by-one
        //    | output tensor | challenge vector #1 ... #N | output running sum #1 ... #N |
        //
        // The length of output_summations is num_output_axes
        // For each summation, there are two tensors: the output value and the random challenge.
        let mut output_dims = output_eq
            .chars()
            .into_iter()
            .map(|c| index_to_size.get(&c).unwrap())
            .rev()
            .copied()
            .collect_vec();
        let mut output_summations = vec![];
        for _ in 0..num_output_axes {
            output_dims.remove(0);
            let num_dot_products = output_dims.iter().product();
            let num_inputs = 2;
            output_summations.push(SummationConfig::new(meta, num_dot_products, num_inputs));
        }
        todo!()
    }
}

/// Each `SummationConfig` corresponds to a summation in the einsum argument,
/// i.e. contraction along a single axis.
/// This consists of multiple running sum (i.e. dot product) arguments.
/// Each `SummationConfig` constraints element-wise multiplication and summation between input tensors.
struct SummationConfig<F: Field> {
    // TODO remove this, bake challenge into constraints
    challenge: Column<Advice>,
    dot_products: Vec<DotProductConfig<F>>,
}

impl<F: Field> SummationConfig<F> {
    fn new(
        meta: &mut ConstraintSystem<F>,
        // The number of dot products
        num_dot_products: usize,
        // The number of tensors in the dot product
        num_inputs: usize,
    ) -> Self {
        // TODO optimise choice of advice columns globally
        let inputs: Vec<_> = (0..num_inputs).map(|_| meta.advice_column()).collect();
        let running_sum = meta.advice_column();
        // TODO remove this, bake challenge into constraints
        let challenge = meta.advice_column_in(SecondPhase);

        let mut dot_products = vec![];
        for _ in 0..num_dot_products {
            dot_products.push(DotProductConfig::new(meta, &inputs, running_sum));
        }

        Self {
            dot_products,
            challenge,
        }
    }

    // vec![vec![0,1,2]]
    //   A0   |   A1   |      A2      |     A3
    //   W    |    Y   |  randomness  | running_sum
    // W_{11} | Y_{11} |      r_1     |
    // ...
    // W_{B1} | Y_{B1} |      r_B     |  out_11
    // W_{12} | Y_{11} |      r_1     |
    // ...
    // W_{B2} | Y_{B1} |      r_B     |  out_12
    // ...
    // W_{1N} | Y_{11} |      r_1     |
    // ...
    // W_{BN} | Y_{B1} |      r_B     |  out_1N
    // ...
    //
    fn assign_input(&self, mut layouter: impl Layouter<F>, inputs: &[&[AssignedCell<F, F>]]) {
        let num_dot_products = self.dot_products.len();
        // bind uncommon indices of input tensors first and pair each of them into tuples

        // for each tuple, expand to the full range of common indices of input tensors
        // and group them into blocks

        // for each block, layout the elements of input tensors
    }

    fn assign_output(
        &self,
        mut layouter: impl Layouter<F>,
        tensor: &[AssignedCell<F, F>],
        powers_of_challenge: Vec<Value<F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let num_dot_products = self.dot_products.len();

        // Witness powers of challenge
        let mut challenge_tensor = vec![];
        layouter.assign_region(
            || "witness powers of challenge",
            |mut region| {
                for (offset, value) in powers_of_challenge.iter().enumerate() {
                    let value = region.assign_advice(|| "", self.challenge, offset, || *value)?;
                    challenge_tensor.push(value);
                }
                Ok(())
            },
        )?;

        // Split tensor and challenge vector into dot products
        let mut dot_product_results = vec![];
        let dot_product_len = powers_of_challenge.len();
        for (idx, tensor) in tensor.chunks_exact(dot_product_len).enumerate() {
            let tensors = vec![tensor.to_vec(), challenge_tensor.clone()];
            let result = self.dot_products[idx].assign(layouter.namespace(|| ""), tensors)?;
            dot_product_results.push(result);
        }

        Ok(dot_product_results)
    }
}

struct DotProductConfig<F: Field> {
    selector: (Selector, Selector),
    inputs: Vec<Column<Advice>>,
    running_sum: Column<Advice>,
    _marker: PhantomData<F>,
}

impl<F: Field> DotProductConfig<F> {
    fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        tensors: Vec<Vec<AssignedCell<F, F>>>,
    ) -> Result<AssignedCell<F, F>, Error> {
        assert_eq!(tensors.len(), self.inputs.len());
        let dot_product_len = tensors[0].len();
        layouter.assign_region(
            || "",
            |mut region| {
                // Copy `tensors` values into appropriate cells
                for (offset, inputs) in izip!(tensors.iter()).enumerate() {
                    for (col, cell) in self.inputs.iter().zip(inputs) {
                        cell.copy_advice(|| "", &mut region, *col, offset)?;
                    }
                }

                let running_sum =
                    izip!(tensors.iter()).scan(Value::known(F::ZERO), |state, inputs| {
                        let multiplied = inputs
                            .iter()
                            .map(|input| input.value())
                            .fold(Value::known(F::ONE), |acc, v| acc * v);
                        *state = *state + multiplied;
                        Some(*state)
                    });

                let mut result = None;
                for (offset, running_sum) in running_sum.enumerate() {
                    let running_sum =
                        region.assign_advice(|| "", self.running_sum, offset, || running_sum)?;

                    if offset == 0 {
                        self.selector.0.enable(&mut region, offset)?;
                    } else {
                        self.selector.1.enable(&mut region, offset)?;
                    }

                    if offset == dot_product_len - 1 {
                        result = Some(running_sum)
                    }
                }

                Ok(result.unwrap())
            },
        )
    }

    fn new(
        meta: &mut ConstraintSystem<F>,
        inputs: &[Column<Advice>],
        running_sum: Column<Advice>,
    ) -> Self {
        // TODO cache and retrieve selectors if using repeated advice columns
        let selector = (meta.selector(), meta.selector());

        Self {
            selector,
            inputs: inputs.to_vec(),
            running_sum,
            _marker: PhantomData,
        }
    }
    // Helper dot product gate
    fn dot_product_gate(&self, meta: &mut ConstraintSystem<F>) {
        meta.create_gate("initialization", |_| {
            let s = self.selector.0.expr();
            let init = self
                .inputs
                .iter()
                .fold(Expression::Constant(F::ONE), |acc, input| acc * input.cur());
            vec![s * (self.running_sum.cur() - init)]
        });

        meta.create_gate("accumulation", |_| {
            let s = self.selector.1.expr();
            let acc = self.running_sum.prev();
            let curr = self
                .inputs
                .iter()
                .fold(Expression::Constant(F::ONE), |acc, input| acc * input.cur());
            vec![s * (self.running_sum.cur() - (acc + curr))]
        });
    }
}
