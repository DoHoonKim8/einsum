use std::marker::PhantomData;

use ff::FromUniformBytes;
use halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    circuit::{floor_planner::V1, Layouter, Value},
    dev::{metadata::Constraint, FailureLocation, MockProver, VerifyFailure},
    plonk::*,
    poly::{
        commitment::{Params, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{EpAffine, Fq};
use itertools::Itertools;
use rand::{rngs::OsRng, RngCore};

use super::freivalds::*;

struct FlattenedTensor<F: Field> {
    dims: Vec<usize>,
    values: Vec<Value<F>>
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
    /// Flattens a tensor along its dimensions into a single vector
    fn flatten(&self) -> Vec<Value<F>> {
        match self {
            Tensor::Scalar(v) => vec![v.clone()],
            Tensor::Vector { inner, .. } => {
                inner.into_iter().flat_map(|t| t.flatten()).collect_vec()
            }
        }
    }
}

struct Config<F: Field> {
    inputs: Vec<Column<Advice>>,
    input_randomness: Vec<Column<Advice>>,
    inputs_running_sum: Vec<Column<Advice>>,
    output: Column<Advice>,
    output_running_sum: Column<Advice>,
    output_randomness: Vec<Column<Advice>>,
    // TODO: selectors
    _marker: PhantomData<F>
}

struct MyCircuit<F: Field> {
    inputs: Vec<FlattenedTensor<F>>,
    output: FlattenedTensor<F>,
    randomness: Vec<Vec<Value<F>>>,   
}

impl<F: Field> Config<F> {
    // Helper dot product gate
    fn dot_product_gate() {

    }

    fn witness_inputs(
        &self,
        input_tensors: &[&Tensor<F>],
        mut layouter: impl Layouter<F>,
    ) -> Vec<FlattenedTensor<F>> {
        assert_eq!(input_tensors.len(), self.inputs.len());
        for (input, col) in input_tensors.iter().zip(self.inputs.iter()) {
            let flattened = input.flatten();
            // Assign flattened to column
        }

        todo!()
    }

    fn witness_output(
        &self,
        output_tensor: &Tensor<F>,
        mut layouter: impl Layouter<F>,
    ) -> FlattenedTensor<F> {
        let flattened = output_tensor.flatten();
        let col = self.output;
        // Assign flattened to column

        todo!()
    }

    fn witness_randomness(
        &self,
        input_tensors: Vec<FlattenedTensor<F>>,
        output_tensor: FlattenedTensor<F>,
        randomness: &[&[Value<F>]],
        mut layouter: impl Layouter<F>,
    ) {
        // witness input randomness
        let input_randomness: Vec<Vec<Value<F>>> = vec![];

        // witness output randomness

        // witness input running sums

        // witness output running sum

        // constrain equality
    }
}