use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

use itertools::Itertools;
use rand::{Rng, rngs::OsRng};

use crate::{Tensor, einsum};

#[derive(Debug)]
pub struct Error;

/// `freivalds` function checks the result of `einsum` function is equal to `output_tensor`.
/// `freivalds` squashes all the axes of output tensor with random challenge vector
/// $[1, r, r^2, \dots, r^{N-1}]$ where $N$ corresponds to the dimension of the axis.
/// Each axis is squashed with the different challenge vectors to ensure soundness.
/// For Hadamard and outer product cases, we do not use this squashing.
pub fn freivalds(
    equation: &str,
    input_tensors: &[&Tensor],
    output_tensor: &Tensor,
) -> Result<(), Error> {
    // parse equation
    let (inputs, output) = equation.split_once("->").unwrap();
    let inputs = inputs.split(",").collect_vec();
    assert_eq!(inputs.len(), input_tensors.len());

    // 1: check whether there exist common indices in the input equation exclusively
    // - 1) if yes, jump to 2.
    // - 2) it no, this means that the operation is either Hadamard or outer product,
    //      then don't squeeze any random vector, jump to 5.
    let mut index_to_dim: HashMap<char, usize> = HashMap::new();
    let mut common_indices_to_inputs = HashSet::new();
    // creates (index -> dim) map
    inputs
        .iter()
        .zip(input_tensors)
        .for_each(|(indices, tensor)| {
            let tensor_dim = tensor.dims();
            indices
                .chars()
                .zip(tensor_dim.iter())
                .for_each(|(index, dim)| {
                    if let std::collections::hash_map::Entry::Vacant(e) = index_to_dim.entry(index)
                    {
                        e.insert(*dim);
                    } else {
                        common_indices_to_inputs.insert(index);
                    }
                });
        });
    let output_index_to_dim = output
        .chars()
        .map(|index| {
            if let Some(dim) = index_to_dim.get(&index) {
                (index, *dim)
            } else {
                // if the index is not found in the inputs, set to 1
                (index, 1)
            }
        })
        .collect_vec();
    let (output_index, _): (Vec<char>, Vec<usize>) = output_index_to_dim.iter().cloned().unzip();
    let common_indices_to_inputs_exclusive = common_indices_to_inputs
        .into_iter()
        .filter(|index| !output_index.contains(index))
        .collect_vec();

    let mut updated_input_equations = inputs
        .iter()
        .copied()
        .map(String::from_str)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut updated_output_equation = vec![output.to_string()];
    if common_indices_to_inputs_exclusive.is_empty() || output.is_empty() {
        // Hadamard or outer product or dot product
        if einsum(equation, input_tensors) == *output_tensor {
            return Ok(());
        } else {
            return Err(Error);
        }
    }
    let mut input_tensors = input_tensors.iter().copied().cloned().collect_vec();
    let mut output_tensors = vec![output_tensor.clone()];

    // 2: for each output index, will squeeze out random vector
    let mut rng = OsRng;
    let challenge_vectors: Vec<Tensor> = output_index_to_dim
        .iter()
        .map(|(_, size)| {
            (0..*size)
                .map(|_| Tensor::Scalar(rng.r#gen::<u16>() as usize))
                .collect_vec()
                .into()
        })
        .collect_vec();

    // 3: augment the output indices to input equations and the output equation
    //    the strategy is to squash the input tensor along the output axes
    output_index.into_iter().for_each(|c| {
        updated_input_equations.push(c.to_string());
        updated_output_equation.push(c.to_string());
    });

    input_tensors.extend_from_slice(challenge_vectors.as_slice());
    output_tensors.extend_from_slice(challenge_vectors.as_slice());

    // 4: do einsum logic on input equations
    // TODO : instead of calling `einsum`, reorder the contraction to make the number of multiplications small
    let mut updated_input_equations = updated_input_equations.join(",");
    updated_input_equations.push_str("->");
    let squashed_input = einsum(
        &updated_input_equations,
        &input_tensors.iter().collect_vec(),
    );

    // A = M x N
    // flatten(A) -> single column, row-major
    // B = L x M x N
    // flatten(B) -> two columns (two phases)
    //  - B_1 -> single column, LMN entries, L groups of M subgroups of N scalars
    //  - B_1 dot product with [r, r^2, ..., r^N] output: L groups of M scalars
    //  - B_2 -> single column, LM entries, L groups of M
    //  - B_2 dot product with [q, q^2, ..., q^M] output: L scalars
    //  - B_3 -> single column, L entries
    //  - B_3 dot product with [s, s^2, ..., s^L] output: b (a scalar)
    // TODO: does q need to be in a separate phase?

    // 5: do einsum logic on output equation
    // TODO : instead of calling `einsum`, reorder the contraction to make the number of multiplications small
    let mut updated_output_equation = updated_output_equation.join(",");
    updated_output_equation.push_str("->");
    let squashed_output = einsum(
        &updated_output_equation,
        &output_tensors.iter().collect_vec(),
    );

    println!("updated_input_equations : {:?}", updated_input_equations);
    println!("updated_output_equation : {:?}", updated_output_equation);

    // 6: check whether two squashed results are the same
    assert_eq!(squashed_input, squashed_output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        Tensor, einsum,
        freivalds::{Error, freivalds},
    };

    #[test]
    fn mat_vec_mul() -> Result<(), Error> {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let b = vec![Tensor::Scalar(0), Tensor::Scalar(1), Tensor::Scalar(2)].into();
        let equation = "ik,k->i";
        let output = einsum(equation, &[&a, &b]);
        freivalds(equation, &[&a, &b], &output)
    }

    #[test]
    fn mat_mat_mul() -> Result<(), Error> {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let b = vec![
            Tensor::array(vec![0, 1, 2, 3, 4]),
            Tensor::array(vec![5, 6, 7, 8, 9]),
            Tensor::array(vec![10, 11, 12, 13, 14]),
        ]
        .into();
        let equation = "ik,kj->ij";
        let output = einsum(equation, &[&a, &b]);
        freivalds(equation, &[&a, &b], &output)
    }

    #[test]
    fn dot_product() -> Result<(), Error> {
        let equation = "i,i->";
        let a = Tensor::array(vec![0, 1, 2]);
        let b = Tensor::array(vec![3, 4, 5]);
        let output = einsum(equation, &[&a, &b]);
        freivalds(equation, &[&a, &b], &output)
    }

    #[test]
    fn batch_mat_mul() -> Result<(), Error> {
        let a = vec![
            vec![
                Tensor::array(vec![0, 1, 2, 3, 4]),
                Tensor::array(vec![5, 6, 7, 8, 9]),
            ]
            .into(),
            vec![
                Tensor::array(vec![10, 11, 12, 13, 14]),
                Tensor::array(vec![15, 16, 17, 18, 19]),
            ]
            .into(),
            vec![
                Tensor::array(vec![20, 21, 22, 23, 24]),
                Tensor::array(vec![25, 26, 27, 28, 29]),
            ]
            .into(),
        ]
        .into();
        let b = vec![
            vec![
                Tensor::array(vec![0, 1, 2]),
                Tensor::array(vec![3, 4, 5]),
                Tensor::array(vec![6, 7, 8]),
                Tensor::array(vec![9, 10, 11]),
                Tensor::array(vec![12, 13, 14]),
            ]
            .into(),
            vec![
                Tensor::array(vec![15, 16, 17]),
                Tensor::array(vec![18, 19, 20]),
                Tensor::array(vec![21, 22, 23]),
                Tensor::array(vec![24, 25, 26]),
                Tensor::array(vec![27, 28, 29]),
            ]
            .into(),
            vec![
                Tensor::array(vec![30, 31, 32]),
                Tensor::array(vec![33, 34, 35]),
                Tensor::array(vec![36, 37, 38]),
                Tensor::array(vec![39, 40, 41]),
                Tensor::array(vec![42, 43, 44]),
            ]
            .into(),
        ]
        .into();
        let equation = "ijk,ikl->ijl";
        // ikl,l->ik
        // ijk,j->ik
        // i,ik,ik->

        // \sum_{i} r_i (\sum_{j} r'_j A_{ijk}) (\sum_{l} r''_l B_{ikl})
        // [\sum_{i} r_i (\sum_{j} r'_j A_{ijk})] • [\sum_{i} r_i (\sum_{l} r''_l B_{ikl})]

        // \sum_{k} A_{ijk}B_{ikl} = C_{ijl}
        // \sum_{i} r_i \sum_{j} r'_j \sum_{l} r''_l C_{ijl} -> output
        // \sum_{i} r_i \sum_{j} r'_j \sum_{l} r''_l \sum_{k} A_{ijk}B_{ikl} -> input

        // A (B r) = Cr
        let output = einsum(equation, &[&a, &b]);
        freivalds(equation, &[&a, &b], &output)
    }

    #[test]
    fn tensor_contraction() -> Result<(), Error> {
        let a: Tensor = vec![
            vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into(),
            vec![Tensor::array(vec![6, 7, 8]), Tensor::array(vec![9, 10, 11])].into(),
        ]
        .into();
        let b = vec![
            Tensor::array(vec![0, 1]),
            Tensor::array(vec![2, 3]),
            Tensor::array(vec![4, 5]),
        ]
        .into();
        let equation = "inj,jk->ik";
        // i,inj->nj
        // jk,k->j
        // nj,j->

        // \sum_{n} (\sum_{j} ((\sum_{i} r_i A_{inj}) (\sum_{k} r'_k B_{jk})))

        // \sum_{n} \sum_{j} A_{inj} B_{jk} = C_{ik}
        // \sum_{i} r_i \sum_{k} r'_k \sum_{n} \sum_{j} A_{inj} B_{jk} = \sum_{i} r_i \sum_{k} r'_k C_{ik}
        let output = einsum(equation, &[&a, &b]);
        freivalds(equation, &[&a, &b], &output)
    }
}
