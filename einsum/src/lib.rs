#![feature(new_range_api)]
use std::{
    collections::{HashMap, HashSet},
    range::Range,
};

use itertools::Itertools;

#[derive(Clone, Debug)]
pub enum Tensor {
    Scalar(usize),
    Vector {
        /// dimensions of tensor
        dims: Vec<usize>,
        inner: Vec<Tensor>,
    },
}

impl From<Vec<Tensor>> for Tensor {
    fn from(value: Vec<Tensor>) -> Self {
        let sub_dims = value[0].dims();
        assert!(value.iter().all(|t| t.dims() == sub_dims));
        let mut dims = vec![value.len()];
        dims.extend_from_slice(sub_dims);
        Self::Vector { dims, inner: value }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Tensor::Scalar(a), Tensor::Scalar(b)) => a == b,
            (Tensor::Scalar(_), Tensor::Vector { .. })
            | (Tensor::Vector { .. }, Tensor::Scalar(_)) => false,
            (
                Tensor::Vector {
                    dims: ldims,
                    inner: linner,
                },
                Tensor::Vector {
                    dims: rdims,
                    inner: rinner,
                },
            ) => ldims == rdims && linner == rinner,
        }
    }
}

impl Tensor {
    pub fn new_zeroes(dims: &[usize]) -> Self {
        if dims.is_empty() {
            Self::Scalar(0)
        } else {
            let inner = (0..dims[0])
                .map(|_| Self::new_zeroes(&dims[1..]))
                .collect_vec();
            Self::Vector {
                dims: dims.to_vec(),
                inner,
            }
        }
    }

    pub fn array(value: Vec<usize>) -> Self {
        Tensor::Vector {
            dims: vec![value.len()],
            inner: value.into_iter().map(Tensor::Scalar).collect_vec(),
        }
    }

    pub fn arange(size: usize) -> Self {
        Self::array((0..size).collect_vec())
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[],
            Self::Vector { dims, .. } => dims,
        }
    }

    /// Return an *owned* tensor containing exactly the elements in `ranges`,
    /// slicing each axis in order.  Panics if you give the wrong number of ranges.
    pub fn slice(&self, ranges: &[Range<usize>]) -> Tensor {
        match self {
            Tensor::Scalar(v) => {
                assert!(ranges.is_empty(), "cannot slice a scalar with ranges");
                Tensor::Scalar(*v)
            }

            Tensor::Vector { dims, inner } => {
                let rank = dims.len();
                assert!(
                    ranges.len() == rank,
                    "must supply exactly {} ranges, got {}",
                    rank,
                    ranges.len()
                );

                // compute the new shape
                let mut new_dims = dims.clone();
                for (i, r) in ranges.iter().enumerate() {
                    assert!(
                        r.start <= r.end && r.end <= dims[i],
                        "range {:?} out of bounds for axis {} (len={})",
                        r,
                        i,
                        dims[i]
                    );
                    new_dims[i] = r.end - r.start;
                }

                // now slice off the first dimension, then recurse
                let first_range = &ranges[0];
                let mut new_inner = Vec::with_capacity(first_range.end - first_range.start);
                for i in *first_range {
                    let child = &inner[i];
                    // for the child, drop the first range and pass the rest
                    new_inner.push(child.slice(&ranges[1..]));
                }

                Tensor::Vector {
                    dims: new_dims,
                    inner: new_inner,
                }
            }
        }
    }

    pub fn get_scalar(&self) -> usize {
        match self {
            Tensor::Scalar(v) => *v,
            Tensor::Vector { dims, inner } => {
                assert!(dims.iter().all(|dim| *dim == 1));
                inner[0].get_scalar()
            }
        }
    }

    pub fn write(&mut self, index: &[usize], value: usize) {
        match self {
            Self::Scalar(v) => *v = value,
            Self::Vector { inner, .. } => {
                inner[index[0]].write(&index[1..], value);
            }
        }
    }
}

pub fn einsum(equation: &str, input_tensors: &[&Tensor]) -> Tensor {
    // parse equation
    let (inputs, output) = equation.split_once("->").unwrap();
    let inputs = inputs.split(",").collect_vec();
    assert_eq!(inputs.len(), input_tensors.len());

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
                    if index_to_dim.contains_key(&index) {
                        common_indices_to_inputs.insert(index);
                    } else {
                        index_to_dim.insert(index, *dim);
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
    let (output_index, output_dim): (Vec<char>, Vec<usize>) =
        output_index_to_dim.iter().cloned().unzip();

    let non_common_indices_to_inputs_exclusive = index_to_dim
        .keys()
        .filter(|c| !common_indices_to_inputs.contains(c) && !output_index.contains(c))
        .cloned()
        .collect_vec();
    // find common indices that are not contained in output equation and their possible ranges
    let common_indices_to_inputs_exclusive = common_indices_to_inputs
        .into_iter()
        .filter(|index| !output_index.contains(index))
        .collect_vec();
    let mut common_indices = common_indices_to_inputs_exclusive
        .iter()
        .map(|index| 0..index_to_dim[index])
        .multi_cartesian_product();
    let mut non_common_indices = non_common_indices_to_inputs_exclusive
        .iter()
        .map(|index| 0..index_to_dim[index])
        .multi_cartesian_product();

    let mut output_tensor = Tensor::new_zeroes(&output_dim);
    // we will iterate over output indices
    let mut output_entries_indices = output_dim
        .into_iter()
        .map(|dim| 0..dim)
        .multi_cartesian_product();
    while let Some(entry_index) = output_entries_indices.next() {
        let mut output_entry = 0;
        // for each input equation, check the index is whether
        // contained in output equation.
        // 1) If yes, bound that index to the same value with output index value
        // 2) If no, range over all possible values for the index
        // Return the slice of each input tensor with indices bounded correctly
        let sliced_inputs = input_tensors
            .iter()
            .zip(inputs.iter())
            .map(|(input_tensor, input_indices)| {
                let mut sliced_dim = vec![];
                input_indices.chars().for_each(|index| {
                    if let Some(pos) = output_index.iter().position(|o| *o == index) {
                        sliced_dim.push((entry_index[pos]..entry_index[pos] + 1).into());
                    } else {
                        sliced_dim.push((0..*index_to_dim.get(&index).unwrap()).into());
                    }
                });
                input_tensor.slice(&sliced_dim)
            })
            .collect_vec();
        // For the indices which aren't contained in output equation,
        // iterate through the range of the common indices to input equations
        // and iterate through the range of uncommon indices, binding all the indices
        // of input equations.
        let mut input_tuples = vec![];
        let mut common_indices_next = common_indices.next();
        loop {
            let sliced_inputs = match common_indices_next {
                Some(common_indices) => sliced_inputs
                    .iter()
                    .enumerate()
                    .map(|(i, slice)| {
                        let mut sliced_dim = vec![];
                        for (c, dim) in inputs[i].chars().zip(slice.dims()) {
                            if let Some(pos) = common_indices_to_inputs_exclusive
                                .iter()
                                .position(|index| *index == c)
                            {
                                sliced_dim
                                    .push((common_indices[pos]..common_indices[pos] + 1).into());
                            } else {
                                sliced_dim.push((0..*dim).into());
                            }
                        }
                        slice.slice(&sliced_dim)
                    })
                    .collect_vec(),
                None => sliced_inputs.clone(),
            };
            // iterate over non common indices
            let mut non_common_indices_next = non_common_indices.next();
            loop {
                match non_common_indices_next {
                    Some(non_common_indices) => {
                        let input_tuple = sliced_inputs
                            .iter()
                            .enumerate()
                            .map(|(i, slice)| {
                                let mut sliced_dim = vec![];
                                for c in inputs[i].chars() {
                                    if let Some(pos) = non_common_indices_to_inputs_exclusive
                                        .iter()
                                        .position(|index| *index == c)
                                    {
                                        sliced_dim.push(
                                            (non_common_indices[pos]..non_common_indices[pos] + 1)
                                                .into(),
                                        );
                                    } else {
                                        sliced_dim.push((0..1).into());
                                    }
                                }
                                slice.slice(&sliced_dim)
                            })
                            .collect_vec();
                        input_tuples.push(input_tuple);
                    }
                    None => {
                        input_tuples.push(sliced_inputs);
                        break;
                    }
                }
                if let Some(non_common_indices) = non_common_indices.next() {
                    non_common_indices_next = Some(non_common_indices);
                } else {
                    break;
                }
            }

            if let Some(common_indices) = common_indices.next() {
                common_indices_next = Some(common_indices);
            } else {
                break;
            }
        }

        output_entry += input_tuples.into_iter().fold(0, |acc, tuple| {
            let term = tuple
                .into_iter()
                .map(|tensor| tensor.get_scalar())
                .product::<usize>();
            acc + term
        });
        output_tensor.write(&entry_index, output_entry);
    }
    output_tensor
}

#[cfg(test)]
mod tests {
    use crate::{Tensor, einsum};

    #[test]
    fn mat_transpose() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let actual = einsum("ij->ji", &[&a]);
        let expected = vec![
            Tensor::array(vec![0, 3]),
            Tensor::array(vec![1, 4]),
            Tensor::array(vec![2, 5]),
        ];
        assert_eq!(actual, expected.into());
    }

    #[test]
    fn sum() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let actual = einsum("ij->", &[&a]);
        let expected = Tensor::Scalar(15);
        assert_eq!(actual, expected);
    }

    #[test]
    fn column_sum() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let actual = einsum("ij->j", &[&a]);
        let expected = vec![Tensor::Scalar(3), Tensor::Scalar(5), Tensor::Scalar(7)].into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn row_sum() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let actual = einsum("ij->i", &[&a]);
        let expected = vec![Tensor::Scalar(3), Tensor::Scalar(12)].into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn mat_vec_mul() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let b = vec![Tensor::Scalar(0), Tensor::Scalar(1), Tensor::Scalar(2)].into();
        let actual = einsum("ik,k->i", &[&a, &b]);
        let expected = vec![Tensor::Scalar(5), Tensor::Scalar(14)].into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn mat_mat_mul() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let b = vec![
            Tensor::array(vec![0, 1, 2, 3, 4]),
            Tensor::array(vec![5, 6, 7, 8, 9]),
            Tensor::array(vec![10, 11, 12, 13, 14]),
        ]
        .into();
        let actual = einsum("ik,kj->ij", &[&a, &b]);
        let expected = vec![
            Tensor::array(vec![25, 28, 31, 34, 37]),
            Tensor::array(vec![70, 82, 94, 106, 118]),
        ]
        .into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn tensor_contraction() {
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
        let actual = einsum("inj,jk->ik", &[&a, &b]);
        let expected = vec![Tensor::array(vec![38, 53]), Tensor::array(vec![110, 161])].into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_product() {
        let a = Tensor::array(vec![0, 1, 2]);
        let b = Tensor::array(vec![3, 4, 5]);
        let actual = einsum("i,i->", &[&a, &b]);
        let expected = Tensor::Scalar(14);
        assert_eq!(actual, expected);
    }

    #[test]
    fn hadamard_product() {
        let a = vec![Tensor::array(vec![0, 1, 2]), Tensor::array(vec![3, 4, 5])].into();
        let b = vec![Tensor::array(vec![6, 7, 8]), Tensor::array(vec![9, 10, 11])].into();
        let actual = einsum("ij,ij->ij", &[&a, &b]);
        let expected = vec![
            Tensor::array(vec![0, 7, 16]),
            Tensor::array(vec![27, 40, 55]),
        ]
        .into();
        assert_eq!(actual, expected);
    }

    #[test]
    fn outer_product() {
        let a = Tensor::array(vec![0, 1, 2]);
        let b = Tensor::array(vec![3, 4, 5, 6]);
        let actual = einsum("i,j->ij", &[&a, &b]);
        let expected = vec![
            Tensor::array(vec![0, 0, 0, 0]),
            Tensor::array(vec![3, 4, 5, 6]),
            Tensor::array(vec![6, 8, 10, 12]),
        ]
        .into();
        assert_eq!(actual, expected);
    }
}
