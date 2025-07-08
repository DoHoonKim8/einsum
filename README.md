# About

Self-study Rust implementation of Pytorch einsum function

## Generalization of Freivalds' algorithm to einsum

```
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
// TODO: does each axis randomness need to be in a separate phase?

// bn,anm,bm->ba
//  W  X  Y -> Z
//
// Output:
// - r_b = [r, r^2, ..., r^b]
// - q_a = [q, q^2, ..., q^a]
// z = r_b Z q_a
//
// Input:
// \sum_{a} q_a \sum_{b} r_b \sum_{n,m} W_{bn} X_{anm} Y_{bm}
// \sum_{n,m} (\sum_{a} q_a X_{anm}) (\sum_{b} r_b W_{bn} Y_{bm})
//
// \sum_{b} r_b W_{bn} Y_{bm}
// = \sum_{i} r_b[i] w_{ij} y_{ik}
// We'll end up with an n x m matrix, call this T
// T_{jk} = \sum_{i} r_b[i] w_{ij} y_{ik}
//
// x = q_a X, anm->nm
// wy = r_b W Y, bn,bm->nm
// nm,nm->
// [
    [1,2],
    [3,4],
    [5,6]
    ]
// [
    [7,8],
    [9,10],
    [11,12]
    ]
// [
    [1*7,2*8],
    [3*9,4*10],
    [5*11,6*12],
]
// Input subconfigs:
// - \sum_{a}
// - \sum_{b}
// - \sum_{n}
// - \sum_{m}
//
// - \sum_{b}
// (\sum_{b} r_b W_{bn} Y_{bm})
```

```rust
struct SubConfig {
    // W, Y
    // We have M blocks, each block does N running sums of length B
    // Each block is of length BN
    // Block 1
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
    // Block 2
    // W_{11} | Y_{12} |      r_1     |
    // ...
    // W_{B1} | Y_{B2} |      r_B     |  out_21
    // ...
    // W_{1N} | Y_{12} |      r_1     |
    // ...
    // W_{BN} | Y_{B2} |      r_B     |  out_2N
    // ...
    // Block M
    inputs: Vec<Column<Advice>>,
    running_sum: Column<Advice>,
    selector: (Selector, Selector),
}
```
Config 0
 a_0   a_1   a_2   a_3
------------------------
| sc0| sc1 | sc2 | sc3 |
|    |     |     |
|    |     |
|    |     |
|    |
|    |
|    |
|    |

Config 1
 a_3   a_2   a_1   a_0
------------------------
|    |     |     |     |
|    |     |     |
|    |     |
|    |     |
|    |
|    |
|    |
|    |


```
// output equation side:
// z = r_b Z q_a
// z = \sum_{a} q_a \sum_{b} r_b Z_{a,b}
// z = \sum_{a} \sum_{b} (q_a r_b) Z_{a,b}
```

The following is the advice columns that we need.

| input tensors | input randomness | input running sum | output tensor | output randomness | output running sum |
| ------------- | ---------------- | ----------------- | ------------- | ----------------- | ------------------ |

For input equation side, the first step is to find the contraction path that has the smallest number of multiplications. This can be found by some contraction planner after augmenting input equation indices with challenge vector indices. (at compile-time)
We will flatten each tensors (including challenge vectors) in row-major into a single column.

If path is given as the vector of `(usize, usize)`, where each indicates the index of column, then we can group top-level circuit configuration into the following multiple sub-configurations. Actually this `SubConfig` constraints the dot product relation between two inputs. However, we should also cache the intermediate result of tensor operation into a separate column. So we also have to consider these intermediate columns before scheduling the path.

```rust
struct SubConfig {
    inputs: [Column<Advice>; 2],
    running_sum: Column<Advice>,
    selector: (Selector, Selector),
}
```

## Reference

- https://github.com/zkonduit/ezkl/blob/main/src/circuit/ops/layouts.rs
- https://hackmd.io/cNJ41EJ4Q2apGkbqulmxAQ?both
