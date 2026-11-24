/// Count global partitions without accepting an empty inferred partition.
pub(crate) fn partition_counts(
    problem: [u32; 3],
    partition: [u32; 3],
) -> Result<[u32; 3], &'static str> {
    if partition.contains(&0) {
        return Err("global matmul partition dimensions must be nonzero");
    }
    Ok([
        problem[0].div_ceil(partition[0]),
        problem[1].div_ceil(partition[1]),
        problem[2].div_ceil(partition[2]),
    ])
}

#[cfg(test)]
mod tests {
    use super::partition_counts;

    #[test]
    fn zero_partitions_are_rejected_on_every_axis() {
        for partition in [[0, 16, 1], [128, 0, 1], [128, 16, 0]] {
            assert!(
                partition_counts([128, 16, 1], partition).is_err(),
                "accepted {partition:?}"
            );
        }
    }

    #[test]
    fn partial_exact_and_empty_problems_keep_their_grid() {
        assert_eq!(partition_counts([129, 17, 3], [8, 16, 2]), Ok([17, 2, 2]));
        assert_eq!(partition_counts([128, 16, 1], [8, 16, 1]), Ok([16, 1, 1]));
        assert_eq!(partition_counts([0, 16, 1], [8, 16, 1]), Ok([0, 1, 1]));
        assert_eq!(
            partition_counts([u32::MAX, 1, 1], [2, 1, 1]),
            Ok([1 << 31, 1, 1])
        );
    }
}
