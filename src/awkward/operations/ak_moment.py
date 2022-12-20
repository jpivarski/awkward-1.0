# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def moment(
    x,
    n,
    weight=None,
    axis=None,
    *,
    keepdims=False,
    mask_identity=False,
    flatten_records=False,
):
    """
    Args:
        x: The data on which to compute the moment (anything #ak.to_layout recognizes).
        n (int): The choice of moment: `0` is a sum of weights, `1` is
            #ak.mean, `2` is #ak.var without subtracting the mean, etc.
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.
        flatten_records (bool): If True, axis=None combines fields from different
            records; otherwise, records raise an error.

    Computes the `n`th moment in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the moment is calculated as

        ak.sum((x*weight)**n) / ak.sum(weight)

    The `n=2` moment differs from #ak.var in that #ak.var also subtracts the
    mean (the `n=1` moment).

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with ak._errors.OperationErrorContext(
        "ak.moment",
        dict(
            x=x,
            n=n,
            weight=weight,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        ),
    ):
        return _impl(x, n, weight, axis, keepdims, mask_identity, flatten_records)


def _impl(x, n, weight, axis, keepdims, mask_identity, flatten_records):
    behavior = ak._util.behavior_of(x, weight)
    x = ak.highlevel.Array(
        ak.operations.to_layout(x, allow_record=False, allow_other=False),
        behavior=behavior,
    )
    if weight is not None:
        weight = ak.highlevel.Array(
            ak.operations.to_layout(weight, allow_record=False, allow_other=False),
            behavior=behavior,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        if weight is None:
            sumw = ak.operations.ak_count._impl(
                x,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
                highlevel=True,
                behavior=behavior,
            )
            sumwxn = ak.operations.ak_sum._impl(
                x**n,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
                highlevel=True,
                behavior=behavior,
            )
        else:
            sumw = ak.operations.ak_sum._impl(
                x * 0 + weight,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
                highlevel=True,
                behavior=behavior,
            )
            sumwxn = ak.operations.ak_sum._impl(
                (x * weight) ** n,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
                highlevel=True,
                behavior=behavior,
            )
        return ak._nplikes.nplike_of(sumwxn, sumw).true_divide(sumwxn, sumw)
