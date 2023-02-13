# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
from awkward._nplikes.numpylike import ArrayLike, NumpyMetadata

np = NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def merge_union_of_records(array, axis=-1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the  innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Simplifies unions of records, e.g.

        >>> array = ak.concatenate(([{"a": 1}], [{"b": 2}]))

    into records of options, i.e.

        >>> ak.merge_union_of_records(array)
        <Array [{a: 1, b: None}, {a: None, ...}] type='2 * {a: ?int64, b: ?int64}'>
    """
    with ak._errors.OperationErrorContext(
        "ak.merge_union_of_records",
        {"array": array, "axis": axis, "highlevel": highlevel, "behavior": behavior},
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.to_layout(array, allow_record=False)

    def invert_record_union(
        tags: ArrayLike, index: ArrayLike, contents, *, backend
    ) -> ak.contents.RecordArray:
        index_nplike = backend.index_nplike
        # First, find all ordered fields, regularising any index-of-record
        # such that we have record-of-index
        seen_fields = set()
        all_fields = []
        for content in contents:
            # Find new fields
            for field in content.fields:
                if field not in seen_fields:
                    seen_fields.add(field)
                    all_fields.append(field)

        # Build unions for each field
        outer_field_contents = []
        for field in all_fields:
            field_tags = index_nplike.asarray(tags, copy=True)
            field_index = index_nplike.asarray(index, copy=True)

            # Build contents for union representing current field
            field_contents = [c.content(field) for c in contents if c.has_field(field)]

            # Find the best location for option type.
            # We will potentially have fewer contents in this per-field union
            # than the original outer union-of-records, because some recordarrays
            # may not have the given field.
            tag_for_missing = 0
            for i, content in enumerate(field_contents):
                if content.is_option:
                    tag_for_missing = i
                    break

            # If at least one recordarray doesn't have this field, we add
            # a special option
            if len(field_contents) < len(contents):
                # Make the tagged content an option, growing by one to ensure we
                # have a known `None` value to index into
                tagged_content = field_contents[tag_for_missing]
                indexedoption_index = backend.index_nplike.arange(
                    tagged_content.length + 1, dtype=np.int64
                )
                indexedoption_index[
                    index_nplike.shape_item_as_index(tagged_content.length)
                ] = -1
                field_contents[
                    tag_for_missing
                ] = ak.contents.IndexedOptionArray.simplified(
                    ak.index.Index64(indexedoption_index), tagged_content
                )

            # Index of None values in tagged content (content with extra None item at end)
            index_missing = index_nplike.shape_item_as_index(
                field_contents[tag_for_missing].length - 1
            )
            # Now build contents for union, by looping over outermost index
            # Overwrite tags to adjust for new contents length
            # and use the tagged content for any missing values
            k = 0
            for j, content in enumerate(contents):
                tag_is_j = field_tags == j

                if content.has_field(field):
                    # Rewrite tags to account for missing fields
                    field_tags[tag_is_j] = k
                    k += 1

                else:
                    # Rewrite tags to point to option content
                    field_tags[tag_is_j] = tag_for_missing
                    # Point each value to missing value
                    field_index[tag_is_j] = index_missing

            outer_field_contents.append(
                ak.contents.UnionArray.simplified(
                    ak.index.Index8(field_tags),
                    ak.index.Index64(field_index),
                    field_contents,
                )
            )
        return ak.contents.RecordArray(
            outer_field_contents, all_fields, backend=backend
        )

    def compact_option_index(index: ArrayLike, *, backend) -> ArrayLike:
        # Find dense (outer) index into non-null items.
        # This is in trivial order: the re-arranging is done by the union (below)
        is_none = index < 0
        num_none = backend.index_nplike.count_nonzero(is_none)
        dense_index = backend.index_nplike.empty(len(index), dtype=index.dtype)
        dense_index[is_none] = -1
        dense_index[~is_none] = backend.index_nplike.arange(
            len(index) - num_none,
            dtype=index.dtype,
        )
        return dense_index

    def apply(layout, depth, backend, **kwargs):
        posaxis = ak._util.maybe_posaxis(layout, axis, depth)
        if depth < posaxis + 1 and layout.is_leaf:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        elif depth == posaxis + 1 and layout.is_union:
            if not all(
                x.is_record or x.is_indexed or x.is_option for x in layout.contents
            ):
                return

            # Any option types need to be re-written
            if any(x.is_option for x in layout.contents):
                # We'll create an outermost indexed-option type, which re-instates the missing values
                outer_option_index = backend.index_nplike.empty(
                    layout.length, dtype=np.int64
                )

                # We'll rebuild the union to include only the non-null items.
                next_union_dense_index_parts = []
                next_contents = []
                next_tags_sparse = backend.index_nplike.asarray(layout.tags, copy=True)
                for tag, content in enumerate(layout.contents):
                    is_this_tag = backend.index_nplike.asarray(layout.tags) == tag

                    # Union arrays for this content
                    tag_index = backend.index_nplike.asarray(layout.index)[is_this_tag]

                    # Optimisation
                    if isinstance(content, ak.contents.UnmaskedArray) or not (
                        content.is_option or content.is_indexed
                    ):
                        next_contents.append(content)
                        next_union_dense_index_parts.append(tag_index)

                    else:
                        content = content.to_IndexedOptionArray64()

                        # Find dense (outer) index into non-null items.
                        outer_option_index[is_this_tag] = compact_option_index(
                            backend.index_nplike.asarray(content.index), backend=backend
                        )

                        # Now find the inner index that actually re-arranges the (non-null) items
                        merged_index = backend.index_nplike.asarray(content.index)[
                            tag_index
                        ]
                        is_non_null = merged_index >= 0
                        next_union_dense_index_parts.append(merged_index[is_non_null])
                        # Mask out tags of items that are missing
                        next_tags_sparse[is_this_tag] = backend.index_nplike.where(
                            is_non_null, tag, -1
                        )

                        # outer_index has same length as layout, so union index should align
                        # union items that are null are set to -1, and those that are not are densified
                        next_contents.append(content.content)

                # Find length of new tags array
                total_length = 0
                for array in next_union_dense_index_parts:
                    total_length += array.size

                # Ignore missing items for inner union
                next_tags = next_tags_sparse[next_tags_sparse >= 0]
                # Inject new dense index
                next_index = backend.index_nplike.empty(total_length, dtype=np.int64)
                for tag, index in enumerate(next_union_dense_index_parts):
                    next_index[next_tags == tag] = index

                next_content = invert_record_union(
                    next_tags, next_index, next_contents, backend=backend
                )
                return ak.contents.IndexedOptionArray(
                    ak.index.Index64(outer_option_index),
                    next_content,
                )

            # Any index types need to be re-written
            elif any(x.is_indexed for x in layout.contents):
                # We'll create an outermost indexed-option type, which re-instates the missing values
                current_index = backend.index_nplike.asarray(layout.index)
                next_index = backend.index_nplike.empty(
                    current_index.size, dtype=np.int64
                )

                # We'll rebuild the union to include only the non-null items.
                next_contents = []
                for tag, content in enumerate(layout.contents):
                    is_this_tag = backend.index_nplike.asarray(layout.tags) == tag

                    # Rewrite union index of indexed types
                    if content.is_indexed:
                        content_index = backend.index_nplike.asarray(content.index)
                        next_index[is_this_tag] = current_index[is_this_tag][
                            content_index
                        ]
                        next_contents.append(content.content)

                    else:
                        next_index[is_this_tag] = current_index[is_this_tag]
                        next_contents.append(content)

                return invert_record_union(
                    backend.index_nplike.asarray(layout.tags),
                    next_index,
                    next_contents,
                    backend=backend,
                )

            else:
                return invert_record_union(
                    backend.index_nplike.asarray(layout.tags),
                    backend.index_nplike.asarray(layout.index),
                    layout.contents,
                    backend=backend,
                )

    out = ak._do.recursively_apply(layout, apply)
    return ak._util.wrap(out, highlevel=highlevel, behavior=behavior)
