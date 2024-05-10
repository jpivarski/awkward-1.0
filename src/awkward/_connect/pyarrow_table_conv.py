from __future__ import annotations

import json

import pyarrow

from .pyarrow import AwkwardArrowType

AWKWARD_INFO_KEY = b"awkward_info"  # metadata field in Table schema


def convert_awkward_arrow_table_to_native(aatable: pyarrow.Table) -> pyarrow.Table:
    """
    aatable: A pyarrow Table created with extensionarray=True
    returns: A pyarrow Table without extensionsarrays, but
      with 'awkward_info' in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    new_fields = []
    metadata = []
    for aacol_field in aatable.schema:
        metadata.append(collect_ak_arr_type_metadata(aacol_field))
        new_field = awkward_arrow_field_to_native(aacol_field)
        new_fields.append(new_field)
    metadata_serial = json.dumps(metadata).encode(errors="surrogatescape")
    new_schema = pyarrow.schema(
        new_fields, metadata={AWKWARD_INFO_KEY: metadata_serial}
    )
    new_table = aatable.cast(new_schema)
    return new_table


def convert_native_arrow_table_to_awkward(table: pyarrow.Table) -> pyarrow.Table:
    """
    table: A pyarrow Table converted with convert_awkward_arrow_table_to_native
    returns: A pyarrow Table without extensionsarrays, but
      with 'awkward_info' in the schema's metadata that can be used to
      convert the resulting table back into one with extensionarrays.
    """
    new_fields = []
    metadata = json.loads(
        table.schema.metadata[AWKWARD_INFO_KEY].decode(errors="surrogatescape")
    )
    for aacol_field, field_metadata in zip(table.schema, metadata):
        new_fields.append(
            native_arrow_field_to_akarraytype(aacol_field, field_metadata)
        )
    new_schema = pyarrow.schema(new_fields, metadata=table.schema.metadata)
    # new_table = table.cast(new_schema)
    # return new_table
    return new_schema


def collect_ak_arr_type_metadata(aafield: pyarrow.Field) -> dict | list | None:
    """
    Given a Field, collect ArrowExtensionArray metadata as an object.
    If that field holds more ArrowExtensionArray types, a "subfield_metadata"
    property is added that holds a list of metadata objects for the sub-fields.
    This recurses down the whole type structure.
    """
    typ = aafield.type
    if not isinstance(typ, AwkwardArrowType):
        return None  # Not expected to reach here
    metadata = typ._metadata_as_dict()
    metadata["field_name"] = aafield.name
    if typ.num_fields == 0:
        # Simple type
        return metadata
    # Compound type
    subfield_metadata_list = []
    for ifield in range(typ.num_fields):
        # Note: You can treat some, but not all, compound pyarrow types as iterators.
        # Note: AwkwardArrowType provides num_fields property but not field() method.
        ak_field = typ.storage_type.field(ifield)
        subfield_metadata_list.append(
            collect_ak_arr_type_metadata(ak_field)  # Recurse
        )
    metadata["subfield_metadata"] = subfield_metadata_list
    return metadata


def awkward_arrow_field_to_native(aafield: pyarrow.Field) -> pyarrow.Field:
    """
    Given a Field with ArrowExtensionArray type, returns a corresponding
    field with only Arrow builtin, or storage, types. Metadata is removed.
    """
    typ = aafield.type
    if not isinstance(typ, AwkwardArrowType):
        # Not expected to reach this. Maybe throw ValueError?
        return aafield

    if typ.num_fields == 0:
        # We have a simple type wrapped in AwkwardArrowType.
        new_field = pyarrow.field(
            aafield.name, type=typ.storage_type, nullable=aafield.nullable
        )
        # print(f"  Returning simple field {new_field.name}: {new_field =}")
        return new_field

    # We have a container/compound type, wrapped in AwkwardArrowType.
    # print(f"field {aafield.name}")
    native_fields = []
    for ifield in range(typ.storage_type.num_fields):
        ak_field = typ.storage_type.field(ifield)
        # print(f"Sub-field {ak_field.name}: {ak_field}")
        native_fields.append(
            awkward_arrow_field_to_native(ak_field)  # Recurse
        )

    typ_cls = typ.storage_type.__class__
    if typ_cls not in _pyarrow_type_builder:
        raise NotImplementedError(f"Class {typ_cls} is not handled for conversion.")
    native_type = _pyarrow_type_builder[typ_cls](*native_fields)

    new_field = pyarrow.field(aafield.name, type=native_type, nullable=aafield.nullable)
    # print(f"Returning new field {new_field.name}: {new_field}")
    return new_field


# TODO: add the remaining Arrow non-primitive types that we use
_pyarrow_type_builder = {
    pyarrow.lib.StructType: lambda *subfields: pyarrow.struct(subfields),
    pyarrow.lib.LargeListType: lambda subfield: pyarrow.large_list(subfield),
}


def native_arrow_field_to_akarraytype(
    ntv_field: pyarrow.Field, metadata: dict
) -> pyarrow.Field:
    if isinstance(ntv_field, AwkwardArrowType):
        raise ValueError(f"field {ntv_field} is already an AwkwardArrowType")
    storage_type = ntv_field.type

    if storage_type.num_fields > 0:
        # We need to replace storage_type with one that contains AwkwardArrowTypes.
        awkwardized_fields = []
        for ifield in range(storage_type.num_fields):
            subfield = storage_type.field(ifield)
            submeta = metadata["subfield_metadata"][ifield]
            awkwardized_fields.append(
                native_arrow_field_to_akarraytype(subfield, submeta)  # Recurse
            )

        typ_cls = storage_type.__class__
        if typ_cls not in _pyarrow_type_builder:
            raise NotImplementedError(f"Class {typ_cls} is not handled for conversion.")
        storage_type = _pyarrow_type_builder[typ_cls](*awkwardized_fields)

    ak_type = AwkwardArrowType._from_metadata_object(storage_type, metadata)
    return pyarrow.field(ntv_field.name, type=ak_type, nullable=ntv_field.nullable)
