# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumpyMetadata.instance()


class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, Index) and tags.dtype == np.dtype(np.int8)
        assert isinstance(index, Index) and index.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        )
        assert isinstance(contents, list)
        assert len(index) >= len(tags)  # usually equal
        self._tags = tags
        self._index = index
        self._contents = contents

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    def __len__(self):
        return len(self._tags)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnionArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._tags._repr(indent + "    ", "<tags>", "</tags>\n"))
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        for i, x in enumerate(self._contents):
            out.append("{0}    <content index={1}>\n".format(indent, repr(str(i))))
            out.append(x._repr(indent + "        ", "", "\n"))
            out.append("{0}    </content>\n".format(indent))
        out.append(indent)
        out.append("</UnionArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        return self._contents[self._tags[where]][self._index[where]]

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return UnionArray(
            Index(self._tags[start:stop]),
            Index(self._index[start:stop]),
            self._contents,
        )

    def _getitem_field(self, where):
        return UnionArray(self._tags, self._index, [x[where] for x in self._contents])

    def _getitem_fields(self, where):
        return UnionArray(self._tags, self._index, [x[where] for x in self._contents])
