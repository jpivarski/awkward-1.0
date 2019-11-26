################################################################ Contents

class Content:
    def __iter__(self):
        def convert(x):
            if isinstance(x, Content):
                return list(x)
            elif isinstance(x, tuple):
                return tuple(convert(y) for y in x)
            else:
                return x

        for i in range(len(self)):
            yield convert(self[i])

    def __repr__(self):
        return self.tostring_part("", "", "").rstrip()

class FloatArray(Content):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.data[where]
        else:
            return FloatArray(self.data[where])

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<FloatArray>\n"
        out += indent + "    " + " ".join(repr(x) for x in self.data) + "\n"
        out += indent + "</FloatArray>" + post
        return out

class ListArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        self.offsets = offsets
        self.content = content

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.content[self.offsets[where]:self.offsets[where + 1]]
        else:
            start = where.start
            stop = where.stop + 1
            return ListArray(self.offsets[start:stop], self.content)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<ListArray>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListArray>" + post
        return out

class UnionArray(Content):
    def __init__(self, tags, offsets, contents):
        assert isinstance(tags, list)
        assert isinstance(offsets, list)
        assert all(isinstance(x, Content) for x in contents)
        self.tags = tags
        self.offsets = offsets
        self.contents = contents

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.contents[self.tags[where]][self.offsets[where]]
        else:
            return UnionArray(self.tags[where], self.offsets[where], self.contents)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<UnionArray>\n"
        out += indent + "    <tags>" + " ".join(repr(x) for x in self.tags) + "</tags>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        for i, content in enumerate(self.contents):
            out += content.tostring_part(indent + "    ", "<content i=\"{}\">".format(i), "</content>\n")
        out += indent + "</UnionArray>" + post
        return out

class OptionArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        self.offsets = offsets
        self.content = content

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, where):
        if isinstance(where, int):
            if self.offsets[where] == -1:
                return None
            else:
                return self.content[self.offsets[where]]
        else:
            return OptionArray(self.offsets[where], self.content)

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<OptionArray>\n"
        out += indent + "    <offsets>" + " ".join(repr(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.tostring_part(indent + "    ", "<content>", "</content>\n")
        out += indent + "</OptionArray>" + post
        return out

class TupleArray(Content):
    def __init__(self, contents):
        assert len(contents) != 0
        assert all(isinstance(x, Content) for x in contents)
        assert all(isinstance(x, EmptyArray) or len(contents[0]) == len(x) for x in contents)
        self.contents = contents

    def __len__(self):
        return len(self.contents[0])

    def __getitem__(self, where):
        if isinstance(where, int):
            return tuple(x[where] for x in self.contents)
        else:
            return TupleArray([x[where] for x in self.contents])

    def tostring_part(self, indent, pre, post):
        out = indent + pre + "<TupleArray>\n"
        for i, content in enumerate(self.contents):
            out += content.tostring_part(indent + "    ", "<content i=\"{}\">".format(i), "</content>\n")
        out += indent + "</TupleArray>" + post
        return out

class EmptyArray(Content):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, where):
        if isinstance(where, int):
            [][where]
        else:
            return EmptyArray()
    def tostring_part(self, indent, pre, post):
        return indent + pre + "<EmptyArray/>" + post

class EmptyTupleArray(Content):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, where):
        if isinstance(where, int):
            if 0 <= where < self.length:
                return ()
            else:
                [][where]
        else:
            return where.stop - where.start
    def tostring_part(self, indent, pre, post):
        return indent + pre + "<EmptyTupleArray/>" + post

################################################################ Content tests

one = OptionArray([0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6], UnionArray([1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3], [FloatArray([100, 200, 300]), ListArray([0, 1, 4, 4, 5], ListArray([0, 3, 3, 5, 6, 9], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])))]))
# print(one)
# print(list(one))
assert list(one) == [[[1.1, 2.2, 3.3]], None, 100, None, [[], [4.4, 5.5], [6.6]], None, 200, None, [], None, 300, None, [[7.7, 8.8, 9.9]]]
# print()

two = ListArray([0, 2, 2, 2, 3], TupleArray([FloatArray([1, 2, 3]), ListArray([0, 3, 3, 5], FloatArray([1.1, 2.2, 3.3, 4.4, 5.5]))]))
# print(two)
# print(list(two))
assert list(two) == [[(1, [1.1, 2.2, 3.3]), (2, [])], [], [], [(3, [4.4, 5.5])]]
# print()

################################################################ Fillables

class FillableArray:
    def __init__(self):
        self.fillable = UnknownFillable.fromempty()

    def fill(self, x):
        if x is None:
            self._maybeupdate(self.fillable.null())
        elif isinstance(x, (int, float)):
            self._maybeupdate(self.fillable.real(x))
        elif isinstance(x, list):
            self._maybeupdate(self.fillable.beginlist())
            for y in x:
                self.fill(y)
            self._maybeupdate(self.fillable.endlist())
        elif isinstance(x, tuple):
            self._maybeupdate(self.fillable.begintuple(len(x)))
            for i, y in enumerate(x):
                self.fillable.index(i)
                self.fill(y)
            self._maybeupdate(self.fillable.endtuple())
        else:
            raise AssertionError(x)

    def active(self):
        return self.fillable.active()

    def null(self):
        self._maybeupdate(self.fillable.null())

    def real(self, x):
        self._maybeupdate(self.fillable.real(x))

    def beginlist(self):
        self._maybeupdate(self.fillable.beginlist())

    def endlist(self):
        self._maybeupdate(self.fillable.endlist())

    def begintuple(self, numfields):
        self._maybeupdate(self.fillable.begintuple(numfields))

    def index(self, i):
        self._maybeupdate(self.fillable.index(i))

    def endtuple(self):
        self._maybeupdate(self.fillable.endtuple())

    def _maybeupdate(self, fillable):
        assert fillable is not None
        if fillable is not self.fillable:
            self.fillable = fillable

    def snapshot(self):
        return self.fillable.snapshot()

class Fillable:
    pass

class UnknownFillable(Fillable):
    def __init__(self, nullcount):
        assert isinstance(nullcount, int)
        self.nullcount = nullcount

    @classmethod
    def fromempty(cls):
        return UnknownFillable(0)

    def snapshot(self):
        if self.nullcount == 0:
            return EmptyArray()
        else:
            return OptionArray([-1] * self.nullcount, EmptyArray())

    def __len__(self):
        return self.nullcount

    def active(self):
        return False

    def null(self):
        self.nullcount += 1
        return self

    def real(self, x):
        if self.nullcount == 0:
            out = FloatFillable.fromempty()
        else:
            out = OptionFillable.fromnulls(self.nullcount, FloatFillable.fromempty())
        out.real(x)
        return out

    def beginlist(self):
        if self.nullcount == 0:
            out = ListFillable.fromempty()
        else:
            out = OptionFillable.fromnulls(self.nullcount, ListFillable.fromempty())
        out.beginlist()
        return out

    def endlist(self):
        raise ValueError("called 'endlist' without corresponding 'beginlist'")

    def begintuple(self, numfields):
        if self.nullcount == 0:
            out = TupleFillable.fromempty()
        else:
            out = OptionFillable.fromnulls(self.nullcount, TupleFillable.fromempty())
        out.begintuple(numfields)
        return out

    def index(self, i):
        raise ValueError("called 'index' without corresponding 'begintuple'")

    def endtuple(self):
        raise ValueError("called 'endtuple' without corresponding 'begintuple'")

class OptionFillable(Fillable):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Fillable)
        self.offsets = offsets
        self.content = content

    @classmethod
    def fromnulls(cls, nullcount, content):
        return cls([-1] * nullcount, content)

    @classmethod
    def fromvalids(cls, content):
        return cls(list(range(len(content))), content)

    def snapshot(self):
        return OptionArray(list(self.offsets), self.content.snapshot())

    def __len__(self):
        return len(self.offsets)

    def active(self):
        return self.content.active()

    def null(self):
        if self.content.active():
            self.content.null()
        else:
            self.offsets.append(-1)
        return self

    def real(self, x):
        if self.content.active():
            self.content.real(x)
        else:
            length = len(self.content)
            self._maybeupdate(self.content.real(x))
            self.offsets.append(length)
        return self

    def beginlist(self):
        if self.content.active():
            self.content.beginlist()
        else:
            self._maybeupdate(self.content.beginlist())
        return self

    def endlist(self):
        if not self.content.active():
            raise ValueError("'endlist' without corresponding 'beginlist'")
        else:
            length = len(self.content)
            self.content.endlist()
            self.offsets.append(length)
        return self

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

    def _maybeupdate(self, fillable):
        assert fillable is not None
        if fillable is not self.content:
            self.content = content

class UnionFillable(Fillable):
    def __init__(self, tags, offsets, contents):
        assert isinstance(tags, list)
        assert isinstance(offsets, list)
        assert all(isinstance(x, Fillable) for x in contents)
        self.tags = tags
        self.offsets = offsets
        self.contents = contents
        self.current = -1

    @classmethod
    def fromsingle(cls, firstcontent):
        return UnionFillable([0] * len(firstcontent),
                             list(range(len(firstcontent))),
                             [firstcontent])

    def snapshot(self):
        return UnionArray(list(self.tags), list(self.offsets), [x.snapshot() for x in self.contents])

    def __len__(self):
        return len(self.tags)

    def active(self):
        if self.current == -1:
            return False
        else:
            return self.contents[self.current].active()

    def null(self):
        raise NotImplementedError

    def real(self, x):
        raise NotImplementedError

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        raise NotImplementedError

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

class ListFillable(Fillable):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Fillable)
        self.offsets = offsets
        self.content = content
        self.begun = False

    @classmethod
    def fromempty(cls):
        return ListFillable([0], UnknownFillable.fromempty())

    def snapshot(self):
        return ListArray(list(self.offsets), self.content.snapshot())

    def __len__(self):
        return len(self.offsets) - 1

    def active(self):
        return self.begun

    def null(self):
        if self.begun:
            self._maybeupdate(self.content.null())
            return self
        else:
            out = OptionFillable.fromvalids(self)
            out.null()
            return out

    def real(self, x):
        if self.begun:
            self._maybeupdate(self.content.real(x))
            return self
        else:
            # make a union
            raise NotImplementedError

    def beginlist(self):
        if self.begun:
            self._maybeupdate(self.content.beginlist())
        else:
            self.begun = True
        return self

    def endlist(self):
        if self.begun and self.content.active():
            self._maybeupdate(self.content.endlist())
        elif self.begun:
            self.offsets.append(len(self.content))
            self.begun = False
        else:
            raise ValueError("called 'endlist' without corresponding 'beginlist'")
        return self

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

    def _maybeupdate(self, fillable):
        assert fillable is not None
        if fillable is not self.content:
            self.content = fillable

class TupleFillable(Fillable):
    def __init__(self):
        self.length = -1
        self.begun = False

    @classmethod
    def fromempty(cls):
        return TupleFillable()

    def snapshot(self):
        assert self.length != -1
        if len(self.contents) == 0:
            return EmptyTupleArray(self.length)
        else:
            return TupleArray([x.snapshot() for x in self.contents])

    def __len__(self):
        return self.length

    def active(self):
        return self.begun

    def null(self):
        raise NotImplementedError

    def real(self, x):
        assert self.length != -1

        if not self.begun:
            # make a union
            raise NotImplementedError

        elif self.nextindex == -1:
            raise ValueError("'real' called immediately after 'begintuple'; needs 'index' or 'endtuple'")

        elif self.contents[self.nextindex].active():
            self.contents[self.nextindex].real(x)

        else:
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].real(x))

        return self

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        raise NotImplementedError

    def begintuple(self, numfields):
        if self.length == -1:
            self.contents = [UnknownFillable.fromempty() for i in range(numfields)]
            self.length = 0

        if not self.begun and numfields == len(self.contents):
            self.begun = True
            self.nextindex = -1

        elif not self.begun:
            # make a union
            raise NotImplementedError

        elif self.nextindex == -1:
            raise ValueError("'begintuple' called immediately after 'begintuple'; needs 'index' or 'endtuple'")

        elif self.contents[self.nextindex].active():
            self.contents[self.nextindex].begintuple(numfields)

        else:
            self._maybeupdate(self.nextindex, self.contents[self.nextindex].begintuple(numfields))

        return self
            
    def index(self, i):
        assert self.length != -1

        if not self.begun:
            raise ValueError("'index' called without corresponding 'begintuple'")

        elif self.nextindex != -1 and self.contents[self.nextindex].active():
            self.contents[self.nextindex].index(i)

        else:
            self.nextindex = i

        return self

    def endtuple(self):
        assert self.length != -1

        if not self.begun:
            raise ValueError("'endtuple' called without corresponding 'begintuple'")

        elif self.nextindex != -1 and self.contents[self.nextindex].active():
            self.contents[self.nextindex].endtuple()

        else:
            for i in range(len(self.contents)):
                if len(self.contents[i]) == self.length:
                    self._maybeupdate(i, self.contents[i].null())
                assert len(self.contents[i]) == self.length + 1
            self.length += 1
            self.begun = False
            
        return self

    def _maybeupdate(self, index, fillable):
        assert fillable is not None
        if fillable is not self.contents[index]:
            self.contents[index] = fillable

class FloatFillable(Fillable):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    @classmethod
    def fromempty(cls):
        return FloatFillable([])

    def snapshot(self):
        return FloatArray(list(self.data))

    def __len__(self):
        return len(self.data)

    def active(self):
        return False

    def null(self):
        out = OptionFillable.fromvalids(self)
        out.null()
        return out

    def real(self, x):
        self.data.append(x)
        return self

    def beginlist(self):
        raise NotImplementedError

    def endlist(self):
        raise NotImplementedError

    def begintuple(self, numfields):
        raise NotImplementedError

    def index(self, i):
        raise NotImplementedError

    def endtuple(self):
        raise NotImplementedError

################################################################ Fillable tests

fillable = FillableArray()
assert list(fillable.snapshot()) == []
fillable.begintuple(2)
fillable.index(0)
fillable.real(1.1)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None)]
fillable.begintuple(2)
fillable.index(0)
fillable.real(2.2)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None), (2.2, None)]
fillable.begintuple(2)
fillable.index(1)
fillable.real(3.3)
fillable.endtuple()
assert list(fillable.snapshot()) == [(1.1, None), (2.2, None), (None, 3.3)]

datasets = [
    [],
    [None],
    [None, None, None],
    [1.1, 2.2, 3.3],
    [None, 1.1, 2.2, 3.3],
    [1.1, None, 2.2, 3.3],
    [None, 1.1, None, 2.2, 3.3],
    [1.1, 2.2, 3.3, None],
    [1.1, 2.2, None, 3.3, None],
    [None, 1.1, 2.2, 3.3, None],
    [(1, 1.1), (2, 2.2), (3, 3.3)],
    [(1, (2, 3)), (10, (20, 30)), (100, (200, 300))],
    [(1, (2, 3, 4)), (10, (20, 30, 40)), (100, (200, 300, 400))],
    [((1, 2), (3, 4)), ((10, 20), (30, 40)), ((100, 200), (300, 400))],
    [((1, 2, 3), (4, 5)), ((10, 20, 30), (40, 50)), ((100, 200, 300), (400, 500))],
    [((1, 2, 3), (4, 5, 6)), ((10, 20, 30), (40, 50, 60)), ((100, 200, 300), (400, 500, 600))],
    [(1, (2, (3, 4))), (10, (20, (30, 40))), (100, (200, (300, 400)))],
    [(1, ((2, 3), 4)), (10, ((20, 30), 40)), (100, ((200, 300), 400))],
    [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3]],
    [None, [1.1], [1.1, 2.2], [1.1, 2.2, 3.3]],
    [[1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3]],
    [None, [1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3]],
    [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [[1.1], None, [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [None, [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], None],
    [[1.1], [1.1, 2.2], [1.1, None, 3.3]],
    [None, [1.1], [1.1, 2.2], [1.1, None, 3.3]],
    ]

for dataset in datasets:
    fillable = FillableArray()
    for x in dataset:
        fillable.fill(x)
    if list(fillable.snapshot()) != dataset:
        print(dataset)
        print(list(fillable.snapshot()))
        raise AssertionError

# fillable = FillableArray()
# fillable.fill(None)
# fillable.fill([1.1])
# fillable.fill([2.2, 3.3, 4.4])
# print(fillable.snapshot())
# print(list(fillable.snapshot()))
