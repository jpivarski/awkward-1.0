import math
import random

class Content:
    def __iter__(self):
        def convert(x):
            if isinstance(x, Content):
                return list(x)
            elif isinstance(x, tuple):
                return tuple(convert(y) for y in x)
            elif isinstance(x, dict):
                return {n: convert(y) for n, y in x.items()}
            else:
                return x

        for i in range(len(self)):
            yield convert(self[i])

    @staticmethod
    def random(minlen=0, choices=None):
        "Generate a random array from a set of possible classes."

        if choices is None:
            choices = [x for x in globals().values() if isinstance(x, type) and issubclass(x, Content)]
        else:
            choices = list(choices)
        if minlen != 0 and EmptyArray in choices:
            choices.remove(EmptyArray)
        assert len(choices) > 0
        cls = random.choice(choices)
        return cls.random(minlen, choices)

    def tolist(self):
        return list(self)

def random_number():
    return round(random.gauss(5, 3), 1)

def random_length(minlen=0, maxlen=None):
    if maxlen is None:
        return minlen + int(math.floor(random.expovariate(0.1)))
    else:
        return random.randint(minlen, maxlen)

class RawArray(Content):
    def __init__(self, ptr):
        assert isinstance(ptr, list)
        self.ptr = ptr

    @staticmethod
    def random(minlen=0, choices=None):
        return RawArray([random_number() for i in range(random_length(minlen))])

    def __len__(self):
        return len(self.ptr)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.ptr[where]
        elif isinstance(where, slice) and where.step is None:
            return RawArray(self.ptr[where])
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        return RawArray([self.ptr[i] for i in index])

    def purelist_depth(self):
        return 1

    def minmax_depth(self):
        return 1, 1

    def branch_depth(self):
        return False, 1

    def __repr__(self):
        return "RawArray(" + repr(self.ptr) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RawArray>\n"
        out += indent + "    <ptr>" + " ".join(repr(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "</RawArray>" + post
        return out

class NumpyArray(Content):
    def __init__(self, ptr, shape, strides, offset):
        assert isinstance(ptr, list)
        assert isinstance(shape, list)
        assert isinstance(strides, list)
        for x in ptr:
            assert isinstance(x, (bool, int, float))
        assert len(shape) > 0
        assert len(strides) == len(shape)
        for x in shape:
            assert isinstance(x, int)
            assert x >= 0
        for x in strides:
            assert isinstance(x, int)
        assert isinstance(offset, int)
        if all(x != 0 for x in shape):
            assert 0 <= offset < len(ptr)
            last = offset
            for sh, st in zip(shape, strides):
                last += (sh - 1) * st
            assert last <= len(ptr)
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.offset = offset

    @staticmethod
    def random(minlen=0, choices=None):
        shape = [random_length(minlen)]
        for i in range(random_length(0, 2)):
            shape.append(random_length(1, 3))
        strides = [1]
        for x in shape[:0:-1]:
            skip = random_length(0, 2)
            strides.insert(0, x * strides[0] + skip)
        offset = random_length()
        ptr = [random_number() for i in range(shape[0] * strides[0] + offset)]
        return NumpyArray(ptr, shape, strides, offset)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            offset = self.offset + self.strides[0] * where
            if len(self.shape) == 1:
                return self.ptr[offset]
            else:
                return NumpyArray(self.ptr, self.shape[1:], self.strides[1:], offset)
        elif isinstance(where, slice) and where.step is None:
            offset = self.offset + self.strides[0] * where.start
            shape = [where.stop - where.start] + self.shape[1:]
            return NumpyArray(self.ptr, shape, self.strides, offset)
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        shape = [len(index)] + self.shape[1:]
        strides = [1]
        for x in shape[:0:-1]:
            strides = [strides[0] * x] + strides
        if len(self.shape) == 1:
            ptr = [self.ptr[self.offset + self.strides[0] * i] for i in index]
        else:
            o = self.offset
            sh0, sh1 = self.shape[0], self.shape[1]
            st0, st1 = self.strides[0], self.strides[1]
            ptr = sum([self.ptr[o + st0*i : o + st0*i + st1*sh1] for i in index], [])
        return NumpyArray(ptr, shape, strides, 0)

    def purelist_depth(self):
        return len(self.shape)

    def minmax_depth(self):
        return len(self.shape), len(self.shape)

    def branch_depth(self):
        return False, len(self.shape)

    def __repr__(self):
        return "NumpyArray(" + repr(self.ptr) + ", " + repr(self.shape) + ", " + repr(self.strides) + ", " + repr(self.offset) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<NumpyArray>\n"
        out += indent + "    <ptr>" + " ".join(str(x) for x in self.ptr) + "</ptr>\n"
        out += indent + "    <shape>" + " ".join(str(x) for x in self.shape) + "</shape>\n"
        out += indent + "    <strides>" + " ".join(str(x) for x in self.strides) + "</strides>\n"
        out += indent + "    <offset>" + str(self.offset) + "</offset>\n"
        out += indent + "</NumpyArray>" + post
        return out

class EmptyArray(Content):
    def __init__(self):
        pass

    @staticmethod
    def random(minlen=0, choices=None):
        assert minlen == 0
        return EmptyArray()

    def __len__(self):
        return 0

    def __getitem__(self, where):
        if isinstance(where, int):
            assert False
        elif isinstance(where, slice) and where.step is None:
            return EmptyArray()
        elif isinstance(where, str):
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def carry(self, index):
        # this will always fail
        assert all(0 <= i < len(self) for i in index)

    def purelist_depth(self):
        return 1

    def minmax_depth(self):
        return 1, 1

    def branch_depth(self):
        return False, 1

    def __repr__(self):
        return "EmptyArray()"

    def toxml(self, indent="", pre="", post=""):
        return indent + pre + "<EmptyArray/>" + post

class RegularArray(Content):
    def __init__(self, content, size):
        assert isinstance(content, Content)
        assert isinstance(size, int)
        assert size > 0
        self.content = content
        self.size = size

    @staticmethod
    def random(minlen=0, choices=None):
        size = random_length(1, 5)
        return RegularArray(Content.random(random_length(minlen) * size, choices), size)

    def __len__(self):
        return len(self.content) // self.size   # floor division

    def __getitem__(self, where):
        if isinstance(where, int):
            return self.content[(where) * self.size:(where + 1) * self.size]
        elif isinstance(where, slice) and where.step is None:
            start = where.start * self.size
            stop = where.stop * self.size
            return RegularArray(self.content[start:stop], self.size)
        elif isinstance(where, str):
            return RegularArray(self.content[where], self.size)
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        nextcarry = [None] * len(index) * self.size
        for i in range(len(index)):
            for j in range(self.size):
                nextcarry[i*self.size + j] = index[i]*self.size + j
        return RegularArray(self.content.carry(nextcarry), self.size)

    def purelist_depth(self):
        return self.content.purelist_depth() + 1

    def minmax_depth(self):
        min, max = self.content.minmax_depth()
        return min + 1, max + 1

    def branch_depth(self):
        branch, depth = self.content.branch_depth()
        if branch:
            return branch, depth
        else:
            return False, depth + 1

    def __repr__(self):
        return "RegularArray(" + repr(self.content) + ", " + repr(self.size) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RegularArray>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "    <size>" + str(self.size) + "</size>\n"
        out += indent + "</RegularArray>" + post
        return out

class ListOffsetArray(Content):
    def __init__(self, offsets, content):
        assert isinstance(offsets, list)
        assert isinstance(content, Content)
        assert len(offsets) != 0
        for i in range(len(offsets) - 1):
            start = offsets[i]
            stop = offsets[i + 1]
            assert isinstance(start, int)
            assert isinstance(stop, int)
            if start != stop:
                assert start < stop   # i.e. start <= stop
                assert start >= 0
                assert stop <= len(content)
        self.offsets = offsets
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        counts = [random_length() for i in range(random_length(minlen))]
        offsets = [random_length()]
        for x in counts:
            offsets.append(offsets[-1] + x)
        return ListOffsetArray(offsets, Content.random(offsets[-1], choices))

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.offsets[where]:self.offsets[where + 1]]
        elif isinstance(where, slice) and where.step is None:
            offsets = self.offsets[where.start : where.stop + 1]
            if len(offsets) == 0:
                offsets = [0]
            return ListOffsetArray(offsets, self.content)
        elif isinstance(where, str):
            return ListOffsetArray(self.offsets, self.content[where])
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        # starts = [self.offsets[i] for i in index]
        # stops = [self.offsets[i + 1] for i in index]
        # return ListArray(starts, stops, self.content)

        # ordinarily, we'd output a ListArray, but I want fewer things to worry about right now
        nextoffsets = [None] * (len(index) + 1)
        nextoffsets[0] = 0
        sumcounts = 0
        for i in range(len(index)):
            count = self.offsets[index[i] + 1] - self.offsets[index[i]]
            sumcounts += count
            nextoffsets[i + 1] = nextoffsets[i] + count
        nextindex = [None] * sumcounts
        k = 0
        for i in range(len(index)):
            for j in range(self.offsets[index[i]], self.offsets[index[i] + 1]):
                nextindex[k] = j
                k += 1
        return ListOffsetArray(nextoffsets, self.content.carry(nextindex))

    def purelist_depth(self):
        return self.content.purelist_depth() + 1

    def minmax_depth(self):
        min, max = self.content.minmax_depth()
        return min + 1, max + 1

    def branch_depth(self):
        branch, depth = self.content.branch_depth()
        if branch:
            return branch, depth
        else:
            return False, depth + 1

    def __repr__(self):
        return "ListOffsetArray(" + repr(self.offsets) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ListOffsetArray>\n"
        out += indent + "    <offsets>" + " ".join(str(x) for x in self.offsets) + "</offsets>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListOffsetArray>" + post
        return out

class ListArray(Content):
    def __init__(self, starts, stops, content):
        assert isinstance(starts, list)
        assert isinstance(stops, list)
        assert isinstance(content, Content)
        assert len(stops) >= len(starts)   # usually equal
        for i in range(len(starts)):
            start = starts[i]
            stop = stops[i]
            assert isinstance(start, int)
            assert isinstance(stop, int)
            if start != stop:
                assert start < stop   # i.e. start <= stop
                assert start >= 0
                assert stop <= len(content)
        self.starts = starts
        self.stops = stops
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        content = Content.random(0, choices)
        length = random_length(minlen)
        if len(content) == 0:
            starts = [random.randint(0, 10) for i in range(length)]
            stops = list(starts)
        else:
            starts = [random.randint(0, len(content) - 1) for i in range(length)]
            stops = [x + min(random_length(), len(content) - x) for x in starts]
        return ListArray(starts, stops, content)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.starts[where]:self.stops[where]]
        elif isinstance(where, slice) and where.step is None:
            starts = self.starts[where.start:where.stop]
            stops = self.stops[where.start:where.stop]
            return ListArray(starts, stops, self.content)
        elif isinstance(where, str):
            return ListArray(self.starts, self.stops, self.content[where])
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        starts = [self.offsets[i] for i in index]
        stops = [self.offsets[i + 1] for i in index]
        return ListArray(starts, stops, self.content)

    def purelist_depth(self):
        return self.content.purelist_depth() + 1

    def minmax_depth(self):
        min, max = self.content.minmax_depth()
        return min + 1, max + 1

    def branch_depth(self):
        branch, depth = self.content.branch_depth()
        if branch:
            return branch, depth
        else:
            return False, depth + 1

    def __repr__(self):
        return "ListArray(" + repr(self.starts) + ", " + repr(self.stops) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ListArray>\n"
        out += indent + "    <starts>" + " ".join(str(x) for x in self.starts) + "</starts>\n"
        out += indent + "    <stops>" + " ".join(str(x) for x in self.stops) + "</stops>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ListArray>" + post
        return out

class IndexedArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, list)
        assert isinstance(content, Content)
        for x in index:
            assert isinstance(x, int)
            assert 0 <= x < len(content)   # index[i] must not be negative
        self.index = index
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        if minlen == 0:
            content = Content.random(0, choices)
        else:
            content = Content.random(1, choices)
        if len(content) == 0:
            index = []
        else:
            index = [random.randint(0, len(content) - 1) for i in range(random_length(minlen))]
        return IndexedArray(index, content)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.content[self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return IndexedArray(self.index[where.start:where.stop], self.content)
        elif isinstance(where, str):
            return IndexedArray(self.index, self.content[where])
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        index = [self.index[i] for i in index]
        return IndexedArray(index, self.content)

    def purelist_depth(self):
        return self.content.purelist_depth()

    def minmax_depth(self):
        return self.content.minmax_depth()

    def branch_depth(self):
        return self.content.branch_depth()

    def __repr__(self):
        return "IndexedArray(" + repr(self.index) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<IndexedArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedArray>\n"
        return out

class IndexedOptionArray(Content):
    def __init__(self, index, content):
        assert isinstance(index, list)
        assert isinstance(content, Content)
        for x in index:
            assert isinstance(x, int)
            assert x < len(content)   # index[i] may be negative
        self.index = index
        self.content = content

    @staticmethod
    def random(minlen=0, choices=None):
        content = Content.random(0, choices)
        index = []
        for i in range(random_length(minlen)):
            if len(content) == 0 or random.randint(0, 4) == 0:
                index.append(-random_length(1))   # a random number, but not necessarily -1
            else:
                index.append(random.randint(0, len(content) - 1))
        return IndexedOptionArray(index, content)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            if self.index[where] < 0:
                return None
            else:
                return self.content[self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return IndexedOptionArray(self.index[where.start:where.stop], self.content)
        elif isinstance(where, str):
            return IndexedOptionArray(self.index, self.content[where])
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        index = [self.index[i] for i in index]
        return IndexedOptionArray(index, self.content)

    def purelist_depth(self):
        return self.content.purelist_depth()

    def minmax_depth(self):
        return self.content.minmax_depth()

    def branch_depth(self):
        return self.content.branch_depth()

    def __repr__(self):
        return "IndexedOptionArray(" + repr(self.index) + ", " + repr(self.content) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<IndexedOptionArray>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</IndexedOptionArray>\n"
        return out

class ByteMaskedArray(Content):
    def __init__(self, mask, content, validwhen):
        assert isinstance(mask, list)
        assert isinstance(content, Content)
        assert isinstance(validwhen, bool)
        assert len(mask) <= len(content)
        for x in mask:
            assert isinstance(x, bool)
        self.mask = mask
        self.content = content
        self.validwhen = validwhen

    @staticmethod
    def random(minlen=0, choices=None):
        validwhen = random.choice([False, True])
        mask = [(random.randint(0, 4) == 0) ^ validwhen for i in range(random_length(minlen))]   # 80% are valid
        content = Content.random(len(mask), choices)
        return ByteMaskedArray(mask, content, validwhen)

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            if self.mask[where] == self.validwhen:
                return self.content[where]
            else:
                return None
        elif isinstance(where, slice) and where.step is None:
            return ByteMaskedArray(self.mask[where.start:where.stop], self.content[where.start:where.stop], self.validwhen)
        elif isinstance(where, str):
            return ByteMaskedArray(self.mask, self.content[where], self.validwhen)
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        mask = [self.mask[i] for i in index]
        return ByteMaskedArray(mask, self.content.carry(index), self.validwhen)

    def purelist_depth(self):
        return self.content.purelist_depth()

    def minmax_depth(self):
        return self.content.minmax_depth()

    def branch_depth(self):
        return self.content.branch_depth()

    def __repr__(self):
        return "ByteMaskedArray(" + repr(self.mask) + ", " + repr(self.content) + ", " + repr(self.validwhen) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<ByteMaskedArray validwhen=\"" + repr(self.validwhen) + "\">\n"
        out += indent + "    <mask>" + " ".join(str(x) for x in self.mask) + "</mask>\n"
        out += self.content.toxml(indent + "    ", "<content>", "</content>\n")
        out += indent + "</ByteMaskedArray>\n"
        return out

class RecordArray(Content):
    def __init__(self, contents, recordlookup, length):
        assert isinstance(contents, list)
        if len(contents) == 0:
            assert isinstance(length, int)
            assert length >= 0
        else:
            assert length is None
            for x in contents:
                assert isinstance(x, Content)
        assert recordlookup is None or isinstance(recordlookup, list)
        if isinstance(recordlookup, list):
            assert len(recordlookup) == len(contents)
            for x in recordlookup:
                assert isinstance(x, str)
        self.contents = contents
        self.recordlookup = recordlookup
        self.length = length

    @staticmethod
    def random(minlen=0, choices=None):
        length = random_length(minlen)
        contents = []
        for i in range(random.randint(0, 2)):
            contents.append(Content.random(length, choices))
        if len(contents) != 0:
            length = None
        if random.randint(0, 1) == 0:
            recordlookup = None
        else:
            recordlookup = ["x" + str(i) for i in range(len(contents))]
        return RecordArray(contents, recordlookup, length)

    def __len__(self):
        if len(self.contents) == 0:
            return self.length
        else:
            return min(len(x) for x in self.contents)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            record = [x[where] for x in self.contents]
            if self.recordlookup is None:
                return tuple(record)
            else:
                return dict(zip(self.recordlookup, record))
        elif isinstance(where, slice) and where.step is None:
            if len(self.contents) == 0:
                start = min(max(where.start, 0), self.length)
                stop = min(max(where.stop, 0), self.length)
                if stop < start:
                    stop = start
                return RecordArray([], self.recordlookup, stop - start)
            else:
                return RecordArray([x[where] for x in self.contents], self.recordlookup, self.length)
        elif isinstance(where, str):
            if self.recordlookup is None:
                try:
                    i = int(where)
                except ValueError:
                    pass
                else:
                    if i < len(self.contents):
                        return self.contents[i]
            else:
                try:
                    i = self.recordlookup.index(where)
                except ValueError:
                    pass
                else:
                    return self.contents[i]
            raise ValueError("field " + repr(where) + " not found")
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        if len(self.contents) == 0:
            return RecordArray([], self.recordlookup, len(index))
        else:
            return RecordArray([x.carry(index) for x in self.contents], self.recordlookup, self.length)

    def purelist_depth(self):
        return 1

    def minmax_depth(self):
        min, max = None, None
        for content in self.contents:
            thismin, thismax = content.minmax_depth()
            if min is None or thismin < min:
                min = thismin
            if max is None or thismax > max:
                max = thismax
        return min, max

    def branch_depth(self):
        if len(self.contents) == 0:
            return False, 1
        else:
            anybranch = False
            mindepth = -1
            for content in self.contents:
                branch, depth = content.branch_depth()
                if mindepth == -1:
                    mindepth = depth
                if branch or mindepth != depth:
                    anybranch = True
                if mindepth > depth:
                    mindepth = depth
            return anybranch, mindepth

    def __repr__(self):
        return "RecordArray([" + ", ".join(repr(x) for x in self.contents) + "], " + repr(self.recordlookup) + ", " + repr(self.length) + ")"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<RecordArray>\n"
        if len(self.contents) == 0:
            out += indent + "    <istuple>" + str(self.recordlookup is None) + "</istuple>\n"
            out += indent + "    <length>" + str(self.length) + "</length>\n"
        if self.recordlookup is None:
            for i, content in enumerate(self.contents):
                out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        else:
            for i, (key, content) in enumerate(zip(self.recordlookup, self.contents)):
                out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\" key=\"" + repr(key) + "\">", "</content>\n")
        out += indent + "</RecordArray>" + post
        return out

class UnionArray(Content):
    def __init__(self, tags, index, contents):
        assert isinstance(tags, list)
        assert isinstance(index, list)
        assert isinstance(contents, list)
        assert len(index) >= len(tags)   # usually equal
        for x in tags:
            assert isinstance(x, int)
            assert 0 <= x < len(contents)
        for i, x in enumerate(index):
            assert isinstance(x, int)
            assert 0 <= x < len(contents[tags[i]])
        self.tags = tags
        self.index = index
        self.contents = contents

    @staticmethod
    def random(minlen=0, choices=None):
        contents = []
        unshuffled_tags = []
        unshuffled_index = []
        for i in range(random.randint(1, 3)):
            if minlen == 0:
                contents.append(Content.random(0, choices))
            else:
                contents.append(Content.random(1, choices))
            if len(contents[-1]) != 0:
                thisindex = [random.randint(0, len(contents[-1]) - 1) for i in range(random_length(minlen))]
                unshuffled_tags.extend([i] * len(thisindex))
                unshuffled_index.extend(thisindex)
        permutation = list(range(len(unshuffled_tags)))
        random.shuffle(permutation)
        tags = [unshuffled_tags[i] for i in permutation]
        index = [unshuffled_index[i] for i in permutation]
        return UnionArray(tags, index, contents)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, where):
        if isinstance(where, int):
            assert 0 <= where < len(self)
            return self.contents[self.tags[where]][self.index[where]]
        elif isinstance(where, slice) and where.step is None:
            return UnionArray(self.tags[where], self.index[where], self.contents)
        elif isinstance(where, str):
            return UnionArray(self.tags, self.index, [x[where] for x in self.contents])
        else:
            raise AssertionError(where)

    def carry(self, index):
        assert all(0 <= i < len(self) for i in index)
        return UnionArray([self.tags[i] for i in index], [self.index[i] for i in index], self.contents)

    def purelist_depth(self):
        return 1

    def minmax_depth(self):
        min, max = None, None
        for content in self.contents:
            thismin, thismax = content.minmax_depth()
            if min is None or thismin < min:
                min = thismin
            if max is None or thismax > max:
                max = thismax
        return min, max

    def branch_depth(self):
        anybranch = False
        mindepth = -1
        for content in self.contents:
            branch, depth = content.branch_depth()
            if mindepth == -1:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return anybranch, mindepth

    def __repr__(self):
        return "UnionArray(" + repr(self.tags) + ", " + repr(self.index) + ", [" + ", ".join(repr(x) for x in self.contents) + "])"

    def toxml(self, indent="", pre="", post=""):
        out = indent + pre + "<UnionArray>\n"
        out += indent + "    <tags>" + " ".join(str(x) for x in self.tags) + "</tags>\n"
        out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
        for i, content in enumerate(self.contents):
            out += content.toxml(indent + "    ", "<content i=\"" + str(i) + "\">", "</content>\n")
        out += indent + "</UnionArray>" + post
        return out

###################################################################### use NumPy to define correct behavior

import numpy

# (My example reducer will be 'prod' on prime numbers; can't get the right answer accidentally.)
primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]

nparray = numpy.array(primes[:2*3*5]).reshape(2, 3, 5)

assert (nparray.tolist() ==
    [[[  2,   3,   5,   7,  11],
      [ 13,  17,  19,  23,  29],
      [ 31,  37,  41,  43,  47]],
     [[ 53,  59,  61,  67,  71],
      [ 73,  79,  83,  89,  97],
      [101, 103, 107, 109, 113]]])
assert nparray.shape == (2, 3, 5)

assert (numpy.prod(nparray, axis=-3).tolist() ==
    [[ 106,  177,  305,  469, 781],
     [ 949, 1343, 1577, 2047, 2813],
     [3131, 3811, 4387, 4687, 5311]])
assert numpy.prod(nparray, axis=0).shape == (3, 5)

assert (numpy.prod(nparray, axis=-2).tolist() ==
    [[   806,   1887,   3895,   6923, 14993],
     [390769, 480083, 541741, 649967, 778231]])
assert numpy.prod(nparray, axis=1).shape == (2, 5)

assert (numpy.prod(nparray, axis=-1).tolist() ==
    [[     2310,    2800733,    95041567],
     [907383479, 4132280413, 13710311357]])
assert numpy.prod(nparray, axis=2).shape == (2, 3)

# (axis=3 is a numpy.AxisError; axis<0 counts backward from the end.)

assert numpy.prod(nparray, axis=None) == 7581744426003940878
# (but that should be a special case because it results in a scalar.)

########################################################################

def Content_reduce(self, axis):
    negaxis = -axis   # easier to think about it this way because negaxis aligns with depth
    branch, depth = self.branch_depth()

    if branch:
        if negaxis <= 0:
            raise ValueError("cannot use non-negative axis on a nested list structure of variable depth (negative axis counts from the leaves of the tree)")
        if negaxis > depth:
            raise ValueError("cannot use axis={0} on a nested list structure that splits into different depths, the minimum of which is {1} from the leaves".format(axis, depth))
    else:
        if negaxis <= 0:
            negaxis += depth
        if not (0 < negaxis <= depth):
            raise ValueError("axis={0} exceeds the depth of the nested list structure ({1})".format(axis, depth))

    parents = [None] * len(self)
    for i in range(len(self)):
        parents[i] = 0
    return self.reduce_next(negaxis, parents, 1)[0]

Content.reduce = Content_reduce

def RawArray_reduce_next(self, negaxis, parents, length):
    assert negaxis == 1
    ptr = [1] * length
    for i in range(len(parents)):
        ptr[parents[i]] *= self.ptr[i]
    return RawArray(ptr)

RawArray.reduce_next = RawArray_reduce_next

def RegularArray_toListOffsetArray(self):
    nextoffsets = [None] * (len(self) + 1)
    for i in range(len(nextoffsets)):
        nextoffsets[i] = i * self.size
    return ListOffsetArray(nextoffsets, self.content)

RegularArray.toListOffsetArray = RegularArray_toListOffsetArray

def RegularArray_reduce_next(self, negaxis, parents, length):
    return self.toListOffsetArray().reduce_next(negaxis, parents, length)

RegularArray.reduce_next = RegularArray_reduce_next

def ListArray_toListOffsetArray(self):
    contentlen = 0
    for i in range(len(self.starts)):
        contentlen += self.stops[i] - self.starts[i]

    nextoffsets = [None] * (len(self.starts) + 1)
    nextoffsets[0] = 0
    nextcarry = [None] * contentlen
    k = 0
    for i in range(len(self.starts)):
        nextoffsets[i + 1] = nextoffsets[i] + (self.stops[i] - self.starts[i])
        for j in range(self.starts[i], self.stops[i]):
            nextcarry[k] = j
            k += 1

    return ListOffsetArray(nextoffsets, self.content.carry(nextcarry))

ListArray.toListOffsetArray = ListArray_toListOffsetArray

def ListArray_reduce_next(self, negaxis, parents, length):
    return self.toListOffsetArray().reduce_next(negaxis, parents, length)

ListArray.reduce_next = ListArray_reduce_next

def ListOffsetArray_reduce_next(self, negaxis, parents, length):
    branch, depth = self.branch_depth()

    if not branch and negaxis == depth:
        maxcount = 0
        for i in range(len(self.offsets) - 1):
            count = self.offsets[i + 1] - self.offsets[i]
            if count > maxcount:
                maxcount = count

        offsetscopy = list(self.offsets)

        nextcarry = [None] * (self.offsets[-1] - self.offsets[0])
        nextparents = [None] * (self.offsets[-1] - self.offsets[0])
        maxnextparents = 0
        distincts = [-1] * (maxcount * length)
        k = 0
        while k < len(nextcarry):
            j = 0
            for i in range(len(offsetscopy) - 1):
                if offsetscopy[i] < self.offsets[i + 1]:
                    count = self.offsets[i + 1] - self.offsets[i]
                    diff = offsetscopy[i] - self.offsets[i]

                    nextcarry[k] = offsetscopy[i]
                    nextparents[k] = parents[i]*maxcount + diff
                    if maxnextparents < nextparents[k]:
                        maxnextparents = nextparents[k]

                    if distincts[nextparents[k]] == -1:
                        distincts[nextparents[k]] = j
                        j += 1

                    k += 1
                    offsetscopy[i] += 1

        nextcontent = self.content.carry(nextcarry)
        outcontent = nextcontent.reduce_next(negaxis - 1, nextparents, maxnextparents + 1)

        gaps = [None] * length
        k = 0
        last = -1
        for i in range(len(parents)):
            if last < parents[i]:
                gaps[k] = parents[i] - last
                k += 1
                last = parents[i]

        outstarts = [None] * length
        outstops = [None] * length
        maxdistinct = -1
        j = 0
        k = 0
        for i in range(len(distincts)):
            if maxdistinct < distincts[i]:
                maxdistinct = distincts[i]
                for x in range(gaps[j]):
                    outstarts[k] = i
                    outstops[k] = i
                    k += 1
                j += 1
            if distincts[i] != -1:
                outstops[k - 1] = i + 1

        return ListArray(outstarts, outstops, outcontent)

    else:
        nextparents = [None] * (self.offsets[-1] - self.offsets[0])
        # k = 0
        for i in range(len(self.offsets) - 1):
            for j in range(self.offsets[i], self.offsets[i + 1]):
                nextparents[j] = i

        trimmed = self.content[self.offsets[0]:self.offsets[-1]]
        outcontent = trimmed.reduce_next(negaxis, nextparents, len(self.offsets) - 1)

        outoffsets = [None] * (length + 1)
        outoffsets[-1] = len(parents)
        k = 0
        last = -1
        for i in range(len(parents)):
            while last < parents[i]:
                outoffsets[k] = i
                k += 1
                last += 1

        return ListOffsetArray(outoffsets, outcontent)

ListOffsetArray.reduce_next = ListOffsetArray_reduce_next

def RecordArray_reduce_next(self, negaxis, parents, length):
    if len(self.contents) == 0:
        return RecordArray(self.contents, self.recordlookup, length)
    else:
        contents = []
        for content in self.contents:
            trimmed = content[0:len(self)]
            contents.append(trimmed.reduce_next(negaxis, parents, length))
        return RecordArray(contents, self.recordlookup, self.length)

RecordArray.reduce_next = RecordArray_reduce_next

def IndexedOptionArray_reduce_next(self, negaxis, parents, length):
    numnull = 0
    for i in range(len(self.index)):
        if self.index[i] < 0:
            numnull += 1

    nextparents = [None] * (len(self.index) - numnull)
    nextcarry = [None] * (len(self.index) - numnull)
    k = 0
    for i in range(len(self.index)):
        if self.index[i] >= 0:
            nextcarry[k] = self.index[i]
            nextparents[k] = parents[i]
            k += 1

    next = self.content.carry(nextcarry)
    return next.reduce_next(negaxis, nextparents, length)

    # if True:  # isinstance(next, (RawArray, NumpyArray, EmptyArray)):
    #     print("index", self.index)
    #     print("parents", parents)
    #     print("length", length)
    #     print("next", list(next))
    #     print(next.toxml())
    #     print("reduced", list(reduced))
    #     print(reduced.toxml())

    #     raise Exception

    # else:
    #     return reduced

IndexedOptionArray.reduce_next = IndexedOptionArray_reduce_next

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 25, 30], RawArray(primes[:2*3*5])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([1, 4, 7], ListOffsetArray([0, 1, 6, 11, 16, 21, 26, 31], RawArray([123] + primes[:2*3*5])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 25, 29], RawArray(primes[:2*3*5 - 1])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109     ]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687,   47]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 25, 28], RawArray(primes[:2*3*5 - 2])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107,         ]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387,   43,   47]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 24, 28], RawArray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 101, 103, 107, 109])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,    ],
     [101, 103, 107, 109     ]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469, 781],
    [ 949, 1343, 1577, 2047,  29],
    [3131, 3811, 4387, 4687,  47]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 4, 9, 14, 19, 24, 29], RawArray(primes[1:2*3*5])))
assert depth2.tolist() == [
    [[  3,   5,   7,  11     ],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 159,  295,  427,  737,   71],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 3, 8, 13, 18, 23, 28], RawArray(primes[2:2*3*5])))
assert depth2.tolist() == [
    [[  5,   7,  11          ],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 265,  413,  671,   67,   71],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 3, 8, 13, 18, 23, 28], RawArray([3, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113])))
assert depth2.tolist() == [
    [[  3,   5,   7,         ],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 159,  295,  427,   67,   71],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 4, 8, 13, 18, 23, 28], RawArray([3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113])))
assert depth2.tolist() == [
    [[  3,   5,   7,  11     ],
     [ 13,  17,  19,  23     ],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 159,  295,  427,  737,   71],
    [ 949, 1343, 1577, 2047,   97],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 14, 19, 24, 28], RawArray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43     ]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109     ]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047, 2813],
    [3131, 3811, 4387, 4687]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 9, 14, 19, 23, 28], RawArray([2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 101, 103, 107, 109, 113])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23     ],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89     ],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-3)) == [
    [ 106,  177,  305,  469,  781],
    [ 949, 1343, 1577, 2047      ],
    [3131, 3811, 4387, 4687, 5311]]

depth2 = ListOffsetArray([0, 2, 4, 6], ListOffsetArray([0, 3, 4, 6, 6, 7, 9], RawArray(primes[:9])))
assert list(depth2) == [
    [[      2,   3, 5],
     [      7        ]],
    [[     11,  13   ],
     [               ]],
    [[     17        ],
     [     19,  23   ]]]

assert list(depth2.reduce(-3)) == [
    [2*11*17, 3*13, 5],
    [7*19   , 23     ]]

depth2 = ListOffsetArray([0, 2, 3, 5], ListOffsetArray([0, 3, 4, 6, 7, 9], RawArray(primes[:9])))
assert list(depth2) == [
    [[      2,   3, 5],
     [      7        ]],
    [[     11,  13   ]],
    [[     17        ],
     [     19,  23   ]]]

assert list(depth2.reduce(-3)) == [
    [2*11*17, 3*13, 5],
    [7*19   , 23     ]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 3, 5, 6, 8, 9, 10], RawArray(primes[:10])))
assert list(depth2) == [
    [[ 2,  3, 5],
     [ 7, 11   ],
     [13       ]],
    [[17, 19   ],
     [23       ],
     [29       ]]]

assert list(depth2.reduce(-3)) == [
    [ 34, 57, 5],
    [161, 11   ],
    [377       ]]

depth2 = ListOffsetArray([0, 4, 6], ListOffsetArray([0, 3, 3, 5, 6, 8, 9], RawArray(primes[:9])))
assert list(depth2) == [
    [[ 2,  3, 5],
     [         ],
     [ 7, 11   ],
     [13       ]],
    [[17, 19   ],
     [23       ]]]

assert list(depth2.reduce(-3)) == [
    [34, 57, 5],
    [23       ],
    [ 7, 11   ],
    [13       ]]

depth2 = ListOffsetArray([0, 4, 4, 6], ListOffsetArray([0, 3, 3, 5, 6, 8, 9], RawArray(primes[:9])))
assert list(depth2) == [
    [[ 2,  3, 5],
     [         ],
     [ 7, 11   ],
     [13       ]],
    [],
    [[17, 19   ],
     [23       ]]]

assert list(depth2.reduce(-3)) == [
    [34, 57, 5],
    [23       ],
    [ 7, 11   ],
    [13       ]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 25, 30], RawArray(primes[:2*3*5])))
assert depth2.tolist() == [
    [[  2,   3,   5,   7,  11],
     [ 13,  17,  19,  23,  29],
     [ 31,  37,  41,  43,  47]],
    [[ 53,  59,  61,  67,  71],
     [ 73,  79,  83,  89,  97],
     [101, 103, 107, 109, 113]]]

assert list(depth2.reduce(-1)) == [
    [  2 *   3 *   5 *   7 *  11,
      13 *  17 *  19 *  23 *  29,
      31 *  37 *  41 *  43 *  47],
    [ 53 *  59 *  61 *  67 *  71,
      73 *  79 *  83 *  89 *  97,
     101 * 103 * 107 * 109 * 113]]

assert list(depth2.reduce(-2)) == [
    [2*13*31, 3*17*37, 5*19*41, 7*23*43, 11*29*47],
    [53*73*101, 59*79*103, 61*83*107, 67*89*109, 71*97*113]]

depth2 = ListOffsetArray([0, 4, 4, 6], ListOffsetArray([0, 3, 3, 5, 6, 8, 9], RawArray(primes[:9])))
assert list(depth2) == [
    [[ 2,  3, 5],
     [         ],
     [ 7, 11   ],
     [13       ]],
    [],
    [[17, 19   ],
     [23       ]]]

assert list(depth2.reduce(-1)) == [
    [2*3*5, 1, 7*11, 13],
    [],
    [17*19, 23]]

assert list(depth2.reduce(-2)) == [
    [2*7*13, 3*11, 5],
    [],
    [17*23, 19]]

depth1 = ListOffsetArray([0, 4, 8, 12], RawArray(primes[:12]))
assert list(depth1) == [[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37]]

assert list(depth1.reduce(-1)) == [
    2*3*5*7,
    11*13*17*19,
    23*29*31*37]

assert list(depth1.reduce(-2)) == [
    2*11*23,
    3*13*29,
    5*17*31,
    7*19*37]

complicated = ListOffsetArray([0, 1, 1, 3], RecordArray([ListOffsetArray([0, 3, 3, 5], RawArray(primes[:5])), ListOffsetArray([0, 4, 4, 6], ListOffsetArray([0, 3, 3, 5, 6, 8, 9], RawArray(primes[:9])))], ["x", "y"], None))
assert complicated.tolist() == [[{"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}], [], [{"x": [], "y": []}, {"x": [7, 11], "y": [[17, 19], [23]]}]]
assert complicated.minmax_depth() == (3, 4)
assert complicated.branch_depth() == (True, 2)

assert list(complicated["x"]) == [
    [[2, 3, 5]],
    [],
    [[],
     [7, 11]]]
assert list(complicated["y"]) == [
    [[[ 2,  3, 5],
      [         ],
      [ 7, 11   ],
      [13       ]]],
    [             ],
    [[          ],
     [[17, 19   ],
      [23       ]]]]

assert list(complicated["x"].reduce(-1)) == [[30], [], [1, 77]]
assert list(complicated["y"].reduce(-1)) == [[[30, 1, 77, 13]], [], [[], [323, 23]]]
assert list(complicated.reduce(-1)) == [{"x": [30], "y": [[30, 1, 77, 13]]}, {"x": [], "y": []}, {"x": [1, 77], "y": [[], [323, 23]]}]

assert list(complicated["x"].reduce(-2)) == [[2, 3, 5], [], [7, 11]]
assert list(complicated["y"].reduce(-2)) == [[[182, 33, 5]], [], [[], [391, 19]]]

assert list(complicated.reduce(-2)) == [{"x": [2, 3, 5], "y": [[182, 33, 5]]}, {"x": [], "y": []}, {"x": [7, 11], "y": [[], [391, 19]]}]

depth2 = ListOffsetArray([0, 3, 6], IndexedOptionArray([5, 4, 3, 2, 1, 0], ListOffsetArray([0, 5, 10, 15, 20, 25, 30], RawArray(primes[:2*3*5]))))
assert depth2.tolist() == [
    [[101, 103, 107, 109, 113],
     [ 73,  79,  83,  89,  97],
     [ 53,  59,  61,  67,  71]],
    [[ 31,  37,  41,  43,  47],
     [ 13,  17,  19,  23,  29],
     [  2,   3,   5,   7,  11]]]

assert list(depth2.reduce(-1)) == [
    [101 * 103 * 107 * 109 * 113,
      73 *  79 *  83 *  89 *  97,
      53 *  59 *  61 *  67 *  71],
    [ 31 *  37 *  41 *  43 *  47,
      13 *  17 *  19 *  23 *  29,
       2 *   3 *   5 *   7 *  11]]

assert list(depth2.reduce(-2)) == [
    [101*73*53, 103*79*59, 107*83*61, 109*89*67, 113*97*71],
    [  31*13*2,   37*17*3,   41*19*5,   43*23*7,  47*29*11]]

assert list(depth2.reduce(-3)) == [
    [101*31, 103*37, 107*41, 109*43, 113*47],
    [ 73*13,  79*17,  83*19,  89*23,  97*29],
    [  53*2,   59*3,   61*5,   67*7,  71*11]]

depth2 = ListOffsetArray([0, 3, 6], IndexedOptionArray([3, -1, 2, 1, -1, 0], ListOffsetArray([0, 5, 10, 15, 20], RawArray([2, 3, 5, 7, 11, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 101, 103, 107, 109, 113]))))
assert depth2.tolist() == [
    [[101, 103, 107, 109, 113],
     None,
     [ 53,  59,  61,  67,  71]],
    [[ 31,  37,  41,  43,  47],
     None,
     [  2,   3,   5,   7,  11]]]

assert list(depth2.reduce(-1)) == [
    [101 * 103 * 107 * 109 * 113,
      53 *  59 *  61 *  67 *  71],
    [ 31 *  37 *  41 *  43 *  47,
       2 *   3 *   5 *   7 *  11]]

assert list(depth2.reduce(-2)) == [
    [101*53, 103*59, 107*61, 109*67, 113*71],
    [  31*2,   37*3,   41*5,   43*7,  47*11]]

assert list(depth2.reduce(-3)) == [
    [101*31, 103*37, 107*41, 109*43, 113*47],
    [],
    [  53*2,   59*3,   61*5,   67*7,  71*11]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 10, 15, 20, 25, 30], IndexedOptionArray([15, 16, 17, 18, 19, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4], RawArray([2, 3, 5, 7, 11, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 101, 103, 107, 109, 113]))))
assert depth2.tolist() == [
    [[ 101,  103,  107,  109,  113],
     [None, None, None, None, None],
     [  53,   59,   61,   67,   71]],
    [[  31,   37,   41,   43,   47],
     [None, None, None, None, None],
     [   2,    3,    5,    7,   11]]]

assert list(depth2.reduce(-1)) == [
    [101 * 103 * 107 * 109 * 113,
       1 *   1 *   1 *   1 *   1,
      53 *  59 *  61 *  67 *  71],
    [ 31 *  37 *  41 *  43 *  47,
       1 *   1 *   1 *   1 *   1,
       2 *   3 *   5 *   7 *  11]]

assert list(depth2.reduce(-2)) == [
    [101*53, 103*59, 107*61, 109*67, 113*71],
    [  31*2,   37*3,   41*5,   43*7,  47*11]]

assert list(depth2.reduce(-3)) == [
    [101*31, 103*37, 107*41, 109*43, 113*47],
    [     1,      1,      1,      1,      1],
    [  53*2,   59*3,   61*5,   67*7,  71*11]]

depth2 = ListOffsetArray([0, 3, 6], ListOffsetArray([0, 5, 6, 11, 16, 17, 22], IndexedOptionArray([15, 16, 17, 18, 19, -1, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, -1, 0, 1, 2, 3, 4], RawArray([2, 3, 5, 7, 11, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 101, 103, 107, 109, 113]))))
assert depth2.tolist() == [
    [[ 101,  103,  107,  109,  113],
     [None],
     [  53,   59,   61,   67,   71]],
    [[  31,   37,   41,   43,   47],
     [None],
     [   2,    3,    5,    7,   11]]]

assert list(depth2.reduce(-1)) == [
    [101 * 103 * 107 * 109 * 113,
       1,
      53 *  59 *  61 *  67 *  71],
    [ 31 *  37 *  41 *  43 *  47,
       1,
       2 *   3 *   5 *   7 *  11]]

assert list(depth2.reduce(-2)) == [
    [101*53, 103*59, 107*61, 109*67, 113*71],
    [  31*2,   37*3,   41*5,   43*7,  47*11]]

assert list(depth2.reduce(-3)) == [
    [101*31, 103*37, 107*41, 109*43, 113*47],
    [     1],
    [  53*2,   59*3,   61*5,   67*7,  71*11]]

content = RawArray([1,2,3, 4,5,6, 5,6,7, 8,9,10, 9,10,11, 12,13,14, 13,14,15, 16,17,18])
assert content.tolist() == [1,2,3, 4,5,6, 5,6,7, 8,9,10, 9,10,11, 12,13,14, 13,14,15, 16,17,18]

depth1 = RegularArray(content, 3)
depth2 = RegularArray(depth1, 2)
depth3 = RegularArray(depth2, 2)
depth4 = RegularArray(depth3, 2)
print(list(depth4))
assert list(depth4) == [[[[[1, 2, 3],[4, 5, 6]],[[5, 6, 7],[8, 9, 10]]],[[[9, 10, 11],[12, 13, 14]],[[13, 14, 15],[16, 17, 18]]]]]
print("depth1", list(depth1.reduce(-1)))
print("depth2", list(depth2.reduce(-1)))
print("depth3", list(depth3.reduce(-1)))
print("depth4 (-1)", list(depth4.reduce(-1)))
# >>> np.prod(array, axis=-1)
# array([[[   6,  120],
#         [ 210,  720]],
#
#        [[ 990, 2184],
#         [2730, 4896]]])
assert list(depth4.reduce(-1)) == [[[[6,120],[210,720]],[[990,2184],[2730,4896]]]]

print("depth4 (-2)", list(depth4.reduce(-2)))
# >>> np.prod(array, axis=-2)
# array([[[  4,  10,  18],
#         [ 40,  54,  70]],
#
#        [[108, 130, 154],
#         [208, 238, 270]]])
assert list(depth4.reduce(-2)) == [[[[4,10,18],[40,54,70]],[[108,130,154],[208,238,270]]]]

print("depth4 (-3)", list(depth4.reduce(-3)))
# >>> np.prod(array, axis=-3)
# array([[[  5,  12,  21],
#         [ 32,  45,  60]],
#
#        [[117, 140, 165],
#         [192, 221, 252]]])
assert list(depth4.reduce(-3)) == [[[[5,12,21],[32,45,60]],[[117,140,165],[192,221,252]]]]

print("depth4 (-4)", list(depth4.reduce(-4)))
# >>> np.prod(array, axis=-4)
# array([[[  9,  20,  33],
#         [ 48,  65,  84]],
#
#        [[ 65,  84, 105],
#         [128, 153, 180]]])
assert list(depth4.reduce(-4)) == [[[[6,120],[210,720]],[[990,2184],[2730,4896]]]]



# >>> np.prod(array, axis=0)
# array([[[  9,  20,  33],
#         [ 48,  65,  84]],
#
#        [[ 65,  84, 105],
#         [128, 153, 180]]])
# >>> np.prod(array, axis=1)
# array([[[  5,  12,  21],
#         [ 32,  45,  60]],
#
#        [[117, 140, 165],
#         [192, 221, 252]]])
# >>> np.prod(array, axis=2)
# array([[[  4,  10,  18],
#         [ 40,  54,  70]],
#
#        [[108, 130, 154],
#         [208, 238, 270]]])
# >>> np.prod(array, axis=3)
# array([[[   6,  120],
#         [ 210,  720]],
#
#        [[ 990, 2184],
#         [2730, 4896]]])
