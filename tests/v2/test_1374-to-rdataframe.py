# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_wonky():
    import json

    example1 = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    example2 = ak._v2.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}]
    )

    generator1 = ak._v2._connect.cling.togenerator(
        example1.layout.form, flatlist_as_rvec=True
    )
    lookup1 = ak._v2._lookup.Lookup(example1.layout)
    generator1.generate(ROOT.gInterpreter.Declare)

    generator2 = ak._v2._connect.cling.togenerator(
        example2.layout.form, flatlist_as_rvec=True
    )
    lookup2 = ak._v2._lookup.Lookup(example2.layout)
    generator2.generate(ROOT.gInterpreter.Declare)

    dataset_type_one = generator1.class_type()
    entry_type_one = generator1.entry_type()
    dataset_type_two = generator2.class_type()
    entry_type_two = generator2.entry_type()

    assert len(example1) == len(example2)

    done = compiler(
        f"""
namespace awkward {{

using namespace awkward;

class RWonkyDS final : public ROOT::RDF::RDataSource {{
private:
   unsigned int fNSlots = 0U;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;

   ULong64_t fSize = 0ULL;
   ULong64_t fPtrs_one = 0;
   ULong64_t fPtrs_two = 0;

   std::vector<std::string> fColNames{{"one", "two"}};
   std::vector<{entry_type_one}>  slots_one;
   std::vector<{entry_type_one}*> addrs_one;
   std::vector<{entry_type_two}>  slots_two;
   std::vector<{entry_type_two}*> addrs_two;

   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &);

protected:
   std::string AsString() {{ return "trivial data source"; }};

public:
   RWonkyDS(ULong64_t size, ULong64_t ptrs_one, ULong64_t ptrs_two);
   RWonkyDS();
   ~RWonkyDS();
   const std::vector<std::string> &GetColumnNames() const;
   bool HasColumn(std::string_view colName) const;
   std::string GetTypeName(std::string_view) const;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges();
   bool SetEntry(unsigned int slot, ULong64_t entry);
   void SetNSlots(unsigned int nSlots);
   void Initialise();
   std::string GetLabel();
}};

// Make a RDF wrapping a RWonkyDS with the specified amount of entries
ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RWonkyDS> MakeWonkyDataFrame(ULong64_t size, ULong64_t ptrs_one, ULong64_t ptrs_two);
// Make a RDF wrapping a broken RWonkyDS... because we need a zero-argument constructor?
ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RWonkyDS> MakeWonkyDataFrame();

std::vector<void *> RWonkyDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &ti) {{
   std::vector<void *> ret;

   if (colName == "one") {{
      for (auto i : ROOT::TSeqU(fNSlots)) {{
         addrs_one[i] = &slots_one[i];
         ret.emplace_back((void *)(&addrs_one[i]));
      }}
   }}
   else if (colName == "two") {{
      for (auto i : ROOT::TSeqU(fNSlots)) {{
         addrs_two[i] = &slots_two[i];
         ret.emplace_back((void *)(&addrs_two[i]));
      }}
   }}
   else {{
      for (auto i : ROOT::TSeqU(fNSlots)) {{
         ret.emplace_back(nullptr);
      }}
   }}

   return ret;
}}

RWonkyDS::RWonkyDS(ULong64_t size, ULong64_t ptrs_one, ULong64_t ptrs_two) : fSize(size), fPtrs_one(ptrs_one), fPtrs_two(ptrs_two) {{
}}

RWonkyDS::RWonkyDS() : fSize(0), fPtrs_one(0), fPtrs_two(0) {{
}}

RWonkyDS::~RWonkyDS() {{
}}

const std::vector<std::string> &RWonkyDS::GetColumnNames() const {{
   return fColNames;
}}

bool RWonkyDS::HasColumn(std::string_view colName) const {{
   for (auto name : fColNames) {{
      if (colName == name) {{
         return true;
      }}
   }}
   return false;
}}

std::string RWonkyDS::GetTypeName(std::string_view colName) const {{
   if (colName == "one") {{
      return {json.dumps(entry_type_one)};
   }}
   else if (colName == "two") {{
      return {json.dumps("awkward::" + entry_type_two)};
   }}
   else {{
      return "no such column";   // should break whatever tries to use it as a type
   }}
}}

std::vector<std::pair<ULong64_t, ULong64_t>> RWonkyDS::GetEntryRanges() {{
   // empty fEntryRanges so we'll return an empty vector on subsequent calls
   auto ranges = std::move(fEntryRanges);
   return ranges;
}}

bool RWonkyDS::SetEntry(unsigned int slot, ULong64_t entry) {{

   slots_one[slot] = {dataset_type_one}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_one))[entry];
   slots_two[slot] = {dataset_type_two}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_two))[entry];

   return true;
}}

void RWonkyDS::SetNSlots(unsigned int nSlots) {{
   R__ASSERT(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");
   fNSlots = nSlots;

   slots_one.resize(fNSlots);
   addrs_one.resize(fNSlots);
   slots_two.resize(fNSlots);
   addrs_two.resize(fNSlots);

}}

void RWonkyDS::Initialise() {{
   // initialize fEntryRanges
   const auto chunkSize = fSize / fNSlots;
   auto start = 0UL;
   auto end = 0UL;
   for (auto i : ROOT::TSeqUL(fNSlots)) {{
      start = end;
      end += chunkSize;
      fEntryRanges.emplace_back(start, end);
      (void)i;
   }}
   // TODO: redistribute reminder to all slots
   fEntryRanges.back().second += fSize % fNSlots;
}}

std::string RWonkyDS::GetLabel() {{
   return "WonkyDS";
}}

ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RWonkyDS> MakeWonkyDataFrame(ULong64_t size, ULong64_t ptrs_one, ULong64_t ptrs_two) {{
   auto lm = std::make_unique<ROOT::RDF::RDFDetail::RLoopManager>(std::make_unique<RWonkyDS>(size, ptrs_one, ptrs_two), ROOT::RDF::RDFInternal::ColumnNames_t{{}});
   return ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RWonkyDS>(std::move(lm));
}}

}} // namespace awkward
"""
    )
    assert done is True

    data_frame = ROOT.awkward.MakeWonkyDataFrame(
        len(example1), lookup1.arrayptrs.ctypes.data, lookup2.arrayptrs.ctypes.data
    )
    assert list(data_frame.GetColumnNames()) == ["one", "two"]
    assert data_frame.GetColumnType("one") == "double"
    assert data_frame.GetColumnType("two").startswith("awkward::Record_")
    assert data_frame.Count().GetValue() == len(example1)


def test_two_columns_as_rvecs():
    ak_array_1 = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    ak_array_2 = ak._v2.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}]
    )

    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}, flatlist_as_rvec=True
    )
    assert list(data_frame.GetColumnNames()) == ["x", "y"]
    assert data_frame.GetColumnType("x") == "double"
    assert data_frame.GetColumnType("y").startswith("awkward::Record_")

    cpp_list_x = ", ".join(str(e) for e in ak_array_1.to_list())
    cpp_list_y = ", ".join(str(e) for e in ak_array_2.x.to_list())

    done = compiler(
        f"""
        int ix = 0;
        double x_val[5] = {{ {cpp_list_x} }};

        template<typename T>
        struct CheckX {{
            void operator()(T const& x) {{
                R__ASSERT(x == x_val[ix++]);
            }}
        }};

        int iy = 0;
        double y_val[5] = {{ {cpp_list_y} }};

        template<typename T>
        struct CheckY {{
            void operator()(T const& y) {{
                R__ASSERT(y.x() == y_val[iy++]);
            }}
        }};
        """
    )
    assert done is True

    f_x = ROOT.CheckX[data_frame.GetColumnType("x")]()
    f_y = ROOT.CheckY[data_frame.GetColumnType("y")]()

    data_frame.Foreach(f_x, ["x"])
    data_frame.Foreach(f_y, ["y"])


def test_list_array():
    ak_array = ak._v2.Array([[1.1], [2.2, 3.3, 4.4], [5.5, 6.6]])

    data_frame = ak._v2.operations.convert.to_rdataframe({"x": ak_array})

    assert data_frame.GetColumnType("x") == "ROOT::RVec<double>"

    done = compiler(
        """
        int64_t row = 0;

        ROOT::RVec<ROOT::RVec<double>> row_vals =
        {{ 1.1},
         { 2.2, 3.3, 4.4 },
         { 5.5, 6.6 }
        };

        template<typename T>
        struct MyFunctor_RVec {
            void operator()(T const& x) {
                for( int64_t j = 0; j < x.size(); j++ ) {
                    R__ASSERT(x[j] == row_vals[row][j]);
                }
                row++;
            }
        };
        """
    )
    assert done is True

    f_x = ROOT.MyFunctor_RVec[data_frame.GetColumnType("x")]()
    data_frame.Foreach(f_x, ["x"])


def test_two_columns_as_vecs():
    ak_array_1 = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    ak_array_2 = ak._v2.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}]
    )

    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}, flatlist_as_rvec=False
    )
    assert list(data_frame.GetColumnNames()) == ["x", "y"]
    assert data_frame.GetColumnType("x") == "double"
    assert data_frame.GetColumnType("y").startswith("awkward::Record_")

    cpp_list_x = ", ".join(str(e) for e in ak_array_1.to_list())
    cpp_list_y = ", ".join(str(e) for e in ak_array_2.x.to_list())

    done = compiler(
        f"""
        int i_xv = 0;
        double xv_val[5] = {{ {cpp_list_x} }};

        template<typename T>
        struct MyFunctorX_Vec {{
            void operator()(T const& x) {{
                R__ASSERT(x == xv_val[i_xv++]);
            }}
        }};

        int i_yv = 0;
        double yv_val[5] = {{ {cpp_list_y} }};

        template<typename T>
        struct MyFunctorY_Vec {{
            void operator()(T const& y) {{
                R__ASSERT(y.x() == yv_val[i_yv++]);
            }}
        }};
        """
    )
    assert done is True

    f_x = ROOT.MyFunctorX_Vec[data_frame.GetColumnType("x")]()
    f_y = ROOT.MyFunctorY_Vec[data_frame.GetColumnType("y")]()

    data_frame.Foreach(f_x, ["x"])
    data_frame.Foreach(f_y, ["y"])


def test_two_columns_transform_filter():
    example1 = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    example2 = ak._v2.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}]
    )

    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"one": example1, "two": example2}
    )
    assert list(data_frame.GetColumnNames()) == ["one", "two"]

    compiler(
        """
template<typename T>
ROOT::RDF::RNode MyTransformation(ROOT::RDF::RNode df) {
    auto myFunc = [](T x){ return -x;};
    return df.Define("neg_one", myFunc, {"one"});
}
"""
    )

    data_frame_transformed = ROOT.MyTransformation[data_frame.GetColumnType("one")](
        ROOT.RDF.AsRNode(data_frame)
    )
    assert list(data_frame_transformed.GetColumnNames()) == ["neg_one", "one", "two"]
    assert data_frame_transformed.Count().GetValue() == 5

    data_frame2 = data_frame.Filter("one > 2.5")
    data_frame2_transformed = ROOT.MyTransformation[data_frame.GetColumnType("one")](
        ROOT.RDF.AsRNode(data_frame2)
    )
    assert data_frame2_transformed.Count().GetValue() == 3


def test_jims_example1():
    array = ak._v2.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}])
    data_frame = ak._v2.to_rdataframe({"some_array": array})
    assert list(data_frame.GetColumnNames()) == ["some_array"]
    data_frame_y = data_frame.Define("y", "some_array.x()")
    assert list(data_frame_y.GetColumnNames()) == ["some_array", "y"]

    cpp_list = ", ".join(str(e) for e in array.x.to_list())

    done = compiler(
        f"""
        int i_y1 = 0;
        double y1_val[5] = {{ {cpp_list} }};

        template<typename T>
        struct MyFunctorY_1 {{
            void operator()(T const& y) {{
                R__ASSERT(y == y1_val[i_y1++]);
            }}
        }};
        """
    )
    assert done is True

    f_y = ROOT.MyFunctorY_1[data_frame_y.GetColumnType("y")]()

    data_frame_y.Foreach(f_y, ["y"])


def test_jims_example2():
    example1 = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    example2 = ak._v2.Array(
        [
            {"x": [1.1, 1.2, 1.3]},
            {"x": [2.2, 2.21]},
            {"x": [3.3]},
            {"x": [4.4, 4.41, 4.42, 4.44]},
            {"x": [5.5]},
        ]
    )

    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"one": example1, "two": example2}
    )

    assert data_frame.GetColumnType("one") == "double"
    assert data_frame.GetColumnType("two").startswith("awkward::Record_")

    done = compiler(
        """
        int i_one = 0;
        double one_val[5] = {1.1, 2.2, 3.3, 4.4, 5.5};

        template<typename T>
        struct check_one {
            void operator()(T const& one) {
                R__ASSERT(one == one_val[i_one++]);
            }
        };

        int i_two = 0;
        ROOT::RVec<ROOT::RVec<double>> two_val =
            {{ 1.1, 1.2, 1.3 },
             { 2.2, 2.21 },
             { 3.3 },
             { 4.4, 4.41, 4.42, 4.44 },
             { 5.5 }
            };

        template<typename T>
        struct check_two {
            void operator()(T const& two) {
                for( int64_t j = 0; j < two.x().size(); j++ ) {
                    R__ASSERT(two.x()[j] == two_val[i_two][j]);
                }
                i_two++;
            }
        };
        """
    )
    assert done is True

    f_one = ROOT.check_one[data_frame.GetColumnType("one")]()
    f_two = ROOT.check_two[data_frame.GetColumnType("two")]()

    data_frame.Foreach(f_one, ["one"])
    data_frame.Foreach(f_two, ["two"])


def test_empty_array():
    array = ak._v2.Array([])
    data_frame = ak._v2.to_rdataframe({"empty_array": array})
    assert data_frame.GetColumnType("empty_array") == "double"
    assert data_frame.Count().GetValue() == 0


def test_empty_list_array():
    array = ak._v2.Array([[], [], []])
    data_frame = ak._v2.to_rdataframe({"empty_list_array": array})
    assert data_frame.GetColumnType("empty_list_array") == "ROOT::RVec<double>"
    assert data_frame.Count().GetValue() == 3
