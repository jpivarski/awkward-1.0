// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/LayoutBuilder.h"

static const unsigned initial = 10;

template <class NODE, class PRIMITIVE, class LENGTH>
void dump(NODE&& node, PRIMITIVE&& ptr, LENGTH&& length) {
  std::cout << node << ": ";
  for (size_t i = 0; i < length; i++) {
    std::cout << +ptr[i] << " ";
  }
  std::cout << std::endl;
}

template<class NODE, class PRIMITIVE, class LENGTH, class ... Args>
void dump(NODE&& node, PRIMITIVE&& ptr, LENGTH&& length, Args&&...args)
{
    dump(node, ptr, length);
    dump(args...);
}

std::map<std::string, void*>
empty_buffers(std::map<std::string, size_t> &names_nbytes)
{
  std::map<std::string, void*> buffers = {};
  for(auto it : names_nbytes) {
    uint8_t* ptr = new uint8_t[it.second];
    buffers[it.first] = (void*)ptr;
  }
  return buffers;
}

using UserDefinedMap = std::map<std::size_t, std::string>;

template<class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<initial, PRIMITIVE>;

template<class BUILDER>
using ListOffsetBuilder = awkward::LayoutBuilder::ListOffset<initial, BUILDER>;

template<class... BUILDERS>
using RecordBuilder = awkward::LayoutBuilder::Record<UserDefinedMap, BUILDERS...>;

template<class BUILDER>
using ListBuilder = awkward::LayoutBuilder::List<initial, BUILDER>;

template<class BUILDER>
using IndexedBuilder = awkward::LayoutBuilder::Indexed<initial, BUILDER>;

template<class BUILDER>
using IndexedOptionBuilder = awkward::LayoutBuilder::IndexedOption<initial, BUILDER>;

template<class BUILDER>
using UnmaskedBuilder = awkward::LayoutBuilder::Unmasked<BUILDER>;

template<bool VALID_WHEN, class BUILDER>
using ByteMaskedBuilder = awkward::LayoutBuilder::ByteMasked<initial, VALID_WHEN, BUILDER>;

template<bool VALID_WHEN, bool LSB_ORDER, class BUILDER>
using BitMaskedBuilder = awkward::LayoutBuilder::BitMasked<initial, VALID_WHEN, LSB_ORDER, BUILDER>;

template <unsigned SIZE, class BUILDER>
using RegularBuilder = awkward::LayoutBuilder::Regular<SIZE, BUILDER>;

template<bool IS_TUPLE>
using EmptyRecordBuilder = awkward::LayoutBuilder::EmptyRecord<IS_TUPLE>;

using EmptyBuilder = awkward::LayoutBuilder::Empty;

template<std::size_t field_name, class BUILDER>
using RecordField = awkward::LayoutBuilder::Field<field_name, BUILDER>;

void
test_Numpy_bool() {

  NumpyBuilder<bool> builder;

  builder.append(true);
  builder.append(false);
  builder.append(true);
  builder.append(true);

  // [True, False, True, True]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-data", (bool*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(bool));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"bool\", "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Numpy_int() {
  NumpyBuilder<int64_t> builder;

  size_t data_size = 10;

  int64_t data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  builder.extend(data, data_size);

 // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-data", (int64_t*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(int64_t));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"int64\", "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Numpy_char() {
  NumpyBuilder<char> builder;

  builder.append('a');
  builder.append('b');
  builder.append('c');
  builder.append('d');

  // ['a', 'b', 'c', 'd']

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-data", (char*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(char));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"char\", "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Numpy_double() {
  NumpyBuilder<double> builder;

  size_t data_size = 9;

  double data[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  builder.extend(data, data_size);

  // [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-data", (double*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"float64\", "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Numpy_complex() {
    NumpyBuilder<std::complex<double>> builder;

  builder.append({1.1, 0.1});
  builder.append({2.2, 0.2});
  builder.append({3.3, 0.3});
  builder.append({4.4, 0.4});
  builder.append({5.5, 0.5});

  // [1.1 + 0.1j, 2.2 + 0.2j, 3.3 + 0.3j, 4.4 + 0.4j, 5.5 + 0.5j]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-data", (std::complex<double>*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(std::complex<double>));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"complex128\", "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_ListOffset() {
  ListOffsetBuilder<NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  builder.end_list();

  // [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_ListOffset_ListOffset() {
  ListOffsetBuilder<ListOffsetBuilder<NumpyBuilder<double>>> builder;

  auto& builder2 = builder.begin_list();

  auto& builder3 = builder2.begin_list();
  builder3.append(1.1);
  builder3.append(2.2);
  builder3.append(3.3);
  builder2.end_list();
  builder2.begin_list();
  builder2.end_list();
  builder.end_list();

  builder.begin_list();
  builder2.begin_list();
  builder3.append(4.4);
  builder3.append(5.5);
  builder2.end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder2.begin_list();
  builder3.append(6.6);
  builder2.end_list();
  builder2.begin_list();
  builder3.append(7.7);
  builder3.append(8.8);
  builder3.append(9.9);
  builder2.end_list();
  builder.end_list();

  // [
  //     [[1.1, 2.2, 3.3], []],
  //     [[4.4, 5.5]],
  //     [],
  //     [[6.6], [7.7, 8.8, 9.9]],
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_EmptyRecord() {
  EmptyRecordBuilder<true> builder;

  builder.append();

  builder.extend(2);

  // [(), (), ()]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 0);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": [], "
      "\"form_key\": \"node0\" "
  "}");
}

void
test_Record()
{
  enum Field : std::size_t {one, two, three};

  UserDefinedMap fields_map({
    {Field::one, "one"},
    {Field::two, "two"},
    {Field::three, "three"}});

    RecordBuilder<
        RecordField<Field::one, NumpyBuilder<double>>,
        RecordField<Field::two, NumpyBuilder<int64_t>>,
        RecordField<Field::three, NumpyBuilder<char>>
    > builder(fields_map);

  std::vector<std::string> fields {"one", "two", "three"};
  auto names = builder.field_names();

  for (size_t i = 0; i < names.size(); i++) {
    assert(names[i] == fields[i]);
  }

  auto& one_builder = builder.field<Field::one>();
  auto& two_builder = builder.field<Field::two>();
  auto& three_builder = builder.field<Field::three>();

  three_builder.append('a');

  one_builder.append(1.1);
  one_builder.append(3.3);

  two_builder.append(2);
  two_builder.append(4);

  std::string error;
  assert (builder.is_valid(error) == false);
  assert (error == "Record node0 has field \"three\" length 1 that differs from the first length 2\n");

  three_builder.append('b');

  // [
  //     {"one": 1.1, "two": 2, "three": 'a'},
  //     {"one": 3.3, "two": 4. "three": 'b'},
  // ]

  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double),
       "node2-data", (int64_t*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(int64_t),
       "node3-data", (char*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(char));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"one\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node1\" "
          "}, "
          "\"two\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"int64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"three\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"char\", "
              "\"form_key\": \"node3\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_ListOffset_Record() {
  enum Field : std::size_t {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  ListOffsetBuilder<
      RecordBuilder<
          RecordField<Field::x, NumpyBuilder<double>>,
          RecordField<Field::y, ListOffsetBuilder<
              NumpyBuilder<int32_t>>
  >>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.set_field_names(fields_map);

  auto& x_builder = subbuilder.field<Field::x>();
  auto& y_builder = subbuilder.field<Field::y>();

  x_builder.append(1.1);
  auto& y_subbuilder = y_builder.begin_list();
  y_subbuilder.append(1);
  y_builder.end_list();

  x_builder.append(2.2);
  y_builder.begin_list();
  y_subbuilder.append(1);
  y_subbuilder.append(2);
  y_builder.end_list();

  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();

  x_builder.append(3.3);
  y_builder.begin_list();
  y_subbuilder.append(1);
  y_subbuilder.append(2);
  y_subbuilder.append(3);
  y_builder.end_list();

  builder.end_list();

  // [
  //     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
  //     [],
  //     [{"x": 3.3, "y": [1, 2, 3]}],
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 4);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

 dump("node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
      "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
      "node3-offsets", (int64_t*)buffers["node3-offsets"], names_nbytes["node3-offsets"]/sizeof(int64_t),
      "node4-data", (int32_t*)buffers["node4-data"], names_nbytes["node4-data"]/sizeof(int32_t));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"RecordArray\", "
          "\"contents\": { "
              "\"x\": { "
                  "\"class\": \"NumpyArray\", "
                  "\"primitive\": \"float64\", "
                  "\"form_key\": \"node2\" "
              "}, "
              "\"y\": { "
                  "\"class\": \"ListOffsetArray\", "
                  "\"offsets\": \"i64\", "
                  "\"content\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"int32\", "
                      "\"form_key\": \"node4\" "
                  "}, "
                  "\"form_key\": \"node3\" "
              "} "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Record_Record()
{
  enum Field0 : std::size_t {x, y};

  UserDefinedMap fields_map0({
    {Field0::x, "x"},
    {Field0::y, "y"}});

  enum Field1 : std::size_t {u, v};

  UserDefinedMap fields_map1({
    {Field1::u, "u"},
    {Field1::v, "v"}});

  enum Field2 : std::size_t {w};

  UserDefinedMap fields_map2({
    {Field2::w, "w"}});

  RecordBuilder<
      RecordField<Field0::x, RecordBuilder<
          RecordField<Field1::u, NumpyBuilder<double>>,
          RecordField<Field1::v, ListOffsetBuilder<NumpyBuilder<int64_t>>>>>,
      RecordField<Field0::y, RecordBuilder<
          RecordField<Field2::w, NumpyBuilder<char>>>>
  > builder;
  builder.set_field_names(fields_map0);

  auto& x_builder = builder.field<Field0::x>();
  x_builder.set_field_names(fields_map1);

  auto& y_builder = builder.field<Field0::y>();
  y_builder.set_field_names(fields_map2);

  auto& u_builder = x_builder.field<Field1::u>();
  auto& v_builder = x_builder.field<Field1::v>();

  auto& w_builder = y_builder.field<Field2::w>();

  u_builder.append(1.1);
  auto& v_subbuilder = v_builder.begin_list();
  v_subbuilder.append(1);
  v_subbuilder.append(2);
  v_subbuilder.append(3);
  v_builder.end_list();

  w_builder.append('a');

  u_builder.append(3.3);
  v_builder.begin_list();
  v_subbuilder.append(4);
  v_subbuilder.append(5);
  v_builder.end_list();

  w_builder.append('b');

  // [
  //     {"x": {"u": 1.1, "v": [1, 2, 3]}, "y": {"w": 'a'}},
  //     {"x": {"u": 3.3, "v": [4, 5]}, "y": {"w": 'b'}},
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 4);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node3-offsets", (int64_t*)buffers["node3-offsets"], names_nbytes["node3-offsets"]/sizeof(int64_t),
       "node4-data", (int64_t*)buffers["node4-data"], names_nbytes["node4-data"]/sizeof(int64_t),
       "node6-data", (char*)buffers["node6-data"], names_nbytes["node6-data"]/sizeof(char));

  auto form = builder.form();

  assert(form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"x\": { "
              "\"class\": \"RecordArray\", "
              "\"contents\": { "
                  "\"u\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"float64\", "
                      "\"form_key\": \"node2\" "
                  "}, "
                  "\"v\": { "
                      "\"class\": \"ListOffsetArray\", "
                      "\"offsets\": \"i64\", "
                      "\"content\": { "
                          "\"class\": \"NumpyArray\", "
                          "\"primitive\": \"int64\", "
                          "\"form_key\": \"node4\" "
                      "}, "
                      "\"form_key\": \"node3\" "
                  "} "
              "}, "
              "\"form_key\": \"node1\" "
          "}, "
          "\"y\": { "
              "\"class\": \"RecordArray\", "
              "\"contents\": { "
                  "\"w\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"char\", "
                      "\"form_key\": \"node6\" "
                  "} "
              "}, "
              "\"form_key\": \"node5\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Record_nested()
{
  enum Field0 : std::size_t {u, v, w};

  UserDefinedMap fields_map0({
    {Field0::u, "u"},
    {Field0::v, "v"},
    {Field0::w, "w"}});

  enum Field1 : std::size_t {i, j};

  UserDefinedMap fields_map1({
    {Field1::i, "i"},
    {Field1::j, "j"}});

  RecordBuilder<
      RecordField<Field0::u, ListOffsetBuilder<
          RecordBuilder<
              RecordField<Field1::i, NumpyBuilder<double>>,
              RecordField<Field1::j, ListOffsetBuilder<
                  NumpyBuilder<int64_t>>>>>>,
      RecordField<Field0::v, NumpyBuilder<int64_t>>,
      RecordField<Field0::w, NumpyBuilder<double>>
  > builder;
  builder.set_field_names(fields_map0);

  auto& u_builder = builder.field<Field0::u>();
  auto& v_builder = builder.field<Field0::v>();
  auto& w_builder = builder.field<Field0::w>();

  auto& u_subbuilder = u_builder.begin_list();
  u_subbuilder.set_field_names(fields_map1);

  auto& i_builder = u_subbuilder.field<Field1::i>();
  auto& j_builder = u_subbuilder.field<Field1::j>();

  i_builder.append(1.1);
  auto& j_subbuilder = j_builder.begin_list();
  j_subbuilder.append(1);
  j_subbuilder.append(2);
  j_subbuilder.append(3);
  j_builder.end_list();

  u_builder.end_list();

  v_builder.append(-1);
  w_builder.append(3.3);

  u_builder.begin_list();

  i_builder.append(2.2);
  j_builder.begin_list();
  j_subbuilder.append(4);
  j_subbuilder.append(5);
  j_builder.end_list();

  u_builder.end_list();

  v_builder.append(-2);
  w_builder.append(4.4);

  // [
  //     {"u": [{"i": 1.1, "j": [1, 2, 3]}], "v": -1, "w": 3.3},
  //     {"u": [{"i": 2.2, "j": [4, 5]}], "v": -2, "w": 4.4},
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 6);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t),
       "node3-data", (double*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(double),
       "node4-offsets", (int64_t*)buffers["node4-offsets"], names_nbytes["node4-offsets"]/sizeof(int64_t),
       "node5-data", (int64_t*)buffers["node5-data"], names_nbytes["node5-data"]/sizeof(int64_t),
       "node6-data", (int64_t*)buffers["node6-data"], names_nbytes["node6-data"]/sizeof(int64_t),
       "node7-data", (double*)buffers["node7-data"], names_nbytes["node7-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"u\": { "
              "\"class\": \"ListOffsetArray\", "
              "\"offsets\": \"i64\", "
              "\"content\": { "
                  "\"class\": \"RecordArray\", "
                  "\"contents\": { "
                      "\"i\": { "
                          "\"class\": \"NumpyArray\", "
                          "\"primitive\": \"float64\", "
                          "\"form_key\": \"node3\" "
                      "}, "
                      "\"j\": { "
                          "\"class\": \"ListOffsetArray\", "
                          "\"offsets\": \"i64\", "
                          "\"content\": { "
                              "\"class\": \"NumpyArray\", "
                              "\"primitive\": \"int64\", "
                              "\"form_key\": \"node5\" "
                          "}, "
                          "\"form_key\": \"node4\" "
                      "} "
                  "}, "
                  "\"form_key\": \"node2\" "
              "}, "
              "\"form_key\": \"node1\" "
          "}, "
          "\"v\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"int64\", "
              "\"form_key\": \"node6\" "
          "}, "
          "\"w\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node7\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_List() {
  ListBuilder<NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  builder.end_list();

  builder.begin_list();
  subbuilder.append(6.6);

  builder.end_list();

  builder.begin_list();
  subbuilder.append(7.7);
  subbuilder.append(8.8);
  subbuilder.append(9.9);
  builder.end_list();

  // [
  //     [1.1, 2.2, 3.3],
  //     [],
  //     [4.4, 5.5],
  //     [6.6],
  //     [7.7, 8.8, 9.9],
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-starts", (int64_t*)buffers["node0-starts"], names_nbytes["node0-starts"]/sizeof(int64_t),
       "node0-stops", (int64_t*)buffers["node0-stops"], names_nbytes["node0-stops"]/sizeof(int64_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListArray\", "
      "\"starts\": \"i64\", "
      "\"stops\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Indexed() {
  IndexedBuilder<NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_index();
  subbuilder.append(1.1);

  builder.append_index();
  subbuilder.append(2.2);

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder.extend(data, 3);

  // [1.1, 2.2, 3.3, 4.4, 5.5]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-index", (int64_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int64_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));


  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"IndexedArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_IndexedOption() {
  IndexedOptionBuilder<NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_index();
  subbuilder.append(1.1);

  builder.append_null();

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);

  // [1.1, None, 3.3, 4.4, 5.5, None, None]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-index", (int64_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int64_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_IndexedOption_Record() {
  enum Field : std::size_t {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  IndexedOptionBuilder<RecordBuilder<
      RecordField<Field::x, NumpyBuilder<double>>,
      RecordField<Field::y, NumpyBuilder<int64_t>>
  >> builder;

  auto& subbuilder = builder.append_index();
  subbuilder.set_field_names(fields_map);

  auto& x_builder = subbuilder.field<Field::x>();
  auto& y_builder = subbuilder.field<Field::y>();

  x_builder.append(1.1);
  y_builder.append(2);

  builder.append_null();

  builder.append_index();
  x_builder.append(3.3);
  y_builder.append(4);

  // [
  //   {x: 1.1, y: 2},
  //   {},
  //   {x: 3.3, y: 4},
  // ]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-index", (int64_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int64_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node3-data", (int64_t*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(int64_t));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
            "\"class\": \"RecordArray\", "
            "\"contents\": { "
                "\"x\": { "
                    "\"class\": \"NumpyArray\", "
                    "\"primitive\": \"float64\", "
                    "\"form_key\": \"node2\" "
                "}, "
                "\"y\": { "
                    "\"class\": \"NumpyArray\", "
                    "\"primitive\": \"int64\", "
                    "\"form_key\": \"node3\" "
                "} "
            "}, "
            "\"form_key\": \"node1\" "
        "}, "
        "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Empty() {
  EmptyBuilder builder;

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 0);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"EmptyArray\" "
  "}");
}

void
test_ListOffset_Empty() {
  ListOffsetBuilder<ListOffsetBuilder<EmptyBuilder>> builder;

  builder.begin_list();
  builder.end_list();

  auto& subbuilder = builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  //  [[], [[], [], []], [[], []], [], [[]]]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"EmptyArray\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Unmasked() {
  UnmaskedBuilder<NumpyBuilder<int64_t>> builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(11);
  subbuilder.append(22);
  subbuilder.append(33);
  subbuilder.append(44);
  subbuilder.append(55);

  // [11, 22, 33, 44, 55]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node1-data", (int64_t*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(int64_t));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"UnmaskedArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_ByteMasked() {
  ByteMaskedBuilder<true, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_null();
  subbuilder.append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-mask", (int8_t*)buffers["node0-mask"], names_nbytes["node0-mask"]/sizeof(int8_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ByteMaskedArray\", "
      "\"mask\": \"i8\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"valid_when\": true, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_BitMasked() {
  BitMaskedBuilder<true, true, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_null();
  subbuilder.append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }

  builder.append_valid();
  subbuilder.append(8);

  builder.append_valid();
  subbuilder.append(9);

  builder.append_valid();
  subbuilder.append(10);

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000, 8, 9, 10]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node0-mask", (uint8_t*)buffers["node0-mask"], names_nbytes["node0-mask"]/sizeof(uint8_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"BitMaskedArray\", "
      "\"mask\": \"u8\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"valid_when\": true, "
      "\"lsb_order\": true, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Regular() {
  RegularBuilder<3, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  subbuilder.append(6.6);
  builder.end_list();

  // [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RegularArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"size\": 3, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

void
test_Regular_size0() {
  RegularBuilder<0, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  // [[], []]

  std::string error;
  assert (builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  dump("node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RegularArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"size\": 0, "
      "\"form_key\": \"node0\" "
  "}");

  std::cout << std::endl;
}

int main(int /* argc */, char ** /* argv */) {
  test_Numpy_bool();
  test_Numpy_int();
  test_Numpy_char();
  test_Numpy_double();
  test_Numpy_complex();
  test_ListOffset();
  test_ListOffset_ListOffset();
  test_EmptyRecord();
  test_Record();
  test_ListOffset_Record();
  test_Record_Record();
  test_Record_nested();
  test_List();
  test_Indexed();
  test_IndexedOption();
  test_IndexedOption_Record();
  test_Empty();
  test_ListOffset_Empty();
  test_Unmasked();
  test_ByteMasked();
  test_BitMasked();
  test_Regular();
  test_Regular_size0();

  return 0;
}
