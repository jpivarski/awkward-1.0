// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/operations.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/type/ArrayType.h"

#include "awkward/Content.h"

namespace awkward {
  Content::Content(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters)
      : identities_(identities)
      , parameters_(parameters) { }

  Content::~Content() { }

  bool Content::isscalar() const {
    return false;
  }

  const std::shared_ptr<Identities> Content::identities() const {
    return identities_;
  }

  const std::string Content::tostring() const {
    return tostring_part("", "", "");
  }

  const std::string Content::tojson(bool pretty, int64_t maxdecimals) const {
    if (pretty) {
      ToJsonPrettyString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
    else {
      ToJsonString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
  }

  void Content::tojson(FILE* destination, bool pretty, int64_t maxdecimals, int64_t buffersize) const {
    if (pretty) {
      ToJsonPrettyFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
    else {
      ToJsonFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
  }

  int64_t Content::nbytes() const {
    // FIXME: this is only accurate if all subintervals of allocated arrays are nested
    // (which is likely, but not guaranteed). In general, it's <= the correct nbytes.
    std::map<size_t, int64_t> largest;
    nbytes_part(largest);
    int64_t out = 0;
    for (auto pair : largest) {
      out += pair.second;
    }
    return out;
  }

  const util::Parameters Content::parameters() const {
    return parameters_;
  }

  void Content::setparameters(const util::Parameters& parameters) {
    parameters_ = parameters;
  }

  const std::string Content::parameter(const std::string& key) const {
    auto item = parameters_.find(key);
    if (item == parameters_.end()) {
      return "null";
    }
    return item->second;
  }

  void Content::setparameter(const std::string& key, const std::string& value) {
    parameters_[key] = value;
  }

  bool Content::parameter_equals(const std::string& key, const std::string& value) const {
    return util::parameter_equals(parameters_, key, value);
  }

  bool Content::parameters_equal(const util::Parameters& other) const {
    return util::parameters_equal(parameters_, other);
  }

  const std::shared_ptr<Content> Content::merge_as_union(const std::shared_ptr<Content>& other) const {
    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index8 tags(mylength + theirlength);
    Index64 index(mylength + theirlength);

    std::vector<std::shared_ptr<Content>> contents({ shallow_copy(), other });

    struct Error err1 = awkward_unionarray_filltags_to8_const(
      tags.ptr().get(),
      0,
      mylength,
      0);
    util::handle_error(err1, classname(), identities_.get());
    struct Error err2 = awkward_unionarray_fillindex_to64_count(
      index.ptr().get(),
      0,
      mylength);
    util::handle_error(err2, classname(), identities_.get());

    struct Error err3 = awkward_unionarray_filltags_to8_const(
      tags.ptr().get(),
      mylength,
      theirlength,
      1);
    util::handle_error(err3, classname(), identities_.get());
    struct Error err4 = awkward_unionarray_fillindex_to64_count(
      index.ptr().get(),
      mylength,
      theirlength);
    util::handle_error(err4, classname(), identities_.get());

    return std::make_shared<UnionArray8_64>(Identities::none(), util::Parameters(), tags, index, contents);
  }

  const std::shared_ptr<Content> Content::getitem(const Slice& where) const {
    std::shared_ptr<Content> next = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), shallow_copy(), length());

    std::shared_ptr<SliceItem> nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    std::shared_ptr<Content> out = next.get()->getitem_next(nexthead, nexttail, nextadvanced);

    if (out.get()->length() == 0) {
      return out.get()->getitem_nothing();
    }
    else {
      return out.get()->getitem_at_nowrap(0);
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at, tail, advanced);
    }
    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range, tail, advanced);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array, tail, advanced);
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      return getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      return getitem_next(*fields, tail, advanced);
    }
    else if (SliceMissing64* missing = dynamic_cast<SliceMissing64*>(head.get())) {
      return getitem_next(*missing, tail, advanced);
    }
    else if (SliceJagged64* jagged = dynamic_cast<SliceJagged64*>(head.get())) {
      return getitem_next(*jagged, tail, advanced);
    }
    else {
      throw std::runtime_error("unrecognized slice type");
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& advanced) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;
    int64_t maxdepth = minmax.second;

    if (tail.length() == 0  ||  (mindepth - 1 == tail.dimlength()  &&  maxdepth - 1 == tail.dimlength())) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead, nexttail, advanced);
    }
    else if (mindepth - 1 == tail.dimlength()  ||  maxdepth - 1 == tail.dimlength()) {
      throw std::invalid_argument("ellipsis (...) can't be used on a data structure of different depths");
    }
    else {
      std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
      std::vector<std::shared_ptr<SliceItem>> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      std::shared_ptr<SliceItem> nexthead = std::make_shared<SliceRange>(Slice::none(), Slice::none(), 1);
      Slice nexttail(items);
      return getitem_next(nexthead, nexttail, advanced);
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return std::make_shared<RegularArray>(Identities::none(), util::Parameters(), getitem_next(nexthead, nexttail, advanced), 1);
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_field(field.key()).get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_fields(fields.keys()).get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceMissing64& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: getitem_next(SliceMissing64)");
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceJagged64& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: getitem_next(SliceJagged64)");
  }

  const std::shared_ptr<Content> Content::getitem_next_array_wrap(const std::shared_ptr<Content>& outcontent, const std::vector<int64_t>& shape) const {
    std::shared_ptr<Content> out = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), outcontent, (int64_t)shape[shape.size() - 1]);
    for (int64_t i = (int64_t)shape.size() - 2;  i >= 0;  i--) {
      out = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), out, (int64_t)shape[(size_t)i]);
    }
    return out;
  }

  const std::string Content::parameters_tostring(const std::string& indent, const std::string& pre, const std::string& post) const {
    if (parameters_.empty()) {
      return "";
    }
    else {
      std::stringstream out;
      out << indent << pre << "<parameters>\n";
      for (auto pair : parameters_) {
        out << indent << "    <param key=" << util::quote(pair.first, true) << ">" << pair.second << "</param>\n";
      }
      out << indent << "</parameters>" << post;
      return out.str();
    }
  }

  const int64_t Content::axis_wrap_if_negative(int64_t axis) const {
    int64_t mindepth = minmax_depth().first;
    int64_t maxdepth = minmax_depth().second;
    int64_t depth = purelist_depth();
    if (axis < 0  &&  mindepth == depth  &&  maxdepth == depth) {
      return (depth - 1 + axis);
    }
    else {
      return axis;
    }
  }
}
