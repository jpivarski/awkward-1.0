// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/StringBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/StringBuilder.h"

namespace awkward {
  const BuilderPtr
  StringBuilder::fromempty(const ArrayBuilderOptions& options,
                           const char* encoding) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    GrowableBuffer<uint8_t> content = GrowableBuffer<uint8_t>::empty(options);
    return std::make_shared<StringBuilder>(options,
                                           offsets,
                                           content,
                                           encoding);
  }

  StringBuilder::StringBuilder(const ArrayBuilderOptions& options,
                               const GrowableBuffer<int64_t>& offsets,
                               const GrowableBuffer<uint8_t>& content,
                               const char* encoding)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , encoding_(encoding) { }

  const std::string
  StringBuilder::classname() const {
    return "StringBuilder";
  };

  const char*
  StringBuilder::encoding() const {
    return encoding_;
  }

  const std::string
  StringBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream outer_form_key;
    std::stringstream inner_form_key;
    outer_form_key << "node" << (form_key_id++);
    inner_form_key << "node" << (form_key_id++);

    container.copy_buffer(outer_form_key.str() + "-offsets",
                          offsets_.ptr().get(),
                          offsets_.length() * sizeof(int64_t));

    container.copy_buffer(inner_form_key.str() + "-data",
                          content_.ptr().get(),
                          content_.length() * sizeof(uint8_t));

    return std::string("{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": ")
           + "{\"class\": \"NumpyArray\", \"primitive\": \"uint8\", \"parameters\": {\"__array__\": \"char\"}, \"form_key\": \"" + inner_form_key.str() + "\"}"
           + ", \"parameters\": {\"__array__\": \"string\"}"
           + ", \"form_key\": \"" + outer_form_key.str() + "\"}";
  }

  int64_t
  StringBuilder::length() const {
    return offsets_.length() - 1;
  }

  void
  StringBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.clear();
  }

  bool
  StringBuilder::active() const {
    return false;
  }

  const BuilderPtr
  StringBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  StringBuilder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  StringBuilder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  StringBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (length < 0) {
      for (int64_t i = 0;  x[i] != 0;  i++) {
        content_.append((uint8_t)x[i]);
      }
    }
    else {
      for (int64_t i = 0;  i < length;  i++) {
        content_.append((uint8_t)x[i]);
      }
    }
    offsets_.append(content_.length());
    return shared_from_this();
  }

  const BuilderPtr
  StringBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  StringBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  StringBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  StringBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

}
