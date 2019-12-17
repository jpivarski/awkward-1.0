// BSD 3-Clause License; see
// https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include <cstdio>

#include "awkward/Identity.h"
#include "awkward/Slice.h"
#include "awkward/cpu-kernels/util.h"
#include "awkward/io/json.h"
#include "awkward/type/Type.h"

namespace awkward {
class Content {
public:
  Content(std::shared_ptr<Identity> id, std::shared_ptr<Type> type)
      : id_(id), type_(type) {}
  virtual ~Content() {}

  virtual bool isscalar() const;
  virtual const std::string classname() const = 0;
  virtual const std::shared_ptr<Identity> id() const;
  virtual void setid() = 0;
  virtual void setid(const std::shared_ptr<Identity> id) = 0;
  virtual const std::string tostring_part(const std::string indent,
                                          const std::string pre,
                                          const std::string post) const = 0;
  virtual void tojson_part(ToJson &builder) const = 0;
  virtual const std::shared_ptr<Type> innertype(bool bare) const = 0;
  virtual const std::shared_ptr<Type> type() const;
  virtual void settype(const std::shared_ptr<Type> type);
  virtual void settype_part(const std::shared_ptr<Type> type) = 0;
  virtual bool accepts(const std::shared_ptr<Type> type) = 0;
  virtual int64_t length() const = 0;
  virtual const std::shared_ptr<Content> shallow_copy() const = 0;
  virtual void check_for_iteration() const = 0;
  virtual const std::shared_ptr<Content> getitem_nothing() const = 0;
  virtual const std::shared_ptr<Content> getitem_at(int64_t at) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_at_nowrap(int64_t at) const = 0;
  virtual const std::shared_ptr<Content> getitem_range(int64_t start,
                                                       int64_t stop) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_range_nowrap(int64_t start, int64_t stop) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_field(const std::string &key) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_fields(const std::vector<std::string> &keys) const = 0;
  virtual const std::shared_ptr<Content> getitem(const Slice &where) const;
  virtual const std::shared_ptr<Content>
  getitem_next(const std::shared_ptr<SliceItem> head, const Slice &tail,
               const Index64 &advanced) const;
  virtual const std::shared_ptr<Content> carry(const Index64 &carry) const = 0;
  virtual const std::pair<int64_t, int64_t> minmax_depth() const = 0;
  virtual int64_t numfields() const = 0;
  virtual int64_t fieldindex(const std::string &key) const = 0;
  virtual const std::string key(int64_t fieldindex) const = 0;
  virtual bool haskey(const std::string &key) const = 0;
  virtual const std::vector<std::string>
  keyaliases(int64_t fieldindex) const = 0;
  virtual const std::vector<std::string>
  keyaliases(const std::string &key) const = 0;
  virtual const std::vector<std::string> keys() const = 0;

  bool isbare() const;
  const std::shared_ptr<Type> baretype() const;
  const std::string tostring() const;
  const std::string tojson(bool pretty, int64_t maxdecimals) const;
  void tojson(FILE *destination, bool pretty, int64_t maxdecimals,
              int64_t buffersize) const;

protected:
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceAt &at, const Slice &tail,
               const Index64 &advanced) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceRange &range, const Slice &tail,
               const Index64 &advanced) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceEllipsis &ellipsis, const Slice &tail,
               const Index64 &advanced) const;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceNewAxis &newaxis, const Slice &tail,
               const Index64 &advanced) const;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceArray64 &array, const Slice &tail,
               const Index64 &advanced) const = 0;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceField &field, const Slice &tail,
               const Index64 &advanced) const;
  virtual const std::shared_ptr<Content>
  getitem_next(const SliceFields &fields, const Slice &tail,
               const Index64 &advanced) const;

  const std::shared_ptr<Content>
  getitem_next_array_wrap(const std::shared_ptr<Content> outcontent,
                          const std::vector<int64_t> &shape) const;

protected:
  std::shared_ptr<Identity> id_;
  std::shared_ptr<Type> type_;
};
} // namespace awkward

#endif // AWKWARD_CONTENT_H_
