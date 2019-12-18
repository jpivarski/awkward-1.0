// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARARRAY_H_
#define AWKWARD_REGULARARRAY_H_

#include <cassert>
#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  class RegularArray: public Content {
  public:
    RegularArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Type> type, const std::shared_ptr<Content> content, int64_t size)
        : Content(id, type)
        , content_(content)
        , size_(size) { }

    const std::shared_ptr<Content> content() const { return content_; }
    int64_t size() const { return size_; }

    const std::string classname() const override;
    void setid() override;
    void setid(const std::shared_ptr<Identity> id) override;
    const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const override;
    void tojson_part(ToJson& builder) const override;
    const std::shared_ptr<Type> innertype(bool bare) const override;
    void settype_part(const std::shared_ptr<Type> type) override;
    bool accepts(const std::shared_ptr<Type> type) override;
    int64_t length() const override;
    const std::shared_ptr<Content> shallow_copy() const override;
    void check_for_iteration() const override;
    const std::shared_ptr<Content> getitem_nothing() const override;
    const std::shared_ptr<Content> getitem_at(int64_t at) const override;
    const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const override;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_field(const std::string& key) const override;
    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const override;
    const std::shared_ptr<Content> carry(const Index64& carry) const override;
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keyaliases(int64_t fieldindex) const override;
    const std::vector<std::string> keyaliases(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

  protected:
    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;

  private:
    const std::shared_ptr<Content> content_;
    int64_t size_;
  };
}

#endif // AWKWARD_REGULARARRAY_H_
