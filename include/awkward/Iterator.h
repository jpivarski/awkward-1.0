// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ITERATOR_H_
#define AWKWARD_ITERATOR_H_

#include "awkward/util.h"
#include "awkward/Content.h"

namespace awkward {
  class Iterator {
  public:
    Iterator(const std::shared_ptr<Content> content)
        : content_(content)
        , where_(0) { }

    const std::shared_ptr<Content> content() const { return content_; }
    const IndexType where() const { return where_; }

    const bool isdone() const { return where_ >= content_.get()->length(); }
    const std::shared_ptr<Content> next() { return content_.get()->get(where_++)->shallow_copy(); }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;

  private:
    const std::shared_ptr<Content> content_;
    IndexType where_;
  };
}

#endif // AWKWARD_ITERATOR_H_
