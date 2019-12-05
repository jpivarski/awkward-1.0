// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARTYPE_H_
#define AWKWARD_REGULARTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class RegularType: public Type {
  public:
    RegularType(const std::shared_ptr<Type> type, int64_t size): type_(type), size_(size) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;
    virtual std::shared_ptr<Type> descend() const;
    virtual std::shared_ptr<Type> descend(std::string key) const;

    const std::shared_ptr<Type> type() const;
    int64_t size() const;

  private:
    const std::shared_ptr<Type> type_;
    const int64_t size_;
  };
}

#endif // AWKWARD_REGULARTYPE_H_
