// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"

namespace awkward {
  class Fillable {
  public:
    virtual const std::shared_ptr<Content> toarray() const = 0;
    virtual int64_t length() const = 0;
    virtual void clear() = 0;

    virtual Fillable* boolean(bool x) = 0;
  };
}

#endif // AWKWARD_FILLABLE_H_
