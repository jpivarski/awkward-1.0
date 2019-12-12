// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTFILLABLE_H_
#define AWKWARD_LISTFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class ListFillable: public Fillable {
  public:
    ListFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, std::shared_ptr<Fillable> content, bool begun): options_(options), offsets_(offsets), content_(std::shared_ptr<Fillable>(content)), begun_(begun) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);

    virtual const std::string classname() const { return "ListFillable"; };
    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual const std::shared_ptr<Fillable> null();
    virtual const std::shared_ptr<Fillable> boolean(bool x);
    virtual const std::shared_ptr<Fillable> integer(int64_t x);
    virtual const std::shared_ptr<Fillable> real(double x);
    virtual const std::shared_ptr<Fillable> beginlist();
    virtual const std::shared_ptr<Fillable> endlist();
    virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields);
    virtual const std::shared_ptr<Fillable> index(int64_t index);
    virtual const std::shared_ptr<Fillable> endtuple();
    virtual const std::shared_ptr<Fillable> beginrecord(int64_t disambiguator);
    virtual const std::shared_ptr<Fillable> field_fast(const char* key);
    virtual const std::shared_ptr<Fillable> field_check(const char* key);
    virtual const std::shared_ptr<Fillable> endrecord();

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> offsets_;
    std::shared_ptr<Fillable> content_;
    bool begun_;

    void maybeupdate(const std::shared_ptr<Fillable>& tmp);
  };
}

#endif // AWKWARD_LISTFILLABLE_H_
