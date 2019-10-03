// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <sstream>

#include "awkward/util.h"
#include "awkward/Identity.h"

namespace awkward {
  namespace util {
    void handle_error(const Error& err, const std::string classname, const Identity* id) {
      if (err.str != nullptr) {
        std::stringstream out;
        out << "in " << classname;
        if (err.location != kSliceNone) {
          assert(err.location > 0);
          out << " at location " << err.location;
          if (id != nullptr  &&  err.location < id->length()) {
            out << " (id " << id->location(err.location) << ")";
          }
        }
        if (err.attempt != kSliceNone) {
          out << " attempting to get " << err.attempt;
        }
        out << ", " << err.str;
        throw std::invalid_argument(out.str());
      }
    }
  }
}
