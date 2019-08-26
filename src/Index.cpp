// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Index.h"

using awkward;

const std::string IndexOf<T>::repr() const {
  std::stringstream out;
  out << "<Index [";
  if (len() <= 10) {
    for (int i = 0;  i < len();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
  }
  else {
    for (int i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
    out << " ... ";
    for (int i = len() - 6;  i < len();  i++) {
      if (i != len() - 6) {
        out << " ";
      }
      out << get(i);
    }
  }
  out << "] at 0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << ">";
  return out.str();
}

T IndexOf<T>::get(T at) {
  assert(0 <= at  &&  at < length_);
  return ptr_.get()[offset_ + at];
}

IndexOf<T> IndexOf<T>::slice(T start, T stop) {
  assert(start == stop  ||  (0 <= start  &&  start < length_));
  assert(start == stop  ||  (0 < stop    &&  stop <= length_));
  assert(start <= stop);
  return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start);
}
