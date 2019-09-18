// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/NumpyArray.h"

using namespace awkward;

ssize_t NumpyArray::ndim() const {
  return shape_.size();
}

bool NumpyArray::isscalar() const {
  return ndim() == 0;
}

bool NumpyArray::isempty() const {
  for (auto x : shape_) {
    if (x == 0) return true;
  }
  return false;  // false for isscalar(), too
}

bool NumpyArray::iscompact() const {
  ssize_t x = itemsize_;
  for (ssize_t i = ndim() - 1;  i >= 0;  i--) {
    if (x != strides_[i]) return false;
    x *= shape_[i];
  }
  return true;  // true for isscalar(), too
}

void* NumpyArray::byteptr() const {
  return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_);
}

ssize_t NumpyArray::bytelength() const {
  if (isscalar()) {
    return itemsize_;
  }
  else {
    return shape_[0]*strides_[0];
  }
}

uint8_t NumpyArray::getbyte(ssize_t at) const {
  return *reinterpret_cast<uint8_t*>(reinterpret_cast<ssize_t>(ptr_.get()) + byteoffset_ + at);
}

void NumpyArray::setid(const std::shared_ptr<Identity> id) {
  id_ = id;
}

void NumpyArray::setid() {
  assert(!isscalar());
  Identity32* id32 = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
  std::shared_ptr<Identity> newid(id32);
  Error err = awkward_identity_new32(length(), id32->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

template <typename T>
void tostring_as(std::stringstream& out, T* ptr, int64_t length) {
  if (length <= 10) {
    for (int64_t i = 0;  i < length;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << ptr[i];
    }
  }
  else {
    for (int64_t i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << ptr[i];
    }
    out << " ... ";
    for (int64_t i = length - 5;  i < length;  i++) {
      if (i != length - 5) {
        out << " ";
      }
      out << ptr[i];
    }
  }
}

const std::string NumpyArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  assert(!isscalar());
  std::stringstream out;
  out << indent << pre << "<NumpyArray format=\"" << format_ << "\" shape=\"";
  for (ssize_t i = 0;  i < ndim();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << shape_[i];
  }
  out << "\" ";
  if (!iscompact()) {
    out << "strides=\"";
    for (ssize_t i = 0;  i < ndim();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << strides_[i];
    }
    out << "\" ";
  }
  out << "data=\"";
#ifdef _MSC_VER
  if (ndim() == 1  &&  format_.compare("l") == 0) {
#else
  if (ndim() == 1  &&  format_.compare("i") == 0) {
#endif
    tostring_as<int32_t>(out, reinterpret_cast<int32_t*>(byteptr()), length());
  }
#ifdef _MSC_VER
  else if (ndim() == 1  &&  format_.compare("q") == 0) {
#else
  else if (ndim() == 1  &&  format_.compare("l") == 0) {
#endif
    tostring_as<int64_t>(out, reinterpret_cast<int64_t*>(byteptr()), length());
  }
  else if (ndim() == 1  &&  format_.compare("f") == 0) {
    tostring_as<float>(out, reinterpret_cast<float*>(byteptr()), length());
  }
  else if (ndim() == 1  &&  format_.compare("d") == 0) {
    tostring_as<double>(out, reinterpret_cast<double*>(byteptr()), length());
  }
  else {
    ssize_t len = bytelength();
    if (len <= 32) {
      for (ssize_t i = 0;  i < len;  i++) {
        if (i != 0  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
    }
    else {
      for (ssize_t i = 0;  i < 16;  i++) {
        if (i != 0  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
      out << " ... ";
      for (ssize_t i = len - 16;  i < len;  i++) {
        if (i != len - 16  &&  i % 4 == 0) {
          out << " ";
        }
        out << std::hex << std::setw(2) << std::setfill('0') << int(getbyte(i));
      }
    }
  }
  out << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get());
  if (id_.get() == nullptr) {
    out << "\"/>" << post;
  }
  else {
    out << "\">\n";
    out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    out << indent << "</NumpyArray>" << post;
  }
  return out.str();
}

int64_t NumpyArray::length() const {
  if (isscalar()) {
    return -1;
  }
  else {
    return (int64_t)shape_[0];
  }
}

const std::shared_ptr<Content> NumpyArray::shallow_copy() const {
  return std::shared_ptr<Content>(new NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_));
}

const std::shared_ptr<Content> NumpyArray::get(int64_t at) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)at);
  const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
  const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
  return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, strides, byteoffset, itemsize_, format_));
}

const std::shared_ptr<Content> NumpyArray::slice(int64_t start, int64_t stop) const {
  assert(!isscalar());
  ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)start);
  std::vector<ssize_t> shape;
  shape.push_back((ssize_t)(stop - start));
  shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new NumpyArray(id, ptr_, shape, strides_, byteoffset, itemsize_, format_));
}

const std::pair<int64_t, int64_t> NumpyArray::minmax_depth() const {
  return std::pair<int64_t, int64_t>((int64_t)shape_.size(), (int64_t)shape_.size());
}






// const std::vector<ssize_t> shape2strides(const std::vector<ssize_t>& shape, ssize_t itemsize) {
//   std::vector<ssize_t> out;
//   for (auto dim = shape.rbegin();  dim != shape.rend();  ++dim) {
//     out.insert(out.begin(), itemsize);
//     itemsize *= *dim;
//   }
//   return out;
// }

// ssize_t shape_product(const std::vector<ssize_t>& shape) {
//   ssize_t out = 1;
//   for (auto dim : shape) {
//     out *= dim;
//   }
//   return out;
// }

// const std::shared_ptr<Content> NumpyArray::getitem(const Slice& slice) const {
//   std::shared_ptr<SliceItem> head = slice.head();
//   Slice tail = slice.tail();
//   return getitem_next(head, tail);
// }
//
// #include <iostream>
//
// const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail) const {
//   if (head.get() == nullptr) {
//     return shallow_copy();
//   }
//
//   else if (SliceAt* h = dynamic_cast<SliceAt*>(head.get())) {
//     if (isscalar()) {
//       throw std::invalid_argument("too many dimensions in index for this array");
//     }
//     std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
//     ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)h->at());
//     std::shared_ptr<SliceItem> nexthead = tail.head();
//     Slice nexttail = tail.tail();
//     std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
//     return next.get()->getitem_next(nexthead, nexttail);
//   }
//
//   else if (SliceStartStop* h = dynamic_cast<SliceStartStop*>(head.get())) {
//     if (isscalar()) {
//       throw std::invalid_argument("too many dimensions in index for this array");
//     }
//     std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
//     ssize_t byteoffset = byteoffset_ + strides_[0]*((ssize_t)h->start());
//     if (tail.length() == 0) {
//       shape.insert(shape.begin(), (ssize_t)(h->stop() - h->start()));
//       return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
//     }
//     else {
//       std::shared_ptr<SliceItem> nexthead = tail.head();
//       Slice nexttail = tail.tail();
//       Index64 nextcarry(h->stop() - h->start());
//       int64_t step = shape_product(std::vector<ssize_t>(shape_.begin() + 1, shape_.end())); //(shape_.size() == 1 ? 1 : (int64_t)shape_[1]);
//       for (int64_t i = 0;  i < nextcarry.length();  i++) {
//         nextcarry.ptr().get()[i] = step*i;
//       }
//       std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset, itemsize_, format_));
//       std::shared_ptr<Content> done = next.get()->getitem_next(nexthead, nexttail, nextcarry);
//       NumpyArray* donearray = dynamic_cast<NumpyArray*>(done.get());
//       if (donearray->length() == nextcarry.length()) {
//         return done;
//       }
//       else {   //  if (dynamic_cast<SliceAt*>(nexthead.get()) != nullptr) {
//         std::vector<ssize_t> doneshape = donearray->shape();
//         std::vector<ssize_t> outshape({ (ssize_t)nextcarry.length(), (ssize_t)(donearray->length() / nextcarry.length()) });
//         outshape.insert(outshape.end(), doneshape.begin() + 1, doneshape.end());
//         return std::shared_ptr<Content>(new NumpyArray(Identity::none(), donearray->ptr(), outshape, shape2strides(outshape, itemsize_), donearray->byteoffset(), itemsize_, format_));
//       }
//       // else {
//       //   std::cout << "WTF " << nextcarry.length() << " " << donearray->length() << std::endl;
//       //   std::vector<ssize_t> doneshape = donearray->shape();
//       //   std::cout << "doneshape " << doneshape.size() << std::endl;
//       //   for (int i = 0;  i < doneshape.size();  i++) {
//       //     std::cout << "doneshape[" << i << "] " << doneshape[i] << std::endl;
//       //   }
//       //
//       //   std::vector<ssize_t> outshape({ (ssize_t)nextcarry.length(), (ssize_t)donearray->length() });
//       //   outshape.insert(outshape.end(), doneshape.begin() + 1, doneshape.end());
//       //   return std::shared_ptr<Content>(new NumpyArray(Identity::none(), donearray->ptr(), outshape, shape2strides(outshape, itemsize_), donearray->byteoffset(), itemsize_, format_));
//       // }
//     }
//   }
//
//   else {
//     assert(false);
//   }
// }
//
// const std::shared_ptr<Content> NumpyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry) const {
//   if (head.get() == nullptr) {
//     std::vector<ssize_t> shape = { 0 };   // reassigned below
//     int64_t skip = itemsize_;
//     int64_t copylen = 1;
//     if (shape_.size() != 0) {
//       shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
//       skip = strides_[0];
//       copylen = shape_[0];
//     }
//     shape[0] = carry.length()*copylen;
//     uint8_t* src = reinterpret_cast<uint8_t*>(ptr_.get());
//     uint8_t* dst = new uint8_t[(size_t)(carry.length()*skip*copylen)];
//
//     for (int64_t i = 0;  i < carry.length();  i++) {
//       std::cout << "COPYING " << skip << "*" << carry.get(i) << " = " << skip*carry.get(i) << " for " << copylen << " which is " << *reinterpret_cast<int64_t*>(&src[(size_t)(byteoffset_ + skip*carry.get(i))]) << std::endl;
//       std::memcpy(&dst[(size_t)(skip*copylen*i)], &src[(size_t)(byteoffset_ + skip*carry.get(i))], skip*copylen);
//     }
//     std::shared_ptr<uint8_t> ptr(dst, awkward::util::array_deleter<uint8_t>());
//     return std::shared_ptr<Content>(new NumpyArray(Identity::none(), ptr, shape, shape2strides(shape, itemsize_), 0, itemsize_, format_));
//   }
//
//   else if (SliceAt* h = dynamic_cast<SliceAt*>(head.get())) {
//     if (isscalar()) {
//       throw std::invalid_argument("too many dimensions in index for this array");
//     }
//     std::shared_ptr<SliceItem> nexthead = tail.head();
//     Slice nexttail = tail.tail();
//     Index64 nextcarry(carry.length());
//     int64_t skip = (shape_.size() > 1 ? shape_[1] : 1);
//     for (int64_t i = 0;  i < nextcarry.length();  i++) {
//       nextcarry.ptr().get()[i] = carry.ptr().get()[i] + skip*h->at();
//     }
//     std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
//     std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset_, itemsize_, format_));
//     return next.get()->getitem_next(nexthead, nexttail, nextcarry);
//   }
//
//   else if (SliceStartStop* h = dynamic_cast<SliceStartStop*>(head.get())) {
//     if (isscalar()) {
//       throw std::invalid_argument("too many dimensions in index for this array");
//     }
//     std::shared_ptr<SliceItem> nexthead = tail.head();
//     Slice nexttail = tail.tail();
//     // Index64 nextcarry(carry.length()*(h->stop() - h->start()));
//     // int64_t k = 0;
//     // for (int64_t i = 0;  i < carry.length();  i++) {
//     //   for (int64_t j = 0;  j < h->stop() - h->start();  j++) {
//     //     nextcarry.ptr().get()[k] = carry.ptr().get()[i] + h->start() + j;
//     //     k++;
//     //   }
//     // }
//
//     // int64_t step = shape_[0];
//     Index64 nextcarry(carry.length());   // *(h->stop() - h->start())
//     // int64_t k = 0;
//     for (int64_t i = 0;  i < carry.length();  i++) {
//       // for (int64_t j = h->start();  j < h->stop();  j++) {
//       //   nextcarry.ptr().get()[k] = carry.ptr().get()[i] + j*step;
//       //   k++;
//       // }
//       nextcarry.ptr().get()[i] = carry.ptr().get()[i] / 5;
//     }
//     std::cout << "carry " << carry.length() << std::endl;
//     for (int i = 0;  i < carry.length();  i++) {
//       std::cout << "carry[" << i << "] " << carry.ptr().get()[i] << std::endl;
//     }
//     std::cout << "nextcarry " << nextcarry.length() << std::endl;
//     for (int i = 0;  i < nextcarry.length();  i++) {
//       std::cout << "nextcarry[" << i << "] " << nextcarry.ptr().get()[i] << std::endl;
//     }
//
//     std::vector<ssize_t> shape({ (ssize_t)(h->stop() - h->start()) });
//     shape.insert(shape.end(), shape_.begin() + 1, shape_.end());
//
//     std::cout << "shape " << shape.size() << std::endl;
//     for (int i = 0;  i < shape.size();  i++) {
//       std::cout << "shape[" << i << "] " << shape[i] << std::endl;
//     }
//     std::vector<ssize_t> strides(shape2strides(shape, itemsize_));
//     std::cout << "strides " << strides.size() << std::endl;
//     for (int i = 0;  i < strides.size();  i++) {
//       std::cout << "strides[" << i << "] " << strides[i] << std::endl;
//     }
//
//     std::shared_ptr<Content> next(new NumpyArray(Identity::none(), ptr_, shape, shape2strides(shape, itemsize_), byteoffset_, itemsize_, format_));
//     return next.get()->getitem_next(nexthead, nexttail, nextcarry);
//   }
//
//   else {
//     assert(false);
//   }
// }




//   else if (SliceStartStopStep* x = dynamic_cast<SliceStartStopStep*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceByteMask* x = dynamic_cast<SliceByteMask*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceIndex32* x = dynamic_cast<SliceIndex32*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceIndex64* x = dynamic_cast<SliceIndex64*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceEllipsis* x = dynamic_cast<SliceEllipsis*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else if (SliceNewAxis* x = dynamic_cast<SliceNewAxis*>(head.get())) {
//     throw std::runtime_error("not implemented");
//   }
//   else {
//     assert(false);
//   }
// }
