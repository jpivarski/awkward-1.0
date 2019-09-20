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

const std::shared_ptr<Content> NumpyArray::getitem(const Slice& where) const {
  assert(!isscalar());

  if (where.isadvanced()) {
    throw std::invalid_argument("FIXME");
  }

  else {
    std::vector<ssize_t> nextshape = { 1 };
    nextshape.insert(nextshape.end(), shape_.begin(), shape_.end());
    std::vector<ssize_t> nextstrides = { shape_[0]*strides_[0] };
    nextstrides.insert(nextstrides.end(), strides_.begin(), strides_.end());
    NumpyArray next(id_, ptr_, nextshape, nextstrides, byteoffset_, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = where.head();
    Slice nexttail = where.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, 1);

    std::vector<ssize_t> outshape(out.shape_.begin() + 1, out.shape_.end());
    std::vector<ssize_t> outstrides(out.strides_.begin() + 1, out.strides_.end());
    return std::shared_ptr<Content>(new NumpyArray(out.id_, out.ptr_, outshape, outstrides, out.byteoffset_, itemsize_, format_));
  }
}

const std::vector<ssize_t> flatten_shape(const std::vector<ssize_t> shape) {
  if (shape.size() == 1) {
    return std::vector<ssize_t>();
  }
  else {
    std::vector<ssize_t> out = { shape[0]*shape[1] };
    out.insert(out.end(), shape.begin() + 2, shape.end());
    return out;
  }
}

const std::vector<ssize_t> flatten_strides(const std::vector<ssize_t> strides) {
  if (strides.size() == 1) {
    return std::vector<ssize_t>();
  }
  else {
    return std::vector<ssize_t>(strides.begin() + 1, strides.end());
  }
}

void set_range(int64_t& start, int64_t& stop, bool posstep, bool hasstart, bool hasstop, int64_t length) {
  if (posstep) {
    if (!hasstart)          start = 0;
    else if (start < 0)     start += length;
    if (start < 0)          start = 0;
    if (start > length)     start = length;

    if (!hasstop)           stop = length;
    else if (stop < 0)      stop += length;
    if (stop < 0)           stop = 0;
    if (stop > length)      stop = length;
    if (stop < start)       stop = start;
  }

  else {
    if (!hasstart)          start = length - 1;
    else if (start < 0)     start += length;
    if (start < -1)         start = -1;
    if (start > length - 1) start = length - 1;

    if (!hasstop)           stop = -1;
    else if (stop < 0)      stop += length;
    if (stop < -1)          stop = -1;
    if (stop > length - 1)  stop = length - 1;
    if (stop > start)       stop = start;
  }
}

const NumpyArray NumpyArray::getitem_bystrides(const std::shared_ptr<SliceItem> head, const Slice& tail, int64_t length) const {
  if (head.get() == nullptr) {
    return NumpyArray(id_, ptr_, shape_, strides_, byteoffset_, itemsize_, format_);
  }

  if (ndim() < 2) {
    throw std::invalid_argument("too many indexes for array");
  }

  if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
    ssize_t nextbyteoffset = byteoffset_ + ((ssize_t)at->at())*strides_[1];
    NumpyArray next(id_, ptr_, flatten_shape(shape_), flatten_strides(strides_), nextbyteoffset, itemsize_, format_);

    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    NumpyArray out = next.getitem_bystrides(nexthead, nexttail, length);

    std::vector<ssize_t> outshape = { (ssize_t)length };
    outshape.insert(outshape.end(), out.shape_.begin() + 1, out.shape_.end());
    return NumpyArray(id_, ptr_, outshape, out.strides_, out.byteoffset_, itemsize_, format_);
  }

  else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
    int64_t start = range->start();
    int64_t stop = range->stop();
    int64_t step = range->step();
    set_range(start, stop, step > 0, range->hasstart(), range->hasstop(), (int64_t)shape_[1]);

    throw std::invalid_argument("HERE");
  }

  else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
    throw std::invalid_argument("getitem_bystrides ellipsis");
  }

  else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
    throw std::invalid_argument("getitem_bystrides newaxis");
  }

  else {
    throw std::runtime_error("unrecognized slice object");
  }
}
