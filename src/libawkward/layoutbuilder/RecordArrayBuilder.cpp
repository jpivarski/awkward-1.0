// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/RecordArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/RecordArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  RecordArrayBuilder::RecordArrayBuilder(const std::vector<FormBuilderPtr>& contents,
                                         const util::RecordLookupPtr recordlookup,
                                         const util::Parameters& parameters,
                                         const std::string& form_key,
                                         const std::string attribute,
                                         const std::string partition)
    : form_recordlookup_(recordlookup),
      parameters_(parameters),
      field_index_(0),
      contents_size_((int64_t) contents.size()) {
    for (auto const& content : contents) {
      contents_.push_back(content);
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_func_.append(contents_.back().get()->vm_func());
      vm_error_.append(contents_.back().get()->vm_error());
    }

    vm_func_name_ = std::string(form_key).append(attribute);

    vm_func_.append(": ")
      .append(vm_func_name_);

    for (auto const& content : contents_) {
      vm_func_.append("\n    ").append(content.get()->vm_func_name())
        .append(" pause");
    }
    // Remove the last pause
    vm_func_.erase(vm_func_.end() - 6, vm_func_.end());
    vm_func_.append("\n;\n\n");
  }

  const std::string
  RecordArrayBuilder::classname() const {
    return "RecordArrayBuilder";
  }

  const std::string
  RecordArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  RecordArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  RecordArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  RecordArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  RecordArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  RecordArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  RecordArrayBuilder::vm_error() const {
    return vm_error_;
  }

  int64_t
  RecordArrayBuilder::field_index() {
    return (field_index_ < contents_size_ - 1) ?
      field_index_++ : (field_index_ = 0);
  }

  void
  RecordArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->boolean(x, builder);
  }

  void
  RecordArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->int64(x, builder);
  }

  void
  RecordArrayBuilder::float64(double x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->float64(x, builder);
  }

  void
  RecordArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->complex(x, builder);
  }

  void
  RecordArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->bytestring(x, builder);
  }

  void
  RecordArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    contents_[(size_t)field_index()].get()->string(x, builder);
  }

  void
  RecordArrayBuilder::begin_list(LayoutBuilder* builder) {
    list_field_index_.emplace_back(field_index_);
    contents_[(size_t)field_index_].get()->begin_list(builder);
  }

  void
  RecordArrayBuilder::end_list(LayoutBuilder* builder) {
    field_index_ = list_field_index_.back();
    contents_[(size_t)field_index_].get()->end_list(builder);
    list_field_index_.pop_back();
    field_index();
  }

  bool
  RecordArrayBuilder::active() {
    if (!list_field_index_.empty()) {
      return contents_[(size_t)list_field_index_.back()].get()->active();
    }
    else {
      for(auto content : contents_) {
        if (content.get()->active()) {
          return true;
        }
      }
    }
    return false;
  }

}
