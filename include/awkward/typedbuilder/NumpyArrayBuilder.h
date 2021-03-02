// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_NUMPYARRAYBUILDER_H_
#define AWKWARD_NUMPYARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class NumpyForm;
  using NumpyFormPtr = std::shared_ptr<NumpyForm>;

  /// @class NumpyArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL NumpyArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a NumpyArrayBuilder from a full set of parameters.
    NumpyArrayBuilder(const NumpyFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOutputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

    /// @brief
    const std::string
      vm_output() const override;

    /// @brief
    const std::string
      vm_func() const override;

    /// @brief
    const std::string
      vm_func_name() const override;

    /// @brief
    const std::string
      vm_from_stack() const override;

  private:
    const NumpyFormPtr form_;
    const FormKey form_key_;

    std::string vm_output_;
    std::string vm_output_data_;
    std::string vm_func_;
    std::string vm_func_name_;
    std::string vm_data_from_stack_;
  };

}

#endif // AWKWARD_NUMPYARRAYBUILDER_H_
