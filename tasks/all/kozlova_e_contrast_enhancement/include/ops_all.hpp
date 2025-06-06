#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kozlova_e_contrast_enhancement_all {

class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_;
  size_t width_ = 0;
  size_t height_ = 0;
  std::vector<uint8_t> output_;
  boost::mpi::communicator world_;
};

}  // namespace kozlova_e_contrast_enhancement_all