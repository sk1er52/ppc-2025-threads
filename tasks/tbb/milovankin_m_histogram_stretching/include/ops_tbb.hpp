#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_histogram_stretching_tbb {

class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> img_;
};

}  // namespace milovankin_m_histogram_stretching_tbb
