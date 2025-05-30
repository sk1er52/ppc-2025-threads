#pragma once

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_cannon_omp {
class CannonOMP : public ppc::core::Task {
 public:
  explicit CannonOMP(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int N_;
  int block_size_;
  int num_blocks_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;

  void InitialShift();
  void BlockMultiply();
  void ShiftBlocks();
};
}  // namespace vavilov_v_cannon_omp
