# pragma once
# include <string>
# include <sycl/sycl.hpp>


struct selector_attribute
{
    std::string platform_name;
    std::string platform_version;
    std::string device_name;
    sycl::id<3> max_work_item_size;
    uint32_t max_compute_unit;
    uint32_t max_workgroup_size;
    auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
    auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
    auto d_name = device.get_info<sycl::info::device::name>();
    auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto maxWorkItemSize1 = device.get_info<sycl::info::device::max_work_item_sizes<1>>();
    auto maxWorkItemSize2 = device.get_info<sycl::info::device::max_work_item_sizes<2>>();
    auto maxWorkItemSize3 = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();  
};

void get_workspace(const int& num_workitem, int& output) {

   // Calculate the number of work groups
  auto localworkspace = (maxWorkItemSize1 + max_work_group_size - 1) / max_work_group_size;

  // Calculate the global size of the workspace
  auto globalworkspace = localworkspace * max_work_group_size;

}