#pragma once

#include "domain/ppo/gaussian_policy.h"

namespace orbital::rl::policy {

using GaussianPolicyDistribution = nmc::domain::ppo::GaussianPolicyDistribution;
using GaussianPolicyImpl = nmc::domain::ppo::GaussianPolicyImpl;
using GaussianPolicy = nmc::domain::ppo::GaussianPolicy;

}  // namespace orbital::rl::policy
