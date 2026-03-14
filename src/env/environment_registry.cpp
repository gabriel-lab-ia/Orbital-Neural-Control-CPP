#include "env/environment_registry.h"

#include <stdexcept>

#include "env/point_mass_env.h"

#if defined(NMC_ENABLE_MUJOCO)
#include "env/mujoco_cartpole_env.h"
#endif

namespace nmc {
namespace {

std::unique_ptr<Environment> make_single_environment(const EnvironmentSelection& selection) {
    if (selection.kind == "point_mass") {
        return std::make_unique<PointMassEnv>();
    }

#if defined(NMC_ENABLE_MUJOCO)
    if (selection.kind == "mujoco_cartpole") {
        return std::make_unique<MuJoCoCartPoleEnv>(selection.mujoco_model_path);
    }
#endif

    throw std::runtime_error(
        "unsupported environment kind: " + selection.kind
    );
}

std::string display_name_for(const EnvironmentSelection& selection) {
    if (selection.kind == "point_mass") {
        return "PointMassEnv";
    }
    if (selection.kind == "mujoco_cartpole") {
        return "MuJoCoCartPoleEnv";
    }
    return selection.kind;
}

}  // namespace

EnvironmentPack make_environment_pack(const EnvironmentSelection& selection, int64_t num_envs) {
    if (num_envs <= 0) {
        throw std::runtime_error("num_envs must be positive");
    }

    auto first_environment = make_single_environment(selection);
    EnvironmentPack pack;
    pack.display_name = display_name_for(selection);
    pack.observation_dim = first_environment->observation_dim();
    pack.action_dim = first_environment->action_dim();
    pack.environments.reserve(static_cast<std::size_t>(num_envs));
    pack.environments.push_back(std::move(first_environment));

    for (int64_t index = 1; index < num_envs; ++index) {
        pack.environments.push_back(make_single_environment(selection));
    }

    return pack;
}

bool mujoco_support_enabled() {
#if defined(NMC_ENABLE_MUJOCO)
    return true;
#else
    return false;
#endif
}

}  // namespace nmc
