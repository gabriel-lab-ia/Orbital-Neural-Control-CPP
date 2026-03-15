##Neuro Motor CPP

`Neuro Motor CPP` is a C++20 reinforcement learning project that combines:

- `LibTorch` for neural policy and value networks
- `PPO` for on-policy optimization
- `MuJoCo` for continuous-control simulation
- `HTML` and `JSON` exports for interactive neural visualization

- <svg width="1600" height="520" viewBox="0 0 1600 520" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">
  <title id="title">Neuro Motor CPP repository banner</title>
  <desc id="desc">Banner for a C plus plus PPO reinforcement learning project with MuJoCo and a 3D neural viewer.</desc>
  <defs>
    <linearGradient id="bg" x1="80" y1="40" x2="1500" y2="500" gradientUnits="userSpaceOnUse">
      <stop stop-color="#071826"/>
      <stop offset="0.48" stop-color="#0F2740"/>
      <stop offset="1" stop-color="#09131F"/>
    </linearGradient>
    <linearGradient id="beam" x1="170" y1="148" x2="1435" y2="148" gradientUnits="userSpaceOnUse">
      <stop stop-color="#5FD4FF"/>
      <stop offset="0.55" stop-color="#73F2C2"/>
      <stop offset="1" stop-color="#F0C15B"/>
    </linearGradient>
    <linearGradient id="panel" x1="1100" y1="116" x2="1450" y2="410" gradientUnits="userSpaceOnUse">
      <stop stop-color="#11273B"/>
      <stop offset="1" stop-color="#09131E"/>
    </linearGradient>
    <filter id="glow" x="0" y="0" width="1600" height="520" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
      <feGaussianBlur stdDeviation="22"/>
    </filter>
  </defs>

  <rect width="1600" height="520" rx="28" fill="url(#bg)"/>
  <circle cx="219" cy="116" r="124" fill="#5FD4FF" opacity="0.10"/>
  <circle cx="1340" cy="412" r="160" fill="#73F2C2" opacity="0.10"/>
  <circle cx="1186" cy="92" r="116" fill="#F0C15B" opacity="0.08"/>

  <g opacity="0.40" filter="url(#glow)">
    <path d="M152 334C258 255 356 229 440 238C573 251 609 347 726 354C844 361 907 259 1027 232C1148 205 1266 244 1448 332" stroke="url(#beam)" stroke-width="7" stroke-linecap="round"/>
  </g>

  <rect x="1036" y="88" width="382" height="328" rx="28" fill="url(#panel)" stroke="rgba(255,255,255,0.10)"/>
  <rect x="1080" y="130" width="298" height="172" rx="18" fill="#08131D" stroke="rgba(255,255,255,0.08)"/>
  <path d="M1108 279C1148 225 1180 201 1216 201C1268 201 1287 262 1333 262C1364 262 1393 236 1418 207" stroke="#73F2C2" stroke-width="4" stroke-linecap="round"/>
  <path d="M1108 239C1157 181 1217 151 1280 151C1328 151 1369 167 1418 197" stroke="#5FD4FF" stroke-width="4" stroke-linecap="round"/>
  <circle cx="1216" cy="201" r="10" fill="#5FD4FF"/>
  <circle cx="1333" cy="262" r="10" fill="#73F2C2"/>
  <circle cx="1280" cy="151" r="10" fill="#F0C15B"/>
  <text x="1080" y="340" fill="#EAF4FF" font-family="DejaVu Sans, Arial, sans-serif" font-size="24" font-weight="700">Policy inspection viewer</text>
  <text x="1080" y="372" fill="#A9C4D8" font-family="DejaVu Sans, Arial, sans-serif" font-size="18">3D neural graph • live activations • static web export</text>

  <text x="88" y="132" fill="#7EE7C0" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="700" letter-spacing="2">C++20 • PPO • LIBTORCH • MUJOCO</text>
  <text x="88" y="214" fill="#F6FBFF" font-family="DejaVu Sans, Arial, sans-serif" font-size="56" font-weight="700">Neuro Motor CPP</text>
  <text x="88" y="268" fill="#D5E6F2" font-family="DejaVu Sans, Arial, sans-serif" font-size="28">High-performance reinforcement learning and neural policy visualization</text>
  <text x="88" y="322" fill="#A8C3D8" font-family="DejaVu Sans, Arial, sans-serif" font-size="24">Modern C++ PPO baseline with MuJoCo-ready control tasks, metric exports,</text>
  <text x="88" y="356" fill="#A8C3D8" font-family="DejaVu Sans, Arial, sans-serif" font-size="24">and a browser-facing 3D viewer for policy structure and activations.</text>

  <g>
    <rect x="88" y="404" width="214" height="48" rx="24" fill="#11283D" stroke="rgba(255,255,255,0.10)"/>
    <rect x="318" y="404" width="214" height="48" rx="24" fill="#11283D" stroke="rgba(255,255,255,0.10)"/>
    <rect x="548" y="404" width="236" height="48" rx="24" fill="#11283D" stroke="rgba(255,255,255,0.10)"/>
    <text x="130" y="435" fill="#E7F2FF" font-family="DejaVu Sans, Arial, sans-serif" font-size="20" font-weight="700">continuous control</text>
    <text x="372" y="435" fill="#E7F2FF" font-family="DejaVu Sans, Arial, sans-serif" font-size="20" font-weight="700">policy export</text>
    <text x="588" y="435" fill="#E7F2FF" font-family="DejaVu Sans, Arial, sans-serif" font-size="20" font-weight="700">public neural demo</text>
  </g>
</svg>


The repository provides a formal PPO baseline in C++, supports a MuJoCo cart-pole environment, exports learning metrics and benchmarks, and generates a browser-based 3D viewer that renders the trained policy network and its live activations.

Live demo: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/`

Direct 3D viewer: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/neural_network_3d.html`

Highlights

- PPO implementation in modern C++ with `LibTorch`
- Optional MuJoCo integration through a clean `Environment` interface
- Critic stabilization with value clipping and robust value loss
- Low-latency lightweight policy with benchmark export
- CSV metrics and SVG learning curves
- Live rollout capture from the trained policy
- Standalone 3D HTML viewer for policy structure and activations
- Touch-friendly mobile interaction in the public 3D viewer
- Static web publishing path through `docs/` for GitHub Pages

Repository Layout

- `src/app/` application entrypoints and orchestration
- `src/env/` environment interface and implementations
- `src/model/` PPO policy and value network
- `src/train/` rollout collection, GAE, and PPO updates
- `src/utils/` logging and export utilities
- `assets/mujoco/` project MuJoCo XML assets
- `tools/` local scripts for plotting, viewing, setup, and publishing
- `docs/` static web site for GitHub Pages
- `notebooks/` analysis notebooks

Requirements

- GCC 13+
- CMake 3.24+
- LibTorch 2.2.2
- Eigen
- MuJoCo 3.2.6 or newer for MuJoCo environments

Quick Start

This repository does not need to commit LibTorch binaries. If `lib/libtorch/` is missing, install the CPU package locally with:

```bash
bash tools/setup_libtorch_cpu.sh
```

Configure:

```bash
cmake --preset dev
```

Build:

```bash
cmake --build --preset build
```

Run the default PPO baseline:

```bash
./build/motor
```

Generate the learning-curve SVG:

```bash
python3 tools/plot_learning_curve.py artifacts/learning_curve.csv artifacts/learning_curve.svg
```

MuJoCo Training

Build with MuJoCo support:

```bash
cmake --preset dev -DNMC_ENABLE_MUJOCO=ON -DNMC_MUJOCO_ROOT=$HOME/.local/mujoco-3.2.6
cmake --build --preset build
```

Train the PPO agent in MuJoCo:

```bash
NMC_ENV=mujoco_cartpole ./build/motor
```

Run a live policy rollout after training:

```bash
NMC_ENV=mujoco_cartpole NMC_LIVE_POLICY=1 NMC_LIVE_STEPS=64 ./build/motor
```

Visualization

Open the generated 3D network viewer:

```bash
xdg-open artifacts/neural_network_3d.html
```

Open the MuJoCo viewer for the project cart-pole:

```bash
./tools/view_mujoco.sh
```

Public links:

- Project site: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/`
- 3D neural viewer: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/neural_network_3d.html`
- Learning curve: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/learning_curve.svg`
- Benchmark summary: `https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/benchmark_summary.json`

Web Publishing

This repository is prepared for static deployment through GitHub Pages.

Sync the current demo into `docs/`:

```bash
python3 tools/publish_demo.py
```

The committed web assets are expected under `docs/demo/`.

After pushing the repository to GitHub and enabling Pages, the generated site exposes:

- `/` landing page
- `/demo/neural_network_3d.html` direct 3D neural viewer
- `/demo/learning_curve.svg` exported learning curve
- `/demo/benchmark_summary.json` benchmark and efficiency snapshot

- https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/
- https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/neural_network_3d.html
- https://gabriel-lab-ia.github.io/PPO_Neural-Control-cpp/demo/benchmark_summary.json

Generated Outputs

- `artifacts/learning_curve.csv`
- `artifacts/learning_curve.svg`
- `artifacts/live_rollout.csv`
- `artifacts/benchmark_summary.json`
- `artifacts/neural_network_3d.html`
- `artifacts/neural_network_3d.json`

Project Status

This is a serious research-engineering foundation, not a general-purpose RL framework yet. The current codebase is strongest as:

- a C++ PPO reference implementation
- a MuJoCo-ready training baseline
- a neural-visualization demo for policy inspection

Development Notes

- The default CI build targets the non-MuJoCo baseline so public builds stay lightweight.
- MuJoCo support is optional and activated through `NMC_ENABLE_MUJOCO`.
- The static site is generated from local artifacts; `docs/` is the publishable output.

License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
