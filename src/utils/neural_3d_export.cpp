#include "utils/neural_3d_export.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nmc {
namespace {

void write_float_vector(std::ostream& stream, const std::vector<float>& values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << values[index];
    }
    stream << ']';
}

void write_tensor_flat(std::ostream& stream, const torch::Tensor& tensor) {
    const auto flat = tensor.contiguous().view({-1}).to(torch::kCPU);
    stream << '[';
    for (int64_t index = 0; index < flat.size(0); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << flat[index].item<float>();
    }
    stream << ']';
}

std::string build_snapshot_json(
    const std::string& environment_name,
    PPOAgent& agent,
    const std::vector<PPOTrainer::LiveStep>& live_steps
) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6);

    const auto layer_names = agent->visualization_layer_names();
    const auto layer_sizes = agent->visualization_layer_sizes();
    const auto weights = agent->visualization_weights();

    stream << "{\n";
    stream << "\"meta\":{";
    stream << "\"environment\":\"" << environment_name << "\",";
    stream << "\"policy_std\":" << agent->policy_std_scalar() << ',';
    stream << "\"frame_count\":" << live_steps.size();
    stream << "},\n";

    stream << "\"layers\":[";
    for (std::size_t index = 0; index < layer_names.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << "{\"name\":\"" << layer_names[index] << "\",\"size\":" << layer_sizes[index] << '}';
    }
    stream << "],\n";

    stream << "\"connections\":[";
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        const auto tensor = weights[index];
        stream << '{';
        stream << "\"from_layer\":" << index << ',';
        stream << "\"to_layer\":" << (index + 1) << ',';
        stream << "\"rows\":" << tensor.size(0) << ',';
        stream << "\"cols\":" << tensor.size(1) << ',';
        stream << "\"weights\":";
        write_tensor_flat(stream, tensor);
        stream << '}';
    }
    stream << "],\n";

    stream << "\"frames\":[";
    for (std::size_t step_index = 0; step_index < live_steps.size(); ++step_index) {
        if (step_index > 0) {
            stream << ',';
        }

        const auto& step = live_steps[step_index];
        const auto observation = torch::tensor(step.observation, torch::TensorOptions().dtype(torch::kFloat32));
        const auto activations = agent->visualization_activations(observation);

        stream << '{';
        stream << "\"step\":" << step.step << ',';
        stream << "\"reward\":" << step.reward << ',';
        stream << "\"action\":" << step.action << ',';
        stream << "\"value\":" << step.value << ',';
        stream << "\"terminated\":" << (step.terminated ? "true" : "false") << ',';
        stream << "\"truncated\":" << (step.truncated ? "true" : "false") << ',';
        stream << "\"observation\":";
        write_float_vector(stream, step.observation);
        stream << ',';
        stream << "\"activations\":[";
        for (std::size_t activation_index = 0; activation_index < activations.size(); ++activation_index) {
            if (activation_index > 0) {
                stream << ',';
            }
            write_tensor_flat(stream, activations[activation_index]);
        }
        stream << ']';
        stream << '}';
    }
    stream << "]\n";
    stream << "}\n";

    return stream.str();
}

std::string build_viewer_html(const std::string& json_payload) {
    std::ostringstream html;
    html << R"(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Neuro Motor PPO 3D Network</title>
  <style>
    :root {
      --bg: #07111a;
      --panel: rgba(8, 18, 30, 0.78);
      --text: #e9f3ff;
      --muted: #8ea6bd;
      --accent: rgba(255, 255, 255, 0.5);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      overflow: hidden;
      font-family: "DejaVu Sans", system-ui, sans-serif;
      background:
        radial-gradient(circle at top, rgba(46, 112, 173, 0.22), transparent 32%),
        radial-gradient(circle at bottom right, rgba(120, 255, 176, 0.15), transparent 28%),
        linear-gradient(180deg, #040912 0%, #07111a 100%);
      color: var(--text);
    }
    canvas { display: block; width: 100vw; height: 100vh; }
    .hud {
      position: fixed;
      top: 18px;
      left: 18px;
      width: 360px;
      padding: 18px 18px 14px;
      background: var(--panel);
      border: 1px solid rgba(146, 189, 226, 0.18);
      border-radius: 18px;
      backdrop-filter: blur(16px);
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
    }
    .hud h1 { margin: 0 0 8px; font-size: 20px; letter-spacing: 0.02em; }
    .hud p { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .controls { display: flex; gap: 8px; margin-bottom: 12px; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
    }
    button:hover { background: rgba(255, 255, 255, 0.14); }
    input[type="range"] { width: 100%; }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 14px;
    }
    .stat { padding: 10px 12px; border-radius: 12px; background: rgba(255, 255, 255, 0.04); }
    .stat .label {
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat .value { display: block; margin-top: 4px; font-size: 16px; }
    .legend {
      display: flex;
      justify-content: space-between;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .signature {
      position: fixed;
      right: 18px;
      bottom: 12px;
      color: var(--accent);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <div class="hud">
    <h1>3D Neural Environment</h1>
    <p>Spatial rendering of the trained PPO network with animated neurons, synapses, and activations sampled from a live rollout.</p>
    <div class="controls">
      <button id="toggle">Pause</button>
      <button id="stepBack">Prev</button>
      <button id="stepForward">Next</button>
    </div>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <div class="stats">
      <div class="stat"><span class="label">Environment</span><span class="value" id="envValue">-</span></div>
      <div class="stat"><span class="label">Frame</span><span class="value" id="frameValue">-</span></div>
      <div class="stat"><span class="label">Reward</span><span class="value" id="rewardValue">-</span></div>
      <div class="stat"><span class="label">Action</span><span class="value" id="actionValue">-</span></div>
    </div>
    <div class="legend">
      <span>blue = lower activation</span>
      <span>orange = higher activation</span>
    </div>
  </div>
  <div class="signature">feito por Gabriel Arantes</div>
  <script id="snapshot-data" type="application/json">)";
    html << json_payload;
    html << R"(</script>
  <script>
    const snapshot = JSON.parse(document.getElementById('snapshot-data').textContent);
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const scrub = document.getElementById('scrub');
    const envValue = document.getElementById('envValue');
    const frameValue = document.getElementById('frameValue');
    const rewardValue = document.getElementById('rewardValue');
    const actionValue = document.getElementById('actionValue');
    const toggleButton = document.getElementById('toggle');

    const state = { frame: 0, playing: true, yaw: 0.7, pitch: -0.25, mouseDown: false, lastX: 0, lastY: 0 };
    const layerSpacing = 280;
    const nodes = [];

    function resize() {
      canvas.width = window.innerWidth * window.devicePixelRatio;
      canvas.height = window.innerHeight * window.devicePixelRatio;
      canvas.style.width = window.innerWidth + 'px';
      canvas.style.height = window.innerHeight + 'px';
      ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function activationColor(value) {
      const normalized = clamp((value + 1) / 2, 0, 1);
      const r = Math.round(78 + normalized * 177);
      const g = Math.round(120 + normalized * 66);
      const b = Math.round(255 - normalized * 140);
      return `rgb(${r}, ${g}, ${b})`;
    }

    function weightColor(weight) {
      const alpha = Math.min(0.45, 0.06 + Math.abs(weight) * 0.18);
      return weight >= 0 ? `rgba(141,255,159,${alpha})` : `rgba(78,183,255,${alpha})`;
    }

    function buildNodes() {
      snapshot.layers.forEach((layer, layerIndex) => {
        const columns = Math.ceil(Math.sqrt(layer.size));
        const rows = Math.ceil(layer.size / columns);
        const x = (layerIndex - (snapshot.layers.length - 1) / 2) * layerSpacing;
        for (let index = 0; index < layer.size; index++) {
          const row = Math.floor(index / columns);
          const column = index % columns;
          const y = (row - (rows - 1) / 2) * 26;
          const z = (column - (columns - 1) / 2) * 26;
          nodes.push({ layerIndex, nodeIndex: index, label: layer.name, position: { x, y, z } });
        }
      });
    }

    function rotated(point) {
      const cosY = Math.cos(state.yaw);
      const sinY = Math.sin(state.yaw);
      const cosX = Math.cos(state.pitch);
      const sinX = Math.sin(state.pitch);
      const xzX = point.x * cosY - point.z * sinY;
      const xzZ = point.x * sinY + point.z * cosY;
      const yzY = point.y * cosX - xzZ * sinX;
      const yzZ = point.y * sinX + xzZ * cosX + 700;
      return { x: xzX, y: yzY, z: yzZ };
    }

    function project(point) {
      const perspective = 780 / Math.max(180, point.z);
      return {
        x: window.innerWidth / 2 + point.x * perspective,
        y: window.innerHeight / 2 + point.y * perspective,
        scale: perspective,
        depth: point.z,
      };
    }

    function buildSparseEdges() {
      const sparse = [];
      snapshot.connections.forEach((connection) => {
        const perTarget = connection.rows > 64 ? 8 : 12;
        for (let target = 0; target < connection.rows; target++) {
          const local = [];
          for (let source = 0; source < connection.cols; source++) {
            const weight = connection.weights[target * connection.cols + source];
            local.push({ source, target, weight });
          }
          local.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
          sparse.push(...local.slice(0, perTarget).map((edge) => ({
            fromLayer: connection.from_layer,
            toLayer: connection.to_layer,
            source: edge.source,
            target: edge.target,
            weight: edge.weight,
          })));
        }
      });
      return sparse;
    }

    const sparseEdges = buildSparseEdges();

    function draw(frame) {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      ctx.fillStyle = 'rgba(6, 11, 18, 0.22)';
      ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

      const projectedNodes = nodes.map((node) => {
        const rotatedPoint = rotated(node.position);
        return { ...node, projected: project(rotatedPoint) };
      });

      sparseEdges.forEach((edge) => {
        const from = projectedNodes.find((node) => node.layerIndex === edge.fromLayer && node.nodeIndex === edge.source);
        const to = projectedNodes.find((node) => node.layerIndex === edge.toLayer && node.nodeIndex === edge.target);
        if (!from || !to) {
          return;
        }

        const activationA = frame.activations[edge.fromLayer][edge.source] || 0;
        const activationB = frame.activations[edge.toLayer][edge.target] || 0;
        const intensity = Math.min(1, Math.abs(activationA) * 0.7 + Math.abs(activationB) * 0.7);
        ctx.strokeStyle = weightColor(edge.weight);
        ctx.lineWidth = 0.5 + intensity * 1.6;
        ctx.beginPath();
        ctx.moveTo(from.projected.x, from.projected.y);
        ctx.lineTo(to.projected.x, to.projected.y);
        ctx.stroke();
      });

      projectedNodes
        .sort((a, b) => b.projected.depth - a.projected.depth)
        .forEach((node) => {
          const activation = frame.activations[node.layerIndex][node.nodeIndex] || 0;
          const radius = Math.max(2.0, 3.8 * node.projected.scale + Math.abs(activation) * 3.4);
          ctx.beginPath();
          ctx.fillStyle = activationColor(activation);
          ctx.shadowBlur = 18;
          ctx.shadowColor = ctx.fillStyle;
          ctx.arc(node.projected.x, node.projected.y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        });

      ctx.fillStyle = 'rgba(233,243,255,0.78)';
      ctx.font = '12px DejaVu Sans, sans-serif';
      snapshot.layers.forEach((layer, index) => {
        const anchor = projectedNodes.find((node) => node.layerIndex === index && node.nodeIndex === 0);
        if (anchor) {
          ctx.fillText(layer.name, anchor.projected.x - 22, anchor.projected.y - 20);
        }
      });
    }

    function syncHud(frame) {
      envValue.textContent = snapshot.meta.environment;
      frameValue.textContent = `${state.frame + 1} / ${snapshot.frames.length}`;
      rewardValue.textContent = frame.reward.toFixed(4);
      actionValue.textContent = frame.action.toFixed(4);
      scrub.value = String(state.frame);
    }

    function render() {
      if (state.playing && snapshot.frames.length > 0) {
        state.frame = (state.frame + 1) % snapshot.frames.length;
      }
      const frame = snapshot.frames[state.frame];
      if (frame) {
        draw(frame);
        syncHud(frame);
      }
      state.yaw += state.mouseDown ? 0.0 : 0.0028;
      requestAnimationFrame(render);
    }

    toggleButton.addEventListener('click', () => {
      state.playing = !state.playing;
      toggleButton.textContent = state.playing ? 'Pause' : 'Play';
    });
    document.getElementById('stepBack').addEventListener('click', () => {
      state.playing = false;
      toggleButton.textContent = 'Play';
      state.frame = (state.frame - 1 + snapshot.frames.length) % snapshot.frames.length;
      draw(snapshot.frames[state.frame]);
      syncHud(snapshot.frames[state.frame]);
    });
    document.getElementById('stepForward').addEventListener('click', () => {
      state.playing = false;
      toggleButton.textContent = 'Play';
      state.frame = (state.frame + 1) % snapshot.frames.length;
      draw(snapshot.frames[state.frame]);
      syncHud(snapshot.frames[state.frame]);
    });
    scrub.addEventListener('input', (event) => {
      state.playing = false;
      toggleButton.textContent = 'Play';
      state.frame = Number(event.target.value);
      draw(snapshot.frames[state.frame]);
      syncHud(snapshot.frames[state.frame]);
    });

    canvas.addEventListener('mousedown', (event) => {
      state.mouseDown = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
    });
    window.addEventListener('mouseup', () => { state.mouseDown = false; });
    window.addEventListener('mousemove', (event) => {
      if (!state.mouseDown) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.yaw += dx * 0.005;
      state.pitch = clamp(state.pitch + dy * 0.004, -1.1, 1.1);
      state.lastX = event.clientX;
      state.lastY = event.clientY;
    });

    resize();
    buildNodes();
    scrub.max = String(Math.max(snapshot.frames.length - 1, 0));
    envValue.textContent = snapshot.meta.environment;
    window.addEventListener('resize', resize);
    if (snapshot.frames.length > 0) {
      draw(snapshot.frames[0]);
      syncHud(snapshot.frames[0]);
    }
    requestAnimationFrame(render);
  </script>
</body>
</html>
)";
    return html.str();
}

}  // namespace

void write_neural_3d_visualization(
    const std::filesystem::path& html_path,
    const std::filesystem::path& json_path,
    const std::string& environment_name,
    PPOAgent& agent,
    const std::vector<PPOTrainer::LiveStep>& live_steps
) {
    std::filesystem::create_directories(html_path.parent_path());
    std::filesystem::create_directories(json_path.parent_path());

    const auto json_payload = build_snapshot_json(environment_name, agent, live_steps);
    const auto html_payload = build_viewer_html(json_payload);

    std::ofstream json_stream(json_path, std::ios::out | std::ios::trunc);
    if (!json_stream.is_open()) {
        throw std::runtime_error("unable to open 3d network json file: " + json_path.string());
    }
    json_stream << json_payload;

    std::ofstream html_stream(html_path, std::ios::out | std::ios::trunc);
    if (!html_stream.is_open()) {
        throw std::runtime_error("unable to open 3d network html file: " + html_path.string());
    }
    html_stream << html_payload;
}

}  // namespace nmc
