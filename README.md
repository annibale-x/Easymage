
## 🚀 Easymage  v0.6.3: Generative Imaging & Prompt Engineering Filter

Professional-grade Open WebUI filter designed to streamline the image generation workflow and automate post-generation technical analysis. By simply prepending the `img ` keyword to any message, you trigger an advanced end-to-end pipeline.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/annibale-x/Easymage)
![Open WebUI Plugin](https://img.shields.io/badge/Open%20WebUI-Plugin-blue?style=flat&logo=openai)
![License](https://img.shields.io/github/license/annibale-x/Easymage?color=green)
<!-- ![Stars](https://img.shields.io/github/stars/annibale-x/Easymage?style=social) -->

---

### 💡Usage

* **`img [prompt]`** **Full Execution Mode**: Triggers the comprehensive end-to-end workflow. This includes multilingual detection, AI-driven prompt expansion, image generation via the active engine, and a final technical audit (Vision QC) to evaluate fidelity and execution.

---

### ✨ Key Features

* **Dynamic Engine Detection**: Automatically identifies and interfaces with the active backend (A1111, ComfyUI, OpenAI/Flux).

* **Prompt Engineering**: Expands simple ideas into high-fidelity technical prompts (lighting, camera settings, textures, and artistic styles).

* **Vision Quality Audit (QC)**: Real-time technical critique of the generated image using Vision LLMs, providing a numerical score (0-100%).

* **Multilingual Intelligence**: Detects input language and optimizes the translation for the generation engine.
* **Performance Analytics**: Tracks generation latency, total execution time, and precise throughput (Token/s).

---

### ℹ️ Integration Architecture

Easymage operates as an abstraction layer within the Open WebUI ecosystem. Unlike standard scripts, this filter **does not establish direct connections** to external backends.

Instead, it orchestrates the workflow by interfacing with the native Open WebUI `image_generations` internal API. This ensures:
* **Security**: Follows existing authentication and protocols configured in your OWUI instance.
* **Compatibility**: Leverages global image configurations (`config.py`) and official engine integrations.
* **Consistency**: Parameters are handled through the platform's official routing logic, maintaining a single source of truth.

---

### ⚠️ Developer Notes & Roadmap

* **Intensive Development Phase**: This filter is currently under heavy development. **Logic, valves, and core features are subject to rapid changes** until the stable **v1.x** release.
* **Primary Engine Testing**: Current stability tests are primarily focused on the **Automatic1111** backend.
* **Unified Parameter Subset**: I am developing a standardized set of "backend-agnostic" parameters to bridge the gap in Open WebUI’s interface, enabling advanced configurations not yet available via the standard UI.
* **ComfyUI Dynamic Discovery**: A workflow discovery system for ComfyUI is in progress. The goal is to automatically detect and map workflow structures, removing the need to manually modify JSON files when the graph changes.

---

### 📖 The Philosophy

Easymage was born out of a desire to simplify image generation for "lazy" power users (like myself). I was tired of constantly toggling tools and wanted a seamless experience directly within the chat prompt. 

While the filter is powerful, it is important to remember that since it relies on the OWUI API rather than direct backend hooks, ultra-complex configurations must still be managed at the backend level (Comfy, A1111, etc.).

---

### 📌 Output & Citations

Technical data is organized into three distinct reference tiers displayed under the generated image:
1.  `[🚀 PROMPT]`: The final, high-fidelity processed prompt.
2.  `[🟢 SCORE: xx%]`: The technical critique and vision-based scoring.
3.  `[🔍 DETAILS]`: A summary of engine parameters and performance metrics.

---

### 📊 Quality Indicators (Vision Audit)

The audit system uses color-coded scoring to evaluate adherence and quality:
* 🟢 **80-100%**: Exceptional quality; high prompt adherence.
* 🔵 **70-79%**: High quality; minor technical artifacts.
* 🟡 **60-69%**: Acceptable quality; lacks fine detail or nuance.
* 🟠 **40-59%**: Visible defects or low adherence to the enhanced prompt.
* 🔴 **< 40%**: Technical failure or complete misinterpretation.

---

### 🔧 Configuration Parameters (Valves) v0.6.3

The filter behavior is managed through "Valves" in the Open WebUI settings. These parameters allow you to override global defaults and control the generation pipeline.

| Valve | Default | Description |
| :--- | :---: | :--- |
| Enhanced Prompt | True | **Enrich prompt details**. When enabled, the filter expands the user's input into a professional, detailed prompt (lighting, camera, style) before generation. |
| Generation Quality Audit | True | **Post-generation Image Quality Audit**. Triggers a technical critique and scoring (0-100%) of the image using a Vision-capable model. |
| Generation Size Override |  | **Force image dimensions**. Allows you to specify a fixed resolution (e.g., `1024x1024`) overriding the system's global configuration. |
| Generation Model Override |  | **Force generation model**. Manually specifies which image generation checkpoint or model to use overriding the system’s global configuration. |
| Generation Steps Override |  | **Force step count**. Sets the number of sampling steps for the generation engine (if supported by the backend). |
| Persistent Vision Cache | False | **Optimized Capability Probing**. Saves the results of vision capability tests to `data/easymage_vision_cache.json` to avoid redundant probes in future sessions. |
| Debug | False | **Developer Logging**. Prints detailed execution logs, engine maps, and internal state dumps to the Docker/container console. |

---

### ⚙️ Technical Implementation Notes

* **Override Priority**: Valves have priority over the global Open WebUI `config.py` settings. If a Valve is set to `None` or left empty, the filter falls back to the system's default values.
* **Resource Optimization**: The `Persistent Vision Cache` is particularly useful in environments with many models, as it eliminates the initial image probe delay during the first generation of a session.
* **Clean UI**: When generation is active, the filter automatically suppresses standard LLM text output to keep the interface focused on the image and technical citations.

---

### 🚧 Upcoming Features (WIP)

1.  **Delimiter Parser (`:`)**: Use `img [styles] : [subject]` (e.g., `img cyberpunk, neon : a cat`) to prioritize stylistic directives.
2.  **Command Modifiers**: 
    * `+/-e`: Toggle Prompt Enhancement on the fly.
    * `+/-a`: Toggle Vision Audit on the fly.
    * `+N`: Batch generation (e.g., `img +4 a sunset`).
    * `WxH`: Resolution override (e.g., `img 1920x1080 a forest`).
3.  **Advanced `aud` Mode**: Perform a vision audit on manually uploaded images without generating a new one.
4.  **NSFW Toggle**: Safety valve for explicit content management and automated LoRA activation.

---

### 📜 Changelog

#### v0.6.3 (2026-01-27)
* **Modular Architecture**: Complete refactor to a method-based class structure for better maintainability.
* **Precise Metrics**: Replaced heuristic token estimation with exact data from Open WebUI API responses.
* **Vision Cache Persistence**: Introduced `persistent_vision_cache` to store model capabilities locally.
* **Audit Optimization**: Refined Vision Audit prompts for objective scoring and artifact detection.

#### v0.5.1 (2026-01-25)
* **Multimodal Validation**: Implemented base64 reference image probing to verify vision capabilities.
* **Debug Mode**: Added a toggle for detailed console logging.

#### v0.4.83 (2026-01-23)
* **Vision Audit System**: Initial integration of real-time image analysis.
* **Unicode UI**: Optimized technical details for environments with restricted Markdown.
