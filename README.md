
# 🚀 Easymage v0.8.1: Easy Image Generator & Prompt Engineer

Easymage is a professional-grade filter for **Open WebUI** designed to transform your image generation workflow into a unified and intelligent experience. By simply prepending `img ` or `imgx ` to any message, you activate an advanced pipeline that handles everything from prompt engineering to multi-engine generation and post-creation technical analysis.

This filter acts as an intelligent router, unlocking advanced, engine-specific parameters like `seed`, `style`, `quality`, and `negative prompts` that are not available through the standard Open WebUI interface.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/annibale-x/Easymage)
![Open WebUI Filter](https://img.shields.io/badge/Open%20WebUI-Filter-blue?style=flat&logo=openai)
![License](https://img.shields.io/github/license/annibale-x/Easymage?color=green)

---

### ✨ Key Features

*   **Intelligent Multi-Engine Router**: Native support for `Forge (A1111)`, `OpenAI (DALL-E)`, and `Gemini (Imagen)`, with a standard fallback for `ComfyUI`. Easymage translates universal commands into the specific "dialect" of each API.
*   **Automated Prompt Engineering**: Expands simple ideas into high-fidelity technical prompts, enriching them with details on lighting, camera angles, textures, and artistic styles.
*   **Vision Quality Audit (QC)**: Provides a real-time technical critique of the generated image using Vision models, assigning a numerical score (0-100%) and an analysis of defects.
*   **Universal Syntax**: A single command set (`img ...`) allows you to control advanced parameters across different backends without needing to learn individual APIs.
*   **Performance Analytics**: Tracks generation latency, total execution time, and precise throughput (Tokens/s).

---

### 💡 Core Usage & Syntax

Easymage is activated with two main commands, followed by a flexible combination of parameters, styles, and prompts.

#### Base Commands
*   `img [prompt]`: **Full Generation Mode**. Triggers the entire pipeline: prompt enhancement, image generation, and quality audit.
*   `imgx [prompt]`: **Text-Only Mode**. Executes only the prompt enhancement and returns it as text, without generating an image. Useful for crafting prompts to use elsewhere.

#### Command Structure
The general format is `img [parameters] [flags] [styles] -- [subject] --no [negative prompt]`.

| Part | Example | Description |
| :--- | :--- | :--- |
| **Parameters** | `sz=1024x1024 stp=30` | `key=value` pairs to control the engine (size, steps, etc.). |
| **Flags** | `+h -a` | Toggles for features like **H**igh-Res (`+h`) or **A**udit (`-a`). |
| **Styles** | `cinematic, dark fantasy` | Stylistic keywords that are added to the subject prompt. |
| **Subject** | `-- a red car` | The main subject of your image. **The `-- ` separator is required if using styles or parameters.** |
| **Negative Prompt**| `--no blurry, ugly` | Elements to exclude from the image. The `--no ` is the separator. |

---

### ⚙️ Command Line Parameters

Use these `key=value` pairs to control the generation process.

| Parameter | Example | Description | Supported Engines |
| :--- | :--- | :--- | :--- |
| `ge` | `ge=o` | **G**eneration **E**ngine. Selects the backend. | All |
| `mdl` | `mdl=dall-e-3`| **M**o**d**e**l**. Specifies the model/checkpoint. | All |
| `sz` | `sz=1792x1024`| **S**i**z**e. Image dimensions in pixels. | All |
| `ar` | `ar=16:9` | **A**spect **R**atio. Sets the aspect ratio. | All |
| `stp` | `stp=25` | **St**e**p**s. Number of sampling steps. | A1111 |
| `sd` | `sd=12345` | **S**ee**d**. Value for reproducibility. | A1111, Gemini |
| `cs` | `cs=7.5` | **C**FG **S**cale. How strictly the prompt should be followed. | A1111 |
| `stl` | `stl=n` | **St**y**l**e. `v` (vivid) or `n` (natural). | OpenAI |
| `n` | `n=4` | **N**umber. How many images to generate. | A1111, Gemini |

#### Command Line Flags
Use `+` to enable and `-` to disable a feature for a single generation.

| Flag | Description |
| :--- | :--- |
| `+h` / `-h` | Toggles **H**igh-Res mode. For OpenAI, this maps to `quality: hd`. |
| `+a` / `-a` | Toggles the Vision Quality **A**udit. |
| `+p` / `-p` | Toggles **P**rompt Enhancement. |
| `+d` / `-d` | Toggles **D**ebug mode, printing state info in the chat. |

---

### 📚 Shortcut Tables

Easymage includes shortcuts for common settings to speed up your workflow.

<details>
<summary><strong>Engine Shortcuts (`ge=...`)</strong></summary>

| Shortcut | Engine |
| :--- | :--- |
| `a` | `automatic1111` |
| `o` | `openai` |
| `g` | `gemini` |
| `c` | `comfyui` |
</details>

<details>
<summary><strong>Aspect Ratio Shortcuts (`ar=...`)</strong></summary>

| Shortcut | Ratio |
| :--- | :--- |
| `1` | `1:1` |
| `16` | `16:9` |
| `9` | `9:16` |
| `4` | `4:3` |
| `3` | `3:4` |
| `21` | `21:9` |
</details>

<details>
<summary><strong>Full Sampler Shortcut List (`smp=...`)</strong></summary>

| Shortcut | Full Name |
| :--- | :--- |
| `d3s` | DPM++ 3M SDE |
| `d2sh` | DPM++ 2M SDE Heun |
| `d2s` | DPM++ 2M SDE |
| `d2m` | DPM++ 2M |
| `d2sa`| DPM++ 2S a |
| `ds` | DPM++ SDE |
| `ea` | Euler a |
| `e` | Euler |
| `l` | LMS |
| `h` | Heun |
| `d2` | DPM2 |
| `d2a` | DPM2 a |
| `df` | DPM fast |
| `dad` | DPM adaptive |
| `r` | Restart |
| `h2` | HeunPP2 |
| `ip` | IPNDM |
| `ipv` | IPNDM_V |
| `de` | DEIS |
| `u` | UniPC |
| `lcm` | LCM |
| `di` | DDIM |
| `dic` | DDIM CFG++ |
| `dp` | DDPM |
</details>

<details>
<summary><strong>Full Scheduler Shortcut List (`sch=...`)</strong></summary>

| Shortcut | Full Name |
| :--- | :--- |
| `a` | Automatic |
| `u` | Uniform |
| `k` | Karras |
| `e` | Exponential |
| `pe` | Polyexponential |
| `su` | SGM Uniform |
| `ko` | KL Optimal |
| `ays` | Align Your Steps |
| `aysg`| Align Your Steps GITS |
| `ays11`| Align Your Steps 11 |
| `ays32`| Align Your Steps 32 |
| `s` | Simple |
| `n` | Normal |
| `di` | DDIM |
| `b` | Beta |
| `t` | Turbo |
</details>

---

### 🪄 Practical Examples

#### Example 1: Simple & Fast
You just want an image of a cat.
```
img a cat
```
*Easymage will expand this into a detailed prompt before generating the image.*

#### Example 2: Style and Subject
You want a red car in a specific style, excluding blurriness.
```
img cinematic, cyberpunk -- a red car --no blurry
```
*The styles "cinematic" and "cyberpunk" are added to the subject "a red car", with a negative prompt.*

#### Example 3: Full Technical Control (Forge/A1111)
You want a fantasy portrait with specific technical parameters.
```
img stp=30 cs=8 ar=3:4 smp=d3s -- portrait of an elf --no ugly hands
```
*This sets 30 steps, CFG 8, a 3:4 aspect ratio, and uses the DPM++ 3M SDE sampler.*

#### Example 4: OpenAI Specifics (DALL-E 3)
You want a high-quality image with a "natural" style.
```
img +h stl=n ge=o -- a dog playing in a park
```
*`+h` activates `quality:hd`, `stl=n` sets `style:natural`, and `ge=o` ensures it runs on OpenAI.*

#### Example 5: Multiple Generations (Gemini)
You want 4 variations of a landscape with a specific seed.
```
img n=4 sd=12345 ge=g -- vast landscape of an alien planet
```
*`n=4` is translated to `sampleCount: 4` and the `seed` is passed to the Gemini API.*

#### Example 6: Text-Only Mode
You need a detailed prompt to use in another application.
```
imgx epic space battle
```
*Easymage returns only the enhanced text prompt, without generating any images.*

---

### 📐 Size & Aspect Ratio Logic

Easymage normalizes dimensions for each engine:
*   **If you use `ar` (e.g., `ar=16:9`):** Easymage calculates the closest supported pixel dimensions.
    *   **OpenAI**: Converts `16:9` to `1792x1024`.
    *   **Gemini**: Passes the `16:9` ratio string directly.
    *   **A1111/Forge**: Calculates pixel dimensions based on the default width (e.g., 1024), maintaining the ratio and rounding to a multiple of 8 (e.g., 1024x576).
*   **If you use `sz` (e.g., `sz=800x600`):**
    *   **OpenAI**: Snaps to the closest supported API dimension (e.g., `1024x1024`).
    *   **Gemini**: Calculates the approximate aspect ratio (e.g., `4:3`) and passes it to the API.
    *   **A1111/Forge**: Uses the exact pixel values.

---

### 🔧 Filter Configuration (Valves)

Control the default behavior of Easymage from your Open WebUI profile settings (`Settings > Filters > Easymage`).

| Valve | Default | Description |
| :--- | :---: | :--- |
| **Enhanced Prompt** | `True` | If enabled, the LLM expands your prompt with technical details by default. |
| **Quality Audit** | `True` | If enabled, a Vision model analyzes the final image and provides a quality score. |
| **Strict Audit** | `False` | Enables a "ruthless" audit mode that is much stricter in identifying technical flaws. |
| **Persistent Vision Cache**| `False` | Saves model vision capability test results to disk to avoid redundant probes. |
| **Debug** | `False` | Prints detailed execution logs and state dumps to the server/container console. |
| **Model** | `None` | Overrides the globally selected image generation model (e.g., `sd_xl_base_1.0.safetensors`). |
| **Steps** | `20` | Default number of sampling steps. |
| **Size** | `1024x1024` | Default image dimensions. |
| **Seed** | `-1` | Default seed (`-1` means random). |
| **CFG Scale** | `1.0` | Default CFG Scale. |
| **Sampler Name** | `Euler` | Default sampler. |
| **Scheduler** | `Simple` | Default scheduler. |
| **Enable HR** | `False` | Default state for High-Res Fix. |
| **N Iter** | `1` | Default number of images to generate. |
| **Batch Size** | `1` | Default batch size. |
| **HR Scale** | `2.0` | Default High-Res upscale factor. |
| **HR Upscaler** | `Latent` | Default High-Res upscaler model. |
| **Denoising Strength**| `0.45` | Default denoising strength for High-Res Fix. |

---

### ⚙️ Engine & Parameter Mapping

Easymage acts as a universal translator. Here is how your commands are mapped to each backend API.

<details>
<summary><strong>➡️ View Full Mapping Tables</strong></summary>

#### Automatic1111 / Forge (Direct HTTPX)
*Logic: Pass-through. Most parameters are sent directly.*
| EasyMage Parameter | Forge API Parameter | Notes |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Final LLM-enhanced prompt. |
| **negative_prompt** | `negative_prompt` | Passed directly. |
| **size** (WxH) | `width`, `height` | Split into two integers. |
| **All others** | *(matches)* | `steps`, `seed`, `cfg_scale`, `sampler_name`, etc., are passed directly. |

#### OpenAI (DALL-E 3) (Direct HTTPX)
*Logic: Strict mapping to DALL-E 3's capabilities.*
| EasyMage Parameter | OpenAI API Parameter | Notes |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Main prompt. |
| **negative_prompt** | `prompt` | ⚠️ **Merged:** Appended as `... . Negative: {text}`. |
| **enable_hr** (`+h`) | `quality` | `True` → `hd`, `False` → `standard`. |
| **style** (`stl=...`) | `style` | `v` → `vivid` (default), `n` → `natural`. |
| **user (Context)** | `user` | Passes the Open WebUI user ID. |
| **n_iter** (`n=...`) | `n` | ⚠️ **Hardcoded to 1**. |
| **-** | `response_format` | ⚠️ **Hardcoded to `b64_json`**. |

#### Gemini (Imagen 3) (Direct HTTPX)
*Logic: Structured for Google Cloud AI Platform.*
| EasyMage Parameter | Gemini API Parameter | Notes |
| :--- | :--- | :--- |
| **enhanced_prompt** | `instances[0].prompt` | Main prompt. |
| **negative_prompt** | `parameters.negativePrompt`| Passed natively. |
| **n_iter** (`n=...`) | `parameters.sampleCount` | Number of images (1-4). |
| **seed** (`sd=...`) | `parameters.seed` | Passed natively. |
| **size** / **aspect_ratio** | `parameters.aspectRatio` | ⚠️ **Calculated:** Pixel dimensions are converted to a ratio string (e.g., "16:9"). |
| **-** | `parameters.addWatermark` | ⚠️ **Hardcoded to `false`** to enable seed usage. |
| **-** | `parameters.personGeneration`| ⚠️ **Hardcoded to `allow_all`**. |
| **-** | `parameters.includeReasoning` | ⚠️ **Hardcoded to `false`**. |
| **-** | `parameters.safetySetting` | ⚠️ **Hardcoded to `block_none`**. |

#### ComfyUI / Fallback (via Open WebUI API)
*Logic: Uses the standard, limited Open WebUI API.*
| EasyMage Parameter | OWUI Form Parameter | Notes |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Main prompt. |
| **model** | `model` | Model name. |
| **size** | `size` | Size string. |
| **n_iter** | `n` | Number of images. |
| **All others** | - | ❌ **Ignored:** Advanced parameters like `seed`, `steps`, `cfg` are not supported by this fallback method. |
</details>

---

### 📌 Output and Citations

Beneath the generated image, you'll find three citations for a complete analysis:
1.  `[🚀 PROMPT]`: The final, enhanced prompt that was sent to the generation engine.
2.  `[🟢 SCORE: xx%]`: The technical critique and score from the Quality Audit.
3.  `[🔍 DETAILS]`: A summary of the configuration and performance metrics.

---

### 📜 Changelog

#### v0.8.1 (2026-02-10)
*   **Intelligent Router**: Implemented a dispatcher with dedicated methods for each engine.
*   **Native OpenAI & Gemini**: Added direct HTTPX support for OpenAI and Gemini with full parameter mapping.
*   **Advanced Mapping**: Implemented logic for `style`/`quality` (OpenAI) and `seed`/`watermark` (Gemini).
*   **UI Cleanup**: Improved code block formatting in chat messages.

#### v0.7.5 (2026-02-08)
*   **SoC Refactor**: Restructured the code into specialized classes (`InferenceEngine`, `PromptParser`, etc.).
*   **Optimized Inference**: Generalized the `_infer` method to handle text and vision calls agnostically.
*   **Local Performance Boost**: Optimized the flow to skip the vision probe when using `imgx`.

