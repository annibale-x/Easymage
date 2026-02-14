
# ✨ Easymage 0.9.2-beta.2: Multilingual Prompt Enhancer & Vision QC

Easymage is a professional-grade orchestration filter for **Open WebUI** designed to transform your image generation workflow into a unified and intelligent experience. By simply prepending `img` triggers to any message, you activate an advanced pipeline that handles everything from multilingual prompt engineering to multi-engine generation and post-creation technical analysis.

This filter acts as an **Intelligent Dispatcher**, unlocking advanced, engine-specific parameters like `seed`, `style`, `quality`, and `distilled CFG` and many others that are not natively exposed through the standard Open WebUI interface. Version 0.9.1 introduces a streamlined **Subcommand Architecture** and a **Unified Help System**, creating a seamless bridge between local power-user tools (**Forge**, **ComfyUI**) and cloud simplicity (**OpenAI**, **Gemini**).

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/annibale-x/Easymage)
![Open WebUI Filter](https://img.shields.io/badge/Open%20WebUI-Filter-blue?style=flat&logo=openai)
![License](https://img.shields.io/github/license/annibale-x/Easymage?color=green)


### 🆕 What's New in v0.9.2-beta.2 (vs v0.6.3)

-   **Subcommand Architecture**: Replaced the legacy `imgx` trigger with a unified syntax (`img:p`, `img:r`, `img ?`). This streamlines the workflow, allowing seamless switching between generation, prompting, and random modes without changing the base command.
-   **Entropy Engine (Random Mode)**: The new `img:r` command utilizes a sophisticated "Mental Dice Roll" logic within the LLM to generate radically diverse prompts across 6 distinct macro-categories (Nature, Street Photography, Art, Pop Culture, Architecture, Abstract), avoiding common AI clichés like jellyfish or nebulas.
-   **Integrated Help System**: Typing `img ?` or simply `img` (with no prompt) now injects a visual manual directly into the chat. It creates interactive citation badges for models, parameters, and engine codes, eliminating the need to memorize syntax.
-   **Expanded High-Res Parameters**: Added granular control for Forge upscaling via CLI, including `hr` (scale), `hru` (upscaler model), `dns` (denoising strength), and `hdcs` (distilled CFG).
-   **High-Performance Connection Pooling**: Implements a global, persistent HTTP client for Ollama, OpenAI, and Forge. Eliminates handshake latency and keeps connections alive for instant response times.
-   **Direct API Dispatch**: Completely bypasses Open WebUI's internal chat completion logic. Requests are sent directly to the backend (Ollama/OpenAI), preventing history pollution and database bloat.
-   **Environment-Agnostic VRAM Management**:
    -   **Standard Cleanup**: Automatically unloads unused models from VRAM before starting image generation, preventing Out-Of-Memory errors on consumer GPUs.
    -   **Extreme Cleanup Valve**: Optional setting to unload *everything* (including the active LLM) for maximum VRAM availability during generation.
-   **Context & Session Stability**: Optimized payload logic prevents unnecessary model reloading in Ollama, solving "VRAM thrashing" issues during complex multi-step workflows.
-   **Engine-Level Determinism**: Custom dispatcher for Ollama/OpenAI ensures 1:1 seed reproducibility by bypassing middleware interference.
-   **Selective Negation Strategy**: Smart logic chooses between native API or LLM integration based on engine support and Hires Fix status.
-   **Universal CLI Syntax**: Full control via `sz=`, `stp=`, `sd=`, `dcs=`, `hdcs=`, and specialized shortcuts.
-   **Dual-Layer Negative Prompts**: Automatic fallback to AVOID protocol for limited engines (OpenAI/Comfy).
-   **Refined Error Interception**: Unified handling for backend timeouts and API failures via Docker logging and UI alerts.

---

### ⚡ Quick Start (Copy & Paste)

New to Easymage? Try these commands to see what it can do.

| Goal | Command | What it does |
| :--- | :--- | :--- |
| **Simple** | `img A futuristic city made of glass` | Generates a 1024x1024 image using your default settings. |
| **Portrait** | `img ar=9:16 An astronaut in a flower field` | Creates a vertical image (Instagram/Reels format). |
| **Advanced** | `img sz=1280x720 +h -- A cyberpunk street rain` | Widescreen HD image with **High-Res Fix** enabled for extra detail. |
| **Random** | `img:r watercolor, peaceful` | Let the AI decide the subject, using "watercolor" as a style guide. |
| **Native Lang**| `img Un gatto che beve caffè` | **Write in any language!** Easymage translates and optimizes it for the model. |

---

### ✨ Key Features

*   **🌍 Multilingual Native**: You don't need to speak English to get professional results. Write your prompt in **any language**. Easymage detects the language, translates it, and expands it into technical English optimized for the specific generation model.
*   **Advanced Multi-Engine Routing**: Native Direct HTTPX support for Forge, `OpenAI (DALL-E-3 / DALL-E-2)`, and the full `Google Ecosystem (Gemini 1.5/2.0/3 Flash/Pro & Imagen 3)`, with a standard API fallback for `ComfyUI`. Easymage translates universal commands into the specific "technical dialect" of each API.
    > **📝 Compatibility Note**: Throughout this documentation, we use **"Forge"** to refer to local Stable Diffusion backends. Easymage is fully compatible with both **WebUI Forge** and the classic **Automatic1111 SD-WebUI**, as they share the same API structure.
*   **VRAM Auto-Optimization**: Automatically manages your GPU memory. Before generating an image, Easymage checks your Ollama server and unloads unused models to ensure Forge or ComfyUI have enough VRAM to operate without crashing.
*   **Zero-Latency Dispatch**: Uses a persistent connection pool to communicate with backends. This reduces the time-to-first-token and image generation start time by eliminating repetitive network handshakes.
*   **Selective Negation Strategy**: A smart logic core that determines how to handle negative prompts. It automatically decides whether to use native API fields (for Gemini or Forge with Hires Fix) or to "inject" the negation into the LLM-enhanced description (for OpenAI and ComfyUI), ensuring perfect visual results.
*   **Automated Prompt Engineering**: Expands minimalist user input into high-fidelity technical prompts. It dynamically incorporates details about lighting, camera angles, textures, environment, and artistic style based on your requirements.
*   **Vision Quality Audit (QC)**: Provides a real-time technical critique of the generated image using Vision LLMs. It assigns a numerical score (0-100%) and evaluates specific artifacts like noise, grain, melting, and aliasing (jaggies).
*   **Universal CLI Syntax**: Control every generation aspect with a single syntax (`sz=`, `stp=`, `ar=`, etc.). No more switching between different interfaces or learning complex API structures.
*   **Performance Analytics**: Every generation tracks precise metrics, including total execution time, image generation latency, and LLM throughput (Tokens per second).

---

### 🧠 The Three-Step Pipeline

1.  **Expansion & Optimization**: The system detects the language, identifies your requested styles and exclusions, and uses an LLM to build a professional prompt.
2.  **Smart Generation (w/ VRAM Handover)**:
    -   Easymage cleans the VRAM (unloading idle LLMs).
    -   The request is routed to the chosen engine via direct, optimized HTTPX calls.
    -   If native parameters are supported, they are passed directly; otherwise, logic fallbacks apply.
3.  **Visual Verification**: If a Vision-capable model is available, the final image is audited for prompt alignment and technical integrity.

---
## Universal CLI (Command Line Interface)

### 💡 Core Usage & Syntax

Easymage is activated by prepending a specific trigger command to your message. It parses technical instructions, stylistic choices, and the subject in a single pass.

#### Subcommand Architecture
*   `img [prompt]`: **Standard Generation**. Triggers the full pipeline: Prompt Enhancement → Image Generation → Vision Audit.
*   `img:p [prompt]`: **Prompt Only Mode**. Executes language detection and prompt enhancement logic but stops *before* generating the image. Useful for iterating on prompt engineering without burning compute credits or time.
*   `img:r [styles]`: **Random / "I'm Feeling Lucky" Mode**. The LLM acts as an "Engine of Total Entropy," rolling virtual dice to select a subject and style from diverse categories (Nature, Street Photography, Art, Pop Culture, Architecture, Abstract). Any text you provide is treated as a **Style Constraint**, not a subject.
*   `img ?`: **Manual Mode**. Displays the help menu, shortcuts, and parameter tables directly in the chat. Typing just `img` (with no prompt) also triggers this mode.

#### Command Structure
The universal format is:
`img:[subcmd] [ flags ] [ parameters ] [ styles -- ] [ subject] [ --no negative prompt ]`

> ⚠️ **CRITICAL SYNTAX RULE**:
> If a parameter value contains **spaces**, you MUST use double quotes!
> *   ✅ Correct: `smp="Euler a"`
> *   ❌ Wrong: `smp=Euler a`

| Part | Example | Description |
| :--- | :--- | :--- |
| **Parameters** | `sz=1024x1024 stp=30` | `key=value` pairs to control the generation (size, steps, etc.). Text values with spaces (e.g. sampler names) must be enclosed in double quotes: `smp="Euler a"`. |
| **Flags** | `+h -a` | Single-character toggles to enable (`+`) or disable (`-`) features. |
| **Styles** | `neon, 8k, macro` | Stylistic keywords that steer the LLM enhancer. |
|  **Subject**  | `-- a lonely robot` | The main content of your image. The `-- ` separator is **mandatory only if you are specifying Styles** to distinguish them from the Subject. |
| **Negative Prompt**| `--no people, blur` | Elements to exclude. The `--no ` (or `—no`) separator triggers the logic fallback system. |

> 💡 **Smart Parsing**: If you only use technical parameters (like `sz=1024` or `+h`), Easymage automatically identifies the Subject as everything following the last parameter. You only need the `-- ` separator when you want to provide descriptive Styles (e.g., `img sz=1024 cinematic, low-angle -- a giant tree`).

#### Configuration Hierarchy & Precedence

Easymage uses an intelligent, three-layer system to determine which settings to apply for each image generation. This tiered approach gives you maximum flexibility, allowing for quick, one-time experiments without altering your preferred defaults. The rule is simple: **more specific settings always override more general ones.**

Here is the order of priority, from highest to lowest:

| Priority | Source | Description |
| :---: | :--- | :--- |
| **1 (Highest)** | **Command Line (CLI)** | Parameters typed directly into your chat message (e.g., `sd=123`). These are temporary and last for only one generation. |
| **2 (Medium)** | **Easymage Valves** | Your personal defaults, configured in `Settings > Filters`. These apply to all your `img` commands unless a CLI parameter is used. |
| **3 (Lowest)** | **Global OWUI Settings** | The server-wide defaults for image generation, usually configured by an administrator. Easymage uses these as a final fallback. |

#### Practical Example

Let's see how this works for the **`size`** parameter:

1.  **Global Setting**: Your Open WebUI server is configured with a default image size of `1024x1024`.
2.  **Valve Override**: You go into your Easymage Valves and set the `Size` to `512x512`. From now on, every time you type `img a cat`, the image will be `512x512`.
3.  **CLI Override**: You want to create a single widescreen image. You type `img sz=1792x1024 a cat`. For this specific request, Easymage will ignore both the Global setting and your Valve, generating a `1792x1024` image.

The next time you type `img a dog`, it will revert to using your Valve setting of `512x512`.

---

### ⚙️ Command Line Parameters

These parameters allow you to override backend settings directly from the chat.

| Parameter | Example | Description | Supported Engines |
| :--- | :--- | :--- | :--- |
| `ge` | `ge=o` | **G**eneration **E**ngine. Selects the generation backend. | All |
| `mdl` | `mdl=d3` | **M**o**d**e**l**. Selects a specific checkpoint or API model version. | All |
| `sz` | `sz=800` | **S**i**z**e. Accepts `WxH` or a single value `N` (auto-converted to square `NxN`). Dimensions are then normalized based on engine constraints and the `ar` parameter.| All |
| `ar` | `ar=16:9` | **A**spect **R**atio. Automatically calculates size based on ratio. | All |
| `stp` | `stp=25` | **St**e**p**s. Number of sampling iterations. | Forge / A1111 |
| `sd` | `sd=42` | **S**ee**d**. Numeric value for reproducible generations. | Forge / A1111, Gemini |
| `cs` | `cs=7.0` | **C**FG **S**cale. Classifier-Free Guidance intensity. | Forge, Comfy |
| `dcs` | `dcs=3.5` | **D**istilled **C**FG **S**cale. Specialized for Flux/SD3 models. | Forge / A1111 |
| `stl` | `stl=v` | **St**y**l**e. Choose between `v` (vivid) or `n` (natural). | OpenAI |
| `smp` | `smp=d2s` | **S**a**m**p**ler**. Selects the sampling algorithm. | Forge / A1111 |
| `sch` | `sch=k` | **Sch**eduler. Selects the noise schedule type. | Forge / A1111 |
| `hr` | `hr=2.0` | **H**igh-**R**es Scale. Enables Hires Fix and sets the multiplier. | Forge / A1111 |
| `hru` | `hru=Latent` | **H**igh-**R**es **U**pscaler. Specifies the upscaling model. | Forge / A1111 |
| `hdcs` | `hdcs=3.5` | **H**ires **D**istilled **C**FG. Distilled scale during the HR pass. | Forge / A1111 |
| `dns` | `dns=0.45` | **D**e**n**oi**s**ing Strength. Intensity of the HR fix pass. | Forge / A1111 |
| `auth` | `auth=sk..`| **Auth**entication. Overrides global/valve keys for this request. | All |

#### Command Line Flags (Toggles)
Flags provide a quick way to override your default **Valves** configuration for a single message.

| Flag | Name | Description |
| :--- | :--- | :--- |
| `+h` / `-h` | **H**igh-Res | Toggles High-Res Fix. For OpenAI, `+h` activates `quality: hd`. |
| `+p` / `-p` | **P**rompt | Toggles the LLM Prompt Enhancer. |
| `+a` / `-a` | **A**udit | Toggles the post-generation Vision Quality Audit. |
| `+d` / `-d` | **D**ebug | Toggles Debug Mode (prints internal state JSON in chat). |

---

### ⚡ Shortcut Tables

Easymage uses optimized shortcodes to minimize typing while maintaining full technical control. However, for those who prefer clarity or are using scripts, **full parameter names are also supported**.

<details>
<summary><strong>Engine Shortcuts (`ge=...`)</strong></summary>

| Shortcut | Engine Name | Backend API |
| :--- | :--- | :--- |
| `f` / `a` | `forge` / `automatic1111` | SD Forge / A1111 (Direct HTTPX) |
| `o` | `openai` | DALL-E 3 (Direct HTTPX) |
| `g` | `gemini` | Imagen 3 (Direct HTTPX) |
| `c` | `comfyui` | ComfyUI (Open WebUI API) |
</details>

<details>
<summary><strong>Model Shortcuts (`mdl=...`)</strong></summary>

| Shortcut | API Model Name / Identifier | Note / Description |
| :--- | :--- | :--- |
| **OpenAI** | | |
| `d3` | `dall-e-3` | Standard DALL-E 3 |
| `d2` | `dall-e-2` | Legacy DALL-E 2 |
| `g4o` | `gpt-4o` | Multimodal |
| `g4om` | `gpt-4o-mini` | Multimodal Mini|
| **Google Imagen Series** | | |
| `i4` | `imagen-4.0-generate-001` | **Imagen 4 (Full)** - Standard 2026 |
| `veo` | `veo-3.0-generate-preview` | **Veo 3** (Frame Gen / 4K) |
| **Google Gemini Series** | | |
| `g2.5f` | `gemini-2.5-flash-image` | **Nano Banana** (Flash / Fast) |
| `g3p` | `gemini-3-pro-image-preview` | **Nano Banana Pro** (High Quality) |
| **Local / Forge** | | |
| `flux` | `flux1-dev.safetensors` | Flux 1 Dev |
| `sdxl` | `sd_xl_base_1.0.safetensors` | SDXL Base |
</details>

<details>
<summary><strong>Aspect Ratio Shortcuts (`ar=...`)</strong></summary>

| Shortcut | Ratio | Common Use Case |
| :--- | :--- | :--- |
| `1` | `1:1` | Square (Social Media) |
| `16` | `16:9` | Cinematic Widescreen |
| `9` | `9:16` | Vertical / Reels |
| `4` | `4:3` | Photography Standard |
| `3` | `3:4` | Portrait |
| `21` | `21:9` | Ultra-Widescreen |
</details>

<details>
<summary><strong>Sampler Shortcuts (`smp=...`)</strong></summary>

| Code | Full Sampler Name | Code | Full Sampler Name |
| :--- | :--- | :--- | :--- |
| `d3s` | DPM++ 3M SDE | `df` | DPM fast |
| `d2sh`| DPM++ 2M SDE Heun | `dad` | DPM adaptive |
| `d2s` | DPM++ 2M SDE | `r` | Restart |
| `d2m` | DPM++ 2M | `h2` | HeunPP2 |
| `d2sa`| DPM++ 2S a | `ip` | IPNDM |
| `ds` | DPM++ SDE | `ipv` | IPNDM_V |
| `ea` | Euler a | `de` | DEIS |
| `e` | Euler | `u` | UniPC |
| `l` | LMS | `lcm` | LCM |
| `h` | Heun | `di` | DDIM |
| `d2` | DPM2 | `dic` | DDIM CFG++ |
| `d2a` | DPM2 a | `dp` | DDPM |
</details>

<details>
<summary><strong>Scheduler Shortcuts (`sch=...`)</strong></summary>

| Code | Full Scheduler Name | Code | Full Scheduler Name |
| :--- | :--- | :--- | :--- |
| `a` | Automatic | `ays` | Align Your Steps |
| `u` | Uniform | `aysg`| Align Your Steps GITS |
| `k` | Karras | `ays11`| Align Your Steps 11 |
| `e` | Exponential | `ays32`| Align Your Steps 32 |
| `pe` | Polyexponential | `s` | Simple |
| `su` | SGM Uniform | `n` | Normal |
| `ko` | KL Optimal | `di` | DDIM |
| `b` | Beta | `t` | Turbo |
</details>

> 💡 **Note**: Text values with spaces (e.g. sampler names) must be enclosed in double quotes: `smp="Euler a"`).

---
## The Selective Negation Logic

### 🧠 The Selective Negation Strategy

One of the most complex challenges in AI image generation is "negation" (telling the AI what *not* to include). Most engines are designed to follow positive instructions and often struggle with negative ones unless they are formatted specifically for their architecture.

Easymage introduces a **Selective Negation Strategy** that automatically chooses the best method to handle your `--no ` requirements.

#### 1. Native API Handling (High Fidelity)
If the generation engine has a dedicated technical field for negative prompts, Easymage passes your exclusions directly to the API. This provides a "surgical" removal of elements without affecting the creative description of the main subject.
*   **Gemini (Imagen 3)**: Uses the `negativePrompt` parameter natively.
*   **Forge (A1111)**: Uses the `negative_prompt` field **only when High-Res Fix (`+h`) is enabled**, ensuring maximum quality during the upscale pass.

#### 2. LLM Fallback Integration (Natural Description)
Many engines (like DALL-E 3) do not have a native "Negative Prompt" field. For these cases, or when Forge is used in standard mode, Easymage uses its **LLM Prompt Engineer** to "digest" the exclusions.
*   Instead of simply appending a list of words, the LLM rewrites the description to ensure those elements are logically absent.
*   *Example*: If you specify `--no people`, the LLM won't just say "no people"; it will describe a "completely deserted, silent landscape where no human presence is visible," which is much more effective for models like DALL-E.

#### Summary Logic Table

| Engine / Condition | Method | How it works |
| :--- | :--- | :--- |
| **Gemini** | Native | Sent to the `negativePrompt` API field. |
| **OpenAI (DALL-E 3)** | LLM Fallback | Integrated into the descriptive flow of the prompt. |
| **Forge (Hires Fix ON)** | Native | Sent to the technical `negative_prompt` field. |
| **Forge (Standard)** | LLM Fallback | Integrated into the descriptive prompt by the LLM. |
| **ComfyUI (Fallback)** | LLM Fallback | Integrated into the descriptive prompt by the LLM. |
| **`img:p` Trigger** | LLM Fallback | Always integrated into the text to provide a ready-to-use natural prompt. |

---

### 🪄 The "AVOID" Protocol
When the LLM Fallback is active, the system prompt for the Enhancer is dynamically updated with the **MANDATORY AVOID** rule. This forces the LLM to verify that the forbidden elements are not just ignored, but that the scene is described in a way that confirms their absence, significantly improving the **Vision Quality Audit** success rate.

---

## Intelligent Engine Router

### ⚙️ Engine & Parameter Mapping

Easymage acts as a high-level abstraction layer. It converts universal parameters into the specific JSON payloads required by each engine. Below are the technical mapping details.

<details>
<summary><strong>➡️ View Full Mapping Tables</strong></summary>

#### 1. Forge / Automatic1111 (Direct HTTPX)
*Connection: Sends a direct POST request to `/sdapi/v1/txt2img`.*
| EasyMage Parameter | Forge API Field | Technical Logic |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Final expanded text from LLM. |
| **negative_prompt** | `negative_prompt` | ⚠️ **Conditional:** Sent natively only if `+h` (Hires Fix) is ON. If OFF, it's integrated into the prompt text. |
| **size** | `width` / `height` | The `WxH` string is split into two integers. |
| **stp** / **sd** / **cs** | `steps` / `seed` / `cfg_scale` | Passed directly to the API. |
| **dcs** | `distilled_cfg_scale` | Used for Flux/SD3. Forces native `cfg_scale` to 1.0. |
| **smp** / **sch** | `sampler_name` / `scheduler` | Converted via `SAMPLER_MAP` and `SCHEDULER_MAP`. |
| **hr** / **hru** / **dns** | `enable_hr` / `hr_upscaler` / `denoising_strength` | Activates the Hires Fix pipeline. |

#### 2. OpenAI (DALL-E 3) (Direct HTTPX)
*Connection: Sends a direct POST request to `/images/generations`.*
| EasyMage Parameter | OpenAI API Field | Technical Logic |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Main prompt. |
| **negative_prompt** | - | ❌ **LLM Fallback:** Always integrated into the descriptive text. |
| **enable_hr** (`+h`) | `quality` | `True` → `hd`, `False` → `standard`. |
| **stl** | `style` | `v` → `vivid` (default), `n` → `natural`. |
| **size** | `size` | Snaps to `1024x1024`, `1792x1024`, or `1024x1792`. |
| **user (Context)** | `user` | Automatically passes your Open WebUI User ID. |
| **-** | `n` | ⚠️ **Hardcoded to 1** (API limitation). |
| **-** | `response_format` | ⚠️ **Hardcoded to `b64_json`**. |

#### 3. Gemini (Imagen 3) (Direct HTTPX)
*Connection: Sends a direct POST request to Google's `:predict` or `:generateContent` endpoint.*
| EasyMage Parameter | Gemini API Field | Technical Logic |
| :--- | :--- | :--- |
| **enhanced_prompt** | `instances[0].prompt` | Main prompt. |
| **negative_prompt** | `parameters.negativePrompt` | ✅ **Always Native:** Supported directly by Gemini. |
| **n** | `parameters.sampleCount` | Number of images (1-4). |
| **sd** | `parameters.seed` | Supported because `addWatermark` is set to false. |
| **ar** / **sz** | `parameters.aspectRatio` | ⚠️ **Calculated:** Pixel dimensions are converted back to a ratio string (`1:1`, `16:9`, etc.). |
| **-** | `parameters.safetySetting` | ⚠️ **Hardcoded to `block_none`** for maximum flexibility. |
| **-** | `parameters.personGeneration`| ⚠️ **Hardcoded to `allow_all`**. |
| **-** | `parameters.addWatermark` | ⚠️ **Hardcoded to `false`**. |
| **-** | `parameters.includeReasoning` | ⚠️ **Hardcoded to `false`**. |

#### 4. ComfyUI / Fallback (via Open WebUI API)
*Connection: Uses the internal `image_generations` router of Open WebUI.*
| EasyMage Parameter | OWUI Form Field | Technical Logic |
| :--- | :--- | :--- |
| **enhanced_prompt** | `prompt` | Main prompt. |
| **negative_prompt** | - | ❌ **LLM Fallback:** Integrated into the descriptive text. |
| **mdl** | `model` | Checkpoint name. |
| **sz** | `size` | Size string. |
| **n** | `n` | Number of images. |
| **All others** | - | ❌ **Unsupported:** The standard OWUI router does not expose seed, steps, or CFG Scale for this method. |
</details>

---

### 📐 Size & Aspect Ratio Logic

Easymage handles dimensions dynamically to satisfy different engine requirements:

1. **Input Parsing**: Using `sz=N` automatically sets the target to `NxN`. Using `sz=WxH` sets specific targets.
2. **Normalization**:
   - **OpenAI (DALL-E 3)**: Ignores specific pixels and "snaps" to the closest supported HD resolution (`1024x1024`, `1792x1024`, or `1024x1792`) based on the calculated aspect ratio.
   - **Gemini (Imagen 3)**: Converts the dimensions into one of the supported ratio strings (`1:1`, `4:3`, `16:9`, etc.).
   - **Forge / A1111**: Uses the requested pixels but rounds them to the nearest multiple of **8** to ensure hardware compatibility.
3. **AR Overriding**: If the `ar` parameter is present, it takes precedence in calculating the final height relative to the requested width.

---
## Configuration & Diagnostics

### 🔧 Filter Configuration (Valves)

Valves allow you to set the "factory defaults" for Easymage. You can find these settings in your Open WebUI profile under `Settings > Filters > Easymage`.

**Note**: Any command sent via CLI (e.g., `sz=512x512`) will temporarily override these global settings for that specific message.

| Valve | Default | Description |
| :--- | :---: | :--- |
| **Enhanced Prompt** | `True` | Enables the LLM Prompt Engineer to expand your input by default. |
| **Quality Audit** | `True` | Enables the post-generation Vision analysis and scoring. |
| **Strict Audit** | `False` | Enables "Ruthless Mode" for the audit, being much more severe with technical flaws. |
| **Persistent Vision Cache**| `False` | Saves the results of the Vision Capability test to disk to speed up subsequent starts. |
| **Extreme VRAM Cleanup** | `False` | If enabled, unloads ALL models (including the current LLM) before image generation. Default is False (unloads only other models). |
| **Debug** | `False` | Prints the full internal state JSON and diagnostic logs to the Docker/Server console. |
| **Model** | `None` | Forces a specific model/checkpoint (e.g., `flux1-dev.safetensors`) for all generations. |
| **Generation Timeout** | `120` | Maximum time (seconds) to wait for the image engine to respond. |
| **Steps** | `20` | Default number of sampling steps. |
| **Size** | `1024x1024` | Default image dimensions. |
| **Seed** | `-1` | Default seed value (`-1` for random). |
| **CFG Scale** | `1.0` | Default Classifier-Free Guidance scale. |
| **Distilled CFG Scale** | `3.5` | Default Distilled CFG for Flux/SD3 models. |
| **Sampler Name** | `Euler` | Default sampling algorithm. |
| **Scheduler** | `Simple` | Default noise schedule type. |
| **Enable HR** | `False` | Enables High-Res Fix (Forge) or HD Quality (OpenAI) by default. |
| **HR Scale** | `2.0` | Default multiplier for the High-Res Fix pass. |
| **HR Upscaler** | `Latent` | Default model used for the Hires upscale pass. |
| **HR Distilled CFG** | `3.5` | Default Distilled CFG used during the Hires pass. |
| **Denoising Strength** | `0.45` | Default intensity of the High-Res Fix pass. |

---

### 📌 Output, Citations & Performance

Easymage provides a transparent output system. Beneath the generated image, you will see three linked citations and a performance status bar.

#### 1. The Citation System
*   `[🚀 PROMPT]`: Shows the **Enhanced Prompt** (the text actually seen by the GPU) followed by a structured recap of your original **Styles** and **Negative Prompt**.
*   `[🟢 SCORE: XX%]`: The result of the Visual Quality Audit. It includes a technical critique and a colored emoji indicator based on the score (80+ 🟢, 70+ 🔵, 60+ 🟡, 40+ 🟠, <40 🔴).
*   `[🔍 DETAILS]`: A full technical recap including the backend Engine used, the active Model, Resolution, and specific latency for each pipeline stage.
*   `[ℹ️ INFO]`: (Help Mode only) Displays version metadata and project links.

#### 2. Real-Time Performance Tracking
The final status bar provides a detailed snapshot of the generation efficiency:
`[Total Time]s total | [Image Gen]s img | [Total Tokens] tk | [Throughput] tk/s`

*   **Total Time**: The entire duration from your message to the final output.
*   **img**: The time spent waiting specifically for the Image Engine.
*   **tk / tk/s**: Token count and speed of the LLM during the Prompt Enhancement phase.

---

### 🛠️ Diagnostics & Debugging

If you encounter issues or want to see how Easymage is "thinking," you can activate **Debug Mode** via the Valve or by adding `+d` to your message.

*   **In-Chat Debug**: Easymage will print two formatted JSON blocks containing the current **Internal Model State** (parsed values, calculated ratios, selected engine) and the current **Valve Configuration**.
*   **Docker Logs**: Detailed "⚡ EASYMAGE DEBUG" logs are printed to the server console, including the raw System Prompts sent to the LLM and the raw responses from the image APIs.
*   **Error Handling**: If an engine fails, Easymage will intercept the error and display a detailed "❌ EASYMAGE ERROR" message in the chat, preventing the filter from crashing.

---

### 🔮 Future Developments

Easymage is constantly evolving. Here are the key features currently in the pipeline:

-   **AWS Nova Canvas Integration**: Integration of Amazon’s Nova Canvas model via AWS Bedrock into the EM image generation pipeline.

-   **Filter Chainability**: Insert EM logic into the native filters sequence, allowing it to interact with, modify, or pass data to other active filters.

-   **Multi-Image & Batching**: Support for generating multiple iterations in a single call and automated batch processing for high-volume workflows.

-   **ComfyUI Native Integration**: Bridging the gap with ComfyUI backends to leverage its node-based power directly through Easymage's streamlined syntax.

-   **Fine-Tuned Control (LoRAs)**: Comprehensive support for custom LoRA injection (A1111/Forge & ComfyUI), enabling precise style and character consistency.

-   **Image-to-Image (Img2Img)**: Implementation of the Img2Img pipeline, allowing users to use reference images as a foundation for Easymage-driven transformations.

---

### 📄 License
Easymage is released under the **MIT License**. Feel free to use, modify, and distribute it within the Open WebUI community.

---

### 🤝 Contributing & Support

**Easymage** is an orchestration layer for a complex and fragmented ecosystem. While developed on high-end hardware, its core mission is universal compatibility and robust control. Given the thousands of possible combinations between LLMs, Image Engines, and UI parameters, this version is a **Public Beta**.

We actively encourage feedback and issue reports regarding:

-   **Engine Mappings:** Incorrect parameter translations or missing features.

-   **Runtime Errors:** Crashes, hangs, or unexpected behavior in the Open WebUI pipeline.

-   **Environment Issues:** Compatibility bugs across different hardware or Docker setups.


Help us harden the orchestration logic by reporting any anomaly you encounter.


If you encounter bugs or have feature requests, please open an issue on the [GitHub Repository](https://github.com/annibale-x/Easymage) or contact the author through the Open WebUI community portal.