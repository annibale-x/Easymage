* 2026-02-16: v0.9.4 - Aspect Ratio & CLI Robustness (Hannibal)  
  * Fixed a critical bug where the default Aspect Ratio User Valve was overriding custom Size settings.
  * Implemented typographic dash normalization in the Prompt Parser to support mobile/iOS auto-corrections.
  * Updated documentation with detailed configuration hierarchy and parameter behavior.

* 2026-02-16: v0.9.3 - Documentation & UX Clarity (Hannibal)  
  * Updated README.md to clarify the configuration hierarchy between CLI, User Valves, Admin Valves, and Global OWUI Settings.
  * Refined explanation of the "Size" parameter location to prevent user confusion.

* 2026-02-15: v0.9.2-beta.2 - Multilingual Prompt Enhancer & Vision QC (Hannibal)  
  * Subcommand Architecture: Unified workflow with `img:p` (Prompt Only), `img:r` (Entropy Randomizer), and `img ?` (Interactive Help).
  * Entropy Engine 2.0: Replaced LLM-based randomizer with a deterministic Python-side "Double Dice" system (20 Categories + 10 Moods) to enforce physical grounding and stylistic diversity.
  * Direct Dispatch Core: Implemented a global persistent HTTP client for zero-latency connection pooling and direct backend execution (bypassing OWUI chat overhead).
  * Smart VRAM Management: Added automatic model unloading and an "Extreme Cleanup" valve to prevent OOM errors on local backends.
  * Forge Power-Ups: Exposed granular High-Res Fix controls via CLI (`hr`, `hru`, `dns`, `hdcs`).
  * Error Intelligence: Now extracts and displays raw API error messages (e.g., OpenAI Safety Violations) directly in the chat.
  * Fixes: Resolved OpenAI size snapping (Error 400), Global Config retrieval for `PersistentConfig` objects, and unified Forge/A1111 terminology.

* 2026-02-10: v0.8.1 - Intelligent Router Refactor (Hannibal)  
  * Intelligent Router: Implemented a dispatcher with dedicated methods for each engine.
  * Native OpenAI & Gemini: Added direct HTTPX support for OpenAI and Gemini with full parameter mapping.
  * Advanced Mapping: Implemented logic for `style`/`quality` (OpenAI) and `seed`/`watermark` (Gemini).
  * UI Cleanup: Improved code block formatting in chat messages.

* 2026-02-08: v0.7.5 - SoC Refactor & Inference Optimization (Hannibal)  
  * SoC Refactor: Restructured the code into specialized classes (`InferenceEngine`, `PromptParser`, etc.).
  * Optimized Inference: Generalized the `_infer` method to handle text and vision calls agnostically.
  * Local Performance Boost: Optimized the flow to skip the vision probe when using `imgx`.

* 2026-01-27: v0.6.3 - Modular Architecture & Vision Audit (Hannibal)  
  * Modular Architecture: Complete refactor to a method-based class structure for better maintainability.
  * Precise Metrics: Replaced heuristic token estimation with exact data from Open WebUI API responses.
  * Vision Cache Persistence: Introduced `persistent_vision_cache` to store model capabilities locally.
  * Audit Optimization: Refined Vision Audit prompts for objective scoring and artifact detection.

* 2026-01-25: v0.5.1 - Multimodal Validation & Debugging (Hannibal)  
  * Multimodal Validation: Implemented base64 reference image probing to verify vision capabilities.
  * Debug Mode: Added a toggle for detailed console logging.

* 2026-01-23: v0.4.83 - Initial Vision Audit Integration (Hannibal)  
  * Vision Audit System: Initial integration of real-time image analysis.
  * Unicode UI: Optimized technical details for environments with restricted Markdown.