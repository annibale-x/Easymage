* 2026-02-16: v0.9.3 - Aspect Ratio, CLI Robustness & Privacy Documentation (Hannibal)  

  * Fixed a critical bug where the default Aspect Ratio User Valve was overriding custom Size settings.
  * Implemented typographic dash normalization in the Prompt Parser to support mobile/iOS auto-corrections (converting — and – back to standard --).
  * Added a dedicated "Privacy & Local Execution" section to the README explaining how to use `easy_cloud_mode` for local-only setups (LM Studio/Local Forge).
  * Updated documentation with a detailed configuration hierarchy and parameter behavior.
  * Refined the "Size" parameter location explanation to guide users to the Chat Controls menu.

* 2026-02-15: v0.9.2-beta.2 - Multilingual Prompt Enhancer & Vision QC (Hannibal)  

  * Subcommand Architecture: Unified workflow with `img:p` (Prompt Only), `img:r` (Entropy Randomizer), and `img ?` (Interactive Help).
  * Entropy Engine 2.0: Replaced LLM-based randomizer with a deterministic Python-side "Double Dice" system (20 Categories + 10 Moods).
  * Direct Dispatch Core: Implemented a global persistent HTTP client for zero-latency connection pooling.
  * Smart VRAM Management: Added automatic model unloading and an "Extreme Cleanup" valve.
  * Forge Power-Ups: Exposed granular High-Res Fix controls via CLI (`hr`, `hru`, `dns`, `hdcs`).
  * Error Intelligence: Extracts and displays raw API error messages (e.g., OpenAI Safety Violations).
  * Technical Fixes: Resolved OpenAI size snapping, Global Config retrieval for PersistentConfig, and Forge terminology.

* 2026-02-10: v0.8.1 - Intelligent Router Refactor (Hannibal)  

  * Intelligent Router: Implemented a dispatcher with dedicated methods for each engine.
  * Native OpenAI & Gemini: Added direct HTTPX support for OpenAI and Gemini with full parameter mapping.
  * UI Cleanup: Improved code block formatting in chat messages.

* 2026-02-08: v0.7.5 - SoC Refactor & Inference Optimization (Hannibal)  

  * SoC Refactor: Restructured the code into specialized classes (`InferenceEngine`, `PromptParser`, etc.).
  * Optimized Inference: Generalized the `_infer` method to handle text and vision calls agnostically.

* 2026-01-27: v0.6.3 - Modular Architecture & Vision Audit (Hannibal)  

  * Modular Architecture: Complete refactor to a method-based class structure for better maintainability.
  * Vision Cache Persistence: Introduced `persistent_vision_cache` to store model capabilities locally.

* 2026-01-25: v0.5.1 - Multimodal Validation & Debugging (Hannibal)  

  * Multimodal Validation: Implemented base64 reference image probing to verify vision capabilities.
  * Debug Mode: Added a toggle for detailed console logging.

* 2026-01-23: v0.4.83 - Initial Vision Audit Integration (Hannibal)  

  * Vision Audit System: Initial integration of real-time image analysis.