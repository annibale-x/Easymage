**v0.8.1 (2026-02-10)**

- Intelligent Router: Implemented a dispatcher with dedicated methods for each engine.
- Native OpenAI & Gemini: Added direct HTTPX support for OpenAI and Gemini with full parameter mapping.
- Advanced Mapping: Implemented logic for `style`/`quality` (OpenAI) and `seed`/`watermark` (Gemini).
- UI Cleanup: Improved code block formatting in chat messages.

**v0.7.5 (2026-02-08)**

- SoC Refactor: Restructured the code into specialized classes (`InferenceEngine`, `PromptParser`, etc.).
- Optimized Inference: Generalized the `_infer` method to handle text and vision calls agnostically.
- Local Performance Boost: Optimized the flow to skip the vision probe when using `imgx`.

**v0.6.3 (2026-01-27)**

- Modular Architecture: Complete refactor to a method-based class structure for better maintainability.
- Precise Metrics: Replaced heuristic token estimation with exact data from Open WebUI API responses.
- Vision Cache Persistence: Introduced `persistent_vision_cache` to store model capabilities locally.
- Audit Optimization: Refined Vision Audit prompts for objective scoring and artifact detection.

**v0.5.1 (2026-01-25)**

- Multimodal Validation: Implemented base64 reference image probing to verify vision capabilities.
- Debug Mode: Added a toggle for detailed console logging.

**v0.4.83 (2026-01-23)**

- Vision Audit System: Initial integration of real-time image analysis.
- Unicode UI: Optimized technical details for environments with restricted Markdown.