**v0.9.1-beta.9 (2026-02-15)**

- Subcommand Architecture: Unified workflow with `img:p` (Prompt Only), `img:r` (Entropy Randomizer), and `img ?` (Interactive Help).
- Entropy Engine 2.0: Replaced LLM-based randomizer with a deterministic Python-side "Double Dice" system (20 Categories + 10 Moods) to enforce physical grounding and stylistic diversity.
- Direct Dispatch Core: Implemented a global persistent HTTP client for zero-latency connection pooling and direct backend execution (bypassing OWUI chat overhead).
- Smart VRAM Management: Added automatic model unloading and an "Extreme Cleanup" valve to prevent OOM errors on local backends.
- Forge Power-Ups: Exposed granular High-Res Fix controls via CLI (`hr`, `hru`, `dns`, `hdcs`).
- Error Intelligence: Now extracts and displays raw API error messages (e.g., OpenAI Safety Violations) directly in the chat.
- Fixes: Resolved OpenAI size snapping (Error 400), Global Config retrieval for `PersistentConfig` objects, and unified Forge/A1111 terminology.


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