"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.9.1-beta.4
repo_url: https://github.com/annibale-x/Easymage
author: Hannibal
author_url: https://openwebui.com/u/h4nn1b4l
author_email: annibale.x@gmail.com
Description: Advanced generation filter with Unified Auth, UserValves, Strict CLI Validation and extensive debugging.
"""

# --- IMPORTS ---

import json
import re
import time
import base64
import os
import sys
import httpx  # type: ignore
from typing import Optional, Any, List, Dict, Tuple, Union
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, CreateImageForm  # type: ignore
from open_webui.models.users import UserModel  # type: ignore

CAPABILITY_CACHE_PATH = "data/easymage_vision_cache.json"

# --- GLOBAL SERVICES ---

# Persistent HTTP client for connection pooling.
# Placed at the top level to act as a singleton for the entire module lifetime.
HTTP_CLIENT = httpx.AsyncClient()


# --- CONFIGURATION & MAPS ---


class EasymageConfig:
    """
    Static mappings, configuration constants, and prompt templates used throughout the application.
    Acts as a central registry for engine-specific parameters and normalization maps.
    """

    class Engines:
        FORGE = "automatic1111"
        OPENAI = "openai"
        GEMINI = "gemini"
        COMFY = "comfyui"

    # Official Cloud Endpoints for Easy Cloud Mode
    OFFICIAL_URLS = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
    }

    # Maps short codes or alternative names to the specific sampler names.
    SAMPLER_MAP = {
        "d3s": "DPM++ 3M SDE",
        "d2sh": "DPM++ 2M SDE Heun",
        "d2s": "DPM++ 2M SDE",
        "d2m": "DPM++ 2M",
        "d2sa": "DPM++ 2S a",
        "ds": "DPM++ SDE",
        "ea": "Euler a",
        "e": "Euler",
        "l": "LMS",
        "h": "Heun",
        "d2": "DPM2",
        "d2a": "DPM2 a",
        "df": "DPM fast",
        "dad": "DPM adaptive",
        "r": "Restart",
        "h2": "HeunPP2",
        "ip": "IPNDM",
        "ipv": "IPNDM_V",
        "de": "DEIS",
        "u": "UniPC",
        "lcm": "LCM",
        "di": "DDIM",
        "dic": "DDIM CFG++",
        "dp": "DDPM",
    }

    # Maps short codes to scheduler names.
    SCHEDULER_MAP = {
        "a": "Automatic",
        "u": "Uniform",
        "k": "Karras",
        "e": "Exponential",
        "pe": "Polyexponential",
        "su": "SGM Uniform",
        "ko": "KL Optimal",
        "ays": "Align Your Steps",
        "aysg": "Align Your Steps GITS",
        "ays11": "Align Your Steps 11",
        "ays32": "Align Your Steps 32",
        "s": "Simple",
        "n": "Normal",
        "di": "DDIM",
        "b": "Beta",
        "t": "Turbo",
    }

    # Mapping between internal engine constants and the configuration keys expected in Open WebUI state.
    ENGINE_MAP = {
        Engines.FORGE: ["AUTOMATIC1111_BASE_URL", "AUTOMATIC1111_PARAMS"],
        Engines.COMFY: ["COMFYUI_WORKFLOW", "COMFYUI_WORKFLOW_NODES"],
        Engines.OPENAI: ["IMAGES_OPENAI_API_BASE_URL", "IMAGES_OPENAI_API_PARAMS"],
        Engines.GEMINI: ["IMAGES_GEMINI_API_KEY", "IMAGES_GEMINI_ENDPOINT_METHOD"],
    }

    # Expanded Model Shortcuts
    MODEL_SHORTCUTS = {
        # OpenAI
        "d3": "dall-e-3",
        "d2": "dall-e-2",
        "g4o": "gpt-4o",
        "g4om": "gpt-4o-mini",
        # Gemini (Smart Chunk Initials)
        "g3pip": "gemini-3-pro-image-preview",
        "g2.5fi": "gemini-2.5-flash-image",
        "g2f": "gemini-2.0-flash",
        "g2fe": "gemini-2.0-flash-exp",
        "i3": "imagen-3.0-generate-001",
        "i3f": "imagen-3.0-fast-generate-001",
        # Local / Forge
        "flux": "flux1-dev.safetensors",
        "sdxl": "sd_xl_base_1.0.safetensors",
    }

    # Auto-Detection Map: Associates specific models to their engines
    # Keys MUST be lowercase for case-insensitive matching.
    MODEL_ENGINE_MAP = {
        "dall-e-3": Engines.OPENAI,
        "dall-e-2": Engines.OPENAI,
        "gemini-3-pro-image-preview": Engines.GEMINI,
        "gemini-2.5-flash-image": Engines.GEMINI,
        "gemini-2.0-flash": Engines.GEMINI,
        "gemini-2.0-flash-exp": Engines.GEMINI,
        "imagen-3.0-generate-001": Engines.GEMINI,
        "imagen-3.0-fast-generate-001": Engines.GEMINI,
        "flux1-dev.safetensors": Engines.FORGE,
        "sd_xl_base_1.0.safetensors": Engines.FORGE,
    }

    # PROMPT TEMPLATES

    AUDIT_STANDARD = """
        RESET: ####################### NEW DATA STREAM ##########################
        ENVIRONMENT: STATELESS SANDBOX.
        TASK: AUDIT ANALYSIS (Audit scores is 0 to 100):
                Compare image with: '{prompt}',
                Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line.
        
        RULE: If any element explicitly forbidden in the prompt is present, the AUDIT score MUST be below 50.
        
        TASK: TECHNICAL EVALUATION:
                Evaluate NOISE, GRAIN, MELTING, JAGGIES.
        MANDATORY: Respond in {lang}. NO MARKDOWN. Use plain text and • for lists and ➔ for headings.
        MANDATORY: Final response MUST end with: SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
    """

    AUDIT_STRICT = """
        RESET: ####################### NEW DATA STREAM ##########################            
        ENVIRONMENT: STATELESS SANDBOX.
        RULE: Context Break. Terminate processing of previous context.
        RULE: Clear your working memory buffer and analyze this input in total isolation.
        TASK: AUDIT ANALYSIS (Audit scores is 0 to 100, where 0 is bad and 100 good):
                Compare the image with the reference prompt: '{prompt}',
                Describe what you actually see in the image.
                Critically evaluate the image's technical execution and its alignment with the prompt's requirements.
                Identify any contradictions, missing elements, or hallucinations.
                Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line.
        
        RULE: Contradictions regarding the presence or absence of objects are CRITICAL FAILURES. 
        RULE: If an element explicitly marked as absent/forbidden is detected, the AUDIT score MUST NOT exceed 30.
        
        RULE: Be extremely severe in technical evaluation. Do not excuse defects.
        TASK: TECHNICAL EVALUATION (Technical scores are 0 to 100, where 0 is LOW and 100 HIGH):
                Perform a ruthless technical audit. Identify every visual flaw.
                Evaluate NOISE, GRAIN, MELTING, JAGGIES.
        MANDATORY: Respond in {lang}. Be objective. NO MARKDOWN. Use plain text and • for lists and ➔ for headings. 
        MANDATORY: Final response MUST end with a single line containing only the following metrics:
        SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
    """

    PROMPT_ENHANCE_BASE = """
        ROLE: You are an expert AI Image Prompt Engineer.
        TASK: Expand the user's input into a professional, highly detailed prompt in {lang}.
        TASK: Add details about lighting, camera angle, textures, environment, and artistic style.
    """

    PROMPT_ENHANCE_STYLES = """
        MANDATORY: Incorporate these style elements naturally into the description: {styles}.
    """

    PROMPT_ENHANCE_NEG = """
        MANDATORY: The description must explicitly ensure that {negative} are NOT present. If necessary, 
        describe the scene in a way that confirms their absence.
    """

    PROMPT_ENHANCE_RULES = """
        RULE: Output ONLY the enhanced prompt.
        RULE: End the response immediately after the last descriptive sentence.
        RULE: Do not add any text, notes, or disclaimers after the prompt.
    """

    PROMPT_RANDOM_GEN = """
        ROLE: You are a Creative Director for generative AI art.
        TASK: Generate a completely random, unique, and highly detailed image description.
        THEME: Vary wildly between Sci-Fi, Fantasy, Photorealistic, Abstract, Surreal, Cyberpunk, or Classical Art.
        {style_instruction}
        RULE: Output ONLY the prompt description. Do not add introductions, quotes or markdown.
    """

    HELP_TEXT = """
### 🪄 Easymage Manual
**Advanced Generation Filter**

Easymage allows granular control over image generation directly from the chat.

**Usage Syntax:**
`img:[cmd] [flags] prompt --no negative_prompt`

**Subcommands:**
- `img`: Standard generation (Text + Vision Audit).
- `img:p`: Prompt Enhancer only (No generation).
- `img:r`: Random "I'm Feeling Lucky" mode.
- `img ?`: Show this help menu.

**Examples:**
- `img A cat in space` (Default)
- `img:r ar=16:9 --no text` (Random Wallpaper)
- `img ?` (Open Manual)
    """


# --- DATA STRUCTURES ---


class Store(dict):
    """
    Dynamic dictionary storage allowing dot notation access.
    Used for holding the Easymage state model.
    """

    def __getattr__(self, item):
        try:
            return self[item]

        except KeyError:
            return None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EasymageState:
    """
    Manages the internal state for a single request lifecycle.
    """

    def __init__(self, valves, user_valves):
        # Initialize the core model with default settings.
        # UserValves take precedence over Admin Valves in the initial merge.

        self.model = Store(
            {
                "trigger": None,
                "subcommand": None,
                "vision": False,
                # Debug logic: start with UserValve, but allow CLI override later
                "debug": user_valves.debug,
                "enhanced_prompt": user_valves.enhanced_prompt,
                "quality_audit": user_valves.quality_audit,
                "persistent_vision_cache": valves.persistent_vision_cache,
                "llm_type": "ollama",
                "api_source": "Global",
                "using_official_url": False,
                # Track keys for validation and override protection
                "captured_keys": [],
                "overridden_keys": [],
                # Intent Tracking Flags
                "_explicit_engine": False,
                "_explicit_model": False,
            }
        )

        self.valves = valves
        self.user_valves = user_valves

        self.performance_stats = []
        self.validation_errors = []
        self.validation_warnings = []
        self.execution_details = {}

        self.cumulative_tokens = 0
        self.cumulative_elapsed_time = 0.0
        self.start_time = time.time()
        self.image_gen_time_int = 0
        self.quality_audit_results = {"score": None, "critique": None, "emoji": "⚪"}
        self.vision_cache = {}
        self.output_content = ""
        self.executed = False

    def register_stat(self, stage_name: str, elapsed: float, token_count: int = 0):
        """
        Records performance metrics for a specific processing stage.
        """

        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tps = token_count / elapsed if elapsed > 0 else 0

        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tps:.1f} tk/s"
        )


# --- SERVICES ---


class EmitterService:
    """
    Handles asynchronous communication with the Open WebUI event emitter.
    """

    def __init__(self, event_emitter, ctx):
        self.emitter = event_emitter
        self.ctx = ctx

    async def emit_message(self, content: str):
        if self.emitter:
            if self.ctx.st.model.debug:
                self.ctx.debug.log(f"[EMIT] Message: {content[:30]}...")

            fmt = (
                content.replace("```", "```\n") if content.endswith("```") else content
            )

            if "```" in content:
                fmt = re.sub(r"(```[^\n]*\n.*?\n```)", r"\1\n", fmt, flags=re.DOTALL)

            await self.emitter({"type": "message", "data": {"content": fmt}})

    async def emit_status(self, description: str, done: bool = False):
        if self.emitter:
            if self.ctx.st.model.debug:
                self.ctx.debug.log(f"[EMIT] Status: {description} (Done: {done})")

            await self.emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def emit_citation(self, name: str, document: str, source: str, cid: str):
        if self.emitter:
            if self.ctx.st.model.debug:
                self.ctx.debug.log(f"[EMIT] Citation {cid}: {name}")

            await self.emitter(
                {
                    "type": "citation",
                    "data": {
                        "source": {"name": name},
                        "document": [document],
                        "metadata": [{"source": source, "id": cid}],
                    },
                }
            )


class DebugService:
    """
    Handles logging to stderr and strictly controlled object dumping.
    Updated to respect dynamic CLI overrides for debugging.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def log(self, msg: str, is_error: bool = False):
        st_debug = (
            self.ctx.st.model.debug if self.ctx.st else self.ctx.user_valves.debug
        )

        if st_debug or is_error:
            # Calculate time delta from start
            delta = 0.0
            if self.ctx.st and self.ctx.st.start_time:
                delta = time.time() - self.ctx.st.start_time

            prefix = f"[{delta:+.2f}s]"

            if is_error:
                print(
                    f"\n\n❌ {prefix} EASYMAGE ERROR: {msg}\n",
                    file=sys.stderr,
                    flush=True,
                )

            else:
                print(f"⚡ {prefix} EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def error(self, e: Any):
        self.log(str(e), is_error=True)
        await self.ctx.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def dump(self, data: Any = None, label: str = "DUMP"):
        st_debug = (
            self.ctx.st.model.debug if self.ctx.st else self.ctx.user_valves.debug
        )

        if not st_debug:
            return

        header = "—" * 60 + f"\n📦 EASYMAGE {label}:\n" + "—" * 60
        print(header, file=sys.stderr, flush=True)

        def universal_resolver(obj):
            if hasattr(obj, "value"):
                return obj.value

            if hasattr(obj, "model_dump"):
                return obj.model_dump()

            return str(obj)

        try:
            clean_json = json.dumps(data, indent=2, default=universal_resolver)
            print(clean_json, file=sys.stderr, flush=True)

        except Exception as e:
            print(f"❌ DUMP ERROR: {str(e)}", file=sys.stderr, flush=True)

        print("—" * 60, file=sys.stderr, flush=True)


class NetworkService:
    """
    Centralized HTTP wrapper for handling requests, error logging, and payload dumping.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    async def post(
        self, url: str, payload: dict, headers: dict = None, timeout: int = 60
    ) -> httpx.Response:
        try:
            r = await HTTP_CLIENT.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
            return r

        except httpx.HTTPStatusError as e:
            # CRITICAL FIX: Always dump error body on HTTP failure, regardless of debug mode.
            # This is essential to see OpenAI/Gemini specific error messages (e.g. Safety, Length).
            print(
                f"\n❌ HTTP ERROR {e.response.status_code} | URL: {url}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"📄 RESPONSE BODY: {e.response.text}\n",
                file=sys.stderr,
                flush=True,
            )
            raise e

        except Exception as e:
            if not self.ctx.st.model.debug:
                print(
                    f"\n❌ NETWORK ERROR: {str(e)} | URL: {url}",
                    file=sys.stderr,
                    flush=True,
                )
            raise e


class InferenceEngine:
    """
    Abstracts LLM interactions for both Text and Vision tasks.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    async def _infer(
        self, task: Optional[str], data: Dict[str, Any], creative_mode: bool = False
    ) -> str:
        start = time.time()

        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[INFER] Starting Task: {task}")

        if task:
            await self.ctx.em.emit_status(f"{task}..")

        inference_data = {
            "system_prompt": None,
            "user_prompt": None,
            "image_b64": None,
        }

        if sys_c := data.get("system"):
            inference_data["system_prompt"] = re.sub(r"\s+", " ", sys_c).strip()

        user_c = data.get("user")

        if isinstance(user_c, dict):
            if i_url := user_c.get("image_url"):
                inference_data["image_b64"] = i_url.split(",")[-1]

            if t_c := user_c.get("text"):
                inference_data["user_prompt"] = re.sub(r"\s+", " ", t_c).strip()

        else:
            inference_data["user_prompt"] = re.sub(r"\s+", " ", str(user_c)).strip()

        if creative_mode:
            temperature = 0.7
            s_val = (
                int(self.ctx.st.model.seed)
                if self.ctx.st.model.seed is not None
                else -1
            )
            seed = None if s_val == -1 else s_val

        else:
            seed, temperature = 42, 0.0

        backend_type = getattr(self.ctx.st.model, "llm_type", "ollama")

        try:
            if self.ctx.st.model.debug:
                self.ctx.debug.log(
                    f"[INFER] Dispatching to Backend: {backend_type} (Seed: {seed})"
                )

            if backend_type == "ollama":
                content, usage = await self._infer_ollama(
                    inference_data, seed, temperature
                )

            else:
                content, usage = await self._infer_openai(
                    inference_data, seed, temperature
                )

            content = content.split("</think>")[-1].strip().strip('"').strip()

            if task:
                self.ctx.st.register_stat(task, time.time() - start, usage)

            if self.ctx.st.model.debug:
                self.ctx.debug.log(f"[INFER] Completed: {task}. Usage: {usage}")

            return content

        except Exception as e:
            await self.ctx.debug.error(f"INFERENCE ERROR ({task}): {e}")
            return ""

    async def _infer_ollama(
        self, inference_data: Dict, seed: Optional[int], temperature: float
    ) -> Tuple[str, int]:
        """
        Constructs and sends the payload to an Ollama-compatible API.
        """
        base_urls = getattr(self.ctx.request.app.state.config, "OLLAMA_BASE_URLS", [])

        if not base_urls:
            raise Exception("Ollama Base URL is missing from Open WebUI configuration.")

        base_url = base_urls[0].rstrip("/")
        messages = []

        if sys_prompt := inference_data.get("system_prompt"):
            messages.append({"role": "system", "content": sys_prompt})

        user_message = {
            "role": "user",
            "content": inference_data.get("user_prompt", ""),
        }

        if image_data := inference_data.get("image_b64"):
            user_message["images"] = [image_data]

        messages.append(user_message)

        options = {
            "temperature": temperature,
            "mirostat": 0,
        }

        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": self.ctx.st.model.id,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        r = await self.ctx.net.post(
            f"{base_url}/api/chat",
            payload=payload,
            timeout=60,
        )

        res = r.json()

        return (
            res.get("message", {}).get("content", ""),
            res.get("eval_count", 0),
        )

    async def _infer_openai(
        self, inference_data: Dict, seed: Optional[int], temperature: float
    ) -> Tuple[str, int]:
        """
        Constructs and sends the payload to an OpenAI-compatible API.
        """
        conf = self.ctx.request.app.state.config
        base_urls = getattr(conf, "OPENAI_API_BASE_URLS", [])

        if not base_urls:
            raise Exception("OpenAI Base URL is missing from Open WebUI configuration.")

        base_url = base_urls[0].rstrip("/")
        api_keys = getattr(conf, "OPENAI_API_KEYS", [])
        api_key = api_keys[0] if api_keys else ""

        messages = []

        if sys_prompt := inference_data.get("system_prompt"):
            messages.append({"role": "system", "content": sys_prompt})

        content_parts = []

        if user_prompt := inference_data.get("user_prompt"):
            content_parts.append({"type": "text", "text": user_prompt})

        if image_data := inference_data.get("image_b64"):
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            )

        user_content = (
            content_parts if image_data else (inference_data.get("user_prompt") or "")
        )

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.ctx.st.model.id,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }

        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        r = await self.ctx.net.post(
            f"{base_url}/chat/completions",
            payload=payload,
            headers=headers,
            timeout=60,
        )

        res = r.json()

        return (
            res["choices"][0]["message"].get("content", ""),
            res.get("usage", {}).get("completion_tokens", 0),
        )

    async def purge_vram(self, unload_current: bool = False):
        """
        Interrogates Ollama for loaded models and purges them based on Open WebUI config.
        """
        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"[VRAM] Checking purge necessity (Unload Current: {unload_current})"
            )

        if self.ctx.st.model.llm_type != "ollama":
            if self.ctx.st.model.debug:
                self.ctx.debug.log("VRAM Purge skipped (not Ollama backend)")
            return

        base_urls = getattr(self.ctx.request.app.state.config, "OLLAMA_BASE_URLS", [])

        if not base_urls:
            await self.ctx.debug.error("VRAM Purge failed: Ollama Base URL not found.")
            return

        base_url = base_urls[0].rstrip("/")

        try:
            if self.ctx.st.model.debug:
                self.ctx.debug.log("Checking loaded models for VRAM purge...")

            r_ps = await HTTP_CLIENT.get(f"{base_url}/api/ps")
            r_ps.raise_for_status()
            loaded_models = r_ps.json().get("models", [])

            current_model = self.ctx.st.model.id
            unloaded_list = []

            for m in loaded_models:
                m_name = m.get("name")

                if not unload_current and m_name == current_model:
                    continue

                await self.ctx.net.post(
                    f"{base_url}/api/chat",
                    payload={"model": m_name, "keep_alive": 0},
                    timeout=5,
                )

                unloaded_list.append(m_name)

            if unloaded_list:
                self.ctx.debug.log(f"[VRAM] Purge: Unloaded models: {unloaded_list}")

            else:
                if self.ctx.st.model.debug:
                    self.ctx.debug.log("[VRAM] Purge: No models to unload.")

        except Exception as e:
            await self.ctx.debug.error(f"VRAM Purge failed: {str(e)}")


class ImageGenEngine:
    """
    Handles the actual image generation request dispatch with Unified Auth Strategy.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    async def generate(self) -> str:
        """
        Orchestrates the image generation process. Emits the image upon success.
        Returns the generated image URL for reference.
        """
        await self.ctx.em.emit_status("Generating Image..")
        start = time.time()
        eng = self.ctx.st.model.engine
        E = self.ctx.config.Engines

        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[GEN] Starting image generation using engine: {eng}")

        try:
            if eng == E.OPENAI:
                await self._gen_openai()
            elif eng == E.GEMINI:
                await self._gen_gemini()
            elif eng == E.FORGE:
                await self._gen_forge()
            else:
                await self._gen_standard()

            self.ctx.st.image_gen_time_int = int(time.time() - start)

            if self.ctx.st.model.image_url:
                # CRITICAL: Emit the image immediately for the user to see during audit.
                await self.ctx.em.emit_message(
                    f"![Generated Image]({self.ctx.st.model.image_url})"
                )
                return self.ctx.st.model.image_url
            else:
                raise Exception("No image URL returned")

        except Exception as e:
            raise e

    # --- PAYLOAD BUILDERS ---

    def _prepare_forge(self) -> dict:
        """
        Constructs a clean, whitelist-based payload for A1111/Forge.
        Prevents pollution from Open WebUI internal config objects.
        """
        m = self.ctx.st.model

        # Calculate dimensions
        try:
            w, h = map(int, m.get("size", "1024x1024").split("x"))
        except:
            w, h = 1024, 1024

        # Base Payload (Standard Txt2Img)
        payload = {
            "prompt": m.enhanced_prompt,
            "negative_prompt": m.negative_prompt,
            "steps": m.steps,
            "seed": m.seed,
            "sampler_name": m.sampler_name,
            "scheduler": m.scheduler,
            "cfg_scale": m.cfg_scale,
            "width": w,
            "height": h,
            "n_iter": 1,
            "batch_size": 1,
        }

        # Flux / SDXL Specifics
        if m.get("distilled_cfg_scale") is not None:
            payload["distilled_cfg_scale"] = m.distilled_cfg_scale

        # High-Res Fix
        if m.enable_hr:
            payload.update(
                {
                    "enable_hr": True,
                    "hr_scale": m.hr_scale,
                    "hr_upscaler": m.hr_upscaler,
                    "hr_second_pass_steps": m.steps,  # Usually matches base steps
                    "denoising_strength": m.denoising_strength,
                }
            )

            if m.get("hr_distilled_cfg") is not None:
                payload["hr_distilled_cfg"] = m.hr_distilled_cfg

        return payload

    def _prepare_openai(self) -> dict:
        m = self.ctx.st.model
        is_d3 = "dall-e-3" in str(m.model)

        # Truncate prompt: DALL-E 2 limit is 1000 chars, DALL-E 3 is 4000.
        limit = 4000 if is_d3 else 1000
        final_prompt = m.enhanced_prompt

        if len(final_prompt) > limit:
            final_prompt = final_prompt[:limit]

        payload = {
            "model": m.model or "dall-e-3",
            "prompt": final_prompt,
            "n": 1,
            "size": m.size,
            "response_format": "b64_json",
            "user": self.ctx.user.id,
        }

        if is_d3:
            payload["quality"] = "hd" if m.enable_hr else "standard"
            payload["style"] = "natural" if m.style == "natural" else "vivid"

        return payload

    def _prepare_gemini(self, method: str) -> dict:
        """
        Builds the payload for Google Gemini Image Generation based on the detected method.
        """
        m = self.ctx.st.model

        if method == "generateContent":
            # Payload for standard Gemini API (Google AI Studio)
            # FIXED: Implemented correct structure for Gemini 2.5 Flash Image
            # using responseModalities and imageConfig.

            payload = {
                "contents": [{"parts": [{"text": m.enhanced_prompt}]}],
                "generationConfig": {
                    "candidateCount": 1,
                    "responseModalities": ["IMAGE"],  # Explicitly request IMAGE mode
                },
            }

            # Inject Image Configuration if specific AR is requested
            if m.aspect_ratio and m.aspect_ratio != "1:1":
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": m.aspect_ratio
                }

            if m.seed and m.seed != -1:
                # Seed is top-level in generationConfig for this endpoint
                payload["generationConfig"]["seed"] = m.seed

            return payload

        else:
            # Payload for Vertex AI or compatible proxies using :predict (Imagen Legacy)
            try:
                w, h = map(int, (m.size or "1024x1024").split("x"))
                rat = w / h
                ar = (
                    "16:9"
                    if rat > 1.7
                    else (
                        "4:3"
                        if rat > 1.3
                        else "9:16" if rat < 0.6 else "3:4" if rat < 0.8 else "1:1"
                    )
                )

            except:
                ar = "1:1"

            params = {
                "sampleCount": 1,
                "negativePrompt": m.negative_prompt,
                "aspectRatio": ar,
                "safetySetting": "block_none",
                "personGeneration": "allow_all",
                "addWatermark": False,
                "outputOptions": {"mimeType": "image/png"},
            }

            if m.seed and m.seed != -1:
                params["seed"] = m.seed

            return {"instances": [{"prompt": m.enhanced_prompt}], "parameters": params}

    # --- HTTP HANDLERS ---

    async def _gen_forge(self):
        """
        Executes request against Forge/A1111 with Basic Auth support.
        """
        conf = self.ctx.request.app.state.config
        valves = self.ctx.valves
        user_valves = self.ctx.user_valves

        url = conf.AUTOMATIC1111_BASE_URL.rstrip("/")

        # Auth Hierarchy: CLI > UserValve > Valve > Global
        cli_auth = self.ctx.st.model.get("cli_auth")
        user_auth = user_valves.automatic1111_auth
        valve_auth = valves.automatic1111_auth
        global_auth = getattr(conf, "AUTOMATIC1111_API_AUTH", "")

        auth_string = cli_auth or user_auth or valve_auth or global_auth

        if cli_auth:
            self.ctx.st.model.api_source = "CLI Override"

        elif user_auth:
            self.ctx.st.model.api_source = "User Profile"

        elif valve_auth:
            self.ctx.st.model.api_source = "Filter Default"

        else:
            self.ctx.st.model.api_source = "System Settings"

        headers = {}

        if auth_string:
            if ":" in auth_string:
                # Standard User:Password Basic Auth
                headers["Authorization"] = (
                    f"Basic {base64.b64encode(auth_string.encode()).decode()}"
                )

            else:
                # Fallback for Token-based proxies
                headers["Authorization"] = f"Bearer {auth_string}"

        payload = self._prepare_forge()

        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[GEN] FORGE PAYLOAD: {json.dumps(payload, indent=2)}")

        r = await self.ctx.net.post(
            f"{url}/sdapi/v1/txt2img",
            payload=payload,
            headers=headers,
            timeout=valves.generation_timeout,
        )

        res = r.json()

        img = res["images"][0].split(",")[-1]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

    async def _gen_openai(self):
        """
        Executes request against OpenAI with Easy Cloud Mode support.
        """
        conf = self.ctx.request.app.state.config
        valves = self.ctx.valves
        user_valves = self.ctx.user_valves

        if valves.easy_cloud_mode:
            base_url = self.ctx.config.OFFICIAL_URLS["openai"]
            self.ctx.st.model.using_official_url = True

        else:
            base_url = getattr(conf, "IMAGES_OPENAI_API_BASE_URL", "").rstrip("/")

            if not base_url:
                raise Exception(
                    "OpenAI Base URL missing in Global Settings (and Easy Mode is OFF)."
                )

        # Auth Hierarchy: CLI > UserValve > Valve > Global
        cli_auth = self.ctx.st.model.get("cli_auth")
        user_auth = user_valves.openai_auth
        valve_auth = valves.openai_auth
        global_keys = getattr(conf, "IMAGES_OPENAI_API_KEYS", [])
        global_key = global_keys[0] if global_keys else ""

        api_key = cli_auth or user_auth or valve_auth or global_key

        if cli_auth:
            self.ctx.st.model.api_source = "CLI Override"

        elif user_auth:
            self.ctx.st.model.api_source = "User Profile"

        elif valve_auth:
            self.ctx.st.model.api_source = "Filter Default"

        else:
            self.ctx.st.model.api_source = "System Settings"

        if not api_key:
            raise Exception(
                "No OpenAI API Key found (Checked: CLI, User, Valves, Global)."
            )

        # 3. Prepare Payload & Execute
        payload = self._prepare_openai()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Task 4: Debug Log - Payload
        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[GEN] OPENAI PAYLOAD: {json.dumps(payload, indent=2)}")

        r = await self.ctx.net.post(
            f"{base_url}/images/generations",
            payload=payload,
            headers=headers,
            timeout=valves.generation_timeout,
        )

        img = r.json()["data"][0]["b64_json"]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

    async def _gen_gemini(self):
        """
        Executes request against Gemini with Easy Cloud Mode support.
        """
        conf = self.ctx.request.app.state.config
        valves = self.ctx.valves
        user_valves = self.ctx.user_valves

        # 1. Determine Base URL
        if valves.easy_cloud_mode:
            base_url = self.ctx.config.OFFICIAL_URLS["gemini"]
            self.ctx.st.model.using_official_url = True

        else:
            base_url = getattr(conf, "IMAGES_GEMINI_API_BASE_URL", "").rstrip("/")

            if not base_url:
                raise Exception(
                    "Gemini Base URL missing in Global Settings (and Easy Mode is OFF)."
                )

        # Auth Hierarchy: CLI > UserValve > Valve > Global
        cli_auth = self.ctx.st.model.get("cli_auth")
        user_auth = user_valves.gemini_auth
        valve_auth = valves.gemini_auth
        global_key = getattr(conf, "IMAGES_GEMINI_API_KEY", "")

        api_key = cli_auth or user_auth or valve_auth or global_key

        if cli_auth:
            self.ctx.st.model.api_source = "CLI Override"

        elif user_auth:
            self.ctx.st.model.api_source = "User Profile"

        elif valve_auth:
            self.ctx.st.model.api_source = "Filter Default"

        else:
            self.ctx.st.model.api_source = "System Settings"

        if not api_key:
            raise Exception(
                "No Gemini API Key found (Checked: CLI, User, Valves, Global)."
            )

        # 3. Smart Method Detection
        # Gemini 2.5 Flash Image and similar use generateContent.
        # Legacy Imagen models use predict.

        model_name = self.ctx.st.model.model or "gemini-2.0-flash"

        if "imagen" in model_name.lower():
            method = "predict"
        elif "generativelanguage.googleapis.com" in base_url:
            method = "generateContent"
        else:
            # Fallback for Vertex Enterprise endpoints
            method = "predict"

        url = f"{base_url}/models/{model_name}:{method}"

        # 4. Auth Headers
        headers = {"Content-Type": "application/json"}

        if "generativelanguage.googleapis.com" in base_url:
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = self._prepare_gemini(method)

        # Task 4: Debug Log - Payload and URL
        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"GEMINI: {url} | Method: {method} | EasyMode: {valves.easy_cloud_mode}"
            )
            self.ctx.debug.log(f"[GEN] GEMINI PAYLOAD: {json.dumps(payload, indent=2)}")

        r = await self.ctx.net.post(
            url,
            payload=payload,
            headers=headers,
            timeout=valves.generation_timeout,
        )

        res = r.json()
        img_b64 = ""

        try:
            if method == "generateContent":
                # Check for candidates
                candidates = res.get("candidates", [])
                if not candidates:
                    if "promptFeedback" in res:
                        raise Exception(f"Request Blocked: {res['promptFeedback']}")
                    raise Exception("No candidates returned from Gemini.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise Exception("Empty content parts in Gemini response.")

                # Iterate ALL parts to find the image (inlineData).
                for part in parts:
                    if "inlineData" in part:
                        img_b64 = part["inlineData"]["data"]
                        break

                # If no image found after scanning all parts, raise error
                if not img_b64:
                    text_content = "No text content"
                    for part in parts:
                        if "text" in part:
                            text_content = part["text"]
                            break
                    raise Exception(
                        f"Model response did not contain an image. Output: {text_content}"
                    )

            else:
                # :predict endpoint (Imagen)
                if "predictions" in res:
                    img_b64 = res["predictions"][0]["bytesBase64Encoded"]
                else:
                    raise Exception(f"Imagen API Error: {json.dumps(res)}")

        except Exception as e:
            if self.ctx.st.model.debug:
                self.ctx.debug.log(
                    f"[GEN] GEMINI RAW ERROR RESPONSE: {json.dumps(res, indent=2)}"
                )
            raise Exception(f"Gemini API Error ({method}): {str(e)}")

        self.ctx.st.model.b64_data = img_b64
        self.ctx.st.model.image_url = f"data:image/png;base64,{img_b64}"

    async def _gen_standard(self):
        """
        Falls back to the standard Open WebUI image generation router.
        """
        if self.ctx.st.model.debug:
            self.ctx.debug.log("[GEN] Using Standard OpenWebUI Router")

        form = CreateImageForm(
            prompt=self.ctx.st.model.enhanced_prompt,
            n=1,
            size=self.ctx.st.model.size,
            model=self.ctx.st.model.model,
        )

        gen = await image_generations(self.ctx.request, form, self.ctx.user)

        if gen:
            self.ctx.st.model.image_url = gen[0]["url"]


class PromptParser:
    """
    Parses user input using Regex to extract command flags and key-value pairs.
    Directly modifies the EasymageState model.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.config = ctx.config
        # Map CLI short keys to Model attributes for Override Protection
        self.cli_map = {
            "sz": "size",
            "stp": "steps",
            "ge": "engine",
            "mdl": "model",
            "sd": "seed",
            "sch": "scheduler",
            "smp": "sampler_name",
            "ar": "aspect_ratio",
            "cs": "cfg_scale",
            "dns": "denoising_strength",
            "dcs": "distilled_cfg_scale",
            "hdcs": "hr_distilled_cfg",
            "hr": "hr_scale",
            "hru": "hr_upscaler",
        }

    def parse(self, user_prompt: str):
        """
        Analyzes the raw user string.
        Extracts parameters including the new 'auth=' key and expands model shortcuts.
        """
        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[PARSER] Input: {user_prompt[:50]}...")

        m_st = self.ctx.st.model
        clean, neg = user_prompt, ""

        # FIX: Robust negative prompt split handling both '--no' and em-dash '—no'
        # Handles case-insensitive splitting.
        split_match = re.search(r"\s+(?:--|—)no\s+", clean, re.IGNORECASE)
        if split_match:
            clean = user_prompt[: split_match.start()]
            neg = user_prompt[split_match.end() :]

        if " -- " in clean:
            prefix, subj = clean.split(" -- ", 1)

        else:
            pat = r'(\b\w+=(?:"[^"]*"|\S+))|(?<!\w)([+-][dpah])(?!\w)'
            matches = list(re.finditer(pat, clean))
            idx = matches[-1].end() if matches else 0
            prefix, subj = clean[:idx], clean[idx:].strip()

        rem_styles = re.sub(
            r'(\b\w+=(?:"[^"]*"|\S+))|(?<!\w)([+-][dpah])(?!\w)', "", prefix
        ).strip()

        rem_styles = re.sub(r"^[\s,]+|[\s,]+$", "", rem_styles)

        kv_pat = r'(\b\w+)=("([^"]*)"|(\S+))'

        for k, _, q, u in re.findall(kv_pat, prefix):
            k, v = k.lower(), (q if q else u)  # Preserve case for API keys

            # CRITICAL FIX: Track keys for validation
            if k not in m_st["captured_keys"]:
                m_st["captured_keys"].append(k)

            # CRITICAL FIX: Track keys for CLI Override Protection
            if k in self.cli_map:
                m_st["overridden_keys"].append(self.cli_map[k])

            if self.ctx.st.model.debug:
                self.ctx.debug.log(f"[PARSER] Found key: {k} = {v}")

            try:
                if k == "en":
                    m_st["engine"] = {
                        "a": self.config.Engines.FORGE,
                        "o": self.config.Engines.OPENAI,
                        "g": self.config.Engines.GEMINI,
                        "c": self.config.Engines.COMFY,
                    }.get(v.lower(), v.lower())

                    m_st["_explicit_engine"] = True
                    # Also mark explicit flags as overridden
                    if "engine" not in m_st["overridden_keys"]:
                        m_st["overridden_keys"].append("engine")

                elif k in ["mdl", "mod", "m"]:
                    v_low = v.lower()
                    m_st["model"] = self.config.MODEL_SHORTCUTS.get(v_low, v_low)

                    m_st["_explicit_model"] = True

                elif k == "auth":
                    m_st["cli_auth"] = v

                elif k == "stp":
                    m_st["steps"] = int(v)

                elif k == "sz":
                    m_st["size"] = v if "x" in str(v) else f"{v}x{v}"

                elif k == "ar":
                    m_st["aspect_ratio"] = {
                        "1": "1:1",
                        "16": "16:9",
                        "9": "9:16",
                        "4": "4:3",
                        "3": "3:4",
                        "21": "21:9",
                    }.get(str(v), str(v))

                elif k == "stl":
                    m_st["style"] = {"v": "vivid", "n": "natural"}.get(
                        v.lower(), v.lower()
                    )

                elif k == "sd":
                    m_st["seed"] = int(v)

                elif k == "smp":
                    m_st["sampler_name"] = self._norm(v, self.config.SAMPLER_MAP)

                elif k == "sch":
                    m_st["scheduler"] = self._norm(v, self.config.SCHEDULER_MAP)

                elif k == "cs":
                    m_st["cfg_scale"] = float(v)

                elif k == "dcs":
                    m_st["distilled_cfg_scale"], m_st["cfg_scale"] = float(v), 1.0

                elif k in ["hr", "hru", "hdcs", "dns"]:
                    m_st["enable_hr"] = True

                    if k == "hr":
                        m_st["hr_scale"] = float(v)

                    elif k == "hru":
                        m_st["hr_upscaler"] = v

                    elif k == "hdcs":
                        m_st["hr_distilled_cfg"] = float(v)

                    elif k == "dns":
                        m_st["denoising_strength"] = float(v)

            except ValueError:
                continue

        for flag in re.findall(r"(?<!\w)([+-][dpah])(?!\w)", prefix):
            v_bool, char = flag[0] == "+", flag[1]

            if char == "d":
                m_st["debug"] = v_bool
                m_st["overridden_keys"].append("debug")

            elif char == "p":
                m_st["enhanced_prompt"] = v_bool
                m_st["overridden_keys"].append("enhanced_prompt")

            elif char == "a":
                m_st["quality_audit"] = v_bool
                m_st["overridden_keys"].append("quality_audit")

            elif char == "h":
                m_st["enable_hr"] = v_bool

        m_st.update(
            {
                "user_prompt": subj.strip(),
                "negative_prompt": neg.strip(),
                "styles": rem_styles,
            }
        )

        if self.ctx.st.model.debug:
            self.ctx.debug.log(f"[PARSER] Overridden Keys: {m_st['overridden_keys']}")

    def _norm(self, name, mapping):
        """
        Normalizes loose input strings to exact mapping keys.
        """

        n = (
            name.lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("++", "")
        )

        return mapping.get(
            n, name.capitalize() if mapping is self.config.SCHEDULER_MAP else name
        )


# --- FILTER ORCHESTRATOR ---


class Filter:
    """
    Main entry point for the Open WebUI Filter system.
    Orchestrates the entire flow: Input interception -> LLM Silence -> Logic Execution -> Output Injection.
    """

    class Valves(BaseModel):
        """
        Global filter settings managed by the Administrator.
        """

        # Admin controls and Defaults
        easy_cloud_mode: bool = Field(
            default=True,
            description="If True, ignores global URLs for Cloud Models (OpenAI/Gemini) and uses official endpoints.",
        )

        extreme_vram_cleanup: bool = Field(
            default=False,
            description="If True, unloads the current LLM as well. If False (default), only unloads other idle models.",
        )

        persistent_vision_cache: bool = Field(
            default=False,
            description="Save vision capability test results to a local JSON file to speed up startup.",
        )

        generation_timeout: int = Field(
            default=120,
            description="Maximum time in seconds allowed for the image generation API request.",
        )

        # Default Auth Keys (Optional Fallback for Users)
        openai_auth: str = Field(
            default="",
            description="Global Default API Key for OpenAI.",
        )

        gemini_auth: str = Field(
            default="",
            description="Global Default API Key for Google Gemini.",
        )

        automatic1111_auth: str = Field(
            default="",
            description="Global Default Auth for Forge (user:password).",
        )

    class UserValves(BaseModel):
        """
        User-specific settings available in the 'Controls' menu.
        These override the global Valves.
        """

        # Personal Auth Overrides
        openai_auth: str = Field(
            default="",
            description="Personal API Key for OpenAI. Overrides Global Default.",
        )

        gemini_auth: str = Field(
            default="",
            description="Personal API Key for Google Gemini. Overrides Global Default.",
        )

        automatic1111_auth: str = Field(
            default="",
            description="Personal Auth for Forge (user:password). Overrides Global Default.",
        )

        # Personal Preferences
        enhanced_prompt: bool = Field(
            default=True,
            description="Enable LLM-based prompt expansion.",
        )

        quality_audit: bool = Field(
            default=True,
            description="Perform a visual quality control (VQC).",
        )

        strict_audit: bool = Field(
            default=False,
            description="Enable severe scoring penalties for hallucinations.",
        )

        model: Optional[str] = Field(
            default=None,
            description="The specific model checkpoint to use.",
        )

        steps: int = Field(
            default=20,
            description="Number of sampling steps.",
        )

        size: str = Field(
            default="1024x1024",
            description="Default image resolution. Format: Width x Height.",
        )

        aspect_ratio: str = Field(
            default="1:1",
            description="Target aspect ratio.",
        )

        seed: int = Field(
            default=-1,
            description="Deterministic seed (-1 for random).",
        )

        cfg_scale: float = Field(
            default=1.0,
            description="Classifier Free Guidance.",
        )

        distilled_cfg_scale: float = Field(
            default=3.5,
            description="CFG scale for distilled models (Flux).",
        )

        sampler_name: str = Field(
            default="Euler",
            description="Sampling algorithm.",
        )

        scheduler: str = Field(
            default="Simple",
            description="Noise scheduler.",
        )

        enable_hr: bool = Field(
            default=False,
            description="Enable High-Resolution Fix pass.",
        )

        hr_scale: float = Field(
            default=2.0,
            description="Multiplier for the final resolution.",
        )

        hr_upscaler: str = Field(
            default="Latent",
            description="The algorithm used for upscaling.",
        )

        hr_distilled_cfg: float = Field(
            default=3.5,
            description="Specific CFG scale applied during the upscaling pass.",
        )

        denoising_strength: float = Field(
            default=0.45,
            description="Controls how much the upscaler changes the original image.",
        )

        debug: bool = Field(
            default=False,
            description="Enable verbose logging and surgical object dumping in the server console/logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.config = EasymageConfig()

        # Explicit type hinting for Pyright: st can be EasymageState or None
        self.st: EasymageState = None  # type: ignore

    def _unwrap(self, obj):
        """
        Recursively unwraps Open WebUI configuration objects.
        """

        if hasattr(obj, "value"):
            return self._unwrap(obj.value)

        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}

        return obj

    def _clean_key(self, key, eng):
        """
        Normalizes environment variable keys.
        """

        k = (
            key.upper()
            .replace("IMAGE_GENERATION_", "")
            .replace("IMAGE_", "")
            .replace("ENABLE_IMAGE_", "ENABLE_")
        )

        return (
            k.replace(f"{eng.upper()}_API_", "")
            .replace(f"{eng.upper()}_", "")
            .replace("IMAGES_", "")
            .lower()
        )

    def _validate_request(self):
        """
        Task 2: Validates the request consistency and engine compatibility.
        """
        if self.st.model.debug:
            self.debug.log("[STEP 3/10] Validating Request...")

        m = self.st.model
        conf = self.config
        errors = self.st.validation_errors
        warnings = self.st.validation_warnings

        # 1. Blocking Errors

        # Whitelist Validation (Unknown Parameters)
        allowed_keys = [
            "en",
            "mdl",
            "mod",
            "m",
            "auth",
            "stp",
            "sz",
            "ar",
            "stl",
            "sd",
            "smp",
            "sch",
            "cs",
            "dcs",
            "hr",
            "hru",
            "hdcs",
            "dns",
        ]

        for k in m.get("captured_keys", []):
            if k not in allowed_keys:
                errors.append(f"Unknown parameter: '{k}'")

        # Empty Subject check
        # MODIFIED: Allow empty subject if subcommand is 'r' (Random)
        is_random_mode = m.subcommand == "r"
        if (not m.user_prompt or len(m.user_prompt.strip()) < 2) and not is_random_mode:
            errors.append("Validation Error: No subject description provided.")

        # Engine/Model Incompatibility (Explicit Override Conflict)
        if m._explicit_engine and m._explicit_model:
            detected_engine = conf.MODEL_ENGINE_MAP.get(str(m.model).lower())

            if detected_engine and detected_engine != m.engine:
                errors.append(
                    f"Validation Error: Engine '{m.engine}' implies specific models, but incompatible model '{m.model}' was requested."
                )

        # 2. Warnings

        # High-Res Fix on unsupported engines
        if m.enable_hr and m.engine == conf.Engines.GEMINI:
            warnings.append("High-Res Fix (+h) is not supported on Gemini. Ignored.")

        if self.st.model.debug:
            if errors:
                self.debug.log(f"Validation FAILED with errors: {errors}")
            else:
                self.debug.log("Validation PASSED.")

    def _mask_key(self, key: str) -> str:
        """
        Helper to mask API keys and Auth strings for display.
        """
        if not key:
            return "None"

        if ":" in key:
            # Handle User:Pass format
            try:
                u, p = key.split(":", 1)
                return f"{u[:2]}...:{p[:2]}..."
            except:
                return "****"

        # Handle Standard API Keys
        if len(key) < 8:
            return "****"

        return f"{key[:4]}...{key[-4:]}"

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:
        """
        Intercepts the incoming request before it reaches the LLM.
        """
        trigger_data = self._check_input(body)

        if not trigger_data:
            self.st = None
            return body

        # Unpack trigger data: trigger='img', raw_p='prompt...', subcommand='p'|'r'|'?'|None
        trigger, raw_p, subcommand = trigger_data

        # 1. IMMEDIATE STATE CREATION
        self.request = __request__
        self.user = UserModel(**__user__)

        # POPULATION OF USERVALVES
        # Check if valves data is already an instance or a dictionary to prevent unpacking errors.
        if __user__ and "valves" in __user__:
            uv_data = __user__["valves"]

            if isinstance(uv_data, dict):
                # If it's a raw dictionary, instantiate the model
                self.user_valves = self.UserValves(**uv_data)

            else:
                # If it's already an instance of UserValves (Pydantic model), assign directly
                self.user_valves = uv_data

        self.st = EasymageState(self.valves, self.user_valves)
        self.st.model.trigger = trigger
        self.st.model.subcommand = subcommand
        self.em = EmitterService(__event_emitter__, self)

        if self.st.model.debug:
            self.debug = DebugService(self)
            self.debug.log(
                f"[STEP 1/10] Inlet triggered. Trigger type: {trigger} | Sub: {subcommand}"
            )

        # --- FAST EXIT: HELP SYSTEM (SILENT MODE) ---
        # Immediate short-circuit if subcommand is '?' or 'help'.
        # We DO NOT emit anything here to avoid flickering. 
        # Visualization is fully deferred to the outlet.
        if self.st.model.subcommand in ["?", "help"]:
            return self._suppress_output(body)
        # ---------------------------------------------

        try:
            metadata = body.get("metadata", {})
            self.st.model.llm_type = metadata.get("model", {}).get("owned_by", "ollama")

        except:
            self.st.model.llm_type = "ollama"

        # 2. IMMEDIATE SUPPRESSION (For standard generation)
        body = self._suppress_output(body)

        # 3. WORKER INITIALIZATION
        self.debug = DebugService(self)
        self.net = NetworkService(self)
        self.inf = InferenceEngine(self)
        self.img_gen = ImageGenEngine(self)
        self.parser = PromptParser(self)

        try:

            # PHASE 1: Immediate Parsing (Technical Extraction)
            if self.st.model.debug:
                self.debug.log("[STEP 2/10] Parsing Input")
            self.parser.parse(raw_p)

            # PHASE 2: Immediate Validation (Fail-Fast)
            self._validate_request()

            if self.st.validation_errors:
                # Task 2: Blocking Errors - Abort generation
                error_msg = "‼️GENERATION BLOCKED:\n" + "\n".join(
                    [f"→ {e}" for e in self.st.validation_errors]
                )
                self.st.output_content = error_msg
                self.st.executed = True

                await self.em.emit_status("Validation Failed", True)
                await self.em.emit_message(error_msg)

                if self.st.model.debug:
                    self.debug.log("Generation aborted due to validation errors.")
                return body

            # PHASE 3: Heavy LLM Logic (Setup Context)
            # Only proceeds if validation passed.
            if self.st.model.debug:
                self.debug.log("[STEP 4/10] Setting Up Context")
            await self._setup_context(body, raw_p)

            # --- EARLY DUMP EMISSION (Sanitized) ---
            if self.st.model.debug:
                self.debug.log("[STEP 4b/10] Emitting Early Debug Dumps")

                # Helper to mask sensitive keys locally
                def _sanitize(data: dict):
                    safe = data.copy()
                    for k, v in safe.items():
                        # Mask if key contains 'auth', 'key' or is specifically 'cli_auth'
                        if isinstance(v, str) and (
                            "auth" in k.lower() or "key" in k.lower()
                        ):
                            safe[k] = self._mask_key(v)
                    return safe

                # Create safe copies for display
                safe_model = _sanitize(self.st.model)
                safe_valves = _sanitize(self.user_valves.model_dump())

                # Prepare formatted JSON blocks
                dump_model = json.dumps(safe_model, indent=2, default=str)
                dump_valves = json.dumps(safe_valves, indent=2, default=str)

                debug_block = (
                    f"\n\n<details>\n<summary>🔍 Debug STATE</summary>\n\n"
                    f"```json\n{dump_model}\n```\n"
                    f"</details>\n"
                    f"<details>\n<summary>🔍 Debug VALVES</summary>\n\n"
                    f"```json\n{dump_valves}\n```\n"
                    f"</details>\n"
                )

                # Append to buffer AND Emit Immediately to stream
                self.st.output_content += debug_block
                await self.em.emit_message(debug_block)

            # --- SMART ENGINE RECONCILIATION ---

            m = self.st.model
            detected_engine = self.config.MODEL_ENGINE_MAP.get(str(m.model).lower())

            if self.st.model.debug:
                self.debug.log(
                    f"Reconciliation -> Explicit Engine: {m._explicit_engine}, Explicit Model: {m._explicit_model}, Current Engine: {m.engine}, Detected from Model: {detected_engine}"
                )

            if m._explicit_model and detected_engine:
                m.engine = detected_engine

            elif m._explicit_engine and not m._explicit_model:
                if detected_engine and detected_engine != m.engine:
                    self.debug.log(
                        f"Conflict: Engine {m.engine} incompatible with model {m.model}. Resetting model."
                    )
                    m.model = None

            elif self.valves.easy_cloud_mode and detected_engine:
                m.engine = detected_engine

            # LOGIC SWITCH: Use subcommand "p" (Prompt Only)
            if self.st.model.subcommand == "p":
                self.st.output_content = (
                    self.st.model.enhanced_prompt + self.st.output_content
                )
                self.st.executed = True

            else:
                # STANDARD GENERATION OR RANDOM (r)

                # VRAM CLEANUP LOGIC
                E = self.config.Engines
                is_cloud_engine = self.st.model.engine in [E.OPENAI, E.GEMINI]

                if not is_cloud_engine:
                    if self.st.model.debug:
                        self.debug.log("[STEP 5/10] Cleaning VRAM")
                    await self.em.emit_status("Cleaning VRAM..")

                    await self.inf.purge_vram(
                        unload_current=self.valves.extreme_vram_cleanup
                    )
                elif self.st.model.debug:
                    self.debug.log("[STEP 5/10] VRAM Purge Skipped (Cloud Engine)")

                # Full Generation Mode
                if self.st.model.debug:
                    self.debug.log("[STEP 6/10] Generating Image")
                await self.img_gen.generate()

                # Initialize content buffer (Image Markdown only)
                img_md = (
                    f"![Generated Image]({self.st.model.image_url})"
                    if self.st.model.image_url
                    else ""
                )

                # Prepend image to existing content
                if self.st.output_content:
                    self.st.output_content = (
                        self.st.output_content.strip() + "\n" + img_md
                    )
                else:
                    self.st.output_content = img_md

                if self.st.model.quality_audit:
                    # User sees only image + status bar here. No placeholders yet.
                    if self.st.model.debug:
                        self.debug.log("[STEP 7/10] Vision Audit")
                    await self._vision_audit()

                # Emit placeholders NOW to the live stream.
                if self.st.model.debug:
                    self.debug.log("[STEP 8/10] Syncing Placeholders")

                await self.em.emit_message("\n\n[1] [2] [3]")

                # Update the persistent buffer for DB save
                self.st.output_content += "\n\n[1] [2] [3]"

                self.st.executed = True

                if self.st.model.debug:
                    self.debug.log("[STEP 9/10] Inlet Finished")

        except Exception as e:
            await self.em.emit_status("Execution Aborted!", True)
            await self.debug.error(e)
            if self.st:
                self.st.executed = False

        return body


    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        """
        Intercepts the LLM's response.
        """
        if getattr(self, "st", None) is not None and self.st.model.debug:
            self.debug.log(f"[STEP 10/10] Outlet Triggered")

        if getattr(self, "st", None) is None or not self.st.executed:
            # Catch-all for when inlet exited early without creating full state (except help)
            # If st exists but not executed, we check if it was a help command handled in inlet?
            # actually, help is handled here in outlet now.
            
            # Re-check logic: if inlet returned early for help, st exists but executed is False?
            # Wait, if inlet returns early for help, st is created.
            pass

        # Check if the process was actually triggered (avoid generic non-img messages)
        # Note: 'st' might be None if check_input failed.
        if not getattr(self, "st", None) or not self.st.model.trigger:
             return body

        if __event_emitter__:
            self.em.emitter = __event_emitter__

        # FIX: Immediate handling for Help/Info commands in Outlet
        # This is where we execute the logic to avoid flickering in Inlet
        if self.st.model.subcommand in ["?", "help"]:
            await self._handle_help()
            # Inject the prepared help content into the message body
            if "messages" in body and len(body["messages"]) > 0:
                body["messages"][-1]["content"] = self.st.output_content
            return body

        # FINAL CONTENT INJECTION (Standard Flow)
        if "messages" in body and len(body["messages"]) > 0:
            body["messages"][-1]["content"] = self.st.output_content

        # CRITICAL FIX: If validation failed, skip citations and show only status
        # LOGIC SWITCH: Check subcommand "p" instead of "imgx"
        if self.st.validation_errors or self.st.model.subcommand == "p":
            await self._output_status_only()
            return body

        # Standard delivery logic
        await self._output_delivery()

        return body

    def _suppress_output(self, body: dict) -> dict:
        """
        Force the LLM to be completely silent using invisible characters.
        REVERTED to robust implementation with stop tokens.
        """

        body["messages"] = [
            {
                "role": "system",
                "content": "Respond only with ' ' (one space). No text.",
            },
            {"role": "user", "content": "\u200b"},
        ]

        body["max_tokens"] = 1
        body["stream"] = True
        body["stop"] = [chr(i) for i in range(128)]

        if self.st and self.st.model.llm_type == "ollama":
            if "options" not in body:
                body["options"] = {}

            body["options"].update(
                {
                    "num_predict": 1,
                    "temperature": 0.0,
                }
            )

        return body

    async def _setup_context(self, body: dict, user_prompt: str):
        """
        Aggregates configuration from Global Settings, Valves, and User Input.
        """

        if self.st.model.debug:
            self.debug.log("Context: Merging Configurations...")

        if "features" in body:
            body["features"]["web_search"] = False

        self.st.model.id = body.get("model", "")

        # 1. Apply Global Settings (Env Vars)
        self._apply_global_settings()

        # 2. Merge User Valves - CRITICAL FIX
        # Do NOT overwrite keys that have been explicitly set via CLI (overridden_keys)
        protected = self.st.model.overridden_keys
        if self.st.model.debug:
            self.debug.log(
                f"Context: Protecting CLI keys from UserValve Overwrite: {protected}"
            )

        uv_dump = self.user_valves.model_dump()
        for k, v in uv_dump.items():
            # Skip if key is protected (e.g., 'size' set by sz=256)
            if k in protected:
                continue
            # Skip auth keys (handled at generation time)
            if k.endswith("_auth"):
                continue

            if v is not None:
                self.st.model[k] = v

        # CHECK VISION: Only if trigger is img and NOT subcommand p (prompt enhancer doesn't need vision)
        if self.st.model.trigger == "img" and self.st.model.subcommand != "p":
            await self._check_vision_capability()

        # 2. Language Detect (Deterministic)
        dl = await self.inf._infer(
            task="Language Detection",
            data={
                "user": self.st.model.user_prompt,
                "system": "Return ONLY the language name of the user text.",
            },
            creative_mode=False,
        )

        self.st.model.language = dl if dl else "English"

        # 3. Prompt Enhancement / Generation Logic
        E = self.config.Engines

        # LOGIC BRANCH: Random Mode (img:r)
        if self.st.model.subcommand == "r":
            if self.st.model.debug:
                self.debug.log("[CTX] Entering Random Mode Generation")

            # Build Style Instructions (Positive & Negative)
            style_parts = []

            if self.st.model.user_prompt and len(self.st.model.user_prompt.strip()) > 1:
                style_parts.append(
                    f"MANDATORY STYLE/THEME: {self.st.model.user_prompt}"
                )

            # FIX: Explicitly forbid negative elements in the generation phase
            if self.st.model.negative_prompt:
                style_parts.append(
                    f"ABSOLUTE PROHIBITION: Do NOT include, mention, or describe: {self.st.model.negative_prompt}."
                )

            style_instruction = "\n".join(style_parts)

            # Generate the random prompt
            rand_prompt = await self.inf._infer(
                task="Randomizing Prompt",
                data={
                    "system": self.config.PROMPT_RANDOM_GEN.format(
                        style_instruction=style_instruction
                    ),
                    "user": "Generate a random image prompt now.",
                },
                creative_mode=True,  # High temperature for variance
            )

            self.st.model.enhanced_prompt = (
                re.sub(r"[\"]", "", rand_prompt)
                if rand_prompt
                else "A random abstract masterpiece."
            )

            # Overwrite the user prompt for display purposes in citations
            self.st.model.user_prompt = (
                f"[Random] {self.st.model.enhanced_prompt[:50]}..."
            )

        # LOGIC BRANCH: Standard Enhancement (img / img:p)
        else:
            native_support = (self.st.model.engine == E.GEMINI) or (
                self.st.model.engine == E.FORGE and self.st.model.enable_hr
            )
            use_llm_neg = (self.st.model.subcommand == "p") or (not native_support)

            if self.st.model.enhanced_prompt or self.st.model.subcommand == "p":
                instructions = [
                    self.config.PROMPT_ENHANCE_BASE.format(lang=self.st.model.language),
                    self.config.PROMPT_ENHANCE_RULES,
                ]

                is_d3 = "dall-e-3" in str(self.st.model.model)

                if self.st.model.engine == E.OPENAI and not is_d3:
                    instructions.append(
                        "RULE: The output MUST be strictly under 950 characters. Do not be verbose."
                    )

                user_content = f"EXPAND THIS PROMPT: {self.st.model.user_prompt}"

                if self.st.model.styles:
                    user_content += f"\nAPPLY THESE STYLES: {self.st.model.styles}"

                if self.st.model.negative_prompt and use_llm_neg:
                    user_content += f"\nAVOID THESE ELEMENTS AT ALL COSTS: {self.st.model.negative_prompt}"

                    instructions.append(
                        self.config.PROMPT_ENHANCE_NEG.format(
                            negative=self.st.model.negative_prompt
                        )
                    )

                enh = await self.inf._infer(
                    task="Prompt Enhancing",
                    data={"system": "\n".join(instructions), "user": user_content},
                    creative_mode=True,
                )

                self.st.model.enhanced_prompt = (
                    re.sub(r"[\"]", "", enh) if enh else self.st.model.user_prompt
                )

            else:
                self.st.model.enhanced_prompt = re.sub(
                    r"[\"]", "", self.st.model.user_prompt
                )

        self._validate_and_normalize()

    async def _vision_audit(self):
        """
        Performs a Visual Quality Control audit using a Vision LLM.
        """

        if not self.st.model.quality_audit or not self.st.model.vision:
            return

        url = (
            f"data:image/png;base64,{self.st.model.b64_data}"
            if self.st.model.b64_data
            else self.st.model.image_url
        )

        tpl = (
            self.config.AUDIT_STRICT
            if self.user_valves.strict_audit
            else self.config.AUDIT_STANDARD
        )

        # Trigger inference for vision analysis
        raw_v = await self.inf._infer(
            task="Visual Quality Audit..",
            data={
                "system": tpl.format(
                    prompt=self.st.model.enhanced_prompt, lang=self.st.model.language
                ),
                "user": {"image_url": url, "text": "Analyze image quality."},
            },
            creative_mode=False,
        )

        m = re.search(r"SCORE:\s*(\d+)", raw_v, re.IGNORECASE)

        if m:
            v = int(m.group(1))

            # Populate results dictionary with specific metrics and emoji
            self.st.quality_audit_results.update(
                {
                    "score": v,
                    "critique": re.sub(r"SCORE:\s*\d+", "", raw_v, flags=re.I).strip(),
                    "emoji": (
                        "🟢"
                        if v >= 80
                        else (
                            "🔵"
                            if v >= 70
                            else "🟡" if v >= 60 else "🟠" if v >= 40 else "🔴"
                        )
                    ),
                }
            )

        else:
            self.st.quality_audit_results["critique"] = raw_v

    async def _output_delivery(self):
        """
        Constructs and emits the final UI components: Citations and Status Bar.
        """
        m = self.st.model
        total = int(time.time() - self.st.start_time)

        # 1. Prompt Citation (Safety cast to string to prevent bool.replace crash)
        prompt_val = (
            m.enhanced_prompt if isinstance(m.enhanced_prompt, str) else m.user_prompt
        )
        cit = str(prompt_val).replace("*", "")

        if m.styles:
            cit += f"\n\nSTYLES\n{m.styles}"

        if m.negative_prompt:
            cit += f"\n\nNEGATIVE\n{m.negative_prompt}"

        await self.em.emit_citation("🚀 PROMPT", cit, "1", "p")

        # 2. Audit Citation
        if m.quality_audit:
            critique = self.st.quality_audit_results.get("critique")

            if not m.vision:
                await self.em.emit_citation(
                    "‼️NO VISION", f"Model {m.id} lacks vision.", "2", "b"
                )

            elif isinstance(critique, str):
                await self.em.emit_citation(
                    f"{self.st.quality_audit_results['emoji']} SCORE: {self.st.quality_audit_results['score']}%",
                    critique.replace("*", ""),
                    "2",
                    "a",
                )

        # 3. Technical Details Citation (Plain Text Format)

        def _mask_key(k):
            return f"{k[:4]}...{k[-4:]}" if k and len(k) > 8 else "****"

        # Determine the correct descriptive line for authentication/source
        if self.st.model.engine == self.config.Engines.FORGE:
            # For Local Engines, report the Auth Status directly
            has_auth = bool(
                getattr(self.request.app.state.config, "AUTOMATIC1111_API_AUTH", None)
            )
            auth_info_line = f"→ Auth Status: {'Credentials Active' if has_auth else 'None Required'}"

        else:
            # For Cloud Engines, report where the API Key came from
            cli_key = self.st.model.get("cli_auth")
            user_auth = (
                self.user_valves.openai_auth
                if self.st.model.engine == "openai"
                else self.user_valves.gemini_auth
            )
            valve_auth = (
                self.valves.openai_auth
                if self.st.model.engine == "openai"
                else self.valves.gemini_auth
            )

            # Simplified source detection for display
            if cli_key:
                src = "CLI Override"
            elif user_auth:
                src = "User Profile"
            elif valve_auth:
                src = "Filter Default"
            else:
                src = "System Settings"

            used_key_display = _mask_key(cli_key or user_auth or valve_auth or "Global")
            auth_info_line = f"→ API Key Source: {src} ({used_key_display})"

        # Determine engine display name and cloud status
        engine_name = self.st.model.engine.upper()
        is_cloud_engine = self.st.model.engine in [
            self.config.Engines.OPENAI,
            self.config.Engines.GEMINI,
        ]
        cloud_info = (
            f" (Easy Mode: {self.valves.easy_cloud_mode})" if is_cloud_engine else ""
        )

        # Construct Plain Text Body (No Markdown formatting like ** or *)
        tech = f"""
⚙️ Configuration
→ Model: {self.st.model.model}
→ Engine: {engine_name}{cloud_info}
{auth_info_line}

🎨 Parameters
→ Size: {self.st.model.size} (AR: {self.st.model.aspect_ratio})
→ Steps: {self.st.model.get('steps')}
→ Guidance: {self.st.model.get('cfg_scale')} (Distilled: {self.st.model.get('distilled_cfg_scale')})
→ Seed: {self.st.model.seed}
→ Sampler: {self.st.model.get('sampler_name')} / {self.st.model.get('scheduler')}

⚡ Execution
→ VRAM Purge: {self.valves.extreme_vram_cleanup}
→ Vision Cache: {self.valves.persistent_vision_cache}

⏱️ Performance
→ Total Time: {total}s
→ Image Gen: {self.st.image_gen_time_int}s
{chr(10).join(self.st.performance_stats)}
"""

        await self.em.emit_citation("🔍 DETAILS", tech.strip(), "3", "d")

        # 4. Advice Badge
        if self.st.validation_warnings:
            advice_content = "\n".join([f"• {w}" for w in self.st.validation_warnings])
            await self.em.emit_citation("⚠️ ADVICE", advice_content, "4", "w")

        # 5. Final Status Update
        tps = (
            self.st.cumulative_tokens / self.st.cumulative_elapsed_time
            if self.st.cumulative_elapsed_time > 0
            else 0
        )

        await self.em.emit_status(
            f"{total}s total | {self.st.image_gen_time_int}s img | {self.st.cumulative_tokens} tk | {tps:.1f} tk/s",
            True,
        )

    async def _output_status_only(self):
        """
        Emits only the performance stats for the 'imgx' (prompt-only) trigger.
        """

        tps = (
            self.st.cumulative_tokens / self.st.cumulative_elapsed_time
            if self.st.cumulative_elapsed_time > 0
            else 0
        )

        await self.em.emit_status(
            f"{int(time.time() - self.st.start_time)}s total | {self.st.cumulative_tokens} tk | {tps:.1f} tk/s",
            True,
        )

    def _check_input(self, body: dict) -> Optional[Tuple[str, str, Optional[str]]]:
        """
        Parses the last user message to detect Easymage triggers.
        Supports:
        - img:p (Prompt Only)
        - img:r (Random)
        - img ? (Help)
        Returns: (trigger, prompt, subcommand)
        """

        if body.get("is_probe"):
            return None

        msgs = body.get("messages", [])

        if not msgs:
            return None

        last = msgs[-1]["content"]
        # Safe extraction for both string and list (multimodal) content
        txt = last[0].get("text", "") if isinstance(last, list) else str(last)
        
        # Robustness: Strip whitespace
        txt = txt.strip()

        # FIX: Split Regex to handle 'img:p/r' vs 'img ?'
        # Group 1: Trigger (img)
        # Group 2: Colon subcommands (p|r) - Operational commands
        # Group 3: Space subcommands (?|help) - Help commands
        m = re.match(r"^(img)(?:(?::\s*(p|r))|(?:\s+(\?|help)))?(?:\s|$)", txt, re.IGNORECASE)

        if m:
            trigger = m.group(1).lower()
            # Capture subcommand from either group (colon or space based)
            sub = (m.group(2) or m.group(3) or "").lower() or None
            return (trigger, txt[m.end() :].strip(), sub)

        return None

    async def _check_vision_capability(self):
        """
        Verifies if the current LLM supports Vision capabilities.
        """

        if self.st.model.debug or not self.st.valves.persistent_vision_cache:
            if os.path.exists(CAPABILITY_CACHE_PATH):
                os.remove(CAPABILITY_CACHE_PATH)

        if os.path.exists(CAPABILITY_CACHE_PATH):
            try:
                with open(CAPABILITY_CACHE_PATH, "r") as f:
                    self.st.vision_cache = json.load(f)

                    if self.st.model.id in self.st.vision_cache:
                        self.st.model.vision = self.st.vision_cache[self.st.model.id]
                        return

            except:
                pass

        # 64x64 Black Pixel Base64
        b = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAS0lEQVQ4jWNkYGB4ycDAwMPAwMDEgAn+ERD7wQLVzIVFITGABZutJAEmBuxOJ8kAil0w8AZgiyr6umAYGDDEA5GFgYHhB5QmB/wAAIcLCBsQodqvAAAAAElFTkSuQmCC"

        res = await self.inf._infer(
            task="Ensure Vision",
            data={
                "system": "You must reply only 1 or 0",
                "user": {
                    "image_url": f"data:image/png;base64,{b}",
                    "text": "Analyze image. Reply 1 if black.",
                },
            },
            creative_mode=False,
        )

        has_v = "1" in res
        self.st.model.vision, self.st.vision_cache[self.st.model.id] = has_v, has_v

        try:
            os.makedirs(os.path.dirname(CAPABILITY_CACHE_PATH), exist_ok=True)

            with open(CAPABILITY_CACHE_PATH, "w") as f:
                json.dump(self.st.vision_cache, f)

        except:
            pass

    def _unwrap(self, obj):
        """
        Recursively unwraps Open WebUI configuration objects.
        """

        if hasattr(obj, "value"):
            return self._unwrap(obj.value)

        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}

        return obj

    def _clean_key(self, key, eng):
        """
        Normalizes environment variable keys.
        """

        k = (
            key.upper()
            .replace("IMAGE_GENERATION_", "")
            .replace("IMAGE_", "")
            .replace("ENABLE_IMAGE_", "ENABLE_")
        )

        return (
            k.replace(f"{eng.upper()}_API_", "")
            .replace(f"{eng.upper()}_", "")
            .replace("IMAGES_", "")
            .lower()
        )

    def _validate_and_normalize(self):
        """
        Ensures model parameters (size, aspect ratio, models) are valid for the selected engine.
        """
        eng, mdl = (
            self.st.model.get("engine"),
            str(self.st.model.get("model", "")).lower(),
        )

        E = self.config.Engines

        # Enforce defaults for specific engines
        if eng == E.OPENAI and "dall-e" not in mdl:
            self.st.model["model"] = "dall-e-3"

        elif eng == E.GEMINI and "imagen" not in mdl and "gemini" not in mdl:
            self.st.model["model"] = "imagen-3.0-fast-generate-001"

        sz, ar = self.st.model.get("size", "1024x1024"), self.st.model.get(
            "aspect_ratio"
        )

        try:
            w, h = map(int, sz.split("x")) if "x" in str(sz) else (int(sz), int(sz))
            r = (
                (int(ar.split(":")[0]) / int(ar.split(":")[1]))
                if ar and ":" in str(ar)
                else w / h
            )

        except:
            w, h, r = 1024, 1024, 1.0

        # CRITICAL FIX: DALL-E 3 supports Rectangles.
        # DALL-E 2 supports ONLY Squares but allows 256, 512, 1024.
        if eng == E.OPENAI or "dall-e" in mdl:
            if "dall-e-3" in mdl:
                if r > 1.2:
                    self.st.model["size"], self.st.model["aspect_ratio"] = (
                        "1792x1024",
                        "16:9",
                    )
                elif r < 0.8:
                    self.st.model["size"], self.st.model["aspect_ratio"] = (
                        "1024x1792",
                        "9:16",
                    )
                else:
                    self.st.model["size"], self.st.model["aspect_ratio"] = (
                        "1024x1024",
                        "1:1",
                    )
            else:
                # DALL-E 2 Logic: Snap to nearest valid square (256, 512, 1024)
                target = 1024
                # Use the width from the requested size (e.g. 256 from 256x256)
                try:
                    req_w = int(self.st.model.size.split("x")[0])
                    if req_w <= 256:
                        target = 256
                    elif req_w <= 512:
                        target = 512
                    else:
                        target = 1024
                except:
                    target = 1024

                self.st.model["size"] = f"{target}x{target}"
                self.st.model["aspect_ratio"] = "1:1"

        else:
            # Forge, Gemini, ComfyUI: Allow any calculated aspect ratio
            self.st.model["size"] = f"{w}x{(int(w / r) // 8) * 8}"

    def _apply_global_settings(self):
        """
        Merges global environment variables from Open WebUI config into the state.
        Fixes the PersistentConfig type error by unwrapping values before casting.
        """
        conf = getattr(self.request.app.state.config, "_state", {})

        # Helper to safely get and unwrap a value
        def get_val(key):
            return self._unwrap(conf.get(key))

        eng = str(get_val("IMAGE_GENERATION_ENGINE") or "none").lower()

        # Protected keys from CLI (Override Protection)
        prot = self.st.model.overridden_keys

        # Apply Engine
        if "engine" not in prot:
            self.st.model.engine = eng

        # Apply Model
        if "model" not in prot:
            val = get_val("IMAGE_GENERATION_MODEL")
            if val:
                self.st.model.model = val

        # Apply Size
        if "size" not in prot:
            val = get_val("IMAGE_SIZE")
            if val:
                self.st.model.size = val

        # Apply Steps (CRITICAL FIX: Unwrap before int casting)
        if "steps" not in prot:
            val = get_val("IMAGE_STEPS")
            if val is not None:
                try:
                    self.st.model.steps = int(val)
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails

        # Apply Engine-Specific Settings
        # These are generally safe as they are stored in the dict, but we unwrap them too.
        for k in self.config.ENGINE_MAP.get(eng, []):
            clean_k = self._clean_key(k, eng)
            if clean_k not in prot:
                val = get_val(k)
                if val is not None:
                    self.st.model[clean_k] = val

        self.st.model.update({})

    async def _handle_help(self):
        """
        Generates Help/Manual content.
        Emits citations immediately (events), but buffers text content for the outlet injection.
        """
        # 1. Prepare Content with Placeholders for Badges
        # FIX: Adding [1] [2] [3] allows Open WebUI to link citations as badges
        full_help_content = self.config.HELP_TEXT + "\n\n**Reference Tables:**\n[1] [2] [3]"
        
        # Store content in state, do NOT emit message here to avoid flickering.
        # The outlet will inject this text atomically.
        self.st.output_content = full_help_content

        # 2. Shortcuts & Engines Table
        shortcuts = "\n".join([f"| `{k}` | {v} |" for k, v in self.config.MODEL_SHORTCUTS.items()])
        engines = "| `en=o` | OpenAI |\n| `en=g` | Gemini |\n| `en=f` | Forge/A1111 |\n| `en=c` | ComfyUI |"
        
        tbl_models = f"""
| Shortcut | Model / Engine |
| :--- | :--- |
{shortcuts}

**Engine Codes**
{engines}
"""
        await self.em.emit_citation("⚡ SHORTCUTS", tbl_models, "1", "help-1")

        # 3. Parameters Table
        tbl_params = """
| Flag | Description | Example |
| :--- | :--- | :--- |
| `sz` | Size (WxH or square) | `sz=1024`, `sz=512x768` |
| `ar` | Aspect Ratio | `ar=16:9`, `ar=1:1` |
| `stp` | Steps | `stp=30` |
| `cfg` | Guidance Scale | `cfg=7.0` |
| `sd` | Seed | `sd=42` |
| `smp` | Sampler | `smp=dpm++2m` |
| `sch` | Scheduler | `sch=karras` |
| `auth` | Auth Override | `auth=sk-proj-123...` |
| `+h` | High-Res Fix | `+h` (Enable) |
| `+p` | Prompt Enhance | `+p` (Force On), `-p` (Off) |
| `+a` | Audit | `+a` (Force On), `-a` (Off) |
"""
        await self.em.emit_citation("🎛️ PARAMETERS", tbl_params, "2", "help-2")

        # 4. Samplers & Schedulers (Forge/Comfy)
        smp_list = ", ".join([f"`{k}`" for k in self.config.SAMPLER_MAP.keys()])
        sch_list = ", ".join([f"`{k}`" for k in self.config.SCHEDULER_MAP.keys()])
        
        tbl_adv = f"""
**Supported Samplers (smp=...)**
{smp_list}

**Supported Schedulers (sch=...)**
{sch_list}
"""
        await self.em.emit_citation("🛠️ ADVANCED", tbl_adv, "3", "help-3")
        
        # Mark as executed
        self.st.executed = True
