"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.9.0-alpha.5
repo_url: https://github.com/annibale-x/Easymage
author: Hannibal
author_url: https://openwebui.com/u/h4nn1b4l
author_email: annibale.x@gmail.com
Description: Image generation filter and prompt enhancer for Open WebUI with deterministic control, Easy Cloud Mode and Smart Engine Reconciliation.
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
# This handles multiple origins (Ollama, OpenAI, Forge, Gemini) simultaneously.
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
    # Used in Easy Cloud Mode to enforce "Model Trumps Engine" logic
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

    def __init__(self, valves):
        # Initialize the core model with default valve settings and empty triggers.
        self.model = Store(
            {
                "trigger": None,
                "vision": False,
                "debug": valves.debug,
                "enhanced_prompt": valves.enhanced_prompt,
                "quality_audit": valves.quality_audit,
                "persistent_vision_cache": valves.persistent_vision_cache,
                "llm_type": "ollama",
                "api_source": "Global",
                "using_official_url": False,
                # Intent Tracking Flags
                "_explicit_engine": False,
                "_explicit_model": False,
            }
        )
        self.valves = valves
        self.performance_stats = []
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

    def __init__(self, event_emitter):
        self.emitter = event_emitter

    async def emit_message(self, content: str):
        if self.emitter:
            fmt = (
                content.replace("```", "```\n") if content.endswith("```") else content
            )

            if "```" in content:
                fmt = re.sub(r"(```[^\n]*\n.*?\n```)", r"\1\n", fmt, flags=re.DOTALL)

            await self.emitter({"type": "message", "data": {"content": fmt}})

    async def emit_status(self, description: str, done: bool = False):
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def emit_citation(self, name: str, document: str, source: str, cid: str):
        if self.emitter:
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
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def log(self, msg: str, is_error: bool = False):
        if self.ctx.valves.debug or is_error:
            if is_error:
                print(f"\n\n❌ EASYMAGE ERROR: {msg}\n", file=sys.stderr, flush=True)
            else:
                print(f"⚡ EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def error(self, e: Any):
        self.log(str(e), is_error=True)
        await self.ctx.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def dump(self, data: Any = None):
        if not self.ctx.valves.debug:
            return

        header = "—" * 60 + "\n📦 EASYMAGE DUMP:\n" + "—" * 60
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

            return content

        except Exception as e:
            self.ctx.debug.log(f"INFERENCE ERROR ({task}): {e}", is_error=True)
            return ""

    async def _infer_ollama(
        self, inference_data: Dict, seed: Optional[int], temperature: float
    ) -> Tuple[str, int]:
        """
        Constructs and sends the payload to an Ollama-compatible API.
        Minimalist version: relies entirely on server-side settings for session management.
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

        r = await HTTP_CLIENT.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=self.ctx.valves.generation_timeout,
        )

        r.raise_for_status()
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

        r = await HTTP_CLIENT.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.ctx.valves.generation_timeout,
        )

        r.raise_for_status()
        res = r.json()

        return (
            res["choices"][0]["message"].get("content", ""),
            res.get("usage", {}).get("completion_tokens", 0),
        )

    async def purge_vram(self, unload_current: bool = False):
        """
        Interrogates Ollama for loaded models and purges them based on Open WebUI config.
        """
        if self.ctx.st.model.llm_type != "ollama":
            return

        base_urls = getattr(self.ctx.request.app.state.config, "OLLAMA_BASE_URLS", [])

        if not base_urls:
            await self.ctx.debug.error("VRAM Purge failed: Ollama Base URL not found.")
            return

        base_url = base_urls[0].rstrip("/")

        try:
            r_ps = await HTTP_CLIENT.get(f"{base_url}/api/ps")
            r_ps.raise_for_status()
            loaded_models = r_ps.json().get("models", [])

            current_model = self.ctx.st.model.id

            for m in loaded_models:
                m_name = m.get("name")

                if not unload_current and m_name == current_model:
                    continue

                await HTTP_CLIENT.post(
                    f"{base_url}/api/chat",
                    json={"model": m_name, "keep_alive": 0},
                    timeout=5,
                )

                self.ctx.debug.log(f"VRAM Purge: Unloaded {m_name}")

        except Exception as e:
            await self.ctx.debug.error(f"VRAM Purge failed: {str(e)}")


class ImageGenEngine:
    """
    Handles the actual image generation request dispatch with Easy Cloud Mode logic.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    async def generate(self) -> str:
        """
        Orchestrates the image generation process based on the selected engine.
        """
        await self.ctx.em.emit_status("Generating Image..")
        start = time.time()
        eng = self.ctx.st.model.engine
        E = self.ctx.config.Engines

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
        p = self.ctx.st.model.copy()
        p["prompt"] = p["enhanced_prompt"]

        if not p.get("enable_hr"):
            p["negative_prompt"] = ""

        try:
            w, h = map(int, p.get("size", "1024x1024").split("x"))
            p["width"], p["height"] = w, h
        except:
            p["width"], p["height"] = 1024, 1024

        for k in [
            "trigger",
            "id",
            "engine",
            "user_prompt",
            "enhanced_prompt",
            "quality_audit",
            "persistent_vision_cache",
            "debug",
            "vision",
            "language",
            "b64_data",
            "model",
            "styles",
            "size",
            "aspect_ratio",
            "llm_type",
            "api_source",
            "using_official_url",
            "cli_api_key",
            "_explicit_engine",
            "_explicit_model",
        ]:
            p.pop(k, None)
        return p

    def _prepare_openai(self) -> dict:
        m = self.ctx.st.model
        return {
            "model": m.model or "dall-e-3",
            "prompt": m.enhanced_prompt,
            "n": 1,
            "size": m.size or "1024x1024",
            "quality": "hd" if m.enable_hr else "standard",
            "style": "natural" if m.style == "natural" else "vivid",
            "response_format": "b64_json",
            "user": self.ctx.user.id,
        }

    def _prepare_gemini(self, method: str) -> dict:
        """
        Builds the payload for Google Gemini Image Generation based on the detected method.
        """
        m = self.ctx.st.model

        if method == "generateContent":
            # Payload for standard Gemini API (Google AI Studio)
            payload = {
                "contents": [{"parts": [{"text": m.enhanced_prompt}]}],
                "generationConfig": {
                    "candidateCount": 1,
                    "negativePrompt": m.negative_prompt,
                },
            }

            if m.seed and m.seed != -1:
                payload["generationConfig"]["seed"] = m.seed

            return payload

        else:
            # Payload for Vertex AI or compatible proxies using :predict
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
        payload = self._prepare_forge()
        conf = self.ctx.request.app.state.config
        url = conf.AUTOMATIC1111_BASE_URL.rstrip("/")
        auth = conf.AUTOMATIC1111_API_AUTH
        headers = {}

        if auth:
            headers["Authorization"] = (
                f"Basic {base64.b64encode(auth.encode()).decode()}"
                if ":" in auth
                else f"Bearer {auth}"
            )

        r = await HTTP_CLIENT.post(
            f"{url}/sdapi/v1/txt2img",
            json=payload,
            headers=headers,
            timeout=self.ctx.valves.generation_timeout,
        )
        r.raise_for_status()
        res = r.json()

        img = res["images"][0].split(",")[-1]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

    async def _gen_openai(self):
        """
        Executes request against OpenAI with Easy Cloud Mode support.
        Hierarchy: CLI > Valve > Global.
        """
        conf = self.ctx.request.app.state.config
        valves = self.ctx.valves

        # 1. Determine Base URL (Easy Mode vs Global)
        if valves.easy_cloud_mode:
            base_url = self.ctx.config.OFFICIAL_URLS["openai"]
            self.ctx.st.model.using_official_url = True
        else:
            base_url = getattr(conf, "IMAGES_OPENAI_API_BASE_URL", "").rstrip("/")
            if not base_url:
                raise Exception(
                    "OpenAI Base URL missing in Global Settings (and Easy Mode is OFF)."
                )

        # 2. Determine API Key (CLI > Valve > Global)
        cli_key = self.ctx.st.model.get("cli_api_key")
        global_keys = getattr(conf, "IMAGES_OPENAI_API_KEYS", [])
        global_key = global_keys[0] if global_keys else ""
        valve_key = valves.openai_api_key

        api_key = cli_key or valve_key or global_key

        # Track Source for Citations
        if cli_key:
            self.ctx.st.model.api_source = "CLI"
        elif valve_key:
            self.ctx.st.model.api_source = "Valve"
        elif global_key:
            self.ctx.st.model.api_source = "Global"
        else:
            self.ctx.st.model.api_source = "Missing"

        if not api_key:
            raise Exception("No OpenAI API Key found (Checked: CLI, Valves, Global).")

        # 3. Prepare Payload & Execute
        payload = self._prepare_openai()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        r = await HTTP_CLIENT.post(
            f"{base_url}/images/generations",
            json=payload,
            headers=headers,
            timeout=valves.generation_timeout,
        )

        r.raise_for_status()
        img = r.json()["data"][0]["b64_json"]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

    async def _gen_gemini(self):
        """
        Executes request against Gemini with Easy Cloud Mode support.
        Handles URL selection and smart method detection.
        """
        conf = self.ctx.request.app.state.config
        valves = self.ctx.valves

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

        # 2. Determine API Key (CLI > Valve > Global)
        cli_key = self.ctx.st.model.get("cli_api_key")
        global_key = getattr(conf, "IMAGES_GEMINI_API_KEY", "")
        valve_key = valves.gemini_api_key

        api_key = cli_key or valve_key or global_key

        # Track Source for Citations
        if cli_key:
            self.ctx.st.model.api_source = "CLI"
        elif valve_key:
            self.ctx.st.model.api_source = "Valve"
        elif global_key:
            self.ctx.st.model.api_source = "Global"
        else:
            self.ctx.st.model.api_source = "Missing"

        if not api_key:
            raise Exception("No Gemini API Key found (Checked: CLI, Valves, Global).")

        # 3. Smart Method Detection (Easy Mode implies standard API)
        is_studio = "generativelanguage.googleapis.com" in base_url
        method = "generateContent" if is_studio else "predict"

        model_name = self.ctx.st.model.model or "gemini-2.0-flash"
        url = f"{base_url}/models/{model_name}:{method}"

        # 4. Auth Headers
        headers = {"Content-Type": "application/json"}

        if is_studio:
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = self._prepare_gemini(method)

        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"GEMINI: {url} | Method: {method} | EasyMode: {valves.easy_cloud_mode}"
            )

        r = await HTTP_CLIENT.post(
            url,
            json=payload,
            headers=headers,
            timeout=valves.generation_timeout,
        )

        r.raise_for_status()
        res = r.json()
        img_b64 = ""

        try:
            if method == "generateContent":
                img_b64 = res["candidates"][0]["content"]["parts"][0]["inlineData"][
                    "data"
                ]
            else:
                img_b64 = res["predictions"][0]["bytesBase64Encoded"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Gemini API Error ({method}): {str(e)}")

        self.ctx.st.model.b64_data = img_b64
        self.ctx.st.model.image_url = f"data:image/png;base64,{img_b64}"

    async def _gen_standard(self):
        """
        Falls back to the standard Open WebUI image generation router.
        """
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

    def parse(self, user_prompt: str):
        """
        Analyzes the raw user string.
        Extracts parameters including the new 'api=' key and expands model shortcuts.
        """
        m_st = self.ctx.st.model
        clean, neg = user_prompt, ""

        if " --no " in clean.lower():
            clean, neg = re.split(r" --no ", clean, maxsplit=1, flags=re.IGNORECASE)

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
            try:
                if k == "ge":
                    m_st["engine"] = {
                        "a": self.config.Engines.FORGE,
                        "o": self.config.Engines.OPENAI,
                        "g": self.config.Engines.GEMINI,
                        "c": self.config.Engines.COMFY,
                    }.get(v.lower(), v.lower())

                    m_st["_explicit_engine"] = True

                elif k in ["mdl", "mod", "m"]:
                    v_low = v.lower()
                    m_st["model"] = self.config.MODEL_SHORTCUTS.get(v_low, v_low)

                    m_st["_explicit_model"] = True

                elif k == "api":
                    m_st["cli_api_key"] = v
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
            elif char == "p":
                m_st["enhanced_prompt"] = v_bool
            elif char == "a":
                m_st["quality_audit"] = v_bool
            elif char == "h":
                m_st["enable_hr"] = v_bool

        m_st.update(
            {
                "user_prompt": subj.strip(),
                "negative_prompt": neg.strip(),
                "styles": rem_styles,
            }
        )

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
        User-configurable settings available in the Open WebUI interface.
        Controls LLM behavior, VRAM management, and Image Generation parameters.
        """

        # --- CLOUD & CONNECTIVITY ---

        easy_cloud_mode: bool = Field(
            default=True,
            description="If True, ignores global URLs for Cloud Models (OpenAI/Gemini) and uses official endpoints. Automatically detects Engine from Model.",
        )

        openai_api_key: str = Field(
            default="",
            description="Personal API Key for OpenAI. Overrides global settings. Can be overridden by 'api=' in CLI.",
        )

        gemini_api_key: str = Field(
            default="",
            description="Personal API Key for Google Gemini. Overrides global settings. Can be overridden by 'api=' in CLI.",
        )

        # --- CORE ENGINE SETTINGS ---

        enhanced_prompt: bool = Field(
            default=True,
            description="Enable LLM-based prompt expansion to add artistic details, lighting, and textures.",
        )

        quality_audit: bool = Field(
            default=True,
            description="Perform a visual quality control (VQC) using a Vision LLM after the image is generated.",
        )

        strict_audit: bool = Field(
            default=False,
            description="Enable severe scoring penalties for hallucinations or negative prompt violations during audit.",
        )

        persistent_vision_cache: bool = Field(
            default=False,
            description="Save vision capability test results to a local JSON file to speed up startup.",
        )

        # --- HARDWARE & PERFORMANCE ---

        extreme_vram_cleanup: bool = Field(
            default=False,
            description="If True, unloads the current LLM as well. If False (default), only unloads other idle models.",
        )

        generation_timeout: int = Field(
            default=120,
            description="Maximum time in seconds allowed for the image generation API request.",
        )

        debug: bool = Field(
            default=False,
            description="Enable verbose logging and surgical object dumping in the server console/logs.",
        )

        # --- IMAGE GENERATION PARAMETERS (FORGE/OPENAI/GEMINI) ---

        model: Optional[str] = Field(
            default=None,
            description="The specific model checkpoint to use (e.g., flux1-dev, dall-e-3, imagen-3).",
        )

        steps: int = Field(
            default=20,
            description="Number of sampling steps. Higher values increase detail but take more time.",
        )

        size: str = Field(
            default="1024x1024",
            description="Default image resolution. Format: Width x Height.",
        )

        aspect_ratio: str = Field(
            default="1:1",
            description="Target aspect ratio. Automatically adjusts width and height if supported by the engine.",
        )

        seed: int = Field(
            default=-1,
            description="Deterministic seed for generation. Use -1 for a random seed.",
        )

        cfg_scale: float = Field(
            default=1.0,
            description="Classifier Free Guidance. Controls how strictly the model follows the prompt.",
        )

        distilled_cfg_scale: float = Field(
            default=3.5,
            description="CFG scale for distilled models (like Flux). Overrides standard CFG if set.",
        )

        sampler_name: str = Field(
            default="Euler",
            description="Sampling algorithm (mostly for Forge/Automatic1111).",
        )

        scheduler: str = Field(
            default="Simple",
            description="Noise scheduler for the sampling process (e.g., Karras, Beta, Simple).",
        )

        # --- HIGH-RES FIX / UPSCALING ---

        enable_hr: bool = Field(
            default=False,
            description="Enable High-Resolution Fix pass (upscaling) during generation.",
        )

        hr_scale: float = Field(
            default=2.0,
            description="Multiplier for the final resolution during the High-Res Fix pass.",
        )

        hr_upscaler: str = Field(
            default="Latent",
            description="The algorithm used for upscaling (e.g., Latent, R-ESRGAN, ESRGAN_4x).",
        )

        hr_distilled_cfg: float = Field(
            default=3.5,
            description="Specific CFG scale applied during the upscaling pass.",
        )

        denoising_strength: float = Field(
            default=0.45,
            description="Controls how much the upscaler changes the original image. 0.0 is none, 1.0 is total.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.config = EasymageConfig()
        # Explicit type hinting for Pyright: st can be EasymageState or None
        self.st: EasymageState = None  # type: ignore

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:
        """
        Intercepts the incoming request before it reaches the LLM.
        Decides if Easymage should trigger ('img' or 'imgx').
        """
        trigger_data = self._check_input(body)

        if not trigger_data:
            self.st = None
            return body

        trigger, raw_p = trigger_data

        # 1. IMMEDIATE STATE CREATION
        self.request = __request__
        self.user = UserModel(**__user__)
        self.st = EasymageState(self.valves)
        self.st.model.trigger = trigger
        self.em = EmitterService(__event_emitter__)

        try:
            metadata = body.get("metadata", {})
            self.st.model.llm_type = metadata.get("model", {}).get("owned_by", "ollama")

        except:
            self.st.model.llm_type = "ollama"

        # 2. IMMEDIATE SUPPRESSION
        body = self._suppress_output(body)

        # 3. WORKER INITIALIZATION
        self.debug = DebugService(self)
        self.inf = InferenceEngine(self)
        self.img_gen = ImageGenEngine(self)
        self.parser = PromptParser(self)

        try:
            # 4. LOGIC EXECUTION
            await self._setup_context(body, raw_p)

            # --- SMART ENGINE RECONCILIATION ---

            m = self.st.model
            detected_engine = self.config.MODEL_ENGINE_MAP.get(str(m.model).lower())

            if self.st.model.debug:
                self.debug.log(
                    f"Reconciliation -> Explicit Engine: {m._explicit_engine}, Explicit Model: {m._explicit_model}, Current Engine: {m.engine}, Detected from Model: {detected_engine}"
                )

            if m._explicit_model and detected_engine:
                # Case 1: User Explicitly chose a Model
                m.engine = detected_engine

            elif m._explicit_engine and not m._explicit_model:
                # Case 2: User Explicitly chose an Engine but NOT a model
                if detected_engine and detected_engine != m.engine:
                    self.debug.log(
                        f"Conflict: Engine {m.engine} incompatible with model {m.model}. Resetting model."
                    )
                    m.model = None

            elif self.valves.easy_cloud_mode and detected_engine:
                # Case 3: Easy Cloud Mode
                m.engine = detected_engine

            if self.st.model.trigger == "imgx":
                self.st.output_content = self.st.model.enhanced_prompt
                self.st.executed = True

            else:
                # MANDATORY VRAM CLEANUP
                await self.em.emit_status("Cleaning VRAM..")

                await self.inf.purge_vram(
                    unload_current=self.valves.extreme_vram_cleanup
                )

                # Full Generation Mode
                # CRITICAL FIX: Capture the URL returned by generate()
                img_url = await self.img_gen.generate()

                # CRITICAL FIX: Explicitly construct the final content buffer.
                # This ensures the Outlet saves both the Image Markdown AND the Citation placeholders
                # to the chat history, fixing the "disappearing image on refresh" bug.
                self.st.output_content = f"![Generated Image]({img_url})\n\n[1] [2] [3]"

                if self.st.model.quality_audit:
                    await self._vision_audit()

                self.st.executed = True

        except Exception as e:
            await self.debug.error(e)

            if self.st:
                self.st.executed = True

        return body

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        """
        Intercepts the LLM's response (which is just empty space/silence).
        Replaces the silence with the actual content generated by Easymage.
        """

        if getattr(self, "st", None) is None or not self.st.executed:
            return body

        if not self.st.model.trigger:
            return body

        if __event_emitter__:
            self.em.emitter = __event_emitter__

        if "messages" in body and len(body["messages"]) > 0:
            body["messages"][-1]["content"] = self.st.output_content

        if self.st.model.trigger == "imgx":
            await self._output_status_only()
        else:
            await self._output_delivery()

        return body

    def _suppress_output(self, body: dict) -> dict:
        """
        Force the LLM to be completely silent using invisible characters.
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
        if "features" in body:
            body["features"]["web_search"] = False

        self.st.model.id = body.get("model", "")
        self._apply_global_settings()
        self.st.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )

        # 1. Parse User Input
        self.parser.parse(user_prompt)

        if self.st.model.trigger == "img":
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

        # 3. Prompt Enhancement
        E = self.config.Engines
        native_support = (self.st.model.engine == E.GEMINI) or (
            self.st.model.engine == E.FORGE and self.st.model.enable_hr
        )
        use_llm_neg = (self.st.model.trigger == "imgx") or (not native_support)

        if self.st.model.enhanced_prompt or self.st.model.trigger == "imgx":
            instructions = [
                self.config.PROMPT_ENHANCE_BASE.format(lang=self.st.model.language),
                self.config.PROMPT_ENHANCE_RULES,
            ]
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
        Performs a Visual Quality Control audit.
        Uses a Vision LLM to analyze the generated image against the prompt.
        """
        if not self.st.model.quality_audit or not self.st.model.vision:
            return

        # await self.em.emit_status("Visual Quality Audit..", False)

        url = (
            f"data:image/png;base64,{self.st.model.b64_data}"
            if self.st.model.b64_data
            else self.st.model.image_url
        )
        tpl = (
            self.config.AUDIT_STRICT
            if self.valves.strict_audit
            else self.config.AUDIT_STANDARD
        )

        raw_v = await self.inf._infer(
            task="Vision Audit",
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
        Constructs and emits the final UI components:
        1. Prompt Citation
        2. Audit/Warning Citation
        3. Technical Details Citation
        """
        total = int(time.time() - self.st.start_time)
        cit = self.st.model.enhanced_prompt.replace("*", "")

        if self.st.model.styles:
            cit += f"\n\nSTYLES\n{self.st.model.styles}"
        if self.st.model.negative_prompt:
            cit += f"\n\nNEGATIVE\n{self.st.model.negative_prompt}"

        await self.em.emit_citation("🚀 PROMPT", cit, "1", "p")

        if self.st.model.quality_audit:
            if not self.st.model.vision:
                await self.em.emit_citation(
                    "‼️NO VISION", f"Model {self.st.model.id} lacks vision.", "2", "b"
                )
            elif self.st.quality_audit_results["critique"]:
                await self.em.emit_citation(
                    f"{self.st.quality_audit_results['emoji']} SCORE: {self.st.quality_audit_results['score']}%",
                    self.st.quality_audit_results["critique"].replace("*", ""),
                    "2",
                    "a",
                )

        tech = (
            f"\n⠀\n𝗖𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻\n → Model: {self.st.model.id}\n → Engine: {self.st.model.engine}\n → Res: {self.st.model.size} | Steps: {self.st.model.get('steps')}\n\n𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲\n → Total Time: {total}s\n → Image Gen: {self.st.image_gen_time_int}s\n"
            + "\n".join(self.st.performance_stats)
        )
        await self.em.emit_citation("🔍 DETAILS", tech, "3", "d")

        tps = (
            self.st.cumulative_tokens / self.st.cumulative_elapsed_time
            if self.st.cumulative_elapsed_time > 0
            else 0
        )
        total = int(time.time() - self.st.start_time)

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

    def _check_input(self, body: dict) -> Optional[Tuple[str, str]]:
        """
        Parses the last user message to detect Easymage triggers ('img' or 'imgx').
        """
        if body.get("is_probe"):
            return None

        msgs = body.get("messages", [])

        if not msgs:
            return None

        last = msgs[-1]["content"]
        txt = last[0].get("text", "") if isinstance(last, list) else last
        m = re.match(r"^(img|imgx)\s", txt, re.IGNORECASE)

        return (m.group(1).lower(), txt[m.end() :].strip()) if m else None

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

        # 1x1 Black Pixel Base64
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

        if eng == E.OPENAI or "dall-e" in mdl:
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
            self.st.model["size"] = f"{w}x{(int(w / r) // 8) * 8}"

    def _apply_global_settings(self):
        """
        Merges global environment variables from Open WebUI config into the state.
        """
        conf = getattr(self.request.app.state.config, "_state", {})
        eng = str(self._unwrap(conf.get("IMAGE_GENERATION_ENGINE", "none"))).lower()
        sets = {"engine": eng}

        for k in ["IMAGE_GENERATION_MODEL", "IMAGE_SIZE", "IMAGE_STEPS"]:
            sets[self._clean_key(k, eng)] = self._unwrap(conf.get(k))

        for k in self.config.ENGINE_MAP.get(eng, []):
            val = self._unwrap(conf.get(k))
            if val is not None:
                sets[self._clean_key(k, eng)] = val

        self.st.model.update(sets)
