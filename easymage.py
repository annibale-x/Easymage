"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.8.20
repo_url: https://github.com/annibale-x/Easymage
author: Hannibal
author_url: https://openwebui.com/u/h4nn1b4l
author_email: annibale.x@gmail.com
Description: Image generation filter and prompt enhancer for Open WebUI with deterministic control.
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

# Create a global client for connection pooling if not exists
# This further reduces latency on high-end systems by keeping sockets open.
HTTP_CLIENT = httpx.AsyncClient(timeout=120)


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

    # Maps short codes or alternative names to the specific sampler names required by backends (mostly A1111/Forge).
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

    # PROMPT TEMPLATES

    # Template for standard vision analysis without strict penalty rules.
    AUDIT_STANDARD = """
        RESET: ####################### NEW DATA STREAM ##########################
        ENVIRONMENT: STATELESS SANDBOX.
        TASK: AUDIT ANALYSIS (Audit scores is 0 to 100):
                Compare image with: '{prompt}',
                Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line.
        
        # Add explicit penalty rule for content contradictions
        RULE: If any element explicitly forbidden in the prompt is present, the AUDIT score MUST be below 50.
        
        TASK: TECHNICAL EVALUATION:
                Evaluate NOISE, GRAIN, MELTING, JAGGIES.
        MANDATORY: Respond in {lang}. NO MARKDOWN. Use plain text and • for lists and ➔ for headings.
        MANDATORY: Final response MUST end with: SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
    """

    # Template for strict vision analysis, penalizing hallucinations and negative prompt violations severely.
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
        
        # Add ruthless penalty rule for negative prompt violations
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

    # Base prompt for the LLM to enhance the user's short description.
    PROMPT_ENHANCE_BASE = """
        ROLE: You are an expert AI Image Prompt Engineer.
        TASK: Expand the user's input into a professional, highly detailed prompt in {lang}.
        TASK: Add details about lighting, camera angle, textures, environment, and artistic style.
    """

    # Injection for specific artistic styles requested by user.
    PROMPT_ENHANCE_STYLES = """
        MANDATORY: Incorporate these style elements naturally into the description: {styles}.
    """

    # Injection for negative constraints.
    PROMPT_ENHANCE_NEG = """
        MANDATORY: The description must explicitly ensure that {negative} are NOT present. If necessary, 
        describe the scene in a way that confirms their absence.
    """

    # Formatting rules to ensure the LLM output is clean for the image generator.
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
    Holds configuration, performance metrics, and generation results.
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
    Used to send status updates, generated images, and citations to the chat UI.
    """

    def __init__(self, event_emitter):
        self.emitter = event_emitter

    async def emit_message(self, content: str):
        """
        Emits a standard text message or markdown content to the chat.
        Handles formatting for code blocks.
        """
        if self.emitter:
            fmt = (
                content.replace("```", "```\n") if content.endswith("```") else content
            )

            if "```" in content:
                fmt = re.sub(r"(```[^\n]*\n.*?\n```)", r"\1\n", fmt, flags=re.DOTALL)

            await self.emitter({"type": "message", "data": {"content": fmt}})

    async def emit_status(self, description: str, done: bool = False):
        """
        Updates the status indicator in the UI (e.g., "Generating Image...").
        """
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def emit_citation(self, name: str, document: str, source: str, cid: str):
        """
        Emits a citation block, used here to display metadata and audit results neatly.
        """
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
    Ensures Pydantic models and Starlette configuration objects are readable.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def log(self, msg: str, is_error: bool = False):
        """
        Prints diagnostic logs to standard error for Docker/Console visibility.
        Only logs if debug mode is enabled in valves, unless it is an error.
        """
        if self.ctx.valves.debug or is_error:
            if is_error:
                print(f"\n\n❌ EASYMAGE ERROR: {msg}\n", file=sys.stderr, flush=True)
            else:
                print(f"⚡ EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def error(self, e: Any):
        """
        Logs exceptions and notifies the chat UI of critical filter errors.
        """
        self.log(str(e), is_error=True)
        await self.ctx.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def dump(self, data: Any = None):
        """
        Indestructible Surgical Dumper.
        Introspects objects to handle Open WebUI's 'PersistentConfig' and Pydantic models
        without crashing on serialization.
        """
        if not self.ctx.valves.debug:
            return

        header = "—" * 60 + "\n📦 EASYMAGE DUMP:\n" + "—" * 60
        print(header, file=sys.stderr, flush=True)

        def universal_resolver(obj):
            # Intercept Open WebUI PersistentConfig objects which hold data in .value
            if hasattr(obj, "value"):
                return obj.value
            # Intercept Pydantic models for serialization
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            # Fallback for unknown types
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
    Supports switching between Ollama and OpenAI backends based on configuration.
    """

    def __init__(self, ctx):
        self.ctx = ctx  # Access to Filter instance

    async def _infer(
        self, task: Optional[str], data: Dict[str, Any], creative_mode: bool = False
    ) -> str:
        """
        Main entry point for LLM inference.
        Standardizes input data, manages timing/stats, and dispatches to specific backend.
        """
        start = time.time()

        if task:
            await self.ctx.em.emit_status(f"{task}..")

        # 1. Prepare a generic, abstract data structure for the prompt
        inference_data = {
            "system_prompt": None,
            "user_prompt": None,
            "image_b64": None,
        }

        if sys_c := data.get("system"):
            inference_data["system_prompt"] = re.sub(r"\s+", " ", sys_c).strip()

        user_c = data.get("user")

        if isinstance(user_c, dict):
            # Handle multimodal request (Text + Image)
            if i_url := user_c.get("image_url"):
                # Extract base64 data from data URI
                inference_data["image_b64"] = i_url.split(",")[-1]

            if t_c := user_c.get("text"):
                inference_data["user_prompt"] = re.sub(r"\s+", " ", t_c).strip()
        else:
            # Handle text-only request
            inference_data["user_prompt"] = re.sub(r"\s+", " ", str(user_c)).strip()

        # 2. Configure Determinism parameters
        if creative_mode:
            # Creative mode uses high temperature and respects user seed if present
            temperature = 0.7
            s_val = (
                int(self.ctx.st.model.seed)
                if self.ctx.st.model.seed is not None
                else -1
            )
            seed = None if s_val == -1 else s_val
        else:
            # Analytical mode uses 0 temp and fixed seed
            seed, temperature = 42, 0.0

        # 3. Dispatch to the specific engine builder based on model ownership
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

            # Clean up thinking tokens (common in reasoning models)
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
        Enforces strict parameter isolation within the 'options' dict.
        """
        base_urls = getattr(self.ctx.request.app.state.config, "OLLAMA_BASE_URLS", [])
        if not base_urls:
            raise Exception("Ollama Base URL is missing from Open WebUI configuration.")

        base_url = base_urls[0].rstrip("/")

        # Build the Ollama-specific payload from the generic data structure
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

        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"OLLAMA: {base_url}/api/chat (s:{seed} t:{temperature})"
            )
            if inference_data.get("image_b64"):
                self.ctx.debug.log(
                    "Ollama vision payload detected, dumping built payload."
                )
                self.ctx.debug.dump(payload)

        r = await HTTP_CLIENT.post(f"{base_url}/api/chat", json=payload)
        # r = await client.post(f"{base_url}/api/chat", json=payload)
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
        Handles image formatting for GPT-4-Vision style endpoints.
        """

        conf = self.ctx.request.app.state.config
        base_urls = getattr(conf, "OPENAI_API_BASE_URLS", [])

        if not base_urls:
            raise Exception("OpenAI Base URL is missing from Open WebUI configuration.")

        base_url = base_urls[0].rstrip("/")
        api_keys = getattr(conf, "OPENAI_API_KEYS", [])
        api_key = api_keys[0] if api_keys else ""

        # Build the OpenAI-specific payload from the generic data structure
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

        # OpenAI expects content as a string for text-only, or a list for multimodal
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

        # Standard headers for OpenAI authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"OPENAI: {base_url}/chat/completions (s:{seed} t:{temperature})"
            )

        # Async request via global connection pool
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

    async def purge_vram(self, unload_all: bool = False):
        """
        Cleans up the VRAM by unloading models from Ollama.
        Always unloads all models except the one currently in use, 
        unless 'unload_all' is True.
        """
        if self.ctx.st.model.llm_type != "ollama":
            return

        base_urls = getattr(self.ctx.request.app.state.config, "OLLAMA_BASE_URLS", [])

        if not base_urls:
            await self.ctx.debug.error("VRAM Purge failed: Ollama Base URL not found.")
            return

        base_url = base_urls[0].rstrip("/")

        try:
            # 1. Fetch currently loaded models from the Ollama server
            r_ps = await HTTP_CLIENT.get(f"{base_url}/api/ps")
            r_ps.raise_for_status()
            loaded_models = r_ps.json().get("models", [])
            
            current_model = self.ctx.st.model.id

            # 2. Iterate and unload targeted models
            for m in loaded_models:
                m_name = m.get("name")

                # Optimization: skip current model unless extreme cleanup is requested
                if not unload_all and m_name == current_model:
                    continue

                # Sending keep_alive: 0 effectively evicts the model from VRAM
                await HTTP_CLIENT.post(
                    f"{base_url}/api/chat",
                    json={"model": m_name, "keep_alive": 0},
                    timeout=5
                )
                
                self.ctx.debug.log(f"VRAM Purge: Evicted model {m_name}")

        except Exception as e:
            # Use the dedicated EM error service to notify the UI and log to stderr
            await self.ctx.debug.error(f"VRAM Purge failed: {str(e)}")

class ImageGenEngine:
    """
    Handles the actual image generation request dispatch.
    Supports Automatic1111 (Forge), OpenAI DALL-E, Gemini, and Standard Open WebUI generation.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    async def generate(self) -> str:
        """
        Orchestrates the image generation process based on the selected engine.
        Updates state with the generated image URL and B64 data.
        """
        await self.ctx.em.emit_status("Generating Image..")
        start = time.time()
        eng = self.ctx.st.model.engine
        E = self.ctx.config.Engines

        try:
            # Dispatch to appropriate handler
            if eng == E.OPENAI:
                await self._gen_openai()
            elif eng == E.GEMINI:
                await self._gen_gemini()
            elif eng == E.FORGE:
                await self._gen_forge()
            else:
                await self._gen_standard()

            self.ctx.st.image_gen_time_int = int(time.time() - start)

            # Notify the UI if successful
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
        """
        Builds the payload for Automatic1111/Forge API.
        Cleans the internal state dictionary to remove non-SD parameters.
        """
        p = self.ctx.st.model.copy()
        p["prompt"] = p["enhanced_prompt"]

        if not p.get("enable_hr"):
            p["negative_prompt"] = ""

        try:
            w, h = map(int, p.get("size", "1024x1024").split("x"))
            p["width"], p["height"] = w, h
        except:
            p["width"], p["height"] = 1024, 1024

        # Remove internal Easymage keys that the backend doesn't understand
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
        ]:
            p.pop(k, None)
        return p

    def _prepare_openai(self) -> dict:
        """
        Builds the payload for OpenAI DALL-E API.
        """
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

    def _prepare_gemini(self) -> dict:
        """
        Builds the payload for Google Gemini Image Generation.
        Calculates aspect ratio strings from dimensions.
        """
        m = self.ctx.st.model
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
            "sampleCount": m.get("n_iter", 1),
            "negativePrompt": m.negative_prompt,
            "aspectRatio": ar,
            "safetySetting": "block_none",
            "personGeneration": "allow_all",
            "addWatermark": False,
            "includeReasoning": False,
            "outputOptions": {"mimeType": "image/png"},
        }

        if m.seed and m.seed != -1:
            params["seed"] = m.seed

        return {"instances": [{"prompt": m.enhanced_prompt}], "parameters": params}

    # --- HTTP HANDLERS ---

    async def _gen_forge(self):
        """
        Executes request against Automatic1111/Forge txt2img endpoint.
        """
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
        Executes request against OpenAI images/generations endpoint.
        """
        payload = self._prepare_openai()
        conf = self.ctx.request.app.state.config
        headers = {
            "Authorization": f"Bearer {conf.IMAGES_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        r = await HTTP_CLIENT.post(
            f"{conf.IMAGES_OPENAI_API_BASE_URL}/images/generations",
            json=payload,
            headers=headers,
            timeout=self.ctx.valves.generation_timeout,
        )
        r.raise_for_status()
        img = r.json()["data"][0]["b64_json"]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

    async def _gen_gemini(self):
        """
        Executes request against Google VertexAI/Gemini predict endpoint.
        """
        payload = self._prepare_gemini()
        conf = self.ctx.request.app.state.config
        url = f"{conf.IMAGES_GEMINI_API_BASE_URL}/models/{self.ctx.st.model.model}:predict"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": conf.IMAGES_GEMINI_API_KEY,
        }

        r = await HTTP_CLIENT.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.ctx.valves.generation_timeout,
        )
        r.raise_for_status()
        img = r.json()["predictions"][0]["bytesBase64Encoded"]
        self.ctx.st.model.b64_data = img
        self.ctx.st.model.image_url = f"data:image/png;base64,{img}"

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
        1. Splits negative prompt (after --no).
        2. Separates parameter block from description.
        3. Parses KV pairs (e.g., ar=16:9) and flags (e.g., +d).
        """
        m_st = self.ctx.st.model  # Direct state access
        clean, neg = user_prompt, ""

        # Split Negative Prompt
        if " --no " in clean.lower():
            clean, neg = re.split(r" --no ", clean, maxsplit=1, flags=re.IGNORECASE)

        # Split Parameters from Prompt Description
        if " -- " in clean:
            prefix, subj = clean.split(" -- ", 1)
        else:
            # Detect parameter block end via regex looking for key=value or flags
            pat = r'(\b\w+=(?:"[^"]*"|\S+))|(?<!\w)([+-][dpah])(?!\w)'
            matches = list(re.finditer(pat, clean))
            idx = matches[-1].end() if matches else 0
            prefix, subj = clean[:idx], clean[idx:].strip()

        # Isolate the style string
        rem_styles = re.sub(
            r'(\b\w+=(?:"[^"]*"|\S+))|(?<!\w)([+-][dpah])(?!\w)', "", prefix
        ).strip()
        rem_styles = re.sub(r"^[\s,]+|[\s,]+$", "", rem_styles)

        # Parse Key-Value Pairs
        kv_pat = r'(\b\w+)=("([^"]*)"|(\S+))'

        for k, _, q, u in re.findall(kv_pat, prefix):
            k, v = k.lower(), (q if q else u).lower()
            try:
                # Map short keys to state model keys
                if k == "ge":
                    m_st["engine"] = {
                        "a": self.config.Engines.FORGE,
                        "o": self.config.Engines.OPENAI,
                        "g": self.config.Engines.GEMINI,
                        "c": self.config.Engines.COMFY,
                    }.get(v, v)
                elif k == "mdl":
                    m_st["model"] = {
                        "d3": "dall-e-3",
                        "d2": "dall-e-2",
                        "i3": "imagen-3.0-generate-001",
                        "i3f": "imagen-3.0-fast-generate-001",
                    }.get(v, v)
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
                    m_st["style"] = {"v": "vivid", "n": "natural"}.get(v, v)
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

        # Parse Boolean Flags (+d, -p, etc.)
        for flag in re.findall(r"(?<!\w)([+-][dpah])(?!\w)", prefix):
            v, char = flag[0] == "+", flag[1]
            if char == "d":
                m_st["debug"] = v
            elif char == "p":
                m_st["enhanced_prompt"] = v
            elif char == "a":
                m_st["quality_audit"] = v
            elif char == "h":
                m_st["enable_hr"] = v

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
        """

        enhanced_prompt: bool = Field(default=True)
        quality_audit: bool = Field(default=True)
        strict_audit: bool = Field(default=False)
        persistent_vision_cache: bool = Field(default=False)
        extreme_vram_cleanup: bool = Field(
            default=False, 
            description="If True, unloads the current LLM as well. If False (default), only unloads other idle models."
        )
        generation_timeout: int = Field(default=120)
        debug: bool = Field(default=False)
        model: Optional[str] = Field(default=None)
        steps: int = 20
        size: str = "1024x1024"
        aspect_ratio: str = "1:1"
        seed: int = -1
        cfg_scale: float = 1.0
        distilled_cfg_scale: float = 3.5
        sampler_name: str = "Euler"
        scheduler: str = "Simple"
        enable_hr: bool = False
        hr_scale: float = 2.0
        hr_upscaler: str = "Latent"
        hr_distilled_cfg: float = 3.5
        denoising_strength: float = 0.45

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
        If triggered, silences the actual LLM and runs the generation logic internally.
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

        # Metadata Discovery (Required for _suppress_output context)
        try:
            metadata = body.get("metadata", {})
            self.st.model.llm_type = metadata.get("model", {}).get("owned_by", "ollama")
        except:
            self.st.model.llm_type = "ollama"

        # 2. IMMEDIATE SUPPRESSION (Neutralize the LLM while EM works)
        body = self._suppress_output(body)

        # 3. WORKER INITIALIZATION
        self.debug = DebugService(self)
        self.inf = InferenceEngine(self)
        self.img_gen = ImageGenEngine(self)
        self.parser = PromptParser(self)

        try:
            # 4. LOGIC EXECUTION
            await self._setup_context(body, raw_p)

            # Fork logic based on trigger type
            if self.st.model.trigger == "imgx":
                # Prompt Engineering Mode only
                self.st.output_content = self.st.model.enhanced_prompt
                self.st.executed = True
            else:

                # Ensure the GPU is clear of 'ghost' models before calling Forge/Comfy
                await self.em.emit_status("Cleaning VRAM..")
                await self.inf.purge_vram(unload_all=self.valves.extreme_vram_cleanup)

                # Full Generation Mode
                await self.img_gen.generate()
                # Save the image reference to the buffer immediately
                self.st.output_content = (
                    f"![Generated Image]({self.st.model.image_url})"
                )

                if self.st.model.quality_audit:
                    await self._vision_audit()

                # Add citation placeholders for outlet to fill
                self.st.output_content += "\n\n[1] [2] [3]"
                self.st.executed = True

        except Exception as e:
            await self.debug.error(e)
            if self.st:
                self.st.executed = True  # Allows the outlet to display the error

        return body  # body has already been modified by _suppress_output

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        """
        Intercepts the LLM's response (which is just empty space/silence).
        Replaces the silence with the actual content generated by Easymage.
        Delivers citations/status updates.
        """

        if getattr(self, "st", None) is None or not self.st.executed:
            return body

        if not self.st.model.trigger:
            return body

        if __event_emitter__:
            self.em.emitter = __event_emitter__

        # FINAL OVERWRITE
        # Whatever the LLM generated (the invisible space), is now replaced by EM's definitive content.
        if "messages" in body and len(body["messages"]) > 0:
            body["messages"][-1]["content"] = self.st.output_content

        if self.st.model.trigger == "imgx":
            await self._output_status_only()
        else:
            await self._output_delivery()

        return body

    def _suppress_output(self, body: dict) -> dict:
        """
        Force the LLM to be completely silent using invisible characters
        and strict system instructions. Prevents double-response issues.
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

        # Specific handling for Ollama to minimize resource usage
        if self.st and self.st.model.llm_type == "ollama":
            if "options" not in body:
                body["options"] = {}
            body["options"].update({"num_predict": 1, "temperature": 0.0})

        return body

    async def _setup_context(self, body: dict, user_prompt: str):
        """
        Aggregates configuration from Global Settings, Valves, and User Input.
        Runs Prompt Enhancement if enabled.
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

        # 3. Prompt Enhancement (Creative Mode)
        E = self.config.Engines
        # Determine if the backend supports negative prompts natively
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

    def _apply_global_settings(self):
        """
        Merges global environment variables from Open WebUI config into the state.
        Mapping is handled by keys defined in EasymageConfig.
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

    def _validate_and_normalize(self):
        """
        Ensures model parameters (size, aspect ratio, models) are valid for the selected engine.
        Enforces specific constraints for DALL-E 3 (valid aspect ratios).
        """
        eng, mdl = (
            self.st.model.get("engine"),
            str(self.st.model.get("model", "")).lower(),
        )
        E = self.config.Engines

        # Enforce defaults for specific engines
        if eng == E.OPENAI and "dall-e" not in mdl:
            self.st.model["model"] = "dall-e-3"
        elif eng == E.GEMINI and "imagen" not in mdl:
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

        # Adjust dimensions for DALL-E specific aspect ratios
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

    def _check_input(self, body: dict) -> Optional[Tuple[str, str]]:
        """
        Parses the last user message to detect Easymage triggers ('img' or 'imgx').
        Handles both string and multimodal list content robustly.
        """
        if body.get("is_probe"):
            return None

        msgs = body.get("messages", [])
        if not msgs:
            return None

        # Isolate the content of the last message
        last = msgs[-1].get("content", "")

        # Robust text extraction for multimodal compatibility
        if isinstance(last, list):
            # Find the first text block in the multimodal list, regardless of its position.
            # This is crucial for inputs where an image precedes the text prompt.
            txt = next(
                (
                    item.get("text", "")
                    for item in last
                    if isinstance(item, dict) and item.get("type") == "text"
                ),
                "",
            )
        else:
            # Handle standard string content
            txt = last

        # Now `txt` is guaranteed to be a string, safe for re.match
        m = re.match(r"^(img|imgx)\s", str(txt), re.IGNORECASE)

        return (m.group(1).lower(), txt[m.end() :].strip()) if m else None

    async def _check_vision_capability(self):
        """
        Verifies if the current LLM supports Vision capabilities.
        Uses a small black pixel test image to probe the model.
        Results are cached to disk to improve performance.
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
        Recursively unwraps Open WebUI configuration objects (SimpleNamespace/PersistentConfig) to get raw values.
        """
        if hasattr(obj, "value"):
            return self._unwrap(obj.value)
        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}
        return obj

    def _clean_key(self, key, eng):
        """
        Normalizes environment variable keys to match Easymage internal state keys.
        Removes engine prefixes and standard prefixes.
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
