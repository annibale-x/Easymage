"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.8.16
repo_url: https://github.com/annibale-x/Easymage
author: Hannibal
author_url: https://openwebui.com/u/h4nn1b4l
author_email: annibale.x@gmail.com
Description: Image generation filter and prompt enhancer for Open WebUI with deterministic control.
"""

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

# --- CONFIGURATION & MAPS ---


class EasymageConfig:
    """Static mappings and prompt templates."""

    class Engines:
        FORGE = "automatic1111"
        OPENAI = "openai"
        GEMINI = "gemini"
        COMFY = "comfyui"

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

    ENGINE_MAP = {
        Engines.FORGE: ["AUTOMATIC1111_BASE_URL", "AUTOMATIC1111_PARAMS"],
        Engines.COMFY: ["COMFYUI_WORKFLOW", "COMFYUI_WORKFLOW_NODES"],
        Engines.OPENAI: ["IMAGES_OPENAI_API_BASE_URL", "IMAGES_OPENAI_API_PARAMS"],
        Engines.GEMINI: ["IMAGES_GEMINI_API_KEY", "IMAGES_GEMINI_ENDPOINT_METHOD"],
    }

    # PROMPT TEMPLATES
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
    """Dynamic dictionary storage with dot notation access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EasymageState:
    """Manages internal state, configuration, and performance tracking."""

    def __init__(self, valves):
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

    def register_stat(self, stage_name: str, elapsed: float, token_count: int = 0):
        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tps = token_count / elapsed if elapsed > 0 else 0
        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tps:.1f} tk/s"
        )


# --- SERVICES ---


class EmitterService:
    """Handles communication with event emitter."""

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
    """Handles stderr logging, error reporting, and safe object dumping."""

    def __init__(self, ctx):
        self.ctx = ctx

    def log(self, msg: str, is_error: bool = False):
        """Prints diagnostic logs to standard error for Docker logs visibility."""
        if self.ctx.valves.debug or is_error:
            if is_error:
                print(f"\n\n❌ EASYMAGE ERROR: {msg}\n", file=sys.stderr, flush=True)
            else:
                print(f"⚡ EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def error(self, e: Any):
        """Logs exceptions and notifies the chat UI of critical filter errors."""
        self.log(str(e), is_error=True)
        await self.ctx.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def dump(self, data: Any = None):
        """
        Indestructible Surgical Dumper
        Handles the new PersistentConfig objects from Open WebUI
        """
        if not self.ctx.valves.debug:
            return

        header = "—" * 60 + "\n📦 EASYMAGE DUMP:\n" + "—" * 60
        print(header, file=sys.stderr, flush=True)

        def universal_resolver(obj):
            # Intercept Open WebUI PersistentConfig objects
            if hasattr(obj, "value"):
                return obj.value
            # Intercept Pydantic models
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            # Fallback
            return str(obj)

        try:
            clean_json = json.dumps(data, indent=2, default=universal_resolver)
            print(clean_json, file=sys.stderr, flush=True)
        except Exception as e:
            print(f"❌ DUMP ERROR: {str(e)}", file=sys.stderr, flush=True)

        print("—" * 60, file=sys.stderr, flush=True)


class InferenceEngine:
    """Handles LLM interactions (Text & Vision) with context access."""

    def __init__(self, ctx):
        self.ctx = ctx  # Access to Filter instance

    async def _infer(
        self, task: Optional[str], data: Dict[str, Any], creative_mode: bool = False
    ) -> str:
        start = time.time()
        if task:
            await self.ctx.em.emit_status(f"{task}..")

        # Prepare Messages
        msgs = []
        if sys_c := data.get("system"):
            clean_sys = re.sub(r"\s+", " ", sys_c).strip()
            msgs.append({"role": "system", "content": clean_sys})

        user_c = data.get("user")
        if isinstance(user_c, dict):
            cl = []
            if i_url := user_c.get("image_url"):
                cl.append({"type": "image_url", "image_url": {"url": i_url}})
            if t_c := user_c.get("text"):
                clean_text = re.sub(r"\s+", " ", t_c).strip()
                cl.append({"type": "text", "text": clean_text})
            msgs.append({"role": "user", "content": cl})
        else:
            clean_user = re.sub(r"\s+", " ", str(user_c)).strip()
            msgs.append({"role": "user", "content": clean_user})

        # Configure Determinism
        if creative_mode:
            # Mode: PROMPT ENHANCING
            # Temp 0.7 allows the Seed to influence the choice path.
            temperature = 0.7

            s_val = (
                int(self.ctx.st.model.seed)
                if self.ctx.st.model.seed is not None
                else -1
            )

            if s_val == -1:
                seed = None  # True Random
            else:
                seed = s_val  # Deterministic Randomness
        else:
            # Mode: TECHNICAL / AUDIT
            # Temp 0.0 forces Greedy Decoding (Seed is technically ignored but we pass 42)
            seed, temperature = 42, 0.0

        # Dispatch
        backend_type = getattr(self.ctx.st.model, "llm_type", "ollama")

        try:
            if backend_type == "ollama":
                content, usage = await self._infer_ollama(msgs, seed, temperature)
            else:
                content, usage = await self._infer_openai(msgs, seed, temperature)

            content = content.split("</think>")[-1].strip().strip('"').strip()

            if task:
                self.ctx.st.register_stat(task, time.time() - start, usage)

            return content

        except Exception as e:
            self.ctx.debug.log(f"INFERENCE ERROR ({task}): {e}", is_error=True)
            return ""

    async def _infer_ollama(
        self, messages: List[Dict], seed: Optional[int], temperature: float
    ) -> Tuple[str, int]:
        conf = self.ctx.request.app.state.config
        base_urls = getattr(conf, "OLLAMA_BASE_URLS", [])
        base_url = base_urls[0].rstrip("/") if base_urls else "http://localhost:11434"

        options = {
            "temperature": temperature,
            "num_ctx": 4096,
            "mirostat": 0,
        }

        if seed is not None:
            options["seed"] = seed
            # CRITICAL FIX: Removed options["top_k"] = 1
            # We let the model sample (top_k default), controlled by the seed.

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

        async with httpx.AsyncClient(
            timeout=self.ctx.valves.generation_timeout
        ) as client:
            r = await client.post(f"{base_url}/api/chat", json=payload)
            r.raise_for_status()
            res = r.json()
            return (
                res.get("message", {}).get("content", ""),
                res.get("eval_count", 0),
            )

    async def _infer_openai(
        self, messages: List[Dict], seed: Optional[int], temperature: float
    ) -> Tuple[str, int]:
        conf = self.ctx.request.app.state.config
        base_urls = getattr(conf, "OPENAI_API_BASE_URLS", [])
        api_keys = getattr(conf, "OPENAI_API_KEYS", [])
        base_url = (
            base_urls[0].rstrip("/") if base_urls else "https://api.openai.com/v1"
        )
        api_key = api_keys[0] if api_keys else ""

        payload = {
            "model": self.ctx.st.model.id,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }

        if seed is not None:
            payload["seed"] = seed
            # CRITICAL FIX: Removed payload["top_p"] restrictions.
            # Allowing full sampling controlled by seed.

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if self.ctx.st.model.debug:
            self.ctx.debug.log(
                f"OPENAI: {base_url}/chat/completions (s:{seed} t:{temperature})"
            )

        async with httpx.AsyncClient(
            timeout=self.ctx.valves.generation_timeout
        ) as client:
            r = await client.post(
                f"{base_url}/chat/completions", json=payload, headers=headers
            )
            r.raise_for_status()
            res = r.json()
            return (
                res["choices"][0]["message"].get("content", ""),
                res.get("usage", {}).get("completion_tokens", 0),
            )


class ImageGenEngine:
    """Handles image generation logic with direct context access."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def generate(self) -> str:
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

    def _prepare_gemini(self) -> dict:
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

        async with httpx.AsyncClient() as c:
            r = await c.post(
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
        payload = self._prepare_openai()
        conf = self.ctx.request.app.state.config
        headers = {
            "Authorization": f"Bearer {conf.IMAGES_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as c:
            r = await c.post(
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
        payload = self._prepare_gemini()
        conf = self.ctx.request.app.state.config
        url = f"{conf.IMAGES_GEMINI_API_BASE_URL}/models/{self.ctx.st.model.model}:predict"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": conf.IMAGES_GEMINI_API_KEY,
        }
        async with httpx.AsyncClient() as c:
            r = await c.post(
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
    """Regex based input parsing directly modifying state model."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.config = ctx.config

    def parse(self, user_prompt: str):
        m_st = self.ctx.st.model  # Direct state access
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

        # Parse Pairs
        kv_pat = r'(\b\w+)=("([^"]*)"|(\S+))'
        for k, _, q, u in re.findall(kv_pat, prefix):
            k, v = k.lower(), (q if q else u).lower()
            try:
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

        # Parse Flags
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
    class Valves(BaseModel):
        enhanced_prompt: bool = Field(default=True)
        quality_audit: bool = Field(default=True)
        strict_audit: bool = Field(default=False)
        persistent_vision_cache: bool = Field(default=False)
        debug: bool = Field(default=False)
        model: Optional[str] = Field(default=None)
        generation_timeout: int = Field(default=120)
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

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:
        """Entry Point. Acts as the Context Container for all sub-engines."""

        trigger_data = self._check_input(body)
        if not trigger_data:
            return body

        trigger, raw_p = trigger_data

        # 1. SETUP CONTEXT (The Filter itself is the Context)
        self.request = __request__
        self.user = UserModel(**__user__)
        self.st = EasymageState(self.valves)
        self.em = EmitterService(__event_emitter__)
        self.st.model.trigger = trigger

        # Metadata Discovery
        try:
            metadata = body.get("metadata", {})
            self.st.model.llm_type = metadata.get("model", {}).get("owned_by", "ollama")
        except:
            self.st.model.llm_type = "ollama"

        # 2. INITIALIZE WORKERS (Passing 'self' as Context)
        # Note: DebugService comes first to handle early errors if needed
        self.debug = DebugService(self)
        self.inf = InferenceEngine(self)
        self.img_gen = ImageGenEngine(self)
        self.parser = PromptParser(self)

        try:
            # 3. CONTEXT POPULATION & LOGIC
            await self._setup_context(body, raw_p)

            if self.st.model.debug:
                self.debug.log(f"STATE: {json.dumps(self.st.model, indent=2)}")

            # 4. EXECUTION
            if self.st.model.trigger == "img":
                await self.img_gen.generate()
                await self._vision_audit()
                await self._output_delivery()

        except Exception as e:
            await self.debug.error(e)

        # self.debug.dump(self.st.model)
        return self._suppress_output(body)

    async def _setup_context(self, body: dict, user_prompt: str):
        if "features" in body:
            body["features"]["web_search"] = False

        self.st.model.id = body.get("model", "")
        self._apply_global_settings()
        self.st.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )

        # 1. Parse Input
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
                re.sub(r"['\"]", "", enh) if enh else self.st.model.user_prompt
            )
        else:
            self.st.model.enhanced_prompt = re.sub(
                r"['\"]", "", self.st.model.user_prompt
            )

        self._validate_and_normalize()

    async def _vision_audit(self):
        if not self.st.model.quality_audit or not self.st.model.vision:
            return
        await self.em.emit_status("Visual Quality Audit..", False)

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
            task=None,
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
        await self.em.emit_message("\n\n[1] [2] [3]")
        await self.em.emit_status(
            f"{total}s total | {self.st.image_gen_time_int}s img | {self.st.cumulative_tokens} tk | {tps:.1f} tk/s",
            True,
        )

    async def _output_status_only(self):
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
        eng, mdl = (
            self.st.model.get("engine"),
            str(self.st.model.get("model", "")).lower(),
        )
        E = self.config.Engines

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

        b = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAS0lEQVQ4jWNkYGB4ycDAwMPAwMDEgAn+ERD7wQLVzIVFITGABZutJAEmBuxOJ8kAil0w8AZgiyr6umAYGDDEA5GFgYHhB5QmB/wAAIcLCBsQodqvAAAAAElFTkSuQmCC"
        res = await self.inf._infer(
            task=None,
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
        if hasattr(obj, "value"):
            return self._unwrap(obj.value)
        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}
        return obj

    def _clean_key(self, key, eng):
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

    def _suppress_output(self, body: dict) -> dict:
        """Cache-friendly suppression: preserves prefix to reuse KV cache."""
        # Segnaliamo all'outlet che deve pulire l'output residuo

        self.st.easymage_skip = True

        # NON svuotiamo body["messages"]. Manteniamo il prefisso originale.
        # Questo permette a Ollama di rispondere in <100ms usando la cache.
        body["max_tokens"] = 1
        body["stream"] = False

        # Forza il backend a fermarsi immediatamente
        if self.st.model.llm_type == "ollama":
            body["options"] = {
                "num_predict": 1,
                "temperature": 0.0,
                "num_ctx": 4096,  # Deve corrispondere a quello usato in precedenza
            }

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Cleans up the residual token generated by the suppressed backend call."""
        if self.st.easymage_skip == True:
            if "messages" in body and len(body["messages"]) > 0:
                body["messages"][-1]["content"] = self.st.model.enhanced_prompt
            await self._output_status_only()
            return body

        return body
