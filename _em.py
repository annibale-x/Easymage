"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.8.12
repo_url: https://github.com/annibale-x/Easymage
author: Hannibal
author_url: https://openwebui.com/u/h4nn1b4l
author_email: annibale.x@gmail.com
Description: Image generation filter and prompt enhancer for Open WebUI.
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
from open_webui.main import generate_chat_completion  # type: ignore

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
        """Logs duration and token usage for performance metrics."""

        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tps = token_count / elapsed if elapsed > 0 else 0
        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tps:.1f} tk/s"
        )


# --- SERVICES ---


class EmitterService:
    """Handles communication with event emitter. Formatting for UI."""

    def __init__(self, event_emitter):

        self.emitter = event_emitter

    async def emit_message(self, content: str):
        """Sends a message update to the chat UI with code block formatting."""

        if self.emitter:
            fmt = (
                content.replace("```", "```\n") if content.endswith("```") else content
            )

            if "```" in content:
                fmt = re.sub(r"(```[^\n]*\n.*?\n```)", r"\1\n", fmt, flags=re.DOTALL)

            await self.emitter({"type": "message", "data": {"content": fmt}})

    async def emit_status(self, description: str, done: bool = False):
        """Sends a status indicator update to the chat UI."""

        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def emit_citation(self, name: str, document: str, source: str, cid: str):
        """Sends a formal citation record to the chat UI."""

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


class InferenceEngine:
    """Generalized inference engine for Text, Vision, and Probe tasks."""

    def __init__(self, request, user, state: EasymageState, emitter: EmitterService):

        self.request, self.user, self.state, self.emitter = (
            request,
            user,
            state,
            emitter,
        )

    async def _infer(self, task: Optional[str], data: Dict[str, Any]) -> str:
        """Executes LLM chat completions and extracts content, cleaning reasoning tags."""

        start = time.time()

        if task:
            await self.emitter.emit_status(f"{task}..")

        msgs = []

        if sys_c := data.get("system"):
            msgs.append({"role": "system", "content": sys_c})

        user_c = data.get("user")

        if isinstance(user_c, dict):
            cl = []

            if i_url := user_c.get("image_url"):
                cl.append({"type": "image_url", "image_url": {"url": i_url}})

            if t_c := user_c.get("text"):
                cl.append({"type": "text", "text": t_c})

            msgs.append({"role": "user", "content": cl})

        else:
            msgs.append({"role": "user", "content": str(user_c)})

        payload = {
            "model": self.state.model.id,
            "messages": msgs,
            "stream": False,
            "seed": 42,
            "temperature": 0.0,
            "is_probe": True,
        }

        try:
            resp = await generate_chat_completion(self.request, payload, self.user)

            if resp:
                cont = resp["choices"][0]["message"].get("content", "").strip()
                cont = cont.split("</think>")[-1].strip()

                if task:
                    self.state.register_stat(
                        task,
                        time.time() - start,
                        resp.get("usage", {}).get("total_tokens", 0),
                    )

                return cont.strip('"').strip()

            return ""

        except Exception as e:
            print(f"Inference Error ({task}): {e}", file=sys.stderr)
            return ""


class PromptParser:
    """Regex based input parsing and parameter normalization."""

    def __init__(self, config: EasymageConfig):

        self.config = config

    def parse(self, user_prompt: str, model_state: Store):
        """Deconstructs user input to extract triggers, technical params, and subject."""

        clean, neg = user_prompt, ""

        if " --no " in clean.lower():
            clean, neg = re.split(r" --no ", clean, maxsplit=1, flags=re.IGNORECASE)

        if " -- " in clean:
            prefix, subj = clean.split(" -- ", 1)

        else:
            pat = r'(\b\w+=(?:"[^"]*"|\S+))|([+-][dpah])'
            matches = list(re.finditer(pat, clean))
            idx = matches[-1].end() if matches else 0
            prefix, subj = clean[:idx], clean[idx:].strip()

        rem_styles = re.sub(r'(\b\w+=(?:"[^"]*"|\S+))|([+-][dpah])', "", prefix).strip()
        rem_styles = re.sub(r"^[\s,]+|[\s,]+$", "", rem_styles)

        # Parse key=value pairs
        kv_pat = r'(\b\w+)=("([^"]*)"|(\S+))'

        for k, _, q, u in re.findall(kv_pat, prefix):
            k, v = k.lower(), (q if q else u).lower()

            try:

                if k == "ge":
                    model_state["engine"] = {
                        "a": EasymageConfig.Engines.FORGE,
                        "o": EasymageConfig.Engines.OPENAI,
                        "g": EasymageConfig.Engines.GEMINI,
                        "c": EasymageConfig.Engines.COMFY,
                    }.get(v, v)

                elif k == "mdl":
                    model_state["model"] = {
                        "d3": "dall-e-3",
                        "d2": "dall-e-2",
                        "i3": "imagen-3.0-generate-001",
                        "i3f": "imagen-3.0-fast-generate-001",
                    }.get(v, v)

                elif k == "stp":
                    model_state["steps"] = int(v)

                elif k == "sz":
                    model_state["size"] = v if "x" in str(v) else f"{v}x{v}"

                elif k == "ar":
                    model_state["aspect_ratio"] = {
                        "1": "1:1",
                        "16": "16:9",
                        "9": "9:16",
                        "4": "4:3",
                        "3": "3:4",
                        "21": "21:9",
                    }.get(str(v), str(v))

                elif k == "stl":
                    model_state["style"] = {"v": "vivid", "n": "natural"}.get(v, v)

                elif k == "sd":
                    model_state["seed"] = int(v)

                elif k == "smp":
                    model_state["sampler_name"] = self._norm(v, self.config.SAMPLER_MAP)

                elif k == "sch":
                    model_state["scheduler"] = self._norm(v, self.config.SCHEDULER_MAP)

                elif k == "n":
                    model_state["n_iter"] = int(v)

                elif k == "b":
                    model_state["batch_size"] = int(v)

                elif k == "cs":
                    model_state["cfg_scale"] = float(v)

                elif k == "dcs":
                    model_state["distilled_cfg_scale"], model_state["cfg_scale"] = (
                        float(v),
                        1.0,
                    )

                elif k in ["hr", "hru", "hdcs", "dns"]:
                    model_state["enable_hr"] = True

                    if k == "hr":
                        model_state["hr_scale"] = float(v)

                    elif k == "hru":
                        model_state["hr_upscaler"] = v

                    elif k == "hdcs":
                        model_state["hr_distilled_cfg"] = float(v)

                    elif k == "dns":
                        model_state["denoising_strength"] = float(v)

            except ValueError:
                continue

        # Parse flags like +d, -p
        for flag in re.findall(r"([+-][dpah])", prefix):
            v, char = flag[0] == "+", flag[1]

            if char == "d":
                model_state["debug"] = v

            elif char == "p":
                model_state["enhanced_prompt"] = v

            elif char == "a":
                model_state["quality_audit"] = v

            elif char == "h":
                model_state["enable_hr"] = v

        model_state.update(
            {
                "user_prompt": subj.strip(),
                "negative_prompt": neg.strip(),
                "styles": rem_styles,
            }
        )

    def _norm(self, name, mapping):
        """Helper to normalize technical names into engine-compatible strings."""

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
        generation_timeout: int = Field(
            default=120, description="HTTP Timeout for Image Gen (seconds)"
        )
        steps: int = 20
        size: str = "1024x1024"
        seed: int = -1
        cfg_scale: float = 1.0
        distilled_cfg_scale: float = 3.5
        sampler_name: str = "Euler"
        scheduler: str = "Simple"
        enable_hr: bool = False
        n_iter: int = 1
        batch_size: int = 1
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
        """Core entry point for the filter middleware."""

        trigger_data = self._check_input(body)

        if not trigger_data:
            return body

        trigger, raw_p = trigger_data

        # Initialize internal state and services
        self.st, self.em = EasymageState(self.valves), EmitterService(__event_emitter__)
        self.st.model.trigger, self.request, self.user = (
            trigger,
            __request__,
            UserModel(**__user__),
        )
        self.inf, self.parser = InferenceEngine(
            self.request, self.user, self.st, self.em
        ), PromptParser(self.config)

        try:
            # Process prompt and configuration
            await self._setup_context(body, raw_p)

            if self.st.model.debug:
                await self.em.emit_message(
                    f"```json\nDEBUG MODEL: {json.dumps(self.st.model, indent=2)}\n```\n"
                )
                await self.em.emit_message(
                    f"```json\nDEBUG VALVES: {json.dumps(self.valves.model_dump(), indent=2)}\n```\n"
                )

            # Workflow execution based on trigger
            if self.st.model.trigger == "imgx":
                await self.em.emit_message(self.st.model.enhanced_prompt)
                await self._output_status_only()

            else:
                await self._generate_image()
                await self._vision_audit()
                await self._output_delivery()

        except Exception as e:
            await self._err(e)

        # Final state dump for console monitoring
        self._dmp()

        return self._suppress_output(body)

    async def _setup_context(self, body: dict, user_prompt: str):
        """Prepares metadata, performs language detection, and enhances prompt via LLM."""

        if "features" in body:
            body["features"]["web_search"] = False

        self.st.model.id = body.get("model", "")
        self._apply_global_settings()
        self.st.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )

        # 1. Parse input to override defaults with user choices
        self.parser.parse(user_prompt, self.st.model)

        if self.st.model.trigger == "img":
            await self._check_vision_capability()

        # 2. Language Detection
        dl = await self.inf._infer(
            task="Language Detection",
            data={
                "user": self.st.model.user_prompt,
                "system": "Return ONLY the language name of the user text.",
            },
        )
        self.st.model.language = dl if dl else "English"

        # 3. Decide strategy for negative prompt handling
        E = self.config.Engines
        native_support = (self.st.model.engine == E.GEMINI) or (
            self.st.model.engine == E.FORGE and self.st.model.enable_hr
        )
        use_llm_neg = (self.st.model.trigger == "imgx") or (not native_support)

        # 4. LLM-based Prompt Enhancement
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

            self._dbg("SYSTEM PROMPT\n" + "".join(instructions))
            self._dbg(f"USER PROMPT: {user_content}")

            enh = await self.inf._infer(
                task="Prompt Enhancing",
                data={
                    "system": "\n".join(instructions),
                    "user": user_content,
                },
            )
            self.st.model.enhanced_prompt = (
                re.sub(r"['\"]", "", enh) if enh else self.st.model.user_prompt
            )

        else:
            self.st.model.enhanced_prompt = re.sub(
                r"['\"]", "", self.st.model.user_prompt
            )

        self._validate_and_normalize()

    async def _generate_image(self):
        """Dispatcher for triggering the correct generation backend."""

        await self.em.emit_status("Generating Image..")
        start = time.time()
        eng = self.st.model.engine
        E = self.config.Engines

        try:

            if eng == E.OPENAI:
                await self._gen_openai()

            elif eng == E.GEMINI:
                await self._gen_gemini()

            elif eng == E.FORGE:
                await self._gen_forge()

            else:
                await self._gen_standard()

            self.st.image_gen_time_int = int(time.time() - start)

            if self.st.model.image_url:
                await self.em.emit_message(
                    f"![Generated Image]({self.st.model.image_url})"
                )

            else:
                raise Exception("No image URL returned")

        except Exception as e:
            await self._err(f"Gen Failed ({eng}): {e}")

    # --- PAYLOAD BUILDERS (SoC) ---

    def _prepare_payload_forge(self) -> dict:
        """Constructs the JSON payload for Automatic1111/Forge."""

        p = self.st.model.copy()
        p["prompt"] = p["enhanced_prompt"]

        if not p.get("enable_hr"):
            p["negative_prompt"] = ""

        try:
            w, h = map(int, p.get("size", "1024x1024").split("x"))
            p["width"], p["height"] = w, h

        except:
            p["width"], p["height"] = 1024, 1024

        exclude = [
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
        ]

        for k in exclude:
            p.pop(k, None)

        return p

    def _prepare_payload_openai(self) -> dict:
        """Constructs the JSON payload for OpenAI DALL-E API."""

        return {
            "model": self.st.model.model or "dall-e-3",
            "prompt": self.st.model.enhanced_prompt,
            "n": 1,
            "size": self.st.model.size or "1024x1024",
            "quality": "hd" if self.st.model.enable_hr else "standard",
            "style": "natural" if self.st.model.style == "natural" else "vivid",
            "response_format": "b64_json",
            "user": self.user.id,
        }

    def _prepare_payload_gemini(self) -> dict:
        """Constructs the JSON payload for Google Gemini Imagen API."""

        try:
            w, h = map(int, (self.st.model.size or "1024x1024").split("x"))
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
            "sampleCount": self.st.model.get("n_iter", 1),
            "negativePrompt": self.st.model.negative_prompt,
            "aspectRatio": ar,
            "safetySetting": "block_none",
            "personGeneration": "allow_all",
            "addWatermark": False,
            "includeReasoning": False,
            "outputOptions": {"mimeType": "image/png"},
        }

        if self.st.model.seed and self.st.model.seed != -1:
            params["seed"] = self.st.model.seed

        return {
            "instances": [{"prompt": self.st.model.enhanced_prompt}],
            "parameters": params,
        }

    # --- HTTP HANDLERS ---

    async def _gen_forge(self):
        """Executes the HTTP request to a local or remote Forge backend."""

        payload = self._prepare_payload_forge()
        conf = self.request.app.state.config
        url = conf.AUTOMATIC1111_BASE_URL.rstrip("/")
        auth = conf.AUTOMATIC1111_API_AUTH
        headers = (
            {
                "Authorization": (
                    f"Basic {base64.b64encode(auth.encode()).decode()}"
                    if auth and ":" in auth
                    else f"Bearer {auth}"
                )
            }
            if auth
            else {}
        )

        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{url}/sdapi/v1/txt2img",
                json=payload,
                headers=headers,
                timeout=self.valves.generation_timeout,
            )
            r.raise_for_status()
            res = r.json()

        img = res["images"][0].split(",")[-1]
        self.st.model.b64_data, self.st.model.image_url = (
            img,
            f"data:image/png;base64,{img}",
        )

    async def _gen_openai(self):
        """Executes the HTTP request to the OpenAI image generation endpoint."""

        payload = self._prepare_payload_openai()
        conf = self.request.app.state.config
        headers = {
            "Authorization": f"Bearer {conf.IMAGES_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{conf.IMAGES_OPENAI_API_BASE_URL}/images/generations",
                json=payload,
                headers=headers,
                timeout=self.valves.generation_timeout,
            )
            r.raise_for_status()
            img = r.json()["data"][0]["b64_json"]

            self.st.model.b64_data, self.st.model.image_url = (
                img,
                f"data:image/png;base64,{img}",
            )

    async def _gen_gemini(self):
        """Executes the HTTP request to the Google Gemini predict endpoint."""

        payload = self._prepare_payload_gemini()
        conf = self.request.app.state.config
        url = f"{conf.IMAGES_GEMINI_API_BASE_URL}/models/{self.st.model.model}:predict"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": conf.IMAGES_GEMINI_API_KEY,
        }

        async with httpx.AsyncClient() as c:
            r = await c.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.valves.generation_timeout,
            )
            r.raise_for_status()
            img = r.json()["predictions"][0]["bytesBase64Encoded"]

            self.st.model.b64_data, self.st.model.image_url = (
                img,
                f"data:image/png;base64,{img}",
            )

    async def _gen_standard(self):
        """Uses the built-in Open WebUI image router as a fallback backend."""

        form = CreateImageForm(
            prompt=self.st.model.enhanced_prompt,
            n=1,
            size=self.st.model.size,
            model=self.st.model.model,
        )
        gen = await image_generations(self.request, form, self.user)

        if gen:
            self.st.model.image_url = gen[0]["url"]

    async def _vision_audit(self):
        """Analyzes the generated image for quality and alignment using Vision LLM."""

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
        """Final output step: emits citations, metrics, and summary to the UI."""

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
        """Minimal output emission when no image generation is performed."""

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
        """Fetches and normalizes initial image settings from OWUI global state."""

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
        """Sanitizes model names and dimensions based on engine-specific constraints."""

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
        """Scans incoming chat body for active filter trigger commands."""

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
        """Probes model's ability to process image inputs via diagnostic test."""

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
        """Utility for recursively extracting values from complex configuration objects."""

        if hasattr(obj, "value"):
            return self._unwrap(obj.value)

        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}

        return obj

    def _clean_key(self, key, eng):
        """Normalizes global app state keys to match internal state dictionary keys."""

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
        """Modifies the request payload to halt default LLM text response generation."""

        body["messages"] = [{"role": "assistant", "content": ""}]
        body["max_tokens"], body["stop"] = 1, [chr(i) for i in range(128)]

        return body

    def _dbg(self, msg: str):
        """Prints diagnostic logs to standard error for Docker logs visibility."""

        if self.valves.debug:
            print(f"⚡ EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def _err(self, e: Any):
        """Logs exceptions and notifies the chat UI of critical filter errors."""

        self._dbg(f"ERROR: {e}")
        await self.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def _dmp(self):
        """Performs a full state serialization and dump to stderr for debugging."""

        if not self.valves.debug:
            return

        print(
            f"{'—'*40}\n📦 EASYMAGE DUMP\n{json.dumps(self.st.model, indent=2)}\n{'—'*40}",
            file=sys.stderr,
            flush=True,
        )
