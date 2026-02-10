"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.7.5
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
    """Static mappings for engine parameters and engines. Dictionaries are unalterable."""

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
        "automatic1111": ["AUTOMATIC1111_BASE_URL", "AUTOMATIC1111_PARAMS"],
        "comfyui": ["COMFYUI_WORKFLOW", "COMFYUI_WORKFLOW_NODES"],
        "openai": ["IMAGES_OPENAI_API_BASE_URL", "IMAGES_OPENAI_API_PARAMS"],
        "gemini": ["IMAGES_GEMINI_API_KEY", "IMAGES_GEMINI_ENDPOINT_METHOD"],
    }

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
                Identify any contradictions, missing elements, or hallucinations (like objects that shouldn4t be there).
                Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line.
        RULE: Be extremely severe in technical evaluation. Do not excuse defects as limitations of resolution or scale.
        TASK: TECHNICAL EVALUATION (Technical scores are 0 to 100, where 0 is LOW and 100 HIGH):
                Perform a ruthless technical audit. Identify every visual flaw.
                Evaluate NOISE as random pixel color variations.
                Evaluate GRAIN as textural salt-and-pepper luminance noise.
                Evaluate MELTING as lack of structural integrity, blurred textures, or wax-like surfaces.
                Evaluate JAGGIES as staircase artifacts and aliasing on diagonal lines and edges.
        MANDATORY: Respond in {lang}. Be objective. NO MARKDOWN. Use plain text and • for lists and ➔ for headings. 
        MANDATORY: Final response MUST end with a single line containing only the following metrics:
        SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
    """

    PROMPT_ENHANCE = """
        ROLE: You are an expert AI Image Prompt Engineer.
        TASK: Expand the user's input into a professional, highly detailed prompt in {lang}.
        TASK: Add details about lighting, camera angle, textures, environment, and artistic style.
        RULE: Output ONLY the enhanced prompt.
        RULE: End the response immediately after the last descriptive sentence.
        RULE: Do not add any text, notes, or disclaimers after the prompt.
        RULE: Any text after the prompt is a violation of your protocol.
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
    """Manages the internal state and configuration of a single request."""

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
        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tps = token_count / elapsed if elapsed > 0 else 0
        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tps:.1f} tk/s"
        )


# --- SERVICES ---


class EmitterService:
    """Handles communication with the Open WebUI event emitter."""

    def __init__(self, event_emitter):
        self.emitter = event_emitter

    async def emit_message(self, content: str):
        if self.emitter:
            # Ensure \n after any code block closing
            formatted_content = (
                content.replace("```", "```\n") if content.endswith("```") else content
            )
            if "```" in content:
                formatted_content = re.sub(
                    r"(```[^\n]*\n.*?\n```)",
                    r"\1\n",
                    formatted_content,
                    flags=re.DOTALL,
                )

            await self.emitter(
                {"type": "message", "data": {"content": formatted_content}}
            )

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


class InferenceEngine:
    """Handles all LLM interactions with generalized expansion for Text and Vision."""

    def __init__(self, request, user, state: EasymageState, emitter: EmitterService):
        self.request = request
        self.user = user
        self.state = state
        self.emitter = emitter

    async def _infer(self, task: Optional[str], data: Dict[str, Any]) -> str:
        """Generalized inference. Expands 'data' dictionary into standard message roles."""
        start_time = time.time()
        if task:
            await self.emitter.emit_status(f"{task}..")

        # 1. Message Construction Logic
        messages = []
        if sys_content := data.get("system"):
            messages.append({"role": "system", "content": sys_content})

        user_raw = data.get("user")
        if isinstance(user_raw, dict):
            # Vision / Multi-content handling
            content_list = []
            if img_url := user_raw.get("image_url"):
                content_list.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            if text_content := user_raw.get("text"):
                content_list.append({"type": "text", "text": text_content})
            messages.append({"role": "user", "content": content_list})
        else:
            # Simple string handling
            messages.append({"role": "user", "content": str(user_raw)})

        # 2. Payload Preparation
        payload = {
            "model": self.state.model.id,
            "messages": messages,
            "stream": False,
            "seed": 42,
            "temperature": 0.0,
            "is_probe": True,
        }

        # 3. Request Execution
        try:
            response = await generate_chat_completion(self.request, payload, self.user)
            if response:
                content = response["choices"][0]["message"].get("content", "").strip()
                # Maintain internal logic: Clean thinking tags
                content = content.split("</think>")[-1].strip()
                # content = re.sub(r"</?text>", "", content).strip()

                if task:
                    self.state.register_stat(
                        task,
                        time.time() - start_time,
                        response.get("usage", {}).get("total_tokens", 0),
                    )
                return content.strip('"')
            return ""
        except Exception as e:
            print(f"Inference Error ({task}): {e}", file=sys.stderr)
            return ""


class PromptParser:
    """Handles prompt parsing, regex extraction and parameter normalization."""

    def __init__(self, config: EasymageConfig):
        self.config = config

    def parse_input(self, user_prompt: str, model_state: Store):
        clean_prompt = user_prompt
        negative_prompt = ""
        if " --no " in clean_prompt.lower():
            clean_prompt, negative_prompt = re.split(
                r" --no ", clean_prompt, maxsplit=1, flags=re.IGNORECASE
            )

        if " -- " in clean_prompt:
            prefix, subject = clean_prompt.split(" -- ", 1)
        else:
            tech_pattern = r'(\b\w+=(?:"[^"]*"|\S+))|([+-][dpah])'
            matches = list(re.finditer(tech_pattern, clean_prompt))
            split_idx = matches[-1].end() if matches else 0
            prefix, subject = clean_prompt[:split_idx], clean_prompt[split_idx:].strip()

        tech_and_flags_pattern = r'(\b\w+=(?:"[^"]*"|\S+))|([+-][dpah])'
        remaining_styles = re.sub(tech_and_flags_pattern, "", prefix).strip()
        remaining_styles = re.sub(r"^[\s,]+|[\s,]+$", "", remaining_styles)

        param_pattern = r'(\b\w+)=("([^"]*)"|(\S+))'
        for k, _, q_val, u_val in re.findall(param_pattern, prefix):
            k, val = k.lower(), (q_val if q_val else u_val).lower()
            try:
                if k == "ge":
                    model_state["engine"] = {
                        "a": "automatic1111",
                        "o": "openai",
                        "g": "gemini",
                        "c": "comfyui",
                    }.get(val, val)
                elif k == "mdl":
                    model_state["model"] = {
                        "d3": "dall-e-3",
                        "d2": "dall-e-2",
                        "i3": "imagen-3.0-generate-001",
                        "i3f": "imagen-3.0-fast-generate-001",
                    }.get(val, val)
                elif k == "stp":
                    model_state["steps"] = int(val)
                elif k == "sz":
                    model_state["size"] = val if "x" in str(val) else f"{val}x{val}"
                elif k == "ar":
                    model_state["aspect_ratio"] = {
                        "1": "1:1",
                        "16": "16:9",
                        "9": "9:16",
                        "4": "4:3",
                        "3": "3:4",
                        "21": "21:9",
                    }.get(str(val), str(val))
                elif k == "stl":
                    model_state["style"] = {"v": "vivid", "n": "natural"}.get(val, val)
                elif k == "sd":
                    model_state["seed"] = int(val)
                elif k == "smp":
                    model_state["sampler_name"] = self._normalize(
                        val, self.config.SAMPLER_MAP
                    )
                elif k == "sch":
                    model_state["scheduler"] = self._normalize(
                        val, self.config.SCHEDULER_MAP
                    )
                elif k == "n":
                    model_state["n_iter"] = int(val)
                elif k == "b":
                    model_state["batch_size"] = int(val)
                elif k == "cs":
                    model_state["cfg_scale"] = float(val)
                elif k == "dcs":
                    model_state["distilled_cfg_scale"] = float(val)
                    model_state["cfg_scale"] = 1.0
                elif k in ["hr", "hru", "hdcs", "dns"]:
                    model_state["enable_hr"] = True
                    if k == "hr":
                        model_state["hr_scale"] = float(val)
                    elif k == "hru":
                        model_state["hr_upscaler"] = val
                    elif k == "hdcs":
                        model_state["hr_distilled_cfg"] = float(val)
                    elif k == "dns":
                        model_state["denoising_strength"] = float(val)
            except ValueError:
                continue

        for flag in re.findall(r"([+-][dpah])", prefix):
            val, char = flag[0] == "+", flag[1]
            if char == "d":
                model_state["debug"] = val
            elif char == "p":
                model_state["enhanced_prompt"] = val
            elif char == "a":
                model_state["quality_audit"] = val
            elif char == "h":
                model_state["enable_hr"] = val

        model_state.update(
            {
                "user_prompt": subject.strip(),
                "negative_prompt": negative_prompt.strip(),
                "styles": remaining_styles,
            }
        )

    def _normalize(self, name, mapping):
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


# --- FILTER LOGIC ---


class Filter:
    class Valves(BaseModel):
        enhanced_prompt: bool = Field(
            default=True, description="Enrich prompt details."
        )
        quality_audit: bool = Field(
            default=True, description="Post-generation Image Quality Audit."
        )
        strict_audit: bool = Field(
            default=False,
            description="Enable ruthless technical evaluation (Strict Mode).",
        )
        persistent_vision_cache: bool = Field(
            default=False, description="Saves vision probe results to disk."
        )
        debug: bool = Field(default=False, description="Enable debug mode.")
        model: Optional[str] = Field(
            default=None, description="Force generation model."
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
        trigger_data = self._check_input(body)
        if not trigger_data:
            return body

        trigger, user_prompt_raw = trigger_data

        # SoC Initialization
        self.st = EasymageState(self.valves)
        self.st.model.trigger = trigger
        self.em = EmitterService(__event_emitter__)
        self.request = __request__
        self.user = UserModel(**__user__)
        self.inf = InferenceEngine(self.request, self.user, self.st, self.em)
        self.parser = PromptParser(self.config)

        try:
            await self._setup_context(body, user_prompt_raw)

            # Debug Emit in Chat
            if self.st.model.debug:
                await self.em.emit_message(
                    f"```json\nDEBUG MODEL: {json.dumps(self.st.model, indent=2)}\n```\n"
                )
                await self.em.emit_message(
                    f"```json\nDEBUG VALVES: {json.dumps(self.valves.model_dump(), indent=2)}\n```\n"
                )

            if self.st.model.trigger == "imgx":
                await self.em.emit_message(self.st.model.enhanced_prompt)
                await self._output_status_only()
            else:
                await self._generate_image()
                await self._vision_audit()
                await self._output_delivery()
        except Exception as e:
            await self._err(e)

        self._dmp()
        return self._suppress_output(body)

    async def _setup_context(self, body: dict, user_prompt: str):
        if "features" in body:
            body["features"]["web_search"] = False
        self.st.model.id = body.get("model", "")
        self._apply_global_settings()
        self.st.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )
        self.parser.parse_input(user_prompt, self.st.model)

        # Vision Capability Probe - LOCAL OPTIMIZATION: skip if only text enhancement
        if self.st.model.trigger == "img":
            await self._check_vision_capability()

        # Language Detection - DO NOT ALTER INSTRUCTION
        detected_lang = await self.inf._infer(
            task="Language Detection",
            data={
                "user": self.st.model.user_prompt,
                "system": "Return ONLY the language name of the user text.",
            },
        )
        self.st.model.language = detected_lang if detected_lang else "English"

        # Enhance Prompt - DO NOT ALTER INSTRUCTION
        if self.st.model.enhanced_prompt or self.st.model.trigger == "imgx":

            instruction = self.config.PROMPT_ENHANCE.format(lang=self.st.model.language)




            enhanced = await self.inf._infer(
                task="Prompt Enhancing",
                data={
                    "system": instruction,
                    "user": f"Expand this prompt: {self.st.model.user_prompt}",
                },
            )
            self.st.model.enhanced_prompt = (
                re.sub(r"['\"]", "", enhanced)
                if enhanced
                else self.st.model.user_prompt
            )
        else:
            self.st.model.enhanced_prompt = re.sub(
                r"['\"]", "", self.st.model.user_prompt
            )

        self._validate_and_normalize()

    async def _check_vision_capability(self):
        """Probe model for vision support."""
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

        b64_pixels = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAS0lEQVQ4jWNkYGB4ycDAwMPAwMDEgAn+ERD7wQLVzIVFITGABZutJAEmBuxOJ8kAil0w8AZgiyr6umAYGDDEA5GFgYHhB5QmB/wAAIcLCBsQodqvAAAAAElFTkSuQmCC"

        # Probe Prompt - DO NOT ALTER PROMPT
        res = await self.inf._infer(
            task="Ensure Vision Capability",
            data={
                "system": "You must reply only 1 or 0",
                "user": {
                    "image_url": f"data:image/png;base64,{b64_pixels}",
                    "text": "Analyze this image. If the image is completely black, reply with 1. Otherwise, reply with 0. Reply 0 if you can't see the image.",
                },
            },
        )
        has_vision = "1" in res
        self.st.model.vision, self.st.vision_cache[self.st.model.id] = (
            has_vision,
            has_vision,
        )
        try:
            os.makedirs(os.path.dirname(CAPABILITY_CACHE_PATH), exist_ok=True)
            with open(CAPABILITY_CACHE_PATH, "w") as f:
                json.dump(self.st.vision_cache, f)
        except:
            pass

    async def _generate_image(self):
        start = time.time()
        engine = self.request.app.state.config.IMAGE_GENERATION_ENGINE
        await self.em.emit_status("Generating Image..")
        try:
            if engine in ["automatic1111", ""]:
                payload = self._sanitize_payload()
                base_url = self.request.app.state.config.AUTOMATIC1111_BASE_URL.rstrip(
                    "/"
                )
                headers = {}
                api_auth = self.request.app.state.config.AUTOMATIC1111_API_AUTH
                if api_auth:
                    headers["Authorization"] = (
                        f"Basic {base64.b64encode(api_auth.encode()).decode()}"
                        if ":" in api_auth
                        else f"Bearer {api_auth}"
                    )
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{base_url}/sdapi/v1/txt2img",
                        json=payload,
                        headers=headers,
                        timeout=None,
                    )
                    r.raise_for_status()
                    res = r.json()
                img_b64 = res["images"][0].split(",")[-1]
                self.st.model.b64_data, self.st.model.image_url = (
                    img_b64,
                    f"data:image/png;base64,{img_b64}",
                )
            else:
                form = CreateImageForm(
                    prompt=self.st.model.enhanced_prompt,
                    n=1,
                    size=self.st.model.size,
                    model=self.st.model.model,
                )
                gen = await image_generations(self.request, form, self.user)
                if gen:
                    self.st.model.image_url = gen[0]["url"]

            self.st.image_gen_time_int = int(time.time() - start)
            await self.em.emit_message(f"![Generated Image]({self.st.model.image_url})")
        except Exception as e:
            await self._err(f"Image Gen Failed: {e}")

    async def _vision_audit(self):
        if not self.st.model.quality_audit or not self.st.model.vision:
            return
        img_url = (
            f"data:image/png;base64,{self.st.model.b64_data}"
            if self.st.model.b64_data
            else self.st.model.image_url
        )

        # Audit Instruction
        template = (
            self.config.AUDIT_STRICT
            if self.valves.strict_audit
            else self.config.AUDIT_STANDARD
        )

        audit_instruction = template.format(
            prompt=self.st.model.enhanced_prompt, lang=self.st.model.language
        )

        raw_v_text = await self.inf._infer(
            task="Visual Quality Audit",
            data={
                "system": audit_instruction,
                "user": {
                    "image_url": img_url,
                    "text": "Analyze technical quality and prompt alignment. Return objective report and metrics.",
                },
            },
        )
        score_match = re.search(r"SCORE:\s*(\d+)", raw_v_text, re.IGNORECASE)
        if score_match:
            val = int(score_match.group(1))
            self.st.quality_audit_results.update(
                {
                    "score": val,
                    "critique": re.sub(
                        r"SCORE:\s*\d+", "", raw_v_text, flags=re.I
                    ).strip(),
                    "emoji": (
                        "🟢"
                        if val >= 80
                        else (
                            "🔵"
                            if val >= 70
                            else "🟡" if val >= 60 else "🟠" if val >= 40 else "🔴"
                        )
                    ),
                }
            )
        else:
            self.st.quality_audit_results["critique"] = raw_v_text

    async def _output_delivery(self):
        total = int(time.time() - self.st.start_time)
        await self.em.emit_citation(
            "🚀 PROMPT", self.st.model.enhanced_prompt.replace("*", ""), "1", "p"
        )

        if self.st.model.quality_audit:
            if not self.st.model.vision:
                await self.em.emit_citation(
                    "‼️NO VISION",
                    f"Model {self.st.model.id} lacks vision capabilities for audit.",
                    "2",
                    "b",
                )
            elif self.st.quality_audit_results["critique"]:
                await self.em.emit_citation(
                    f"{self.st.quality_audit_results['emoji']} SCORE: {self.st.quality_audit_results['score']}%",
                    self.st.quality_audit_results["critique"].replace("*", ""),
                    "2",
                    "a",
                )

        tech = (
            f"\n⠀\n𝗖𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻\n → Inference Model: {self.st.model.id}\n → Engine Model: {self.valves.model}\n"
            f" → Resolution: {self.st.model.size} | Steps: {self.st.model.get('steps')}\n → Engine: {self.st.model.engine}\n"
            f"\n\n𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 𝗠𝗲𝘁𝗿𝗶𝗰𝘀\n → Total Time: {total}s\n → Image Gen: {self.st.image_gen_time_int}s\n"
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
        conf = self.request.app.state.config
        state_dict = getattr(conf, "_state", {})
        eng = str(
            self._unwrap(state_dict.get("IMAGE_GENERATION_ENGINE", "none"))
        ).lower()
        settings = {"engine": eng}
        for k in ["IMAGE_GENERATION_MODEL", "IMAGE_SIZE", "IMAGE_STEPS"]:
            settings[self._clean_key(k, eng)] = self._unwrap(state_dict.get(k))
        for k in self.config.ENGINE_MAP.get(eng, []):
            val = self._unwrap(state_dict.get(k))
            if val is not None:
                settings[self._clean_key(k, eng)] = val
        self.st.model.update(settings)

    def _validate_and_normalize(self):
        eng, mdl = (
            self.st.model.get("engine"),
            str(self.st.model.get("model", "")).lower(),
        )
        if eng == "openai" and "dall-e" not in mdl:
            self.st.model["model"] = "dall-e-3"
        elif eng == "gemini" and "imagen" not in mdl:
            self.st.model["model"] = "imagen-3.0-fast-generate-001"
        sz, ar = self.st.model.get("size", "1024x1024"), self.st.model.get(
            "aspect_ratio"
        )
        try:
            w, h = map(int, sz.split("x")) if "x" in str(sz) else (int(sz), int(sz))
            ratio = (
                (int(ar.split(":")[0]) / int(ar.split(":")[1]))
                if ar and ":" in str(ar)
                else w / h
            )
        except:
            w, h, ratio = 1024, 1024, 1.0
        if eng == "openai" or "dall-e" in mdl:
            if ratio > 1.2:
                self.st.model["size"], self.st.model["aspect_ratio"] = (
                    "1792x1024",
                    "16:9",
                )
            elif ratio < 0.8:
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
            self.st.model["size"] = f"{w}x{(int(w / ratio) // 8) * 8}"

    def _sanitize_payload(self) -> dict:
        p = self.st.model.copy()
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
        ]:
            p.pop(k, None)
        p["prompt"] = self.st.model.get("enhanced_prompt", "")
        w, h = map(int, self.st.model.get("size", "1024x1024").split("x"))
        p["width"], p["height"] = w, h
        return p

    def _check_input(self, body: dict) -> Optional[Tuple[str, str]]:
        if body.get("is_probe"):
            return None
        messages = body.get("messages", [])
        if not messages:
            return None
        last = messages[-1]["content"]
        text = last[0].get("text", "") if isinstance(last, list) else last
        match = re.match(r"^(img|imgx)\s", text, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower(), text[match.end() :].strip()

    def _unwrap(self, obj):
        if hasattr(obj, "value"):
            return self._unwrap(obj.value)
        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}
        return obj

    def _clean_key(self, key, engine):
        k = (
            key.upper()
            .replace("IMAGE_GENERATION_", "")
            .replace("IMAGE_", "")
            .replace("ENABLE_IMAGE_", "ENABLE_")
        )
        pre = f"{engine.upper()}_"
        return (
            k.replace(f"{pre}API_", "").replace(pre, "").replace("IMAGES_", "").lower()
        )

    def _suppress_output(self, body: dict) -> dict:
        """Modify request body to suppress default generation output."""
        body["messages"] = [{"role": "assistant", "content": ""}]
        body["max_tokens"] = 1
        body["stop"] = [chr(i) for i in range(128)]
        return body

    def _dbg(self, msg: str):
        if self.valves.debug:
            print(f"⚡ EASYMAGE DEBUG: {msg}", file=sys.stderr, flush=True)

    async def _err(self, e: Any):
        self._dbg(f"ERROR: {e}")
        await self.em.emit_message(f"\n\n❌ EASYMAGE ERROR: {str(e)}\n")

    def _dmp(self):
        if not self.valves.debug:
            return
        print(
            f"{'—'*40}\n📦 EASYMAGE DUMP\n{json.dumps(self.st.model, indent=2)}\n{'—'*40}",
            file=sys.stderr,
            flush=True,
        )
