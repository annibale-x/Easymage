"""
title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
version: 0.6.15
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
from typing import Optional, Any, List, Dict, Tuple
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, CreateImageForm  # type: ignore
from open_webui.models.users import UserModel  # type: ignore
from open_webui.main import generate_chat_completion  # type: ignore

CAPABILITY_CACHE_PATH = "data/easymage_vision_cache.json"

# --- DATA STRUCTURES ---


class Store(dict):
    """Dynamic dictionary storage with dot notation access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            # Return None instead of raising AttributeError to prevent crashes on optional keys
            return None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EasymageState:
    """Manages the internal state and configuration of a single request."""

    def __init__(self, valves):
        self.model = Store(
            {
                "trigger": None,
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
            await self.emitter({"type": "message", "data": {"content": content}})

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
    """Handles all LLM interactions (Text and Vision) using a generalized method."""

    def __init__(self, request, user, state: EasymageState, emitter: EmitterService):
        self.request = request
        self.user = user
        self.state = state
        self.emitter = emitter

    async def _infer(self, task: Optional[str], messages: List[Dict]) -> str:
        """Generalized inference method for all LLM calls (replaces direct generate_chat_completion)."""
        start_time = time.time()
        if task:
            await self.emitter.emit_status(f"{task}..")

        payload = {
            "model": self.state.model.id,
            "messages": messages,
            "stream": False,
            "seed": 42,
            "temperature": 0.0,
            "is_probe": True,
        }

        try:
            response = await generate_chat_completion(self.request, payload, self.user)
            if response:
                content = response["choices"][0]["message"].get("content", "").strip()
                # Clean reasoning tags
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                content = re.sub(r"</?text>", "", content).strip()

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
    """Handles prompt parsing, regex extraction and mappings."""

    def __init__(self, sampler_map, scheduler_map):
        self.sampler_map = sampler_map
        self.scheduler_map = scheduler_map

    def parse_input(self, user_prompt: str, model_state: Store):
        clean_prompt = user_prompt

        # 1. Negative Prompt
        negative_prompt = ""
        if " --no " in clean_prompt.lower():
            clean_prompt, negative_prompt = re.split(
                r" --no ", clean_prompt, maxsplit=1, flags=re.IGNORECASE
            )

        # 2. KV Params & Flags
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

        # 3. Parameter Mapping
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
                    model_state["sampler_name"] = self._normalize(val, self.sampler_map)
                elif k == "sch":
                    model_state["scheduler"] = self._normalize(val, self.scheduler_map)
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
            n, name.capitalize() if mapping is self.scheduler_map else name
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

    _SAMPLER_MAP = {
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
    _SCHEDULER_MAP = {
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
    _ENGINE_MAP = {
        "automatic1111": ["AUTOMATIC1111_BASE_URL", "AUTOMATIC1111_PARAMS"],
        "comfyui": ["COMFYUI_WORKFLOW", "COMFYUI_WORKFLOW_NODES"],
        "openai": ["IMAGES_OPENAI_API_BASE_URL", "IMAGES_OPENAI_API_PARAMS"],
        "gemini": ["IMAGES_GEMINI_API_KEY", "IMAGES_GEMINI_ENDPOINT_METHOD"],
    }

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:
        # 1. Early check and extraction
        trigger_data = self._check_input(body)
        if not trigger_data:
            return body

        trigger, user_prompt_raw = trigger_data

        # 2. Initialize SoC Components
        self.st = EasymageState(self.valves)
        self.st.model.trigger = trigger

        self.em = EmitterService(__event_emitter__)
        self.request = __request__
        self.user = UserModel(**__user__)
        self.inf = InferenceEngine(self.request, self.user, self.st, self.em)
        self.parser = PromptParser(self._SAMPLER_MAP, self._SCHEDULER_MAP)

        try:
            await self._setup_context(body, user_prompt_raw)

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

        # Base model info
        self.st.model.id = body.get("model", "")

        # Load settings
        self._apply_global_settings()
        self.st.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )

        # Parse user overrides
        self.parser.parse_input(user_prompt, self.st.model)

        # Vision Capability Probe
        await self._check_vision_capability()

        # Detect Language
        detected_lang = await self.inf._infer(
            task="Language Detection",
            messages=[{"role": "user", "content": self.st.model.user_prompt}],
        )
        self.st.model.language = detected_lang if detected_lang else "English"

        # Enhance Prompt
        if self.st.model.enhanced_prompt or self.st.model.trigger == "imgx":
            enhanced = await self.inf._infer(
                task="Prompt Enhancing",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI Image Prompt Engineer. "
                            f"Your task is to expand the user's input into a professional, highly detailed prompt in {self.st.model.language}. "
                            "Add details about lighting, camera angle, textures, environment, and artistic style. "
                            "Return ONLY the enhanced prompt text, no introductions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Expand this prompt: {self.st.model.user_prompt}",
                    },
                ],
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
        probe_msgs = [
            {"role": "system", "content": "You must reply only 1 or 0"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_pixels}"},
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image. If the image is completely black, reply with 1. Otherwise, reply with 0.",
                    },
                ],
            },
        ]

        res = await self.inf._infer(
            task="Ensure Vision Capability", messages=probe_msgs
        )
        has_vision = "1" in res
        self.st.model.vision = has_vision
        self.st.vision_cache[self.st.model.id] = has_vision

        try:
            os.makedirs(os.path.dirname(CAPABILITY_CACHE_PATH), exist_ok=True)
            with open(CAPABILITY_CACHE_PATH, "w") as f:
                json.dump(self.st.vision_cache, f)
        except:
            pass

    async def _generate_image(self):
        image_gen_start = time.time()
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
                    if ":" in api_auth:
                        auth_b64 = base64.b64encode(api_auth.encode()).decode()
                        headers["Authorization"] = f"Basic {auth_b64}"
                    else:
                        headers["Authorization"] = f"Bearer {api_auth}"

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
                self.st.model.b64_data = img_b64
                self.st.model.image_url = f"data:image/png;base64,{img_b64}"
            else:
                form_data = CreateImageForm(
                    prompt=self.st.model.enhanced_prompt,
                    n=1,
                    size=self.st.model.size,
                    model=self.st.model.model,
                )
                gen_res = await image_generations(self.request, form_data, self.user)
                if gen_res:
                    self.st.model.image_url = gen_res[0]["url"]

            self.st.image_gen_time_int = int(time.time() - image_gen_start)
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

        audit_instruction = f"""
            RESET: ######################################### NEW DATA STREAM ###########################################            
            ENVIRONMENT: STATELESS SANDBOX.
            TASK: AUDIT ANALYSIS (Audit scores is 0 to 100):
                    Compare image with: '{self.st.model.enhanced_prompt}',
                    Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line.
            TASK: TECHNICAL EVALUATION:
                    Evaluate NOISE, GRAIN, MELTING, JAGGIES.
            MANDATORY: Respond in {self.st.model.language}. NO MARKDOWN. Use plain text and • for lists.
            MANDATORY: Final response MUST end with: SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
        """

        raw_v_text = await self.inf._infer(
            task="Visual Quality Audit",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": audit_instruction},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
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
        total_time = int(time.time() - self.st.start_time)
        await self.em.emit_citation(
            "🚀 PROMPT", self.st.model.enhanced_prompt.replace("*", ""), "1", "p"
        )

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
            f"\n⠀\n𝗖𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻\n → Inference Model: {self.st.model.id}\n → Engine: {self.st.model.engine}\n"
            f" → Res: {self.st.model.size} | Steps: {self.st.model.get('steps')}\n\n𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲\n → Total: {total_time}s\n"
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
            f"{total_time}s total | {self.st.image_gen_time_int}s img | {self.st.cumulative_tokens} tk | {tps:.1f} tk/s",
            True,
        )

    async def _output_status_only(self):
        tps = (
            self.st.cumulative_tokens / self.st.cumulative_elapsed_time
            if self.st.cumulative_elapsed_time > 0
            else 0
        )
        summary = f"{int(time.time() - self.st.start_time)}s total | {self.st.cumulative_tokens} tk | {tps:.1f} tk/s"
        await self.em.emit_status(summary, True)

    # --- HELPERS ---

    def _apply_global_settings(self):
        conf = self.request.app.state.config
        state_dict = getattr(conf, "_state", {})
        engine = str(
            self._unwrap(state_dict.get("IMAGE_GENERATION_ENGINE", "none"))
        ).lower()

        settings = {"engine": engine}
        for k in ["IMAGE_GENERATION_MODEL", "IMAGE_SIZE", "IMAGE_STEPS"]:
            settings[self._clean_key(k, engine)] = self._unwrap(state_dict.get(k))

        for k in self._ENGINE_MAP.get(engine, []):
            val = self._unwrap(state_dict.get(k))
            if val is not None:
                settings[self._clean_key(k, engine)] = val

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

        sz = self.st.model.get("size", "1024x1024")
        ar = self.st.model.get("aspect_ratio")
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
            h = (int(w / ratio) // 8) * 8
            self.st.model["size"] = f"{w}x{h}"

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
            "—" * 80
            + "\n📦 EASYMAGE DUMP\n"
            + json.dumps(self.st.model, indent=2)
            + "\n"
            + "—" * 80,
            file=sys.stderr,
            flush=True,
        )
