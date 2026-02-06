"""
Title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
Version: 0.6.13
Repository: https://github.com/annibale-x/Easymage
Author: Hannibal
Author_url: https://openwebui.com/u/h4nn1b4l
Author_email: annibale.x@gmail.com
Description: Image generation filter for Open WebUI.
"""

import json
import re
import time
import base64
import os
import sys
import requests
from typing import Optional, Any, Callable
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, CreateImageForm
from open_webui.models.users import UserModel
from open_webui.models.files import Files
from open_webui.main import generate_chat_completion

CAPABILITY_CACHE_PATH = "data/easymage_vision_cache.json"


class Store(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Store has no attribute '{item}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Filter:

    class Valves(BaseModel):

        # --- Base settings ---

        enhanced_prompt: bool = Field(
            default=True, description="Enrich prompt details."
        )

        quality_audit: bool = Field(
            default=True, description="Post-generation Image Quality Audit."
        )

        # --- Advanced settings ---

        # unload_other_models: bool = Field(
        #     default=False,
        #     description="Unload all models from VRAM except the current one.",
        # )

        persistent_vision_cache: bool = Field(
            default=False,
            description="Saves vision probe results to disk to avoid re-testing",
        )

        debug: bool = Field(
            default=False,
            description="Enable debug mode to print logs to the Docker console.",
        )

        # --- Global Engine Settings ---

        model: Optional[str] = Field(
            default=None,
            description="Force generation model.",
        )

        steps: int = 20
        size: str = "1024x1024"

        # --- Advanced Engine Settings

        seed: int = -1
        cfg_scale: float = 1.0
        distilled_cfg_scale: float = 3.5
        sampler_name: str = "Euler"
        scheduler: str = "Simple"

        # --- Advanced Engine Settings (High-Res and Batching)
        enable_hr: bool = False
        n_iter: int = 1
        batch_size: int = 1
        hr_scale: float = 2.0
        hr_upscaler: str = "Latent"
        hr_distilled_cfg: float = 3.5
        denoising_strength: float = 0.45

    def __init__(self):

        self.model = Store({})
        self.input = Store({})
        self.engine = Store({})
        self.valves = self.Valves()
        self.performance_stats = []
        self.vision_cache = {}
        self.cumulative_tokens = 0
        self.cumulative_elapsed_time = 0.0

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:

        if not (user_prompt := self._check_input(body)):
            return body

        self.emitter = __event_emitter__
        try:
            self._dbg(f"Started")
            await self._create_model(__request__, __user__, body, user_prompt)
            self._dbg(f"Raw user prompt: {self.model.user_prompt}")

        except Exception as e:
            await self._err(e)

        if self.model.trigger == "imgx":
            await self._output_prompt()
        else:
            await self._generate_image(body)
            await self._vision_audit()
            await self._output_delivery()

        self._dmp()
        return self._suppress_output(body)

    async def _create_model(
        self,
        __request__: Any,
        __user__: Any,
        body: dict,
        user_prompt: str,
    ):

        if "features" in body:
            body["features"] = {}
            body["features"]["web_search"] = False

        self.request = __request__
        self.context = UserModel(**__user__)
        self.performance_stats = []
        self.cumulative_tokens = 0
        self.cumulative_elapsed_time = 0.0
        self.start_time = time.time()
        self.model.id = body.get("model", "")

        # Apply global settings to model
        self._apply_global_settings(body)

        # Apply valves settings to model
        self._apply_valves_settings()

        # Apply input overrides to model
        self._apply_user_input(user_prompt)

        await self._check_vision()
        await self._detect_language()
        await self._enhance_prompt()

        if self.model.debug:
            await self.emitter(
                {
                    "type": "message",
                    "data": {
                        "content": f"```\nModel: {json.dumps(self.model, indent=4)}\n```\n"
                        f"```\nValves: {json.dumps(self.valves.model_dump(), indent=4)}\n```\n"
                    },
                }
            )

        return

    def _apply_global_settings(self, body):

        conf = self.request.app.state.config
        state_dict = getattr(conf, "_state", {})

        active_engine = self._unwrap(
            state_dict.get("IMAGE_GENERATION_ENGINE", "none")
        ).lower()

        common_keys = [
            "IMAGE_GENERATION_ENGINE",
            "IMAGE_GENERATION_MODEL",
            "IMAGE_SIZE",
            "IMAGE_STEPS",
        ]
        engine_map = {
            "automatic1111": ["AUTOMATIC1111_BASE_URL", "AUTOMATIC1111_PARAMS"],
            "comfyui": [
                "COMFYUI_BASE_URL",
                "COMFYUI_WORKFLOW",
                "COMFYUI_WORKFLOW_NODES",
            ],
            "openai": ["IMAGES_OPENAI_API_BASE_URL", "IMAGES_OPENAI_API_PARAMS"],
            "gemini": [
                "IMAGES_GEMINI_API_BASE_URL",
                "IMAGES_GEMINI_API_KEY",
                "IMAGES_GEMINI_ENDPOINT_METHOD",
            ],
        }

        common_settings = {
            self._clean_key(k, active_engine): self._unwrap(state_dict.get(k))
            for k in common_keys
        }
        engine_settings = {}

        if active_engine in engine_map:
            for k in engine_map[active_engine]:
                val = self._unwrap(state_dict.get(k))
                if val is not None:
                    engine_settings[self._clean_key(k, active_engine)] = val

        self.engine = {**common_settings, **engine_settings}
        extra_params = (
            engine_settings.pop("params", {})
            if isinstance(engine_settings, dict)
            else {}
        )
        self.model.update(common_settings)
        self.model.update(engine_settings)
        self.model.update(extra_params)

    def _apply_valves_settings(self):
        # Travaso dinamico: le Valves sovrascrivono i Global Settings.
        # Escludiamo i None per permettere ai Global di sopravvivere se la Valve è vuota.
        self.model.update(
            {k: v for k, v in self.valves.model_dump().items() if v is not None}
        )

    def _apply_user_input(self, user_prompt):

        # --- PHASE 1: NO HARDCODED DICT ---
        # A questo punto self.model contiene già i Global Settings
        # e (se chiamata prima) i valori delle Valves.


        # --- PHASE 2: PARSE PROMPT (Identica a prima) ---
        clean_prompt = re.sub(
            r"^imgx?\s*", "", user_prompt.strip(), flags=re.IGNORECASE
        )

        # Gestione Negative Prompt
        negative_prompt = ""
        if " --no " in clean_prompt.lower():
            clean_prompt, negative_prompt = re.split(
                r" --no ", clean_prompt, maxsplit=1, flags=re.IGNORECASE
            )

        # Gestione Prefix/Subject
        if " -- " in clean_prompt:
            prefix, subject = clean_prompt.split(" -- ", 1)
        else:
            tech_pattern = r'(\b\w+=(?:"[^"]*"|\S+))|([+-][dpa])'
            matches = list(re.finditer(tech_pattern, clean_prompt))
            split_idx = matches[-1].end() if matches else 0
            prefix, subject = clean_prompt[:split_idx], clean_prompt[split_idx:].strip()

        # --- PHASE 3: SURGICAL OVERRIDES ON SELF.MODEL ---
        # Qui sovrascriviamo DIRETTAMENTE self.model.
        # Se un parametro era nei Global Settings, ora viene aggiornato dal Prompt.
        param_pattern = r'(\b\w+)=("([^"]*)"|(\S+))'
        for k, _, q_val, u_val in re.findall(param_pattern, prefix):
            val = q_val if q_val else u_val
            try:
                if k == "s":
                    self.model["steps"] = int(val)
                elif k == "ge":
                    mapping = {'a': 'automatic1111', 'o': 'openai', 'g': 'gemini'}
                    self.model["engine"] = mapping.get(val)

                elif k == "sd":
                    self.model["seed"] = int(val)
                elif k == "sz":
                    val = str(val).lower()
                    if "x" in val:
                        self.model["size"] = val
                    else:
                        self.model["size"] = f"{val}x{val}"
                elif k == "n":
                    self.model["n_iter"] = int(val)
                elif k == "b":
                    self.model["batch_size"] = int(val)
                elif k == "cs":
                    self.model["cfg_scale"] = float(val)
                elif k == "dcs":
                    self.model["distilled_cfg_scale"] = float(val)
                    self.model["cfg_scale"] = 1.0  # Flux optimization

                # High-Res Logic
                elif k in ["hr", "hru", "hdcs", "dns"]:
                    self.model["enable_hr"] = True
                    if k == "hr":
                        self.model["hr_scale"] = float(val)
                    elif k == "hru":
                        self.model["hr_upscaler"] = val
                    elif k == "hdcs":
                        self.model["hr_distilled_cfg"] = float(val)
                    elif k == "dns":
                        self.model["denoising_strength"] = float(val)
            except ValueError:
                continue

        # Flag Overrides (+p, -d...)
        for flag in re.findall(r"([+-][dpah])", prefix):
            val = flag[0] == "+"
            char = flag[1]
            if char == "d":
                self.model["debug"] = val
            elif char == "p":
                self.model["enhanced_prompt"] = val
            elif char == "a":
                self.model["quality_audit"] = val
            elif char == "h":
                self.model["enable_hr"] = val

        # --- PHASE 4: FINAL COMMIT ---
        self.model.update(
            {"user_prompt": subject.strip(), "negative_prompt": negative_prompt.strip()}
        )

    async def _check_vision(self):

        self._dbg(f"Starting vision probe for: {self.model.id}")

        if self.model.debug or not self.model.persistent_vision_cache:
            self.vision_cache.clear()
            if os.path.exists(CAPABILITY_CACHE_PATH):
                os.remove(CAPABILITY_CACHE_PATH)
                self._dbg("Cache cleared.")

        if self.model.id in self.vision_cache:
            self._dbg(
                f"Found in memory cache: {self.model.id} = {self.vision_cache[self.model.id]}"
            )
            return self.vision_cache[self.model.id]

        if os.path.exists(CAPABILITY_CACHE_PATH):
            try:
                with open(CAPABILITY_CACHE_PATH, "r") as f:
                    data = json.load(f)
                    self.vision_cache.update(data)
                    if self.model.id in self.vision_cache:
                        self._dbg(
                            f"Found on disk: {self.model.id} = {self.vision_cache[self.model.id]}"
                        )
                        return self.vision_cache[self.model.id]
            except Exception as e:
                await self._err(e)

        self._dbg(f"Model {self.model.id} not in cache. Probing...")
        b64_pixels = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAS0lEQVQ4jWNkYGB4ycDAwMPAwMDEgAn+ERD7wQLVzIVFITGABZutJAEmBuxOJ8kAil0w8AZgiyr6umAYGDDEA5GFgYHhB5QmB/wAAIcLCBsQodqvAAAAAElFTkSuQmCC"

        test_messages = [
            {
                "role": "system",
                "content": "You must reply only 1 or 0",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_pixels}"},
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image. If the image is completely black, reply with 1. Otherwise, reply with 0. Reply 0 if you can't see the image.",
                    },
                ],
            },
        ]

        has_vision = False
        try:
            self._dbg(f"Calling generate_chat_completion for {self.model.id}")

            response = await generate_chat_completion(
                request=self.request,
                form_data={
                    "model": self.model.id,
                    "messages": test_messages,
                    "stream": False,
                    "is_probe": True,
                },
                user=self.context,
            )

            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "").strip()
                self._dbg(f"Model {self.model.id} replied: '{content}'")
                has_vision = "1" in content
            else:
                self._dbg(f"Probe response null or invalid for {self.model.id}")

        except Exception as e:
            await self._err(f"Probe failed for {self.model.id}: {e}")
            has_vision = False

        self.vision_cache[self.model.id] = has_vision
        try:
            os.makedirs(os.path.dirname(CAPABILITY_CACHE_PATH), exist_ok=True)
            with open(CAPABILITY_CACHE_PATH, "w") as f:
                json.dump(self.vision_cache, f)
            self._dbg(f"Cache updated on disk for {self.model.id}")
        except Exception as e:
            await self._err(f"Disk write failed: {e}")

        self.model.vision = has_vision
        return

    async def _detect_language(self):

        self._dbg("Starting Language Detection...")

        await self.emitter(
            {
                "type": "status",
                "data": {
                    "description": "Language Detection...",
                    "done": False,
                },
            }
        )

        start_timestamp = time.time()
        detected_target_lang = "English"

        try:
            detect_payload = {
                "model": self.model.id,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return ONLY the language name of the user text.",
                    },
                    {"role": "user", "content": self.model.user_prompt},
                ],
                "stream": False,
                "is_probe": True,
            }

            detect_res = await generate_chat_completion(
                request=self.request, form_data=detect_payload, user=self.context
            )

            detected_target_lang = (
                detect_res["choices"][0]["message"]["content"].strip().replace(".", "")
            )

            self._dbg(f"Detected Language: {detected_target_lang}")

            t_count = detect_res.get("usage", {}).get("total_tokens", 0)

            self._register_stat(
                "Language Detection",
                time.time() - start_timestamp,
                t_count,
            )

        except Exception as e:
            await self._err(f"Language Detection failed: {e}")

        self.model.language = detected_target_lang

        return

    async def _enhance_prompt(self):

        if self.model.enhanced_prompt or self.model.trigger == "imgx":
            self._dbg("Starting Prompt Enhancing...")
            start_timestamp = time.time()
            await self.emitter(
                {
                    "type": "status",
                    "data": {"description": "Prompt Enhancing...", "done": False},
                }
            )

            try:
                system_prompt = (
                    "You are an expert AI Image Prompt Engineer. "
                    f"Your task is to expand the user's input into a professional, highly detailed prompt in {self.model.language}. "
                    "Add details about lighting, camera angle, textures, environment, and artistic style. "
                    "Return ONLY the enhanced prompt text, no introductions."
                )
                user_content = f"Expand this prompt: {self.model.user_prompt}"

                refine_payload = {
                    "model": self.model.id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                }

                refine_res = await generate_chat_completion(
                    request=self.request, form_data=refine_payload, user=self.context
                )

                if "choices" in refine_res:
                    enhanced_prompt = refine_res["choices"][0]["message"][
                        "content"
                    ].strip()

                self._dbg(f"[Inlet] Enhanced Prompt: {enhanced_prompt}")

                t_count = refine_res.get("usage", {}).get("total_tokens", 0)

                self._register_stat(
                    "Prompt Processing",
                    time.time() - start_timestamp,
                    t_count,
                )

            except Exception as e:
                await self._err(f"Enhancer failed: {e}")

                await self.emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Enhancer Error: {e}",
                            "done": False,
                        },
                    }
                )

        else:
            enhanced_prompt = self.model.user_prompt

        self.model.enhanced_prompt = re.sub(r"['\"]", "", enhanced_prompt)

    def _sanitize_payload(self) -> dict:

        # 1. Copia il modello
        payload = self.model.copy()

        # 2. Rimuovi ESATTAMENTE quello che serve solo a Easymage/OWUI
        internal_keys = [
            "trigger",  # Interno Easymage
            "id",  # ID del modello LLM
            "engine",  # Nome del backend
            "base_url",  # URL del server Forge
            "size",  # Sostituito da width/height
            "user_prompt",  # Usiamo 'prompt' per Forge
            "enhanced_prompt",  # Prompt già processato
            "quality_audit",  # Flag interno
            "persistent_vision_cache",  # Cache interna
            "debug",  # Flag interno
            "vision",  # Flag interno
            "language",  # Metadato lingua
            "b64_data",
            "model",
        ]

        for key in internal_keys:
            payload.pop(key, None)

        # 3. Trasferimento Prompt e Dimensioni
        payload["prompt"] = self.model.get("enhanced_prompt", "")

        if "size" in self.model:
            try:
                w, h = map(int, self.model["size"].split("x"))
                payload["width"] = w
                payload["height"] = h
            except Exception:
                payload["width"], payload["height"] = 1024, 1024  # Default sano

        # Nel punto dove prepari il payload per Forge
        if payload.get("enable_hr"):
            # Fix per l'errore 500: Forge vuole una lista, non None
            payload["hr_additional_modules"] = []

            # Sicurezza: se l'upscaler è nullo, Forge dà 500
            if not payload.get("hr_upscaler"):
                payload["hr_upscaler"] = "Latent"

        return payload

    async def _generate_image(self, body):

        image_gen_start = time.time()

        # Detect current engine
        engine = self.request.app.state.config.IMAGE_GENERATION_ENGINE

        try:
            if engine in ["automatic1111", ""]:
                self._dbg("Using Forge Direct Bypass (Force params)...")

                # --- PAYLOAD FORGE ---
                payload = self._sanitize_payload()

                self._dmp(payload)

                # --- CALL FORGE ---
                base_url = self.request.app.state.config.AUTOMATIC1111_BASE_URL.rstrip(
                    "/"
                )
                api_url = f"{base_url}/sdapi/v1/txt2img"

                headers = {}
                api_auth = self.request.app.state.config.AUTOMATIC1111_API_AUTH
                if api_auth:
                    if ":" in api_auth:
                        auth_b64 = base64.b64encode(api_auth.encode()).decode()
                        headers["Authorization"] = f"Basic {auth_b64}"
                    else:
                        headers["Authorization"] = f"Bearer {api_auth}"

                import httpx

                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        api_url, json=payload, headers=headers, timeout=None
                    )
                    r.raise_for_status()
                    res = r.json()

                self._dmp(res["parameters"])

                # --- PROCESS FORGE RESPONSE ---
                if "images" in res and res["images"]:
                    img_b64 = res["images"][0]
                    if "," in img_b64:
                        img_b64 = img_b64.split(",")[1]

                    # Store for Vision Audit
                    self.model.b64_data = img_b64
                    self.model.image_url = f"data:image/png;base64,{img_b64}"
                else:
                    raise Exception("No images in Forge response")

            else:
                self._dbg(f"Using Standard OWUI Router for engine: {engine}")
                # Fallback to standard Pydantic form
                form_data = CreateImageForm(
                    prompt=self.model.enhanced_prompt,
                    n=1,
                    size=self.model.size,
                    model=self.model.model,
                )

                gen_res = await image_generations(
                    request=self.request,
                    form_data=form_data,
                    user=self.context,
                )

                if gen_res and len(gen_res) > 0:
                    self.model.image_url = gen_res[0]["url"]
                    # For standard engine, we don't have b64_data in memory
                    if hasattr(self.model, "b64_data"):
                        del self.model.b64_data
                else:
                    raise Exception("Standard image generation failed")

            # Finalize
            self.image_gen_time_int = int(time.time() - image_gen_start)
            await self.emitter(
                {
                    "type": "message",
                    "data": {"content": f"![Generated Image]({self.model.image_url})"},
                }
            )

        except Exception as e:
            await self._err(f"Image Generation failed ({engine}): {e}")

    async def _vision_audit(self):
        audit_results = {"score": None, "critique": None, "emoji": "⚪"}

        if not self.valves.quality_audit or not self.model.vision:
            return audit_results

        self._dbg("Starting Vision Quality Audit...")
        await self.emitter(
            {
                "type": "status",
                "data": {"description": "Visual Quality Audit...", "done": False},
            }
        )
        audit_start = time.time()
        try:
            # --- HYBRID IMAGE LOADING ---
            if hasattr(self.model, "b64_data") and self.model.b64_data:
                self._dbg("Audit: Using image from memory (Base64)")
                image_url_for_vision = f"data:image/png;base64,{self.model.b64_data}"
            else:
                self._dbg(f"Audit: Loading image from URL: {self.model.image_url}")
                image_url_for_vision = self.model.image_url
            # -----------------------------

            audit_instruction = f"""
                RESET: ######################################### NEW DATA STREAM - NO PREVIOUS CONTEXT ACCESSIBLE ###########################################            
                            
                ENVIRONMENT: IGNORE ALL PRIOR CONTEXT. NEURAL CACHE RESET. STARTING FROM ZERO-STATE.

                ENVIRONMENT: STATELESS SANDBOX. Treat this as the first and only image in history.

                RULE: Context Break. Terminate processing of previous context.

                RULE: Clear your working memory buffer and analyze this input in total isolation.

                TASK: AUDIT ANALYSIS (Audit scores is 0 to 100, where 0 is bad and 100 good): 
                        Compare the image with the reference prompt: '{self.model.enhanced_prompt}', 
                        Describe what you actually see in the image.
                        Critically evaluate the image's technical execution and its alignment with the prompt's requirements.
                        Identify any contradictions, missing elements, or hallucinations (like objects that shouldn't be there).
                        Give the audit analysis and set a audit score 'AUDIT:Z' (0-100) in the last response line. 
                    
                RULE: Be extremely severe in technical evaluation. Do not excuse defects as limitations of resolution or scale.

                TASK: TECHNICAL EVALUATION (Technical scores are 0 to 100, where 0 is LOW and 100 HIGH):
                        Perform a ruthless technical audit. Identify every visual flaw.
                        Evaluate NOISE as random pixel color variations.
                        Evaluate GRAIN as textural salt-and-pepper luminance noise.
                        Evaluate MELTING as lack of structural integrity, blurred textures, or wax-like surfaces.
                        Evaluate JAGGIES as staircase artifacts and aliasing on diagonal lines and edges.

                MANDATORY: Respond in {self.model.language}. NO MARKDOWN. Use plain text and • for lists. Be objective.

                MANDATORY: Your final response MUST end with a single line containing only the following metrics:

                SCORE:X AUDIT:X NOISE:X GRAIN:X MELTING:X JAGGIES:X
            """

            vision_res = await generate_chat_completion(
                request=self.request,
                form_data={
                    "model": self.model.id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": audit_instruction},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url_for_vision},
                                },
                            ],
                        }
                    ],
                    "stream": False,
                },
                user=self.context,
            )

            raw_v_text = vision_res["choices"][0]["message"]["content"].strip()
            score_match = re.search(r"SCORE:\s*(\d+)", raw_v_text, re.IGNORECASE)

            if score_match:
                val = int(score_match.group(1))
                audit_results["score"] = val
                audit_results["critique"] = re.sub(
                    r"SCORE:\s*\d+", "", raw_v_text, flags=re.I
                ).strip()
                audit_results["emoji"] = (
                    "🟢"
                    if val >= 80
                    else (
                        "🔵"
                        if val >= 70
                        else "🟡" if val >= 60 else "🟠" if val >= 40 else "🔴"
                    )
                )
            else:
                audit_results["critique"] = raw_v_text

            t_count = vision_res.get("usage", {}).get("total_tokens", 0)
            self._register_stat("Visual Audit", time.time() - audit_start, t_count)

        except Exception as e:
            await self._err(f"Vision Audit failed: {e}")
            audit_results["critique"] = "Vision Audit failed."

        self.quality_audit = audit_results
        return audit_results

    async def _output_delivery(self):

        total_exec_time = int(time.time() - self.start_time)
        clean_prompt = self.model.enhanced_prompt.replace("*", "")

        await self.emitter(
            {
                "type": "citation",
                "data": {
                    "source": {"name": "🚀 PROMPT"},
                    "document": [clean_prompt],
                    "metadata": [{"source": "1", "id": "p"}],
                },
            }
        )
        if self.model.quality_audit:
            if not self.model.vision:
                await self.emitter(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "‼️NO VISION"},
                            "document": [
                                f"Model {self.model.id} lacks vision capabilities for audit."
                            ],
                            "metadata": [{"source": "2", "id": "b"}],
                        },
                    }
                )
            elif self.quality_audit["critique"]:
                await self.emitter(
                    {
                        "type": "citation",
                        "data": {
                            "source": {
                                "name": f"{self.quality_audit['emoji']} SCORE: {self.quality_audit['score']}%"
                            },
                            "document": [
                                self.quality_audit["critique"].replace("*", "")
                            ],
                            "metadata": [{"source": "2", "id": "a"}],
                        },
                    }
                )

        mm = "multimodal" if self.model.vision else "not multimodal"
        tech_details = (
            f"\n⠀\n𝗖𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻\n → Inference Model: {self.model.id} ({mm})\n → Engine Model: {self.valves.model}\n"
            f" → Resolution: {self.valves.size} | Steps: {self.valves.steps}\n → Engine: {self.model.engine}\n"
            f"\n\n𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 𝗠𝗲𝘁𝗿𝗶𝗰𝘀\n → Total Time: {total_exec_time}s\n → Image Gen: {self.image_gen_time_int}s\n"
            + "\n".join(self.performance_stats)
        )
        await self.emitter(
            {
                "type": "citation",
                "data": {
                    "source": {"name": "🔍 DETAILS"},
                    "document": [tech_details],
                    "metadata": [{"source": "3", "id": "d"}],
                },
            }
        )

        tps = (
            self.cumulative_tokens / self.cumulative_elapsed_time
            if self.cumulative_elapsed_time > 0
            else 0
        )
        summary = f"{total_exec_time}s total | {self.image_gen_time_int}s img | {self.cumulative_tokens} tk | {tps:.1f} tk/s"

        await self.emitter({"type": "message", "data": {"content": "\n\n[1] [2] [3]"}})
        await self.emitter(
            {"type": "status", "data": {"description": summary, "done": True}}
        )

    async def _output_prompt(self):

        # 1. Emit the enhanced prompt as a message
        await self.emitter(
            {
                "type": "message",
                "data": {"content": self.model.enhanced_prompt},
            }
        )

        # 2. Calculate specific timings for imgp (Prompt Enhancement only)
        total_exec_time = int(time.time() - self.start_time)
        tps = (
            self.cumulative_tokens / self.cumulative_elapsed_time
            if self.cumulative_elapsed_time > 0
            else 0
        )

        # 3. Create a specialized summary string for imgp
        summary = (
            f"{total_exec_time}s total | {self.cumulative_tokens} tk | {tps:.1f} tk/s"
        )

        # 4. Close the status correctly
        await self.emitter(
            {"type": "status", "data": {"description": summary, "done": True}}
        )

    def _dmp(self, data: Optional = None):
        if self.valves.debug:
            header = "—" * 80 + "\n📦 EASYMAGE DUMP:\n" + "—" * 80
            print(header, file=sys.stderr, flush=True)
            if isinstance(data, dict):
                print(
                    "dict: " + json.dumps(data, indent=4),
                    file=sys.stderr,
                    flush=True,
                )
            elif isinstance(data, BaseModel):
                print(
                    "BaseModel: " + json.dumps(data.model_dump(), indent=4),
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "...",
                    file=sys.stderr,
                    flush=True,
                )
            print("—" * 80, file=sys.stderr, flush=True)

    def _dbg(self, message: str, error: bool = False):
        if error:
            print(f"❌ EASYMAGE ERROR: {message}", file=sys.stderr, flush=True)
        elif self.valves.debug:
            print(f"⚡ EASYMAGE DEBUG: {message}", file=sys.stderr, flush=True)

    async def _err(self, e: Exception):
        self._dbg(str(e), True)
        if self.emitter:
            await self.emitter(
                {
                    "type": "message",
                    "data": {"content": f"\n\n❌ EASYMAGE ERROR: {str(e)}\n"},
                }
            )

    def _register_stat(self, stage_name: str, elapsed: float, token_count: int = 0):
        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tokens_per_second:.1f} tk/s"
        )

    def _unwrap(self, obj):
        if hasattr(obj, "value"):
            return self._unwrap(obj.value)
        if isinstance(obj, dict):
            return {k: self._unwrap(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._unwrap(i) for i in obj]
        return obj

    def _clean_key(self, key, engine):
        k = key.upper()
        k = (
            k.replace("IMAGE_GENERATION_", "")
            .replace("IMAGE_", "")
            .replace("ENABLE_IMAGE_", "ENABLE_")
        )
        engine_prefix = f"{engine.upper()}_"
        k = (
            k.replace(f"{engine_prefix}API_", "")
            .replace(engine_prefix, "")
            .replace("IMAGES_", "")
        )
        return k.lower()

    def _normalize_sampler(self, name):
        n = (
            name.lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("++", "")
        )

        mapping = {
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
        return mapping.get(n, name)

    def _normalize_scheduler(self, name):
        n = name.lower().replace("_", "").replace(" ", "").replace("-", "")

        mapping = {
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
        return mapping.get(n, name.capitalize())

    def _check_input(self, body: dict) -> str | None:
        if body.get("is_probe"):
            return None

        messages = body.get("messages", [])
        if not messages:
            return None

        last_message_content = messages[-1]["content"]
        input_text = (
            last_message_content[0].get("text", "")
            if isinstance(last_message_content, list)
            else last_message_content
        )

        match = re.match(r"^(img|imgx)\s", input_text, re.IGNORECASE)

        if not match:
            return None

        self.model["trigger"] = match.group(1).lower()

        return input_text[match.end() :].strip()

    def _suppress_output(self, body: dict) -> dict:
        body["messages"] = [{"role": "assistant", "content": ""}]
        body["max_tokens"] = 1
        body["stop"] = [chr(i) for i in range(128)]
        return body
