"""
Title: Easymage - Multilingual Prompt Enhancer & Vision QC Image Generator
Version: 0.6.3
https://github.com/annibale-x/Easymage
Author: Hannibal
Author_url: https://openwebui.com/u/h4nn1b4l
Author_email: annibale.x@gmail.com
Description: Professional-grade image generation filter for Open WebUI.

MAIN FEATURES:
- DYNAMIC ENGINE DETECTION: Automatically identifies active image generation engines (A1111/ComfyUI/OpenAI).
- PROMPT ENHANCING: Multi-language detection and English-centric prompt engineering.
- SINGLE-PASS VISION AUDIT: Real-time technical quality analysis with integer-based scoring (0-100%).
- PERFORMANCE ANALYTICS: Precise tracking of Token/s, LLM execution time, and raw image generation latency.
- UI OPTIMIZATION: Custom Unicode formatting for environments where Markdown is restricted.

LOGICAL FLOW:
1. INLET INTERCEPTION: Captures "img " prefixed messages.
2. CONTEXT ANALYSIS: Detects source language and engine parameters.
3. REFINEMENT: Translates and enriches the user's intent into a high-fidelity English prompt.
4. GENERATION: Triggers the backend image engine and records generation latency.
5. VISION AUDIT: Passes the result to a Vision LLM for critique and scoring.
6. DELIVERY: Emits formatted citations and status summaries.
"""

import json
import re
import time
import base64
import os
import sys
from typing import Optional, Any, Callable
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, CreateImageForm
from open_webui.models.users import UserModel
from open_webui.models.files import Files
from open_webui.main import generate_chat_completion

CAPABILITY_CACHE_PATH = "data/easymage_vision_cache.json"


class Generation(BaseModel):
    engine: str = ""
    model: str = ""
    size: str = "1024x1024"
    steps: int = 25
    config: dict = {}
    image_url: str | None = None
    audit: dict = {}


class Model(BaseModel):
    name: str = ""
    vision: bool = False
    generation: Generation = Generation()
    language: str = "English"
    valves: dict = {}
    user_prompt: str | None = None
    enhanced_prompt: str | None = None
    trigger: str | None = None


class Filter:
    class Valves(BaseModel):

        enhanced_prompt: bool = Field(
            default=True, description="Enrich prompt details."
        )

        generation_quality_audit: bool = Field(
            default=True, description="Post-generation Image Quality Audit."
        )

        generation_size_override: Optional[str] = Field(
            default=None, description="Force size (e.g. 1024x1024)."
        )
        generation_model_override: Optional[str] = Field(
            default=None,
            description="Force generation model.",
        )
        generation_steps_override: Optional[int] = Field(
            default=None, description="Force step count (if supported)."
        )
        persistent_vision_cache: bool = Field(
            default=False,
            description="Saves vision probe results to disk to avoid re-testing",
        )
        debug: bool = Field(
            default=False,
            description="Enable debug mode to print logs to the Docker console.",
        )

    def __init__(self):

        self.model = Model()
        self.valves = self.Valves()
        self.performance_stats = []
        self.vision_cache = {}
        self.cumulative_tokens = 0
        self.cumulative_elapsed_time = 0.0

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

    def _register_stat(self, stage_name: str, elapsed: float, token_count: int = 0):
        self.cumulative_tokens += token_count
        self.cumulative_elapsed_time += elapsed
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        self.performance_stats.append(
            f"  → {stage_name}: {int(elapsed)}s | {token_count} tk | {tokens_per_second:.1f} tk/s"
        )

    def _suppress_output(self, body: dict) -> dict:
        body["messages"] = [{"role": "assistant", "content": ""}]
        body["max_tokens"] = 1
        body["stop"] = [chr(i) for i in range(128)]
        return body

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

        match = re.match(r"^(img|exp)\s", input_text, re.IGNORECASE)

        if not match:
            return None

        trigger = match.group(1).lower()

        self.model.trigger = trigger

        return input_text[match.end() :].strip()

    async def _create_model(
        self,
        __event_emitter__: Callable,
        __request__: Any,
        __user__: Any,
        body: dict,
        user_prompt: str,
    ):

        if "features" in body:
            body["features"] = {}
            body["features"]["web_search"] = False

        self.performance_stats = []
        self.cumulative_tokens = 0
        self.cumulative_elapsed_time = 0.0
        self.start_time = time.time()

        conf = __request__.app.state.config
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

        self.model.generation = Generation(**common_settings)
        self.model.generation.config = engine_settings
        self.model.name = body.get("model", "")
        self.model.valves = self.valves
        self.model.user_prompt = user_prompt
        self.context = UserModel(**__user__)
        self.emitter = __event_emitter__
        self.request = __request__

        for key in ["model", "size", "steps"]:
            val = getattr(self.model.valves, f"generation_{key}_override", None)
            if val is not None:
                setattr(self.model.generation, key, val)

        await self._check_vision()
        await self._detect_language()
        await self._enhance_prompt()

        return

    async def _check_vision(self):
        self._dbg(f"Starting vision probe for: {self.model.name}")

        if self.valves.debug or not self.valves.persistent_vision_cache:
            self.vision_cache.clear()
            if os.path.exists(CAPABILITY_CACHE_PATH):
                os.remove(CAPABILITY_CACHE_PATH)
                self._dbg("Cache cleared.")

        if self.model.name in self.vision_cache:
            self._dbg(
                f"Found in memory cache: {self.model.name} = {self.vision_cache[self.model.name]}"
            )
            return self.vision_cache[self.model.name]

        if os.path.exists(CAPABILITY_CACHE_PATH):
            try:
                with open(CAPABILITY_CACHE_PATH, "r") as f:
                    data = json.load(f)
                    self.vision_cache.update(data)
                    if self.model.name in self.vision_cache:
                        self._dbg(
                            f"Found on disk: {self.model.name} = {self.vision_cache[self.model.name]}"
                        )
                        return self.vision_cache[self.model.name]
            except Exception as e:
                await self._err(e)

        self._dbg(f"Model {self.model.name} not in cache. Probing...")
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
            self._dbg(f"Calling generate_chat_completion for {self.model.name}")

            response = await generate_chat_completion(
                request=self.request,
                form_data={
                    "model": self.model.name,
                    "messages": test_messages,
                    "stream": False,
                    "is_probe": True,
                },
                user=self.context,
            )

            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "").strip()
                self._dbg(f"Model {self.model.name} replied: '{content}'")
                has_vision = "1" in content
            else:
                self._dbg(f"Probe response null or invalid for {self.model.name}")

        except Exception as e:
            await self._err(f"Probe failed for {self.model.name}: {e}")
            has_vision = False

        self.vision_cache[self.model.name] = has_vision
        try:
            os.makedirs(os.path.dirname(CAPABILITY_CACHE_PATH), exist_ok=True)
            with open(CAPABILITY_CACHE_PATH, "w") as f:
                json.dump(self.vision_cache, f)
            self._dbg(f"Cache updated on disk for {self.model.name}")
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
                "model": self.model.name,
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

        self.model.enhanced_prompt = self.model.user_prompt

        if self.valves.enhanced_prompt or self.model.trigger == "exp":
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
                    "model": self.model.name,
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

            self.model.enhanced_prompt = re.sub(r"['\"]", "", enhanced_prompt)

        return

    async def _generate_image(self):

        self._dbg(
            f"Requesting Image Generation for model: {self.model.generation.model}"
        )

        await self.emitter(
            {
                "type": "status",
                "data": {
                    "description": f"Generating {self.model.generation.size} Image...",
                    "done": False,
                },
            }
        )

        image_gen_start = time.time()
        try:
            gen_res = await image_generations(
                request=self.request,
                form_data=CreateImageForm(
                    prompt=self.model.enhanced_prompt,
                    n=1,
                    size=self.model.generation.size,
                    model=self.model.generation.model,
                ),
                user=self.context,
            )

            if not gen_res:
                return None

            self.image_gen_time_int = int(time.time() - image_gen_start)
            self.model.generation.image_url = gen_res[0]["url"]
            self._dbg(f"Image Generated. URL: {self.model.generation.image_url}")

            await self.emitter(
                {
                    "type": "message",
                    "data": {
                        "content": f"![Generated Image]({self.model.generation.image_url})"
                    },
                }
            )

        except Exception as e:
            await self._err(f"Image Generation failed: {e}")

        return

    async def _vision_audit(self):

        audit_results = {"score": None, "critique": None, "emoji": "⚪"}

        if not self.valves.generation_quality_audit or not self.model.vision:
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
            img_file_id = (
                self.model.generation.image_url.split("/")[-2]
                if "files" in self.model.generation.image_url
                else None
            )
            if not img_file_id:
                return audit_results

            file_record = Files.get_file_by_id(img_file_id)
            with open(file_record.path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")

            # Modification: Open-ended audit to minimize bias and allow dynamic evaluation
            # audit_instruction = (
            # f"AUDIT TASK: Compare the attached image with this reference prompt: '{self.model.enhanced_prompt}'.\n"
            # f"INSTRUCTIONS:\n"
            # f"1. Describe what you actually see in the image.\n"
            # f"2. Critically evaluate the image's technical execution and its alignment with the prompt's requirements.\n"
            # f"3. Identify any contradictions, missing elements, or hallucinations (like objects that shouldn't be there).\n"
            # f"STRICT RULES: Respond in {self.model.language}. NO MARKDOWN. Use plain text and • for lists. "
            # f"Do not use pre-defined categories if they don't apply. Be objective and impartial. "
            # f"END YOUR RESPONSE WITH 'SCORE:X' (0-100)."
            # )

            audit_instruction = (
                f"AUDIT TASK: Compare the attached image with this reference prompt: '{self.model.enhanced_prompt}'.\n"
                f"INSTRUCTIONS:\n"
                f"1. Describe what you actually see in the image.\n"
                f"2. Critically evaluate the image's technical execution and its alignment with the prompt's requirements.\n"
                f"3. Identify any contradictions, missing elements, or hallucinations.\n"
                f"STRICT RULES: Respond in {self.model.language}. NO MARKDOWN. Use plain text and • for lists. "
                f"Do not use pre-defined categories if they don't apply. Be objective and impartial. "
                f"END YOUR RESPONSE WITH 'SCORE:X' (0-100)."
            )

            vision_res = await generate_chat_completion(
                request=self.request,
                form_data={
                    "model": self.model.name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": audit_instruction},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded_image}"
                                    },
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

        self.model.generation.audit = audit_results
        return

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
        if self.valves.generation_quality_audit:
            if not self.model.vision:
                await self.emitter(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "‼️NO VISION"},
                            "document": [
                                f"Model {self.model.name} lacks vision capabilities for audit."
                            ],
                            "metadata": [{"source": "2", "id": "b"}],
                        },
                    }
                )
            elif self.model.generation.audit["critique"]:
                await self.emitter(
                    {
                        "type": "citation",
                        "data": {
                            "source": {
                                "name": f"{self.model.generation.audit['emoji']} SCORE: {self.model.generation.audit['score']}%"
                            },
                            "document": [
                                self.model.generation.audit["critique"].replace("*", "")
                            ],
                            "metadata": [{"source": "2", "id": "a"}],
                        },
                    }
                )

        mm = "multimodal" if self.model.vision else "not multimodal"
        tech_details = (
            f"\n⠀\n𝗖𝗼𝗻𝗳𝗶𝗴𝘂𝗿𝗮𝘁𝗶𝗼𝗻\n → Inference Model: {self.model.name} ({mm})\n → Engine Model: {self.model.generation.model}\n"
            f" → Resolution: {self.model.generation.size} | Steps: {self.model.generation.steps}\n → Engine: {self.model.generation.engine}\n"
            f"\n\n𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 𝗠𝗲𝘁𝗿𝗶𝗰𝘀\n → Image Gen: {self.image_gen_time_int}s\n → Total: {total_exec_time}s\n"
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

    def _dmp(self):
        if self.valves.debug:
            header = "—" * 80 + "\n📦 EASYMAGE DUMP:\n" + "—" * 80
            print(header, file=sys.stderr, flush=True)
            print(
                json.dumps(self.model.model_dump(), indent=4),
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
                {"type": "message", "data": {"content": f"❌ ERROR: {str(e)}\n"}}
            )

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
        __event_emitter__=None,
    ) -> dict:

        if not (user_prompt := self._check_input(body)):
            return body

        try:
            self._dbg(f"Started")
            await self._create_model(
                __event_emitter__, __request__, __user__, body, user_prompt
            )
            self._dbg(f"Raw user prompt: {self.model.user_prompt}")

        except Exception as e:
            await self._err(e)

        if self.model.trigger == "exp":
            await self._output_prompt()

        else:
            await self._generate_image()
            await self._vision_audit()
            await self._output_delivery()

        self._dmp()
        return self._suppress_output(body)
