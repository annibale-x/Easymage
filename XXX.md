
# 🎨 Easymage: Command Syntax Guide

Easymage intercetta i messaggi che iniziano con `img` o `imgx` e li processa attraverso un parser avanzato che separa flag, parametri tecnici, stili e prompt.

---

## 📐 Struttura del Comando

L'ordine dei parametri prima del soggetto è flessibile, ma la struttura raccomandata per evitare ambiguità è:

`img[x] [FLAGS] [PARAMS] [STYLES] -- [SUBJECT] --no [NEGATIVE_PROMPT]`



---

## 🚀 Triggers (Comandi Base)

| Comando | Modalità | Descrizione |
| :--- | :--- | :--- |
| **`img`** | Standard | Usa le impostazioni definite nelle "Valves" del filtro. |
| **`imgx`** | Express | Forza l'attivazione dell'**Enhanced Prompt** (`+p`), ignorando i default. |

---

## 🚩 Flags (Interruttori)

I flag sono booleani attivati con `+` (ON) o disattivati con `-` (OFF).

| Flag | Nome | Descrizione |
| :--- | :--- | :--- |
| **`+p` / `-p`** | **P**rompt Enhancer | Attiva/Disattiva la riscrittura del prompt tramite LLM. |
| **`+a` / `-a`** | **A**udit Vision | Attiva/Disattiva l'analisi critica dell'immagine generata. |
| **`+d` / `-d`** | **D**ebug | Mostra i log dettagliati nella console Docker. |

> **Esempio:** `img -p +a` (Nessun rewrite del prompt, ma esegui l'audit finale).

---

## ⚙️ Parametri Chiave-Valore (KV Params)

Configurazioni tecniche passate direttamente all'engine (SD/Flux). Sintassi: `key=value`.

| Chiave | Parametro | Esempi Valore | Note |
| :--- | :--- | :--- | :--- |
| **`sz`** | **SiZe** | `1024`, `1024x768` | Se un solo numero è fornito, genera un'immagine quadrata. |
| **`s`** | **Steps** | `20`, `30`, `50` | Numero di passi di campionamento. |
| **`cs`** | **Cfg Scale** | `7.0`, `3.5` | Fedeltà al prompt (Guidance Scale). |
| **`dcs`** | **Distilled Cfg** | `3.5`, `2.0` | Per modelli distillati (Flux/SD3.5). Imposta autom. `cs=1.0`. |
| **`smp`** | **SaMPler** | `euler`, `dpm++` | Supporta shortcode (es. `d2m`, `ea`). |
| **`sch`** | **SCHeduler** | `simple`, `sgm` | Supporta shortcode (es. `s`, `k`). |

### Mapping Shortcode Comuni
* **Sampler (`smp`):** `ea` (Euler a), `d2m` (DPM++ 2M), `d2s` (DPM++ 2M SDE), `lcm` (LCM).
* **Scheduler (`sch`):** `s` (Simple), `k` (Karras), `e` (Exponential), `sgm` (SGM Uniform).

---

## 📝 Parsing Logico: Stili e Soggetto

### 1. Separazione Esplicita (Consigliata)
Utilizza `--` per separare nettamente le configurazioni dal prompt.
> `img sz=1024 -- gatto robot`

### 2. Separazione Implicita (Smart)
Se `--` non è presente, il sistema cerca l'ultimo parametro tecnico (es. `s=20`) e considera tutto ciò che segue come soggetto. Gli elementi non riconosciuti come parametri ma posti prima del soggetto diventano **Styles**.
> `img sz=1024 cinematic lighting gatto robot`

### 3. Negative Prompt
Tutto ciò che segue `--no` viene inviato all'engine come prompt negativo.
> `... --no blur, low quality, text`

---

## 💡 Esempi di Utilizzo

#### Base (Uso Quotidiano)
```bash
img un paesaggio cyberpunk sotto la pioggia
```












```python

==================================================================================
# Inizio integrazione OpenAI, Gemini e ComfyUI
==================================================================================

else:
    self._dbg(f"Using Standard OWUI Router for engine: {engine}")
    
    # Costruiamo il form includendo i campi che il router di OWUI sa mappare
    form_data = CreateImageForm(
        prompt=self.model.enhanced_prompt,
        model=self.model.model,
        size=self.model.size,
        n=1,
        negative_prompt=self.model.get("negative_prompt", ""),
        # Qui puoi iniettare parametri extra se l'engine è OpenAI
        quality=self.model.get("quality", "standard"),
        style=self.model.get("style", "vivid")
    )

    gen_res = await image_generations(
        request=self.request,
        form_data=form_data,
        user=self.context,
    )


```



# Gemini Imagen 3 API Parameters Tree

Struttura gerarchica dei parametri per chiamate dirette via `httpx` agli endpoint Vertex AI / AI Studio.

* **root** (JSON Payload)
    * **instances[]** (Array)
        * `prompt`: (string) Prompt principale migliorato (Enhanced Prompt).
    * **parameters** (Object)
        * `negativePrompt`: (string) Elementi da escludere (es: "blurry, low quality, distorted").
        * `sampleCount`: (int) Numero di immagini generate (1-4).
        * `aspectRatio`: (string) Rapporto di forma richiesto.
            * `1:1` (Quadrato)
            * `9:16` (Verticale)
            * `16:9` (Widescreen)
            * `3:4` (Portrait)
            * `4:3` (Landscape)
        * `seed`: (int) Valore per generazioni deterministiche (necessita `addWatermark: false`).
        * `safetySetting`: (string) Livello di filtraggio contenuti.
            * `block_none` (Permissivo - consigliato per bypass)
            * `block_only_high`
            * `block_medium_and_above`
            * `block_low_and_above`
        * `personGeneration`: (string) Policy per soggetti umani.
            * `allow_all`
            * `allow_adult`
            * `dont_allow`
        * `enhancePrompt`: (boolean) Espansione prompt tramite LLM interno.
        * `addWatermark`: (boolean) Applicazione SynthID invisibile.
        * `includeReasoning`: (boolean) Ritorna i metadati sulle scelte del modello.
        * **outputOptions** (Object)
            * `mimeType`: (string) "image/png" | "image/jpeg"
            * `compressionQuality`: (int) 0-100 (Solo per JPEG)

```python
elif engine == "gemini":
    self._dbg("Using Gemini Direct Bypass (Full Params)...")
    
    # Recuperiamo le config dalle state dell'app (quelle che hai postato nel sorgente)
    api_key = self.request.app.state.config.IMAGES_GEMINI_API_KEY
    base_url = self.request.app.state.config.IMAGES_GEMINI_API_BASE_URL
    method = self.request.app.state.config.IMAGES_GEMINI_ENDPOINT_METHOD or "predict"
    
    # Costruiamo il payload "reale" di Google (Imagen)
    # Qui puoi aggiungere tutto quello che il router standard ignorava
    if method == "predict":
        api_url = f"{base_url}/models/{model}:predict"
        payload = {
            "instances": [
                {
                    "prompt": self.model.enhanced_prompt,
                }
            ],
            "parameters": {
                "sampleCount": self.model.n or 1,
                "negativePrompt": self.model.get("negative_prompt"), # Ecco il fix!
                "aspectRatio": self._map_size_to_gemini_ratio(self.model.size),
                "safetySetting": "block_none", # Esempio di parametro extra
                "outputOptions": {"mimeType": "image/png"}
            }
        }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(api_url, json=payload, headers=headers, timeout=None)
        r.raise_for_status()
        res = r.json()
        
        # Estrazione b64 (logica speculare a quella del router ma gestita qui)
        img_b64 = res["predictions"][0]["bytesBase64Encoded"]
        self.model.b64_data = img_b64
        self.model.image_url = f"data:image/png;base64,{img_b64}"

```

# OpenAI httpx

I parametri ammessi sono solo ed esclusivamente:

- model: (es. dall-e-3 o dall-e-2)

- prompt: (La tua descrizione)

- n: (Numero di immagini, fisso a 1 per DALL-E 3)

- size: (1024x1024, 1024x1792, o 1792x1024)

- quality: (standard o hd)

- style: (vivid o natural)

- response_format: (url o b64_json)

- user: (Un ID univoco opzionale per monitorare gli abusi)


```python

elif engine == "openai":
    self._dbg("Using OpenAI Direct Bypass (DALL-E 3 HD)...")
    
    api_key = self.request.app.state.config.IMAGES_OPENAI_API_KEY
    base_url = self.request.app.state.config.IMAGES_OPENAI_API_BASE_URL
    
    # Costruiamo il payload specifico per OpenAI
    payload = {
        "model": model or "dall-e-3",
        "prompt": self.model.enhanced_prompt,
        "n": 1,
        "size": self.model.size or "1024x1024",
        "quality": self.valves.OPENAI_QUALITY or "hd", # Preso dalle tue Valves
        "style": self.valves.OPENAI_STYLE or "vivid",   # Preso dalle tue Valves
        "response_format": "b64_json" 
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        # endpoint standard OpenAI
        url = f"{base_url}/images/generations"
        r = await client.post(url, json=payload, headers=headers, timeout=None)
        r.raise_for_status()
        
        res = r.json()
        img_b64 = res["data"][0]["b64_json"]
        
        # Salviamo nel modello per l'output finale
        self.model.b64_data = img_b64
        self.model.image_url = f"data:image/png;base64,{img_b64}"

```


| Parametro EasyMage | Forge (A1111) | OpenAI (DALL-E 3) | Gemini (Imagen 3) | ComfyUI |
| :--- | :--- | :--- | :--- | :--- |
| **model** | `model` | `model` | `model` | `Checkpoints` Node |
| **prompt** | `prompt` | `prompt` + `neg` | `prompt` | `CLIP Text Encode` |
| **negative_prompt** | `negative_prompt` | ⚠️ **Iniettato nel Prompt** | `negativePrompt` | `Conditioning` Node |
| **style** | ❌ N/A (via prompt) | ✅ `style` (vivid/natural) | ❌ N/A (via prompt) | `Style` Node / Lora |
| **steps** | `steps` | ❌ N/A | ❌ N/A | `steps` |
| **size** | `width`/`height` | `size` (mappato) | `aspectRatio` (mappato) | `Empty Latent` |
| **seed** | `seed` | ❌ N/A | `seed` | `seed` |
| **cfg_scale** | `cfg_scale` | ❌ N/A | ❌ N/A | `cfg` |
| **distilled_cfg_scale**| `distilled_cfg` | ❌ N/A | ❌ N/A | `pag_scale` |
| **sampler_name** | `sampler_name` | ❌ N/A | ❌ N/A | `sampler_name` |
| **scheduler** | `scheduler` | ❌ N/A | ❌ N/A | `scheduler` |
| **enable_hr** | `enable_hr` | ✅ `quality: hd` | ❌ N/A | `Upscale` Workflow |
| **n_iter / batch** | `n_iter` / `batch` | ❌ N/A (1) | `sampleCount` (1-4) | `batch_size` |
| **hr_scale** | `hr_scale` | ❌ N/A | ❌ N/A | `Scale Factor` |
| **hr_upscaler** | `hr_upscaler` | ❌ N/A | ❌ N/A | `Upscale Model` |
| **denoising_strength** | `denoising_strength` | ❌ N/A | ❌ N/A | `denoise` |

---

# Mapping Table

| Categoria | Parametro Cmdline | Parametro EasyMage | Forge (A1111) | OpenAI (DALL-E 3) | Gemini (Imagen 3) | ComfyUI |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Engine** | `ge` | **engine** | `automatic1111/a` | `openai/o` | `gemini/g` | `comfyui/c`|
| **Model** | `mdl` | **model** | `model` | `model` | `model` | `Checkpoints` Node |
| **Core** | *(subject)* | **prompt** | `prompt` | `prompt` + `neg` | `prompt` | `CLIP Text Encode` |
| **Core** | *(trigger)* | **negative_prompt** | `negative_prompt` | ⚠️ **Iniettato nel Prompt** | `negativePrompt` | `Conditioning` Node |
| **Core** | `stl` | **style** | ❌ N/A (via prompt) | ✅ `style` (vivid/natural) | ❌ N/A (via prompt) | `Style` Node / Lora |
| **Core** | `stp` | **steps** | `steps` | ❌ N/A | ❌ N/A | `steps` |
| **Core** | `sz` | **size** | `width`/`height` | `size` (mappato) | `aspectRatio` (mappato) | `Empty Latent` |
| **Core** | `sd` | **seed** | `seed` | ❌ N/A | `seed` | `seed` |
| **Guidance** | `cs` | **cfg_scale** | `cfg_scale` | ❌ N/A | ❌ N/A | `cfg` |
| **Guidance** | `dcs` | **distilled_cfg_scale** | `distilled_cfg` | ❌ N/A | ❌ N/A | `pag_scale` |
| **Sampling** | `smp` | **sampler_name** | `sampler_name` | ❌ N/A | ❌ N/A | `sampler_name` |
| **Sampling** | `sch` | **scheduler** | `scheduler` | ❌ N/A | ❌ N/A | `scheduler` |
| **High-Res** | `+h` / `-h` | **enable_hr** | `enable_hr` | ✅ `quality: hd` | ❌ N/A | `Upscale` Workflow |
| **Cloud** | `n` / `b` | **n_iter / batch** | `n_iter` / `batch` | ❌ N/A (1) | `sampleCount` (1-4) | `batch_size` |
| **High-Res** | `hr` | **hr_scale** | `hr_scale` | ❌ N/A | ❌ N/A | `Scale Factor` |
| **High-Res** | `hru` | **hr_upscaler** | `hr_upscaler` | ❌ N/A | ❌ N/A | `Upscale Model` |
| **High-Res** | `hdcs` | **hr_distilled_cfg** | `hr_distilled_cfg` | ❌ N/A | ❌ N/A | Custom Node |
| **High-Res** | `dns` | **denoising_strength** | `denoising_strength` | ❌ N/A | ❌ N/A | `denoise` |
| **Debug** | `+d` / `-d` | **debug** | ✅ Internal Log | ❌ N/A | ❌ N/A | ✅ Internal Log |
| **Audit** | `+a` / `-a` | **quality_audit** | ✅ Vision Post | ❌ N/A | ❌ N/A | ✅ Vision Post |
| **Prompt** | `+p` / `-p` | **enhanced_prompt** | ✅ LLM Pre-proc | ✅ LLM Pre-proc | ✅ LLM Pre-proc | ✅ LLM Pre-proc |


**Models**: 
*  a1111: `*.safetensors`
* openai: `dall-e-3`,`dall-e-2`
* gemini: `imagen-3.0-generate-001`,`imagen-3.0-fast-generate-001`
* comfyui: uses the node id