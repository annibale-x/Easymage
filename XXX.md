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
        * `aspectRatio`: (string) Rapporto di forma richiesto. **UNICO VERO PARAMETRO IN PIU, GLI ALTRI SONO TUTTI RICONDUCIBILÎ**
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

# Funzione per calcolo size per Forge, OpenAI e Gemini


```python
def _resolve_params(self, input_size: str, engine: str):
    """
    Normalizza le dimensioni e i parametri per i diversi motori.
    Esegue il calcolo dei pixel per Forge e il mapping AR per Gemini/OpenAI.
    """
    # 1. Default fallback
    base_w, base_h = 1024, 1024
    ar_string = "1:1"

    # 2. Parsing dell'input (accetta "1024x1024" o "16:9")
    if "x" in input_size:
        try:
            parts = input_size.split("x")
            base_w, base_h = int(parts[0]), int(parts[1])
            # Calcolo approssimativo dell'AR per Gemini
            ratio = base_w / base_h
            if ratio > 1.7: ar_string = "16:9"
            elif ratio > 1.3: ar_string = "4:3"
            elif ratio < 0.6: ar_string = "9:16"
            elif ratio < 0.8: ar_string = "3:4"
            else: ar_string = "1:1"
        except: pass
    elif ":" in input_size:
        ar_string = input_size
        # Calcolo pixel per Forge basato su una base di 1024
        mapping = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152)
        }
        base_w, base_h = mapping.get(ar_string, (1024, 1024))

    # 3. Restituzione parametri specifici per motore
    if engine == "gemini":
    	base_url = config.IMAGES_GEMINI_API_BASE_URL
	    api_key = config.IMAGES_GEMINI_API_KEY
	    model_id = self.model.get("model") or "imagen-3.0-generate-001"
	    # Endpoint Gemini (struttura Google)
	    url = f"{base_url.rstrip('/')}/models/{model_id}:predict"
        return {
            "aspectRatio": ar_string,
            "sampleCount": 1,
            "negativePrompt": self.model.get("negative_prompt", ""),
            "safetySetting": "block_none",
            "addWatermark": False,
            "personGeneration": "allow_all"
        }
    
    elif engine == "forge":
        return {
            "width": base_w,
            "height": base_h,
            "seed": self.valves.FORGE_SEED if hasattr(self.valves, 'FORGE_SEED') else -1,
            "negative_prompt": self.model.get("negative_prompt", "")
        }

    elif engine == "openai":

		base_url = config.IMAGES_OPENAI_API_BASE_URL
	    api_key = config.IMAGES_OPENAI_API_KEY
	    # Endpoint specifico per le immagini
	    url = f"{base_url.rstrip('/')}/images/generations"

        # Mapping risoluzioni DALL-E 3 (le uniche 3 ammesse)
        oa_size = "1024x1024"
        if ar_string == "16:9": oa_size = "1792x1024"
        elif ar_string == "9:16": oa_size = "1024x1792"
        
        # Mapping Quality basato su flag HR (High Resolution)
        # Supponendo che 'hr' sia un booleano nel tuo modello
        quality = "hd" if self.model.get("hr", False) else "standard"
        
        return {
            "model": self.model.get("model") or "dall-e-3",
            "prompt": self.model.enhanced_prompt,
            "size": oa_size,
            "quality": quality,
            "style": "vivid", # Hardcoded per coerenza visiva
            "response_format": "b64_json",
            "user": str(self.request.app.state.config.USER_ID) # Esempio di ID utente
        }