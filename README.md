# The Scoop UP

**Version 2.0.0**

A macOS helper that uses **iPhone Mirroring** to automate Hinge and Tinder likes and optional prompt replies.

**Current status:** **Hinge is the only supported, tested app.** A Tinder option still appears in the UI, but Tinder detection and clicking have **not** been checked or verified. Do not rely on Tinder mode.

---

## Requirements

- macOS 15 or later
- iPhone running iOS 18 or later
- Apple **iPhone Mirroring** (keep the window visible while the app runs)
- Python 3.11+ recommended
- A Hinge account. A premium / unlimited-likes subscription is strongly recommended.

---

## Install

clone this repo then continue with the following steps
```bash
cd theScoopUp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Free local LLM (recommended)

Prompt Reply defaults to **Local — Free**.

The local model name lives in `reply_generation.py` as `LOCAL_FREE_MODEL`. This release prefers **qwen3.5:9b**.

#### Preferred: qwen3.5:9b

1. Install Ollama:

   ```bash
   brew install ollama
   ```

   Or download the Mac app from [ollama.com/download](https://ollama.com/download).

2. Pull the preferred model:

   ```bash
   ollama pull qwen3.5:9b
   ```

3. Confirm it is listed:

   ```bash
   ollama list
   ```

   You should see `qwen3.5:9b`. Leave `LOCAL_FREE_MODEL = "qwen3.5:9b"` as-is. The Scoop UP starts Ollama automatically if it is installed.

The model needs about 6.6 GB of disk space. The first launch can take 20–40 seconds.

#### Using a different local model

Prompt Reply reads the mirrored screen (written prompts and photos), so pick a **vision-capable** model. Quality and speed will vary; **qwen3.5:9b** is the one this release is built around.

1. Pull your model and copy the exact name from `ollama list` (including the tag, such as `:latest` or `:11b`):

   ```bash
   ollama pull llama3.2-vision
   ollama list
   ```

2. Open `reply_generation.py` and change the local model constant:

   ```python
   LOCAL_FREE_MODEL = "llama3.2-vision:latest"
   ```

3. Restart The Scoop UP (`python main_scoop.py`). If you already built the standalone app, rebuild it so the new name is included.

If the name does not match an installed model, the app will tell you to run `ollama pull <model>`.

### Optional: paid API replies

Choose **Paid API** in the app, select an available model, then import or paste that provider's API key. **Import** reads a `.txt` or `.env` file such as:

```
OPENAI_API_KEY="your-api-key"
ANTHROPIC_API_KEY="your-api-key"
GEMINI_API_KEY="your-api-key"
XAI_API_KEY="your-api-key"
DEEPSEEK_API_KEY="your-api-key"
```

The key stays in memory for this session and is not written to disk. Matching environment variables are used to pre-fill the field for that provider.

Available paid models are listed in `reply_generation.py` as `PAID_MODELS` and include OpenAI, Claude, Gemini, Grok, and DeepSeek. Prompt Reply sends the first recognized prompt/answer and selected tone as a compact text request — never photos or screenshots. Paid output is accepted as raw text without local content, grounding, length, or retry checks.

---

## Run

1. Connect your iPhone, open **iPhone Mirroring**, and open **Hinge**. Keep that window on screen.
2. Launch the app:

   ```bash
   python main_scoop.py
   ```

3. On first use, allow **Terminal** or **Python** under:
   - **System Settings → Privacy & Security → Accessibility**
   - **System Settings → Privacy & Security → Screen & System Audio Recording**

   Then restart The Scoop UP. If capture is blocked, use **Open Screen Recording Settings** in the app.

4. In the control panel:
   - Dating app: **Hinge** (required for a working run)
   - Workflow: **Auto Like** or **Prompt Reply**
   - For Prompt Reply: reply engine and tone. If you choose **Paid API**, pick a model and import or paste that provider's API key. Optionally check **Save prompt & reply** to append each profile (sent and failed) to a Desktop file named like `Scoop gpt-5-mini 2026-08-19 18-45-30.txt`.
   - Number of rotations (Auto Like = like cycles; Prompt Reply = profiles)

5. Click **Start**. Click **Stop** or press **Esc** to halt.

---

## Workflows (Hinge)

**Auto Like** finds Hinge’s like button, clicks it, then looks for **Send Like** / **Send Priority Like** and clicks that. If a target is not detected, that cycle is skipped.

**Prompt Reply** (Hinge only) finds the first written prompt on a profile, generates a short reply, pastes it only after on-screen verification, then sends. If prompt reply fails, it can fall back to a photo-grounded pickup line or a built-in clean line — then skip to the next profile when recovery fails. **Local — Free** can use local vision on any recoverable failure. **Paid API** sends the first profile photo to the selected vision-capable model (OpenAI, Claude, Gemini, Grok) with the same house rules only when scan finds no prompt; other paid failures use built-in pickup lines only.

Reply tones: **Playful & clean**, **Flirty & bold**, **Dry & clever**. Local — Free replies are capped at 140 characters and checked locally. Paid API replies are used as returned, while the app still verifies the open Hinge prompt and pasted text before clicking Send. In paid mode, photo fallback is attempted only when the scan finds no readable written prompt; failures after a prompt is detected skip the profile instead of redirecting the reply to a photo.

---

## Build a standalone Mac app (optional)

```bash
source .venv/bin/activate
python -m PyInstaller --clean --noconfirm main_scoop.spec
```

The app is written to `dist/The Scoop UP V2.0.0.app`.

If you use a paid model with the standalone build, select the model and import the key in the app. You can still pre-fill the matching field from Terminal:

```bash
OPENAI_API_KEY="your-api-key" \
  "dist/The Scoop UP V2.0.0.app/Contents/MacOS/The Scoop UP V2.0.0"
```

A different `LOCAL_FREE_MODEL` only takes effect in the `.app` after you rebuild. Paid models are chosen in the UI.

---

## Notes

- This tool drives the mirrored iPhone UI. Dating-app rules and account risk are your responsibility.
- Captured screens stay in process memory for detection; they are not written to disk for Prompt Reply. If **Save prompt & reply** is checked, prompt text, model input, and generated replies are appended to a Desktop file named like `Scoop gpt-5-mini 2026-08-19 18-45-30.txt`.
- Local — Free talks only to Ollama on `127.0.0.1`. Paid API is used only when that engine is selected.
- Coordinates are never reused after a click. If a button or heart cannot be found, nothing is clicked.
