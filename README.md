# The Scoop UP

A macOS helper that uses **iPhone Mirroring** to automate Hinge likes and optional prompt replies.

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

clone this repo
```bash
cd theScoopUp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Optional: local replies (no API key)

Prompt Reply defaults to **Local — Free**, which uses [Ollama](https://ollama.com/download) on your Mac. Profile text stays on this computer.

```bash
brew install ollama
ollama pull qwen3.5:9b
```

The model needs about 6.6 GB of disk space. The first launch can take 20–40 seconds. The Scoop UP starts Ollama automatically if it is installed.

### Optional: OpenAI replies

Set a key only if you choose **OpenAI API** in the app:

```bash
export OPENAI_API_KEY="your-api-key"
```

Never commit this key. The OpenAI engine sends recognized prompt/answer text only — not photos or screenshots.

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
   - For Prompt Reply: reply engine and tone
   - Number of rotations (Auto Like = like cycles; Prompt Reply = profiles)

5. Click **Start**. Click **Stop** or press **Esc** to halt.

---

## Workflows (Hinge)

**Auto Like** finds Hinge’s like button, clicks it, then looks for **Send Like** / **Send Priority Like** and clicks that. If a target is not detected, that cycle is skipped.

**Prompt Reply** (Hinge only) finds the first written prompt on a profile, generates a short reply, pastes it only after on-screen verification, then sends. If prompt reply fails, it can fall back to a photo-grounded pickup line, a custom fallback line you typed, or a built-in clean line — then skip to the next profile when recovery fails.

Reply tones: **Playful & clean**, **Flirty & bold**, **Dry & clever**. Replies are capped at 140 characters and checked locally before send.

---

## Build a standalone Mac app (optional)

```bash
source .venv/bin/activate
python -m PyInstaller --clean --noconfirm main_scoop.spec
```

The app is written to `dist/The Scoop UP V1.0.0.app`.

If you use OpenAI with the standalone build, launch it from Terminal so the key is available:

```bash
OPENAI_API_KEY="your-api-key" \
  "dist/The Scoop UP V1.0.0.app/Contents/MacOS/The Scoop UP V1.0.0"
```

---

## Notes

- This tool drives the mirrored iPhone UI. Dating-app rules and account risk are your responsibility.
- Captured screens stay in process memory for detection; they are not written to disk for Prompt Reply.
- Local — Free talks only to Ollama on `127.0.0.1`. OpenAI is used only when that engine is selected.
- Coordinates are never reused after a click. If a button or heart cannot be found, nothing is clicked.
