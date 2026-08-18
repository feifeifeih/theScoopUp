# The Scoop UP

**The Scoop UP** automates likes in Tinder and Hinge and can scan an entire Hinge profile, write a funny response to one profile prompt, verify it, and send it.

---

## System Requirements

- **MacOS (version 15 or later)**
- **iPhone (iOS 18 or later)**
- Uses Apple's **iPhone Mirroring** app to interact with your iPhone.

---

## Important Note

It is **strongly recommended** that you have a premium subscription to the dating app (with unlimited swipes/likes) for the script to perform effectively.

---

## How to Use

1. **Install dependencies:**

   ```bash
   cd /Users/fei/Desktop/SideProject/theScoopUp
   source myenv/bin/activate
   python -m pip install -r requirements.txt
   ```

   Prompt Reply defaults to **Local — Free**, which uses Ollama and does not need an API key or send profile text over the internet. Install the local runtime and model once if they are not already present:

   ```bash
   brew install ollama
   ollama pull qwen3.5:9b
   ```

   The Scoop UP starts the local Ollama service automatically and warms it when Prompt Reply starts. The model requires about 6.6 GB of disk space and may take 20–40 seconds to become ready on first launch.

   The optional **OpenAI API** engine requires a paid API account and key:

   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

   Add that export to your shell profile if you want it available every time you launch the app from Terminal. Never commit the key to this repository.

2. **Open iPhone Mirroring:**
   Connect the iPhone, open Tinder or Hinge, and keep the iPhone Mirroring window visible.

3. **Launch The Scoop UP:**

   ```bash
   python main_scoop.py
   ```

   Local — Free mode can be opened normally from `dist/The Scoop UP V1.0.0.app`. If you select OpenAI API, launch the standalone build from a terminal so it receives the key:

   ```bash
   OPENAI_API_KEY="your-api-key" \
     "dist/The Scoop UP V1.0.0.app/Contents/MacOS/The Scoop UP V1.0.0"
   ```

   Auto Like also does not need an API key.

4. **Choose the App and Workflow:**
   Select **Hinge** or **Tinder**, then choose one of these workflows:
   - **Auto Like:** preserves the original automatic like behavior for either app.
   - **Prompt Reply:** available for Hinge only. It jumps past the profile header, then uses fast native OCR first. If that cannot parse a viewport, `qwen3.5:9b` gets one bounded vision attempt to extract the first written prompt; native OCR and heart detection must still anchor the click target. It ignores non-written cards such as “Let's get together” and stops scanning as soon as the first valid written prompt is found. It generates a reply tied directly to that answer and sends it only after every verification passes. If prompt reply fails and the fallback field is blank, Qwen analyzes an in-memory crop of the first photo and generates a line grounded in a visible pet, activity, food, landmark, object, or setting. Unsafe, ungrounded, or unclear results fall back to the built-in clean list. A custom fallback line still overrides photo generation.

5. **Choose a Reply Tone:**
   Choose **Local — Free** or **OpenAI API**, then select **Playful & clean**, **Flirty & bold**, or **Dry & clever**. These settings are ignored by Auto Like.

6. **Set Rotations:**
   For Auto Like, rotations are action cycles. For Prompt Reply, rotations are profiles with one verified reply sent per profile.

7. **Start the Automation:**
   Click the **Start** button to begin the automated clicking process. The script will:
   - Automatically locate the iPhone Mirroring window before every scan.
   - **Tinder:** find the heart at its current position → click → wait 2 seconds → repeat.
   - **Hinge:** scan for the white-heart/black-circle button for up to 2 seconds → when found, click it 3 times and wait 1 second for the UI → search for Send Priority Like for up to 2 seconds and click it 3 times when found. If the heart is missing, the app tries Send Like directly in case the confirmation UI is already open.
   - Skip a click whenever its required visual target is not detected.
   - **Hinge Prompt Reply:** scroll to the top for the first profile → jump to the likely prompt area → scan overlapping viewports only until the first valid written prompt is found → use Qwen vision once only when native parsing needs help → start generating a reply containing a concrete word from the answer while re-confirming and centering that prompt → wait for the generated reply before opening its heart → confirm the opened dialog matches that prompt → clear any preserved draft → retry paste and fall back to direct typing when Mirroring drops clipboard input → verify only inside the composer → click Send Like up to three times only while fresh captures prove the same verified dialog remains, using verified layout when its label OCR is garbled → poll until the dialog leaves instead of waiting a fixed 1.1 seconds → confirm that the dialog and profile changed.

8. **Stop the Automation:**
   Click **Stop** at any time. You can also press **Esc** while the control panel is focused.

On first use, allow Terminal or Python under **System Settings → Privacy & Security → Accessibility** and **Screen & System Audio Recording**, then restart the program. If screen capture is blocked, the app displays an **Open Screen Recording Settings** button that opens the correct permissions page directly.

---

## Additional Information

- The script uses native macOS window metadata, ScreenCaptureKit capture, and Vision text recognition. Accurate OCR is used to parse prompts and verify replies; Fast OCR is used only to poll for Send Like and comment-field visibility. Captured pixels are copied into process-owned memory immediately so WindowServer image transports are not retained across profiles.
- The OpenAI engine sends only locally recognized prompt/answer text to OpenAI. Profile photos and screenshots are not uploaded.
- Local — Free sends prompt text only to the Ollama service on this Mac and requires no authentication.
- Reply generation uses `gpt-5.6-luna` through the Responses API. API charges and account rate limits apply.
- Replies are limited to 140 characters and validated locally. Generation is retried once, but a Send Like click is never retried when Hinge's resulting state is uncertain.
- Prompt Reply never guesses at an unsafe target. When prompt handling fails before an uncertain send, it returns to the top and sends the user's validated custom pickup line—or a random built-in clean line when the field is blank—through the same entry, OCR, safe-position, and post-send checks. Uncertain send state still stops without another click.
- The temporary clipboard content used for pasting is restored after the paste command.
- Heart matching runs at native Retina capture resolution; only the final click point is converted to desktop coordinates.
- A heart-analysis timeout is retried on the next cycle and is not treated as a completed “no heart” result.
- Target detection runs again after every state-changing click; coordinates are never reused.
- Heart and Send Priority Like clicks target the center of a positively detected visual target.
- For any issues or further customization, please refer to the code comments for guidance.

Happy swiping, and enjoy the time saved with The Scoop UP!

---

## Build the Standalone App

```bash
source myenv/bin/activate
python -m PyInstaller --clean --noconfirm main_scoop.spec
```

The reusable macOS application is created at `dist/The Scoop UP V1.0.0.app`.
