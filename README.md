# The Scoop UP

**The Scoop UP** is an automation script designed to help you like or swipe on everyone in a dating app. This saves you time and effort, allowing you to focus on engaging with the people you want to talk to instead of spending hours sending likes. Make your life more efficient with this tool!

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

2. **Open iPhone Mirroring:**
   Connect the iPhone, open Tinder or Hinge, and keep the iPhone Mirroring window visible.

3. **Launch The Scoop UP:**

   ```bash
   python main_scoop.py
   ```

   Or open the standalone V1.0.0 app at `dist/The Scoop UP V1.0.0.app`.

4. **Choose the App:**
   Select **Hinge** or **Tinder**. No window coordinates, heart location, or button location are required.

5. **Set Rotations:**
   Enter the number of rotations (i.e., how many cycles of actions you want the script to perform).

6. **Start the Simulation:**
   Click the **Start** button to begin the automated clicking process. The script will:
   - Automatically locate the iPhone Mirroring window before every scan.
   - **Tinder:** find the heart at its current position → click → wait 2 seconds → repeat.
   - **Hinge:** scan for the white-heart/black-circle button for up to 2 seconds → when found, click it 3 times and wait 1 second for the UI → search for Send Priority Like for up to 2 seconds and click it 3 times when found. If the heart is missing, the app tries Send Like directly in case the confirmation UI is already open.
   - Skip a click whenever its required visual target is not detected.

7. **Stop the Simulation:**
   Click **Stop** at any time. You can also press **Esc** while the control panel is focused.

On first use, allow Terminal or Python under **System Settings → Privacy & Security → Accessibility** and **Screen & System Audio Recording**, then restart the program. If screen capture is blocked, the app displays an **Open Screen Recording Settings** button that opens the correct permissions page directly.

---

## Additional Information

- The script uses native macOS window metadata, Quartz screen capture, and Vision text recognition.
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
