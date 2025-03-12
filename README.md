# The Scoop UP

**The Scoop UP** is an automation script designed to help you like or swipe on everyone in a dating app. This saves you time and effort, allowing you to focus on engaging with the people you want to talk to instead of spending hours sending likes. Make your life more efficient with this tool!

---

## System Requirements

- **MacOS (version 15 or later)**
- **iPhone (iOS 18 or later)**
- Utilizes the **Projection** feature on MacOS to connect and interact with your iPhone.

---

## Important Note

It is **strongly recommended** that you have a premium subscription to the dating app (with unlimited swipes/likes) for the script to perform effectively.

---

## How to Use

1. **Set up Projection:**  
   Set up the Projection feature on your MacOS to mirror your iPhone, and make sure the app is runing

2. **Launch the Script:**  
   Run the script on your MacOS. A GUI window will appear with an initial size of 400x360.

3. **Coordinate Selection:**
   - **Heart Area Inputs:**  
     Click on your screen to automatically fill in the “Heart area top left” and “Heart area bottom right” fields.
   - **Popup Confirmation:**  
     Once the heart area fields are filled, a popup will appear instructing you:  
     *"You can click the heart to start filling the next 2 location"*  
     Click the **Done** button to continue.
   - **'Sent Like' Inputs:**  
     After dismissing the popup, click again to fill in the **'Sent Like' top left** and **'Send Like' bottom right** fields.

4. **Set Rotations:**  
   Enter the number of rotations (i.e., how many cycles of actions you want the script to perform).

5. **Start the Simulation:**  
   Click the **Start** button to begin the automated clicking process. The script will:
   - Randomly click within the specified Heart area.
   - Then click within the 'Sent Like' area.
   - Repeat this for the number of rotations provided.

6. **Stop the Simulation:**  
   At any time during the simulation, press the **ESC** key to pause/stop the automation. (Must select the program panel before press **ESC**, or it won't stop, so you gotta be fast)

---

## Additional Information

- The script uses **Tkinter** for the GUI and **pynput** for capturing mouse events and simulating clicks.
- The random clicking within the defined areas ensures varied interactions, simulating human behavior.
- For any issues or further customization, please refer to the code comments for guidance.

Happy swiping, and enjoy the time saved with The Scoop UP!