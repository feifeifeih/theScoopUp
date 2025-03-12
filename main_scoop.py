from pynput.mouse import Listener, Controller, Button
import tkinter as tk
import random
import time
import re  # for parsing the position string
import os
import sys
import threading  # to run simulation in a separate thread

class DesktopClickSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Desktop Click Selector - with Restart & ESC Stop")
        # Set fixed initial panel size (400x360); user can still resize manually
        self.root.geometry("400x360")
        
        # Mouse controller for simulating clicks
        self.mouse_controller = Controller()
        
        # Simulation control variables
        self.total_rotations = 20  # default number of rotations
        self.is_running = False    # flag to control simulation loop
        
        # Counter for clicks to determine which input to fill
        self.click_count = 0
        # Flag to pause input until the user clicks "Done" in the popup
        self.waiting_for_done = False

        # ========== GUI Layout ==========
        # (A) Label to display current mouse position
        self.current_pos_label = tk.Label(self.root, text="Current Mouse Position: (0, 0)")
        self.current_pos_label.pack()

        # (B) Button: Restart Program only (removed Stop and Copy buttons)
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        self.restart_button = tk.Button(button_frame, text="Restart Program", command=self.restart_program)
        self.restart_button.grid(row=0, column=0, padx=5, pady=5)

        # (C) Heart Area (First Section)
        self.label_first_rect = tk.Label(self.root, text="Heart area top left:")
        self.label_first_rect.pack()
        self.entry_first_rect_tl = tk.Entry(self.root)
        self.entry_first_rect_tl.pack()

        self.label_first_rect_br = tk.Label(self.root, text="Heart area bottom right:")
        self.label_first_rect_br.pack()
        self.entry_first_rect_br = tk.Entry(self.root)
        self.entry_first_rect_br.pack()

        # (D) 'Sent Like' Area (Second Section)
        self.label_tl = tk.Label(self.root, text="'Sent Like' top left:")
        self.label_tl.pack()
        self.entry_tl = tk.Entry(self.root)
        self.entry_tl.pack()

        self.label_br = tk.Label(self.root, text="'Send Like' bottom right:")
        self.label_br.pack()
        self.entry_br = tk.Entry(self.root)
        self.entry_br.pack()

        # (E) Number of Rotations
        self.label_rotations = tk.Label(self.root, text="Number of Rotations:")
        self.label_rotations.pack()
        self.entry_rotations = tk.Entry(self.root)
        self.entry_rotations.pack()

        # (F) Start Button to begin simulation
        self.start_button = tk.Button(self.root, text="Start", command=self.on_start)
        self.start_button.pack()

        # (G) Info Label for displaying messages
        self.info_label = tk.Label(self.root, text="")
        self.info_label.pack()

        # Bind the Escape key to allow stopping the simulation
        self.root.bind("<Escape>", self.handle_escape_key)

        # Start a mouse listener to capture clicks and movements
        self.listener = Listener(on_move=self.on_move, on_click=self.on_click)
        self.listener.start()

    # -------------------------------
    #    Mouse Listener Callbacks
    # -------------------------------
    def on_move(self, x, y):
        """Continuously update the 'Current Mouse Position' label."""
        self.current_pos_label.config(text=f"Current Mouse Position: ({x}, {y})")

    def on_click(self, x, y, button, pressed):
        """
        Auto-enter the coordinates into the appropriate input field when the user clicks.
        First click: fills Heart area top left.
        Second click: fills Heart area bottom right.
        (After these, a popup appears to instruct the user.)
        Third click: fills 'Sent Like' top left.
        Fourth click: fills 'Send Like' bottom right.
        """
        # Ignore clicks if waiting for the popup to be dismissed
        if self.waiting_for_done:
            return

        if not pressed and button == Button.left:
            coord_str = f"({x}, {y})"
            if self.click_count == 0:
                self.entry_first_rect_tl.delete(0, tk.END)
                self.entry_first_rect_tl.insert(0, coord_str)
                self.info_label.config(text=f"Heart area top left set to {coord_str}")
            elif self.click_count == 1:
                self.entry_first_rect_br.delete(0, tk.END)
                self.entry_first_rect_br.insert(0, coord_str)
                self.info_label.config(text=f"Heart area bottom right set to {coord_str}")
            elif self.click_count == 2:
                self.entry_tl.delete(0, tk.END)
                self.entry_tl.insert(0, coord_str)
                self.info_label.config(text=f"'Sent Like' top left set to {coord_str}")
            elif self.click_count == 3:
                self.entry_br.delete(0, tk.END)
                self.entry_br.insert(0, coord_str)
                self.info_label.config(text=f"'Send Like' bottom right set to {coord_str}")
            else:
                # Ignore further clicks if all coordinate fields have been set
                self.info_label.config(text="All coordinate fields have been set.")
                return

            self.click_count += 1

            # After filling the first 2 fields (Heart area), show the popup
            if self.click_count == 2:
                self.waiting_for_done = True
                self.show_popup()

    # -------------------------------
    #       Popup Functions
    # -------------------------------
    def show_popup(self):
        """Show a popup window instructing the user to click the heart for the next locations."""
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Next Step")
        self.popup.geometry("400x100")
        label = tk.Label(self.popup, text="Click the heart to start filling the next 2 location")
        label.pack(padx=20, pady=10)
        done_button = tk.Button(self.popup, text="Done", command=self.close_popup)
        done_button.pack(pady=(0, 10))
        self.popup.transient(self.root)
        self.popup.grab_set()  # Make the popup modal

    def close_popup(self):
        """Close the popup and allow the user to fill the next two inputs."""
        self.waiting_for_done = False
        self.popup.destroy()

    # -------------------------------
    #       Restart Logic
    # -------------------------------
    def restart_program(self):
        """Restarts the program by re-executing the script."""
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # -------------------------------
    #       Stop Simulation Logic
    # -------------------------------
    def stop_simulation(self):
        """Stops the current simulation."""
        self.is_running = False
        self.info_label.config(text="Simulation stopped.")

    def handle_escape_key(self, event):
        """Callback to handle pressing the ESC key to stop simulation."""
        self.stop_simulation()

    # -------------------------------
    #     Simulation Logic
    # -------------------------------
    def simulate_clicks(self, first_rect, top_left, bottom_right):
        """
        Simulates clicking at random positions within two defined rectangles.
        The first rectangle is used for the Heart area action,
        and the second rectangle is used for the 'Sent Like' action.
        """
        self.is_running = True

        first_x1, first_y1 = first_rect[0]
        first_x2, first_y2 = first_rect[1]
        x1, y1 = top_left
        x2, y2 = bottom_right

        for i in range(self.total_rotations):
            if not self.is_running:
                break

            # Choose a random point within the Heart area
            frx = random.randint(min(first_x1, first_x2), max(first_x1, first_x2))
            fry = random.randint(min(first_y1, first_y2), max(first_y1, first_y2))
            self.mouse_controller.position = (frx, fry)
            self.mouse_controller.click(Button.left, 1)
            print(f"Rotation {i+1}: Clicked at random point in Heart area ({frx}, {fry})")
            time.sleep(1)  # pause between actions

            # Choose a random point within the 'Sent Like' area
            rx = random.randint(min(x1, x2), max(x1, x2))
            ry = random.randint(min(y1, y2), max(y1, y2))
            self.mouse_controller.position = (rx, ry)
            self.mouse_controller.click(Button.left, 1)
            print(f"Rotation {i+1}: Clicked at random point in 'Sent Like' area ({rx}, {ry})")
            time.sleep(1)  # pause between actions

        if self.is_running:
            msg = f"Done performing {self.total_rotations} rotations!"
        else:
            msg = "Simulation stopped before completion."

        self.info_label.config(text=msg)
        print(msg)

    # -------------------------------
    #        Start Simulation
    # -------------------------------
    def on_start(self):
        """
        Parse user inputs for rotations and coordinates, then start the simulation
        in a separate thread so that the GUI remains responsive (allowing ESC to stop simulation).
        """
        # 1) Parse the number of rotations from user input
        try:
            text = self.entry_rotations.get().strip()
            if text:
                self.total_rotations = int(text)
            if self.total_rotations <= 0:
                raise ValueError("Rotations must be > 0")
        except ValueError:
            self.info_label.config(text="Invalid rotation value. Enter a positive integer.")
            return

        # 2) Function to parse coordinates from a string input
        def parse_coords(s):
            parts = s.replace("(", "").replace(")", "").split(",")
            if len(parts) != 2:
                raise ValueError("Must have 2 numbers, e.g. '100, 200'.")
            return (int(float(parts[0].strip())), int(float(parts[1].strip())))

        try:
            first_rect_tl = parse_coords(self.entry_first_rect_tl.get().strip())
            first_rect_br = parse_coords(self.entry_first_rect_br.get().strip())
            first_rect = (first_rect_tl, first_rect_br)
            top_left = parse_coords(self.entry_tl.get().strip())
            bottom_right = parse_coords(self.entry_br.get().strip())
        except ValueError as e:
            self.info_label.config(text=f"Coordinate error: {e}")
            return

        # 3) Inform user and start the simulation in a new thread
        self.info_label.config(text=f"Running simulation for {self.total_rotations} rotations...")
        simulation_thread = threading.Thread(target=self.simulate_clicks, args=(first_rect, top_left, bottom_right))
        simulation_thread.daemon = True  # ensures thread exits when the main program exits
        simulation_thread.start()

    def __del__(self):
        """Stop the listener on object destruction."""
        self.listener.stop()

def main():
    root = tk.Tk()
    app = DesktopClickSelectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()