from pynput.mouse import Listener, Controller, Button
import tkinter as tk
import random
import time
import re  # for parsing the position string
import os
import sys

class DesktopClickSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Desktop Click Selector - with Restart, Stop & Copy")

        # Mouse
        self.mouse_controller = Controller()

        # Default rotations if user leaves blank
        self.total_rotations = 20
        self.is_running = False

        # ========== GUI Layout ==========

        # (A) Current Mouse Position Label
        self.current_pos_label = tk.Label(
            self.root, text="Current Mouse Position: (0, 0)"
        )
        self.current_pos_label.pack()

        # (B) Buttons to Restart, Stop & Copy current mouse position
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        self.restart_button = tk.Button(
            button_frame, text="Restart Program", command=self.restart_program
        )
        self.restart_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = tk.Button(
            button_frame, text="Stop Simulation", command=self.stop_simulation
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.copy_button = tk.Button(
            button_frame, text="Copy Current Position", command=self.copy_current_position
        )
        self.copy_button.grid(row=0, column=2, padx=5, pady=5)

        # (C) First Rectangle Area
        self.label_first_rect = tk.Label(self.root, text="First Rectangle Top-Left:")
        self.label_first_rect.pack()
        self.entry_first_rect_tl = tk.Entry(self.root)
        self.entry_first_rect_tl.pack()

        self.label_first_rect_br = tk.Label(self.root, text="First Rectangle Bottom-Right:")
        self.label_first_rect_br.pack()
        self.entry_first_rect_br = tk.Entry(self.root)
        self.entry_first_rect_br.pack()

        # (D) Rectangle top-left
        self.label_tl = tk.Label(self.root, text="Rectangle Top-Left:")
        self.label_tl.pack()
        self.entry_tl = tk.Entry(self.root)
        self.entry_tl.pack()

        # (E) Rectangle bottom-right
        self.label_br = tk.Label(self.root, text="Rectangle Bottom-Right:")
        self.label_br.pack()
        self.entry_br = tk.Entry(self.root)
        self.entry_br.pack()

        # (F) Number of Rotations
        self.label_rotations = tk.Label(self.root, text="Number of Rotations:")
        self.label_rotations.pack()
        self.entry_rotations = tk.Entry(self.root)
        self.entry_rotations.pack()

        # (G) Start Button
        self.start_button = tk.Button(self.root, text="Start", command=self.on_start)
        self.start_button.pack()

        # (H) Info Label
        self.info_label = tk.Label(self.root, text="")
        self.info_label.pack()

        # Start a mouse listener
        self.listener = Listener(on_move=self.on_move, on_click=self.on_click)
        self.listener.start()

    # -------------------------------
    #    Mouse Listener Callbacks
    # -------------------------------
    def on_move(self, x, y):
        """Continuously update the 'Current Mouse Position' label."""
        self.current_pos_label.config(
            text=f"Current Mouse Position: ({x}, {y})"
        )

    def on_click(self, x, y, button, pressed):
        """Provides feedback on clicks."""
        if not pressed and button == Button.left:
            self.info_label.config(
                text=f"Clicked at position: ({x}, {y}). Enter coordinates manually."
            )

    # -------------------------------
    #       Restart Logic
    # -------------------------------
    def restart_program(self):
        """Restarts the program by re-executing the script."""
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # -------------------------------
    #       Stop Logic
    # -------------------------------
    def stop_simulation(self):
        """Stops the current simulation."""
        self.is_running = False
        self.info_label.config(text="Simulation stopped.")

    # -------------------------------
    #       Copy Logic
    # -------------------------------
    def copy_current_position(self):
        """Copies the numeric portion of 'Current Mouse Position: (x, y)' to the clipboard."""
        label_text = self.current_pos_label.cget("text")
        match = re.search(r"\(([^)]+)\)", label_text)
        coords_str = match.group(1) if match else "0, 0"

        # Put it on the clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(coords_str)
        self.info_label.config(text=f"Copied '{coords_str}' to clipboard.")

    # -------------------------------
    #     Simulation Logic
    # -------------------------------
    def simulate_clicks(self, first_rect, top_left, bottom_right):
        """Simulate clicks at specified locations."""
        self.is_running = True

        first_x1, first_y1 = first_rect[0]
        first_x2, first_y2 = first_rect[1]
        x1, y1 = top_left
        x2, y2 = bottom_right

        for i in range(self.total_rotations):
            if not self.is_running:
                break

            # Random point in the first rectangle
            frx = random.randint(min(first_x1, first_x2), max(first_x1, first_x2))
            fry = random.randint(min(first_y1, first_y1), max(first_y1, first_y2))
            self.mouse_controller.position = (frx, fry)
            self.mouse_controller.click(Button.left, 1)
            print(f"Rotation {i+1}: Clicked at random point in first rectangle ({frx}, {fry})")
            time.sleep(1)

            # Random point in the second rectangle
            rx = random.randint(min(x1, x2), max(x1, x2))
            ry = random.randint(min(y1, y2), max(y1, y2))
            self.mouse_controller.position = (rx, ry)
            self.mouse_controller.click(Button.left, 1)
            print(f"Rotation {i+1}: Clicked at random point in second rectangle ({rx}, {ry})")
            time.sleep(0.5)

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
        """Parse user inputs, then run the automation."""
        # 1) Rotations
        try:
            text = self.entry_rotations.get().strip()
            if text:
                self.total_rotations = int(text)
            if self.total_rotations <= 0:
                raise ValueError("Rotations must be > 0")
        except ValueError:
            self.info_label.config(text="Invalid rotation value. Enter a positive integer.")
            return

        # 2) Coordinates
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

        # 3) Run simulation
        self.info_label.config(text=f"Running simulation for {self.total_rotations} rotations...")
        self.simulate_clicks(first_rect, top_left, bottom_right)

    def __del__(self):
        """Stop the listener on destruction."""
        self.listener.stop()

def main():
    root = tk.Tk()
    app = DesktopClickSelectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# hinge
# 857.3984375, 478.08203125

# 885.578125, 570.20703125

# 709.265625, 647.6875

# 864.84375, 682.765625

# tinder
# 811.54296875, 698.94921875
