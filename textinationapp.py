import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline
import torch

class TextToImageGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Textination")

        # Create Stable Diffusion Pipeline 
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, revision="fp16")
        self.pipeline.to(self.device)

        # GUI Elements
        self.text_entry_label = ttk.Label(root, text="Enter Text:")
        self.text_entry_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.text_entry = ttk.Entry(root, width=50)
        self.text_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        self.generate_button = ttk.Button(root, text="Generate Image", command=self.generate_image)
        self.generate_button.grid(row=0, column=3, padx=5, pady=5)

        self.image_label = ttk.Label(root)
        self.image_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

        # Parameter Adjustment
        self.param_label = ttk.Label(root, text="Guidance Scale:")
        self.param_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.param_scale = ttk.Scale(root, from_=1, to=10, orient="horizontal", length=200)
        self.param_scale.set(5)
        self.param_scale.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

    def generate_image(self):
        text = self.text_entry.get()
        guidance_scale = self.param_scale.get()

        if text:
            with torch.autocast(self.device):
                image = self.pipeline(text, guidance_scale=guidance_scale)["sample"][0]

            # Convert PIL image to Tkinter PhotoImage
            image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=image)
            self.image_label.image = image  # Keep reference to avoid garbage collection
        else:
            print("Please enter some text.")

def main():
    root = tk.Tk() 
    app = TextToImageGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
