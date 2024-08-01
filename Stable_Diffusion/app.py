import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token

import torch
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Corrected CTkEntry without unsupported arguments
prompt = ctk.CTkEntry(app, height=40, width=512, fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Set device to CPU
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float32, use_auth_token=auth_token)
pipe.to(device)

def generate():
    image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

trigger = ctk.CTkButton(app, height=40, width=120, fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
