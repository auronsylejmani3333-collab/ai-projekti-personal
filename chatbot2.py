from pathlib import Path
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np
from PIL import Image, ImageOps, ImageTk
from tensorflow.keras.models import load_model
from groq import Groq
from dotenv import load_dotenv


# =====================================================
#                  CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FOLDER = BASE_DIR / "model"
MODEL_PATH = MODEL_FOLDER / "keras_model.h5"
LABELS_PATH = MODEL_FOLDER / "labels.txt"

IMAGE_SIZE = (224, 224)
PREVIEW_SIZE = (380, 270)

APP_TITLE = "AI Vehicle Classifier"

# =====================================================
#                  COLORS
# =====================================================

BG = "#101827"
HEADER = "#162238"
PANEL = "#1E293B"
BLUE = "#38BDF8"
GREEN = "#4ADE80"
WHITE = "#F8FAFC"
GRAY = "#94A3B8"
INPUT_BG = "#263750"


# =====================================================
#                  GROQ CONFIGURATION
# =====================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "openai/gpt-oss-20b"

client = None

if API_KEY:
    client = Groq(api_key=API_KEY)


SYSTEM_MESSAGE = (
    "You are a friendly AI assistant. "
    "Answer in the language used by the user."
)


# =====================================================
#                  LOAD LABELS
# =====================================================

def load_labels():

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:

        for line in file:

            label = line.strip()

            if not label:
                continue

            parts = label.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(label)

    return labels


# =====================================================
#                  LOAD AI MODEL
# =====================================================

try:

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    class_names = load_labels()

    MODEL_READY = True

except Exception as error:

    model = None
    class_names = []
    MODEL_READY = False

    print("MODEL ERROR:", error)


# =====================================================
#                  IMAGE PREPARATION
# =====================================================

def prepare_image(image_path):

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    image = Image.open(image_path).convert("RGB")

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data[0] = normalized_image_array

    return data


# =====================================================
#                  VEHICLE PREDICTION
# =====================================================

def predict_vehicle(image_path):

    if not MODEL_READY:
        raise Exception("AI model could not be loaded.")

    data = prepare_image(image_path)

    prediction = model.predict(
        data,
        verbose=0
    )

    index = int(np.argmax(prediction[0]))

    vehicle_class = class_names[index]

    confidence = float(prediction[0][index])

    return {
        "vehicle_class": vehicle_class,
        "confidence": confidence
    }


# =====================================================
#                  MAIN APPLICATION
# =====================================================

class VehicleApp:

    def __init__(self, root):

        self.root = root

        self.root.title(APP_TITLE)
        self.root.geometry("750x850")
        self.root.minsize(650, 750)

        self.root.configure(bg=BG)

        self.preview_image = None

        # =============================================
        # HEADER
        # =============================================

        header = tk.Frame(
            root,
            bg=HEADER,
            height=90
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="✦",
            font=("Arial", 30, "bold"),
            bg=HEADER,
            fg=BLUE
        ).pack(
            side="left",
            padx=(25, 12)
        )

        header_info = tk.Frame(
            header,
            bg=HEADER
        )

        header_info.pack(
            side="left",
            pady=18
        )

        tk.Label(
            header_info,
            text="AI VEHICLE CLASSIFIER",
            font=("Segoe UI", 16, "bold"),
            bg=HEADER,
            fg=WHITE
        ).pack(anchor="w")

        tk.Label(
            header_info,
            text="Powered by Artificial Intelligence",
            font=("Segoe UI", 9),
            bg=HEADER,
            fg=GRAY
        ).pack(anchor="w")

        # AI ASSISTANT BUTTON

        assistant_button = tk.Button(
            header,
            text="✦ AI ASSISTANT",
            command=self.open_assistant,
            font=("Segoe UI", 10, "bold"),
            bg="#0284C7",
            fg=WHITE,
            activebackground="#0369A1",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=12
        )

        assistant_button.pack(
            side="right",
            padx=20
        )

        # =============================================
        # MAIN CONTENT
        # =============================================

        main_frame = tk.Frame(
            root,
            bg=BG
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=15
        )

        # TITLE

        tk.Label(
            main_frame,
            text="VEHICLE RECOGNITION",
            font=("Segoe UI", 23, "bold"),
            bg=BG,
            fg=BLUE
        ).pack(pady=(10, 5))

        tk.Label(
            main_frame,
            text="Upload an image and let AI identify the vehicle.",
            font=("Segoe UI", 10),
            bg=BG,
            fg=GRAY
        ).pack(pady=(0, 20))

        # =============================================
        # IMAGE CONTAINER
        # =============================================

        self.image_frame = tk.Frame(
            main_frame,
            bg=PANEL,
            highlightbackground=BLUE,
            highlightthickness=2,
            width=430,
            height=300
        )

        self.image_frame.pack(pady=10)

        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="Upload a Vehicle Image",
            font=("Segoe UI", 14, "bold"),
            bg=PANEL,
            fg=GRAY
        )

        self.image_label.pack(expand=True)

        # =============================================
        # UPLOAD BUTTON
        # =============================================

        self.choose_button = tk.Button(
            main_frame,
            text="📁  CHOOSE VEHICLE IMAGE",
            command=self.choose_image,
            font=("Segoe UI", 11, "bold"),
            bg="#0284C7",
            fg=WHITE,
            activebackground="#0369A1",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=25,
            pady=13,
            borderwidth=0
        )

        self.choose_button.pack(pady=20)

        # =============================================
        # RESULTS
        # =============================================

        self.result_frame = tk.Frame(
            main_frame,
            bg=PANEL,
            highlightbackground="#334155",
            highlightthickness=1,
            width=520,
            height=125
        )

        self.result_frame.pack(pady=5)

        self.result_frame.pack_propagate(False)

        self.result_label = tk.Label(
            self.result_frame,
            text="Vehicle result will appear here",
            font=("Segoe UI", 17, "bold"),
            bg=PANEL,
            fg=WHITE
        )

        self.result_label.pack(pady=(18, 7))

        self.confidence_label = tk.Label(
            self.result_frame,
            text="Confidence: -",
            font=("Segoe UI", 12),
            bg=PANEL,
            fg=BLUE
        )

        self.confidence_label.pack()

        # =============================================
        # IMAGE PATH
        # =============================================

        self.path_label = tk.Label(
            main_frame,
            text="No image selected",
            font=("Segoe UI", 9),
            wraplength=550,
            bg=BG,
            fg=GRAY
        )

        self.path_label.pack(pady=15)

        # =============================================
        # STATUS
        # =============================================

        status_text = (
            "AI Model Ready"
            if MODEL_READY
            else "AI Model Not Loaded"
        )

        status_color = GREEN if MODEL_READY else "#F87171"

        self.status_label = tk.Label(
            main_frame,
            text="● " + status_text,
            font=("Segoe UI", 10, "bold"),
            bg=BG,
            fg=status_color
        )

        self.status_label.pack(pady=5)

        # =============================================
        # FOOTER
        # =============================================

        tk.Label(
            root,
            text="AI VEHICLE CLASSIFIER  •  POWERED BY AI",
            font=("Segoe UI", 8, "bold"),
            bg=BG,
            fg="#475569"
        ).pack(side="bottom", pady=15)

    # =================================================
    #                  CHOOSE IMAGE
    # =================================================

    def choose_image(self):

        image_path = filedialog.askopenfilename(
            title="Choose Vehicle Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not image_path:
            return

        try:
            self.show_preview(image_path)
            self.predict(image_path)

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                str(error)
            )

    # =================================================
    #                  SHOW PREVIEW
    # =================================================

    def show_preview(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image.thumbnail(PREVIEW_SIZE)

        self.preview_image = ImageTk.PhotoImage(image)

        self.image_label.configure(
            image=self.preview_image,
            text=""
        )

        self.path_label.configure(
            text=str(Path(image_path))
        )

    # =================================================
    #                  PREDICT
    # =================================================

    def predict(self, image_path):

        try:

            result = predict_vehicle(image_path)

            self.result_label.configure(
                text=f"Vehicle: {result['vehicle_class']}"
            )

            self.confidence_label.configure(
                text=f"Confidence: {result['confidence']:.2%}"
            )

            self.status_label.configure(
                text="● Prediction completed",
                fg=GREEN
            )

        except Exception as error:

            messagebox.showerror(
                "Prediction Error",
                str(error)
            )


    # =================================================
    #                  AI ASSISTANT
    # =================================================

    def open_assistant(self):

        if not client:

            messagebox.showerror(
                "API Key Error",
                "GROQ_API_KEY was not found!\n\n"
                "Please check your .env file."
            )

            return

        assistant = tk.Toplevel(self.root)

        assistant.title("AI Assistant")
        assistant.geometry("850x700")
        assistant.minsize(600, 500)
        assistant.configure(bg=BG)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            }
        ]

        chat_busy = {"value": False}

        # =============================================
        # HEADER
        # =============================================

        header = tk.Frame(
            assistant,
            bg=HEADER,
            height=75
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="✦ AI ASSISTANT",
            font=("Segoe UI", 18, "bold"),
            bg=HEADER,
            fg=BLUE
        ).pack(
            side="left",
            padx=20,
            pady=20
        )

        # =============================================
        # CHAT DISPLAY
        # =============================================

        chat_display = scrolledtext.ScrolledText(
            assistant,
            bg=BG,
            fg=WHITE,
            font=("Arial", 11),
            wrap=tk.WORD,
            relief="flat",
            padx=15,
            pady=15,
            state=tk.DISABLED,
            insertbackground=WHITE
        )

        chat_display.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        chat_display.tag_configure(
            "user",
            foreground=BLUE,
            font=("Arial", 10, "bold"),
            spacing1=12
        )

        chat_display.tag_configure(
            "assistant",
            foreground=GREEN,
            font=("Arial", 10, "bold"),
            spacing1=12
        )

        chat_display.tag_configure(
            "message",
            foreground=WHITE,
            font=("Arial", 11),
            spacing3=10
        )

        chat_display.tag_configure(
            "error",
            foreground="#F87171",
            font=("Arial", 10)
        )

        # =============================================
        # WELCOME MESSAGE
        # =============================================

        def add_message(sender, message):

            chat_display.config(state=tk.NORMAL)

            if sender == "YOU":

                chat_display.insert(
                    tk.END,
                    "YOU\n",
                    "user"
                )

            elif sender == "AI":

                chat_display.insert(
                    tk.END,
                    "AI ASSISTANT\n",
                    "assistant"
                )

            else:

                chat_display.insert(
                    tk.END,
                    "ERROR\n",
                    "error"
                )

            tag = "error" if sender == "ERROR" else "message"

            chat_display.insert(
                tk.END,
                message + "\n\n",
                tag
            )

            chat_display.see(tk.END)

            chat_display.config(state=tk.DISABLED)

        add_message(
            "AI",
            "Hello! I am your AI Assistant. How can I help you today?"
        )

        # =============================================
        # STATUS
        # =============================================

        status = tk.Label(
            assistant,
            text="● Online and ready",
            font=("Arial", 9),
            bg=BG,
            fg=GREEN
        )

        status.pack(anchor="w", padx=20)

        # =============================================
        # INPUT FRAME
        # =============================================

        input_frame = tk.Frame(
            assistant,
            bg=PANEL
        )

        input_frame.pack(
            fill="x",
            padx=15,
            pady=15
        )

        message_entry = tk.Entry(
            input_frame,
            font=("Arial", 13),
            bg=INPUT_BG,
            fg=WHITE,
            insertbackground=BLUE,
            relief="flat"
        )

        message_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=12,
            pady=15,
            ipady=8
        )

        # =============================================
        # NEW CHAT
        # =============================================

        def new_chat():

            if chat_busy["value"]:
                return

            messages.clear()

            messages.append({
                "role": "system",
                "content": SYSTEM_MESSAGE
            })

            chat_display.config(state=tk.NORMAL)

            chat_display.delete("1.0", tk.END)

            chat_display.config(state=tk.DISABLED)

            add_message(
                "AI",
                "New conversation started! How can I help?"
            )

            status.config(
                text="● Online and ready",
                fg=GREEN
            )

            message_entry.focus_set()

        new_chat_button = tk.Button(
            header,
            text="+ NEW CHAT",
            command=new_chat,
            font=("Arial", 9, "bold"),
            bg=INPUT_BG,
            fg=WHITE,
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=9
        )

        new_chat_button.pack(
            side="right",
            padx=20
        )

        # =============================================
        # SEND MESSAGE
        # =============================================

        def send_message(event=None):

            if chat_busy["value"]:
                return "break"

            user_text = message_entry.get().strip()

            if not user_text:
                return "break"

            message_entry.delete(0, tk.END)

            add_message("YOU", user_text)

            messages.append({
                "role": "user",
                "content": user_text
            })

            chat_busy["value"] = True

            send_button.config(
                state=tk.DISABLED,
                text="WAIT..."
            )

            status.config(
                text="● AI is thinking...",
                fg="#FACC15"
            )

            conversation = messages.copy()

            threading.Thread(
                target=ask_groq,
                args=(conversation,),
                daemon=True
            ).start()

            return "break"

        # =============================================
        # GROQ REQUEST
        # =============================================

        def ask_groq(conversation):

            try:

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation
                )

                answer = response.choices[0].message.content

                assistant.after(
                    0,
                    show_answer,
                    answer,
                    None
                )

            except Exception as error:

                assistant.after(
                    0,
                    show_answer,
                    None,
                    str(error)
                )

        # =============================================
        # SHOW ANSWER
        # =============================================

        def show_answer(answer, error):

            if not assistant.winfo_exists():
                return

            if error:

                add_message("ERROR", error)

                if messages and messages[-1]["role"] == "user":
                    messages.pop()

                status.config(
                    text="● Something went wrong",
                    fg="#F87171"
                )

            else:

                add_message("AI", answer)

                messages.append({
                    "role": "assistant",
                    "content": answer
                })

                status.config(
                    text="● Online and ready",
                    fg=GREEN
                )

            chat_busy["value"] = False

            send_button.config(
                state=tk.NORMAL,
                text="➤ SEND"
            )

            message_entry.focus_set()

        # =============================================
        # SEND BUTTON
        # =============================================

        send_button = tk.Button(
            input_frame,
            text="➤ SEND",
            command=send_message,
            font=("Arial", 10, "bold"),
            bg="#0284C7",
            fg=WHITE,
            activebackground="#0369A1",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=12
        )

        send_button.pack(
            side="right",
            padx=10,
            pady=8
        )

        message_entry.bind(
            "<Return>",
            send_message
        )

        message_entry.focus_set()


# =====================================================
#                  START APPLICATION
# =====================================================

def main():

    root = tk.Tk()

    app = VehicleApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
