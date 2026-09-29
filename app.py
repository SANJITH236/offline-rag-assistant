import os
import sys

# Windowed .exe has no console: give libraries a harmless place to print
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import multiprocessing
multiprocessing.freeze_support()

import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

import rag_core

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

NO_DOCS = "(no documents)"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Offline RAG Assistant")
        self.geometry("1000x680")
        self.minsize(800, 520)
        self.busy = False
        self._removing = ""

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- Sidebar ----------
        side = ctk.CTkFrame(self, width=240, corner_radius=0)
        side.grid(row=0, column=0, rowspan=2, sticky="nsew")
        side.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(side, text="Offline RAG",
                     font=ctk.CTkFont(size=22, weight="bold")
                     ).grid(row=0, column=0, padx=20, pady=(20, 10))

        self.add_btn = ctk.CTkButton(side, text="+ Add files",
                                     command=self.add_files)
        self.add_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.docs_box = ctk.CTkTextbox(side, width=200, wrap="word",
                                       state="disabled")
        self.docs_box.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        self.doc_menu = ctk.CTkOptionMenu(side, values=[NO_DOCS])
        self.doc_menu.grid(row=3, column=0, padx=20, pady=(0, 6), sticky="ew")

        self.remove_btn = ctk.CTkButton(side, text="Remove selected file",
                                        fg_color="#8b2e2e", hover_color="#a33a3a",
                                        command=self.remove_file)
        self.remove_btn.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.status = ctk.CTkLabel(side, text="Starting...", anchor="w",
                                   justify="left", wraplength=200)
        self.status.grid(row=5, column=0, padx=20, pady=(5, 20), sticky="ew")

        # ---------- Chat area ----------
        self.chat = ctk.CTkTextbox(self, wrap="word", state="disabled",
                                   font=ctk.CTkFont(size=14))
        self.chat.grid(row=0, column=1, padx=15, pady=(15, 5), sticky="nsew")
        self.chat.tag_config("user", foreground="#7cc4ff")
        self.chat.tag_config("src", foreground="#9aa0a6")

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=1, column=1, padx=15, pady=(5, 15), sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(bar, height=40,
                                  placeholder_text="Ask a question about your documents...")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send())

        self.send_btn = ctk.CTkButton(bar, text="Send", width=90, height=40,
                                      command=self.send)
        self.send_btn.grid(row=0, column=1)

        self.log("Assistant", "Add your documents with '+ Add files', then ask me anything about them.")
        self.set_busy(True, "Loading models...")
        self.run_bg(self._startup, self._startup_done)

    # ---------- Helpers ----------
    def run_bg(self, work, done):
        def target():
            try:
                result, err = work(), None
            except Exception as e:
                result, err = None, e
            self.after(0, done, result, err)
        threading.Thread(target=target, daemon=True).start()

    def set_busy(self, busy, text=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.send_btn.configure(state=state)
        self.add_btn.configure(state=state)
        self.remove_btn.configure(state=state)
        self.doc_menu.configure(state=state)
        if text is not None:
            self.status.configure(text=text)

    def log(self, who, text, sources=None):
        self.chat.configure(state="normal")
        tag = "user" if who == "You" else None
        self.chat.insert("end", f"{who}: ", tag)
        self.chat.insert("end", text + "\n")
        if sources:
            self.chat.insert("end", "Sources: " + "; ".join(sources) + "\n", "src")
        self.chat.insert("end", "\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def refresh_docs(self, names):
        self.docs_box.configure(state="normal")
        self.docs_box.delete("1.0", "end")
        self.docs_box.insert("end", "Indexed documents:\n\n")
        for n in names:
            self.docs_box.insert("end", f"• {n}\n")
        self.docs_box.configure(state="disabled")

        values = list(names) if names else [NO_DOCS]
        self.doc_menu.configure(values=values)
        self.doc_menu.set(values[0])

    # ---------- Startup ----------
    def _startup(self):
        rag_core._get_embedder()
        docs = rag_core.list_documents()
        try:
            import ollama
            ollama.list()
            ollama_ok = True
            rag_core.warmup()
        except Exception:
            ollama_ok = False
        return docs, ollama_ok

    def _startup_done(self, result, err):
        if err:
            self.set_busy(False, f"Startup error: {err}")
            return
        docs, ollama_ok = result
        self.refresh_docs(docs)
        if ollama_ok:
            self.set_busy(False, "Ready (offline)")
        else:
            self.set_busy(False, "Ollama not running. Open Ollama from the Start menu.")

    # ---------- Add files ----------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[("Documents", "*.pdf *.docx *.txt *.csv")])
        if not paths:
            return
        self.set_busy(True, "Indexing...")

        def work():
            results = []
            for p in paths:
                try:
                    n = rag_core.ingest_file(p)
                    results.append((p, n, None))
                except Exception as e:
                    results.append((p, 0, e))
            return results, rag_core.list_documents()

        self.run_bg(work, self._add_done)

    def _add_done(self, result, err):
        if err:
            self.set_busy(False, f"Error: {err}")
            return
        results, docs = result
        for p, n, e in results:
            name = p.replace("\\", "/").split("/")[-1]
            if e:
                self.log("Assistant", f"Could not index {name}: {e}")
            elif n == 0:
                self.log("Assistant", f"{name} had no readable text (scanned PDF or empty file).")
            else:
                self.log("Assistant", f"Indexed {name} ({n} chunks).")
        self.refresh_docs(docs)
        self.set_busy(False, "Ready (offline)")

    # ---------- Remove file ----------
    def remove_file(self):
        name = self.doc_menu.get()
        if not name or name == NO_DOCS:
            return
        if not messagebox.askyesno("Remove file",
                                   f"Remove '{name}' from the index?\n\n"
                                   "The original file on your disk is not deleted."):
            return
        self._removing = name
        self.set_busy(True, "Removing...")

        def work():
            n = rag_core.remove_document(name)
            return n, rag_core.list_documents()

        self.run_bg(work, self._remove_done)

    def _remove_done(self, result, err):
        if err:
            self.log("Assistant", f"Error: {err}")
        else:
            n, docs = result
            if n:
                self.log("Assistant", f"Removed {self._removing} from the index.")
            else:
                self.log("Assistant", f"No indexed file named '{self._removing}' was found.")
            self.refresh_docs(docs)
        self.set_busy(False, "Ready (offline)")

    # ---------- Ask ----------
    def send(self):
        q = self.entry.get().strip()
        if not q or self.busy:
            return
        self.entry.delete(0, "end")
        self.log("You", q)
        self.set_busy(True, "Thinking...")
        self.run_bg(lambda: rag_core.ask(q), self._ask_done)

    def _ask_done(self, result, err):
        if err:
            self.log("Assistant", f"Error: {err}")
        else:
            answer, sources = result
            self.log("Assistant", answer, sources)
        self.set_busy(False, "Ready (offline)")


if __name__ == "__main__":
    App().mainloop()