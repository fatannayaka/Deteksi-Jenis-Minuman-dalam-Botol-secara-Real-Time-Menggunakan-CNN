import cv2
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
from ultralytics import YOLO

# ================= KONFIGURASI =================
# GANTI PATH INI SESUAI LETAK best.pt ANDA!!!
MODEL_PATH = "best.pt"

# Daftar harga
HARGA_DB = {
    'ISOPLUS': 5000,
    'GOLDA': 6000,
    'MILKU': 4000,
    'FRUIT TEA': 7000,
    'FLORIDINA': 3000
}

# --- PALET WARNA NEON (Format BGR untuk OpenCV) ---
# Kuning, Cyan, Magenta, Hijau Neon, Oranye Terang
NEON_COLORS = [
    (0, 255, 255),  # Kuning
    (255, 255, 0),  # Cyan / Biru Langit
    (255, 0, 255),  # Magenta / Ungu Pink
    (50, 255, 50),  # Hijau Neon
    (0, 165, 255)   # Oranye
]

# Skema Warna GUI
COLOR_BG = "#0f0f1a"
COLOR_PANEL = "#353b6e"
COLOR_TEXT = "#ffffff"

class KasirApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Sistem Kasir Minuman Cerdas (CNN/YOLO)")
        self.root.configure(bg=COLOR_BG)
        self.root.geometry("1280x720")
        
        # Load Model
        print("Loading Model...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

        # Inisialisasi Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Webcam tidak terdeteksi")
            exit()

        self.setup_ui()
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.update_video()

    def setup_ui(self):
        # HEADER
        self.lbl_header = tk.Label(self.root, 
                                   text="Deteksi Jenis Minuman & Informasi Harga Real-Time (CNN)",
                                   bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 14, "bold"), pady=15)
        self.lbl_header.pack(side=tk.TOP, fill=tk.X)

        # CONTAINER
        self.main_container = tk.Frame(self.root, bg=COLOR_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # LEFT FRAME (VIDEO)
        self.left_frame = tk.Frame(self.main_container, bg=COLOR_PANEL, bd=2, relief="flat")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        # Mencegah frame membesar otomatis (Bug Fix GUI bergerak)
        self.left_frame.pack_propagate(False) 
        
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # RIGHT FRAME (INFO)
        self.right_frame = tk.Frame(self.main_container, bg=COLOR_BG, width=350)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_frame.pack_propagate(False) # Agar lebar tetap konsisten
        
        def create_info_box(parent, title):
            frame = tk.Frame(parent, bg=COLOR_PANEL, pady=15, padx=10)
            frame.pack(fill=tk.X, pady=(0, 15)) 
            lbl_title = tk.Label(frame, text=title, bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Arial", 14, "bold"))
            lbl_title.pack()
            lbl_value = tk.Label(frame, text="-", bg=COLOR_PANEL, fg="#00ffff", font=("Arial", 22, "bold")) # Value warna Cyan
            lbl_value.pack(pady=(5, 0))
            return lbl_value

        self.lbl_minuman = create_info_box(self.right_frame, "MINUMAN TERDETEKSI")
        self.lbl_harga = create_info_box(self.right_frame, "HARGA")
        self.lbl_conf = create_info_box(self.right_frame, "CONFIDENCE LEVEL")

        # Footer
        lbl_footer = tk.Label(self.root, text="Tekan ESC untuk keluar", bg=COLOR_BG, fg="gray",
                              font=("Arial", 10, "italic"), pady=10)
        lbl_footer.pack(side=tk.BOTTOM)

    def draw_corner_rect(self, img, x1, y1, x2, y2, color, thickness=3, length=25):
        """
        Fungsi untuk menggambar sudut-sudut kotak saja (agar tidak terlihat penuh/sumpek)
        """
        # Sudut Kiri Atas
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        
        # Sudut Kanan Atas
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

        # Sudut Kiri Bawah
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

        # Sudut Kanan Bawah
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # === MODIFIKASI: BOX RAPAT & AKURAT ===
            # imgsz=640: Memaksa deteksi sesuai resolusi training (kotak jadi pas)
            # conf=0.65: Hanya menampilkan deteksi yang sangat yakin (mengurangi kotak ganda)
            results = self.model(frame, conf=0.65, imgsz=640, verbose=False)[0]
            
            best_det = None
            highest_conf = 0

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])
                price = HARGA_DB.get(label, 0)
                
                # --- PEMILIHAN WARNA ---
                color = NEON_COLORS[cls % len(NEON_COLORS)]

                # --- GAMBAR CUSTOM BOX (Gaya Minimalis) ---
                self.draw_corner_rect(frame, x1, y1, x2, y2, color, thickness=3, length=30)
                
                # Label Background
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text + 10, y1), color, -1) 
                
                # Teks Nama Minuman
                cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Logika Update Panel Kanan
                if conf > highest_conf:
                    highest_conf = conf
                    best_det = {
                        "name": label,
                        "price": price,
                        "conf": conf
                    }

            # Update GUI Panel Kanan
            if best_det:
                self.lbl_minuman.config(text=best_det['name'], fg="#00ff00") 
                self.lbl_harga.config(text=f"Rp {best_det['price']:,}")
                self.lbl_conf.config(text=f"{best_det['conf']:.1%}")
            else:
                self.lbl_minuman.config(text="-", fg=COLOR_TEXT)
                self.lbl_harga.config(text="Rp 0")
                self.lbl_conf.config(text="0%")

            # Convert ke Tkinter (Memakai PIL LANCZOS sesuai request resolusi tinggi)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            w_canvas = self.left_frame.winfo_width()
            h_canvas = self.left_frame.winfo_height()
            
            if w_canvas > 10 and h_canvas > 10:
                # Menggunakan LANCZOS (Kualitas Tertinggi)
                # Dikurangi 4 pixel agar tidak memicu bug window membesar
                img.thumbnail((w_canvas - 4, h_canvas - 4), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = KasirApp(root, MODEL_PATH)
    root.mainloop()