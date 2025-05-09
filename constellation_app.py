#!/usr/bin/env python3
"""
Constellation Recognition - Python Application
Based on the Jupyter Notebook constellation_UI.ipynb
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Frame
from PIL import Image, ExifTags, ImageTk
import time
try:
    from ultralytics import YOLO
except ImportError:
    print("YOLO not installed. Please install using: pip install ultralytics")

# Define the dataset path
DATASET_PATH = "constellation_dataset"

def load_model(model_path="best_constellation_model.pt"):
    """Load the YOLOv8 constellation recognition model"""
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def recognize_constellation(img, model):
    """Recognize constellation in an image using YOLOv8 model"""
    if model is None:
        return img, "Error: Model not loaded", None
    
    # Make a copy for annotation
    display_img = img.copy()
    
    # Run inference
    results = model(img)
    
    # Initialize variables for storing results
    recognized_constellation = "No constellation detected"
    confidence = 0
    
    # Process results
    if len(results) > 0:
        # Get the first result (assuming single image)
        result = results[0]
        
        # Check if any detections
        if len(result.boxes) > 0:
            # Get the detection with highest confidence
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            # Get the index of highest confidence detection
            best_idx = np.argmax(confidences)
            
            # Get the class name and confidence
            class_id = class_ids[best_idx]
            class_name = result.names[class_id]
            confidence = confidences[best_idx]
            
            # Set recognized constellation
            recognized_constellation = class_name
            
            # Draw bounding box on the image
            box = boxes[best_idx].astype(int)
            cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(display_img, f"{class_name} {confidence:.2f}", 
                      (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert the annotated image from BGR to RGB for display
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    
    return display_img_rgb, recognized_constellation, confidence

def preprocess_image(image_input, output_size=(640, 640)):
    """
    Preprocess an image by auto-orienting, resizing it to 640x640 (stretch), and converting it to grayscale.

    Args:
        image_input: Either a path to an image file (str) or an image array (numpy array)
        output_size (tuple): Desired output size (width, height).

    Returns:
        processed_image: The preprocessed grayscale image.
    """
    # Check if input is a path or an array
    if isinstance(image_input, str):
        # Load the image using PIL to handle EXIF orientation
        pil_image = Image.open(image_input)
        
        # Auto-orient the image based on EXIF metadata
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = pil_image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation, None)
                if orientation_value == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation_value == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation_value == 8:
                    pil_image = pil_image.rotate(90, expand=True)
        except Exception as e:
            print(f"Warning: Could not auto-orient image due to: {e}")
        
        # Convert PIL image to OpenCV format (numpy array)
        image = np.array(pil_image)
    else:
        # Input is already an image array
        image = image_input.copy()

    # If the image has an alpha channel, remove it
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Ensure image is in RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Resize the image to 640x640 (stretch)
        resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert the resized image to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale, just resize
        resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        grayscale_image = resized_image

    return grayscale_image

class ConstellationRecognitionApp:    
    def __init__(self, root):
        self.root = root
        self.root.title("Constellation Recognition")
        self.root.geometry("1200x800")
        self.root.minsize(900, 700)  # Set minimum window size
        
        # Configure ttk style for better checkbox visibility
        self.style = ttk.Style()
        self.style.configure("TCheckbutton", background="#f0f0f0", font=('Arial', 10))
        
        # Set up the model
        self.model = load_model()
        
        # Set up the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Constellation Recognition", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Step 1: Image Selection
        image_frame = ttk.LabelFrame(main_frame, text="Step 1: Select an Image", padding="10")
        image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        
        # File upload button
        self.upload_button = ttk.Button(image_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(fill=tk.X, pady=5)
        
        # Sample images section
        ttk.Label(image_frame, text="OR").pack(pady=5)
        
        # Sample dropdown
        self.sample_image_options = ["Select a sample image..."]
        sample_images_dir = os.path.join(DATASET_PATH, "test", "images")
        
        if os.path.exists(sample_images_dir):
            sample_files = [f for f in os.listdir(sample_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Include only first 20 samples to avoid huge dropdown
            self.sample_image_options.extend(sample_files[:20])
        
        self.sample_var = tk.StringVar()
        self.sample_var.set(self.sample_image_options[0])
        self.sample_dropdown = ttk.Combobox(image_frame, textvariable=self.sample_var, values=self.sample_image_options, state="readonly", width=40)
        self.sample_dropdown.pack(fill=tk.X, pady=5)
        self.sample_dropdown.bind("<<ComboboxSelected>>", self.on_sample_change)
        
        # Step 2: Options
        options_frame = ttk.LabelFrame(main_frame, text="Step 2: Options", padding="10")
        options_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nw")
        # Configure the frame to match the checkboxes
        options_frame.configure(style="Options.TLabelframe")
        self.style.configure("Options.TLabelframe", background="#f0f0f0")        # Preprocessing checkbox
        self.preprocess_var = tk.BooleanVar(value=False)
        # Use regular Tkinter checkbox for better visibility
        self.preprocess_checkbox = tk.Checkbutton(
            options_frame, 
            text="Apply preprocessing (auto-orient, resize to 640x640)",
            variable=self.preprocess_var,
            onvalue=True,
            offvalue=False,
            bg="#f0f0f0",
            font=("Arial", 10)
        )
        self.preprocess_checkbox.pack(anchor=tk.W, pady=5)
          # Add a description label for the preprocessing checkbox
        preprocessing_description = ttk.Label(
            options_frame,
            text="When checked, shows both original and preprocessed images side by side",
            font=("Arial", 8, "italic"),
            wraplength=250
        )
        preprocessing_description.pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))
          # Enhancement checkbox
        self.enhance_var = tk.BooleanVar(value=True)
        # Use regular Tkinter checkbox for better visibility
        self.enhance_checkbox = tk.Checkbutton(
            options_frame, 
            text="Enhance stars before recognition",
            variable=self.enhance_var,
            onvalue=True,
            offvalue=False,
            bg="#f0f0f0",
            font=("Arial", 10)
        )
        self.enhance_checkbox.pack(anchor=tk.W, pady=5)
          # Add a description label for the enhancement checkbox
        enhancement_description = ttk.Label(
            options_frame,
            text="Improves star visibility to help with detection.",
            font=("Arial", 8, "italic"),
            wraplength=250
        )
        enhancement_description.pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))
        
        # Step 3: Process
        process_frame = ttk.LabelFrame(main_frame, text="Step 3: Process", padding="10")
        process_frame.grid(row=1, column=2, padx=10, pady=10, sticky="nw")
        
        # Process button
        self.process_button = ttk.Button(process_frame, text="Recognize Constellation", command=self.process_image)
        self.process_button.pack(pady=10)
          # Create a separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Results section
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="nsew")
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        
        # Image display - using a frame to hold potentially two images side by side
        self.images_frame = ttk.Frame(results_frame)
        self.images_frame.pack(pady=10)
          # Original image display (left side)
        self.original_image_frame = ttk.LabelFrame(self.images_frame, text="Original Image")
        self.original_image_frame.pack(side=tk.LEFT, padx=5)
        self.original_image_display = ttk.Label(self.original_image_frame)
        self.original_image_display.pack(padx=5, pady=5)
        
        # Processed image display (right side)
        self.processed_image_frame = ttk.LabelFrame(self.images_frame, text="Processed Image with Detection")
        self.processed_image_frame.pack(side=tk.LEFT, padx=5)
        self.processed_image_display = ttk.Label(self.processed_image_frame)
        self.processed_image_display.pack(padx=5, pady=5)
        
        # For backward compatibility, keep image_display reference pointing to processed_image_display
        self.image_display = self.processed_image_display
        
        # Results text
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10, width=80)
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please select an image.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Variables to store image data
        self.current_image_path = None
        self.current_image = None
        self.original_img = None
        
        # Constellation information
        self.constellations_info = {
            'Aquarius': "The Water Bearer constellation, representing Ganymede, a handsome young man who was carried to Olympus by Zeus disguised as an eagle.",
            'Aries': "The Ram constellation, representing the ram with the Golden Fleece from Greek mythology.",
            'Cancer': "The Crab constellation, representing the crab that Hera sent to distract Heracles during his fight with the Hydra.",
            'Capricornus': "The Sea Goat constellation, representing Pan who transformed into a half-goat, half-fish when escaping the monster Typhon.",
            'Gemini': "The Twins constellation, representing Castor and Pollux, the twin sons of Leda and Zeus.",
            'Leo': "The Lion constellation, representing the Nemean Lion slain by Heracles as one of his twelve labors.",
            'Libra': "The Scales constellation, representing the scales of justice held by the goddess Astraea (Virgo).",
            'Pisces': "The Fish constellation, representing Aphrodite and Eros who transformed into fish to escape Typhon.",
            'Sagittarius': "The Archer constellation, representing a centaur, usually identified as Chiron, who was accidentally wounded by Heracles.",
            'Scorpius': "The Scorpion constellation, representing the scorpion that killed Orion the Hunter.",
            'Taurus': "The Bull constellation, representing Zeus when he took the form of a bull to seduce Europa.",
            'Virgo': "The Maiden constellation, representing several goddesses including Demeter, Persephone, and Astraea."
        }
    
    def upload_image(self):
        """Handle image upload from file system"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )
        
        if file_path:
            self.current_image_path = file_path
            self.sample_var.set(self.sample_image_options[0])  # Reset sample dropdown
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            
            # Preview the image
            self.preview_image(file_path)
    
    def on_sample_change(self, event):
        """Handle sample image selection"""
        selected = self.sample_var.get()
        
        if selected != self.sample_image_options[0]:
            sample_images_dir = os.path.join(DATASET_PATH, "test", "images")
            img_path = os.path.join(sample_images_dir, selected)
            
            self.current_image_path = img_path
            self.status_var.set(f"Selected sample image: {selected}")
              # Preview the image
            self.preview_image(img_path)
    
    def preview_image(self, img_path):
        """Display a preview of the selected image"""
        try:
            image = Image.open(img_path)
            
            # Resize image for preview while maintaining aspect ratio
            width, height = image.size
            max_size = 400
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage for display
            tk_image = ImageTk.PhotoImage(resized_image)
            
            # Update original image display
            self.original_image_display.configure(image=tk_image)
            self.original_image_display.image = tk_image  # Keep a reference to prevent garbage collection
            
            # Clear the processed image display for now
            self.processed_image_display.configure(image="")
            
            # Store OpenCV version of the image for processing
            self.current_image = cv2.imread(img_path)
            self.original_img = self.current_image.copy()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
    
    def process_image(self):
        """Process the selected image with the constellation recognition model"""
        if self.current_image_path is None:
            messagebox.showinfo("Information", "Please select an image first.")
            return
        
        self.status_var.set("Processing image...")
        self.root.update()
        
        try:
            img = None
            source_info = ""
              # Always keep a reference to the original image for display
            original_img = self.current_image.copy()
            
            # Handle preprocessing if requested
            if self.preprocess_var.get():
                processed = preprocess_image(self.current_image_path)
                img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                source_info = f"Source: {os.path.basename(self.current_image_path)} (preprocessed)"
                
                # Display original image
                original_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                
                # Resize original image for display while maintaining aspect ratio
                width, height = original_pil.size
                max_size = 400
                ratio = min(max_size/width, max_size/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                resized_original = original_pil.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage for display
                tk_original = ImageTk.PhotoImage(resized_original)
                self.original_image_display.configure(image=tk_original)
                self.original_image_display.image = tk_original
                  # Make both frames visible
                self.original_image_frame.pack(side=tk.LEFT, padx=5)
                self.original_image_frame.configure(text="Original Image")
                self.processed_image_frame.configure(text="Preprocessed Image with Detection")
                self.processed_image_frame.pack(side=tk.LEFT, padx=5)
            else:
                img = self.current_image.copy()
                source_info = f"Source: {os.path.basename(self.current_image_path)}"
                # Hide original image frame when not doing comparison
                self.original_image_frame.pack_forget()
              # Handle enhancement if requested
            if self.enhance_var.get():
                # Create directories if they don't exist
                os.makedirs("raw_images", exist_ok=True)
                os.makedirs("processed_images", exist_ok=True)
                
                # Generate a unique filename with timestamp
                timestamp = int(time.time())
                filename = f"image_{timestamp}.jpg"
                raw_image_path = os.path.join("raw_images", filename)
                processed_image_path = os.path.join("processed_images", filename)
                
                # Save the original image to raw_images folder
                cv2.imwrite(raw_image_path, img)
                
                # Process the image with preprocess_image function
                processed = preprocess_image(raw_image_path)
                
                # Convert grayscale back to BGR for further processing
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                
                # Save the processed image to processed_images folder
                cv2.imwrite(processed_image_path, processed_bgr)
                
                # Use the processed image for recognition
                img = processed_bgr
                  # We only want to show side-by-side comparison when preprocessing is enabled
                # So we don't do anything special here for enhancement-only mode
            
            # Process the image with the model
            start_time = time.time()
            result_img, constellation, confidence = recognize_constellation(img, self.model)
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display the result image
            result_pil = Image.fromarray(result_img)
            
            # Resize image for display while maintaining aspect ratio
            width, height = result_pil.size
            max_size = 500
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            resized_result = result_pil.resize((new_width, new_height), Image.LANCZOS)
              # Convert to PhotoImage for display
            tk_result = ImageTk.PhotoImage(resized_result)
            
            # Update processed image display
            self.processed_image_display.configure(image=tk_result)
            self.processed_image_display.image = tk_result  # Keep a reference
            
            # Display results in text widget
            self.results_text.delete(1.0, tk.END)
            
            confidence_str = f"{confidence:.2f}" if confidence else "N/A"
            
            result_text = f"""Recognition Results
-------------------
{source_info}
Detected Constellation: {constellation}
Confidence: {confidence_str}
Processing Time: {processing_time:.2f} seconds
Enhancement Applied: {'Yes' if self.enhance_var.get() else 'No'}
Preprocessing Applied: {'Yes' if self.preprocess_var.get() else 'No'}
"""
            
            # Include image paths in the result information if enhancement was applied
            if self.enhance_var.get():
                result_text += f"\nRaw Image Path: {raw_image_path}\n"
                result_text += f"Processed Image Path: {processed_image_path}\n"
            
            # Add constellation information if available
            if constellation in self.constellations_info:
                result_text += f"\nAbout {constellation}:\n{self.constellations_info[constellation]}\n"
            
            self.results_text.insert(tk.END, result_text)
            
            # Update status bar
            self.status_var.set(f"Processed {os.path.basename(self.current_image_path)} - Detected: {constellation}")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.status_var.set("Error processing image")
            
            # Reset image display in case of error
            try:
                # Just display the original image on error
                if self.original_img is not None:
                    original_pil = Image.fromarray(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
                    
                    # Resize original image for display
                    width, height = original_pil.size
                    max_size = 400
                    ratio = min(max_size/width, max_size/height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    resized_original = original_pil.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Display just the original image
                    tk_original = ImageTk.PhotoImage(resized_original)
                    self.processed_image_display.configure(image=tk_original)
                    self.processed_image_display.image = tk_original
                    
                    self.original_image_frame.pack_forget()
                    self.processed_image_frame.configure(text="Image (Error in Processing)")
            except:
                # If even this fails, just clear the displays
                self.processed_image_display.configure(image="")
                self.original_image_display.configure(image="")

def main():
    # Create folders if they don't exist
    os.makedirs("raw_images", exist_ok=True)
    os.makedirs("processed_images", exist_ok=True)
    
    # Initialize the main application window
    root = tk.Tk()
    app = ConstellationRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
