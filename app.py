from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer , AutoModel,Qwen2VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
import torch
from threading import Thread
import gradio as gr
from gradio import FileData
import time
import spaces
import fitz  # PyMuPDF
import io
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and processor
ckpt ="mistral-community/pixtral-12b"
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained(ckpt,trust_remote_code=True)

class DocumentState:
    def __init__(self):
        self.current_doc_images = []
        self.current_doc_text = ""
        self.doc_type = None
        
    def clear(self):
        self.current_doc_images = []
        self.current_doc_text = ""
        self.doc_type = None
        
doc_state = DocumentState()

def process_pdf_file(file_path):
    """Convert PDF to images and extract text using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        images = []
        text = ""
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
                
                zoom = 2.5
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img = img.convert("RGB")
                
                max_size = 1600
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                images.append(img)
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
                
        doc.close()
        
        if not images:
            raise ValueError("No valid images could be extracted from the PDF")
            
        return images, text
        
    except Exception as e:
        logger.error(f"Error processing PDF file: {str(e)}")
        raise

def process_uploaded_file(file):
    """Process uploaded file and update document state."""
    try:
        doc_state.clear()
        
        if file is None:
            return "No file uploaded. Please upload a file."
        
        # Get the file path and extension
        if isinstance(file, dict):
            file_path = file["name"]
        else:
            file_path = file.name
            
        # Get file extension
        file_ext = file_path.lower().split('.')[-1]
        
        # Define allowed extensions
        image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        
        if file_ext == 'pdf':
            doc_state.doc_type = 'pdf'
            try:
                doc_state.current_doc_images, doc_state.current_doc_text = process_pdf_file(file_path)
                return f"PDF processed successfully. Total pages: {len(doc_state.current_doc_images)}. You can now ask questions about the content."
            except Exception as e:
                return f"Error processing PDF: {str(e)}. Please try a different PDF file."
        elif file_ext in image_extensions:
            doc_state.doc_type = 'image'
            try:
                img = Image.open(file_path).convert("RGB")
                max_size = 1600
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                doc_state.current_doc_images = [img]
                return "Image loaded successfully. You can now ask questions about the content."
            except Exception as e:
                return f"Error processing image: {str(e)}. Please try a different image file."
        else:
            return f"Unsupported file type: {file_ext}. Please upload a PDF or image file (PNG, JPG, JPEG, GIF, BMP, WEBP)."
    except Exception as e:
        logger.error(f"Error in process_file: {str(e)}")
        return "An error occurred while processing the file. Please try again."

@spaces.GPU()
def bot_streaming(prompt_option, max_new_tokens=4096):
    try:
        # Define predetermined prompts
        prompts = {
            "Timesheet Details (Full Extraction)": (
                """Extract structured information from the provided timesheet. The extracted details should include:

1. Personnel Details:

Name

Position Title

Work Location

Contractor Status (Yes/No)

NOC ID

Month and Year



2. Service and Activity Summary:

Regular Service Days (ONSHORE)

Standby Days (ONSHORE in Doha)

Offshore Days

Standby & Extended Hitch Days (OFFSHORE)

Extended Hitch Days (ONSHORE Rotational)

Service during Weekends & Public Holidays



3. Overtime and Compensation:

ONSHORE Overtime Hours (Over 8 hours)

OFFSHORE Overtime Hours (Over 12 hours)

Per Diem Days (ONSHORE/OFFSHORE Rotational Personnel)



4. Training and Travel:

Training Days

Travel Days



5. Totals:

Provide totals for all categories where applicable.




Ensure all extracted data is presented in a clean, structured format. Omit any irrelevant or unrecognizable content. Use the exact terminology and units (e.g., 'days,' 'hours') as found in the document."""
            ),
            "Timesheet Details (Basic Extraction)": (
                "Based on the provided timesheet details, extract the following information:\n"
                "   - Full name of the person\n"
                "   - Position title of the person\n"
                "   - Work location\n"
                "   - Contractor's name\n"
                "   - NOC ID\n"
                "   - Month and year (in MM/YYYY format)"
            ),
            "Structured Data Extraction": (
                "You are an advanced data extraction assistant. Your task is to parse structured input text and extract key data points into clearly defined categories. Focus only on the requested details, ensuring accuracy and proper grouping. Below is the format for extracting the data:\n\n"
                "---\n"
                "Project Information\n\n"
                "Project Name:\n\n"
                "Project and Package:\n\n"
                "RPO Number:\n\n"
                "PMC Name:\n\n"
                "Project Location:\n\n"
                "Year:\n\n"
                "Month:\n\n"
                "Timesheet Details\n\n"
                "Week X (Date)\n\n"
                "Holidays:\n\n"
                "Regular Hours:\n\n"
                "Overtime Hours:\n\n"
                "Total Hours:\n\n"
                "Comments:\n\n"
                "Additional Data\n\n"
                "Reviewed By:\n\n"
                "Date of Review:\n\n"
                "Position:\n\n"
                "Supervisor Business:\n\n"
                "Date of Approval:\n\n"
                "---\n\n"
                "Ensure the extracted data strictly follows the format above and is organized by category. Ignore unrelated text. Respond only with the formatted output."
            )
        }
        
        # Get the selected prompt
        selected_prompt = prompts.get(prompt_option, "Invalid prompt selected.")
        
        messages = []
        
        # Include document context
        if doc_state.current_doc_images:
            context = f"\nDocument context:\n{doc_state.current_doc_text}" if doc_state.current_doc_text else ""
            current_msg = f"{selected_prompt}{context}"
            messages.append({"role": "user", "content": [{"type": "text", "text": current_msg}, {"type": "image"}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": selected_prompt}]})

        # Process inputs
        texts = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        try:
            if doc_state.current_doc_images:
                inputs = processor(
                    text=texts,
                    images=doc_state.current_doc_images[0:1],
                    return_tensors="pt"
                ).to("cuda")
            else:
                inputs = processor(text=texts, return_tensors="pt").to("cuda")
                
            streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
                yield buffer
                
        except Exception as e:
            logger.error(f"Error in model processing: {str(e)}")
            yield "An error occurred while processing your request. Please try again."
            
    except Exception as e:
        logger.error(f"Error in bot_streaming: {str(e)}")
        yield "An error occurred. Please try again."

def clear_context():
    """Clear the current document context."""
    doc_state.clear()
    return "Document context cleared. You can upload a new document."

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Analyzer with Predetermined Prompts")
    gr.Markdown("Upload a PDF or image (PNG, JPG, JPEG, GIF, BMP, WEBP) and select a prompt to analyze its contents.")
    
    with gr.Row():
        file_upload = gr.File(
            label="Upload Document",
            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
        )
        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False
        )
    
    with gr.Row():
        prompt_dropdown = gr.Dropdown(
            label="Select Prompt",
            choices=[
                "Timesheet Details (Full Extraction)",
                "Timesheet Details (Basic Extraction)",
                "Structured Data Extraction"
            ],
            value="Timesheet Details (Full Extraction)"
        )
        generate_btn = gr.Button("Generate")
    
    clear_btn = gr.Button("Clear Document Context")
    
    output_text = gr.Textbox(
        label="Output",
        interactive=False
    )
    
    file_upload.change(
        fn=process_uploaded_file,
        inputs=[file_upload],
        outputs=[upload_status]
    )
    
    generate_btn.click(
        fn=bot_streaming,
        inputs=[prompt_dropdown],
        outputs=[output_text]
    )
    
    clear_btn.click(
        fn=clear_context,
        outputs=[upload_status]
    )

# Launch the interface
demo.launch(debug=True)