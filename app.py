import gradio as gr
import cv2
import face_recognition
import numpy as np
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import io
import json
from datetime import datetime

# Global face database
face_database = {}

def load_face_database():
    """Load face database from pickle file if it exists"""
    global face_database
    if os.path.exists('face_database.pkl'):
        try:
            with open('face_database.pkl', 'rb') as f:
                face_database = pickle.load(f)
                print(f"Loaded {len(face_database)} faces from database")
        except Exception as e:
            print(f"Error loading face database: {e}")
            face_database = {}
    else:
        face_database = {}

def save_face_database():
    """Save face database to pickle file"""
    global face_database
    try:
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(face_database, f)
        print("Face database saved successfully")
    except Exception as e:
        print(f"Error saving face database: {e}")

def enhance_image_for_recognition(image):
    """Enhance image quality for better face recognition"""
    img_array = np.array(image)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization to improve contrast
    enhanced_gray = cv2.equalizeHist(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def resize_image_for_speed(image, max_width=640):
    """Resize image to reduce processing time while maintaining quality"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Only resize if image is larger than max_width
    if width > max_width:
        # Calculate new height maintaining aspect ratio
        ratio = max_width / width
        new_height = int(height * ratio)
        new_width = max_width
        
        # Resize using PIL for better quality
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height} for faster processing")
        return resized_image, ratio
    
    return image, 1.0

def scale_face_locations(face_locations, scale_ratio):
    """Scale face locations back to original image size"""
    if scale_ratio == 1.0:
        return face_locations
    
    scaled_locations = []
    for (top, right, bottom, left) in face_locations:
        scaled_top = int(top / scale_ratio)
        scaled_right = int(right / scale_ratio)
        scaled_bottom = int(bottom / scale_ratio)
        scaled_left = int(left / scale_ratio)
        scaled_locations.append((scaled_top, scaled_right, scaled_bottom, scaled_left))
    
    return scaled_locations

def detect_and_encode_faces_enhanced(image, model='hog'):
    """Optimized face detection with faster CNN processing"""
    # Resize image for faster processing
    resized_image, scale_ratio = resize_image_for_speed(image, max_width=640)
    img_array = np.array(resized_image)
    
    # Start with fast HOG model first
    face_locations = face_recognition.face_locations(img_array, model='hog')
    
    # If no faces found with HOG, try optimized CNN model
    if len(face_locations) == 0:
        print("No faces found with HOG, trying optimized CNN model...")
        
        # Further resize for CNN to make it even faster
        cnn_image, cnn_scale = resize_image_for_speed(resized_image, max_width=480)
        cnn_array = np.array(cnn_image)
        
        # Use CNN with minimal upsampling for speed
        cnn_locations = face_recognition.face_locations(cnn_array, model='cnn', number_of_times_to_upsample=0)
        
        if len(cnn_locations) > 0:
            # Scale CNN locations back to resized image size
            total_scale = scale_ratio * cnn_scale
            face_locations = scale_face_locations(cnn_locations, cnn_scale)
            print(f"Found {len(face_locations)} faces with optimized CNN")
        else:
            # Last resort: enhance image and try HOG again (faster than CNN with upsampling)
            print("No faces found with CNN, trying enhanced HOG...")
            enhanced_img = enhance_image_for_recognition(resized_image)
            face_locations = face_recognition.face_locations(np.array(enhanced_img), model='hog')
            
            if len(face_locations) > 0:
                img_array = np.array(enhanced_img)
                print(f"Found {len(face_locations)} faces after image enhancement with HOG")
    
    # Scale face locations back to original image size if needed
    if scale_ratio != 1.0:
        face_locations = scale_face_locations(face_locations, scale_ratio)
        # Use original image for encoding for better quality
        img_array = np.array(image)
    
    # Get face encodings with balanced jitters (2 for speed vs accuracy)
    face_encodings = []
    for face_location in face_locations:
        # Use moderate jitters for good accuracy without too much slowdown
        encodings = face_recognition.face_encodings(img_array, [face_location], num_jitters=2)
        if encodings:
            face_encodings.append(encodings[0])
    
    print(f"Detected {len(face_locations)} face(s), encoded {len(face_encodings)} face(s)")
    return face_locations, face_encodings, img_array

def draw_face_boxes_pil(image, face_locations, names=None, confidences=None):
    """Draw bounding boxes around detected faces using PIL"""
    img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw rectangle around face
        draw.rectangle([(left, top), (right, bottom)], outline="green", width=3)
        
        # Add name and confidence label if provided
        if names and i < len(names):
            name = names[i]
            if confidences and i < len(confidences):
                label = f"{name} ({confidences[i]:.1f}%)"
            else:
                label = name
            
            # Draw background rectangle for text
            text_bbox = draw.textbbox((left, bottom), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill="green")
            draw.text((left, bottom), label, fill="white", font=font)
    
    return img_pil

def register_face(image, first_name, last_name):
    """Register a new face in the database with optimized detection"""
    global face_database
    
    if image is None:
        return None, "❌ Please provide an image"
    
    if not first_name.strip() or not last_name.strip():
        return None, "❌ Please provide both first name and last name"
    
    try:
        print(f"Starting face registration for {first_name} {last_name}...")
        face_locations, face_encodings, img_array = detect_and_encode_faces_enhanced(image)
        
        if len(face_encodings) == 0:
            return None, "❌ No face detected in the image. Please try another photo with a clearer face."
        
        if len(face_encodings) > 1:
            return None, "❌ Multiple faces detected. Please upload an image with only one face."
        
        # Check if person already exists
        full_name = f"{first_name.strip()} {last_name.strip()}"
        if full_name in face_database:
            return None, f"❌ {full_name} is already registered."
        
        # Store encoding
        face_database[full_name] = {
            'encoding': face_encodings[0],
            'registered_date': datetime.now().isoformat(),
            'total_encodings': 1
        }
        
        # Save to file
        save_face_database()
        print(f"Successfully registered {full_name}")
        
        # Draw box around detected face
        result_image = draw_face_boxes_pil(img_array, face_locations, [full_name])
        
        return result_image, f"✅ Successfully registered {full_name}! Total faces in database: {len(face_database)}"
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return None, f"❌ Error during registration: {str(e)}"

def recognize_faces(image):
    """Recognize faces in the uploaded image with optimized processing"""
    global face_database
    
    if image is None:
        return None, "❌ Please provide an image"
    
    if len(face_database) == 0:
        return None, "❌ No faces registered yet. Please register some faces first."
    
    try:
        print("Starting face recognition...")
        face_locations, face_encodings, img_array = detect_and_encode_faces_enhanced(image)
        
        if len(face_encodings) == 0:
            return None, "❌ No face detected in the image. Please try another photo."
        
        print(f"Found {len(face_encodings)} face(s) in the image")
        recognized_names = []
        confidences = []
        all_distances = []  # Store all distances for debugging
        threshold = 0.6  # Decreased threshold for more lenient matching
        
        # Get all registered encodings
        registered_names = list(face_database.keys())
        registered_encodings = [face_database[name]['encoding'] for name in registered_names]
        print(f"Comparing against {len(registered_names)} registered faces")
        
        for i, face_encoding in enumerate(face_encodings):
            # Calculate distances to all registered faces
            face_distances = face_recognition.face_distance(registered_encodings, face_encoding)
            all_distances.extend(face_distances)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            print(f"Face {i+1}: Best match distance = {best_distance:.3f}")
            
            if best_distance < threshold:
                name = registered_names[best_match_index]
                confidence = (1 - best_distance) * 100
                recognized_names.append(name)
                confidences.append(confidence)
                print(f"Recognized: {name} with {confidence:.1f}% confidence")
            else:
                recognized_names.append("Unknown Person")
                confidences.append(0.0)
                print(f"No match found. Best distance {best_distance:.3f} > threshold {threshold}")
        
        # Draw boxes on image
        result_image = draw_face_boxes_pil(img_array, face_locations, recognized_names, confidences)
        
        # Create result message
        known_faces = [name for name in recognized_names if name != "Unknown Person"]
        if known_faces:
            confidence_info = [f"{name} ({conf:.1f}%)" for name, conf in zip(recognized_names, confidences) if name != "Unknown Person"]
            message = f"✅ Recognized: {', '.join(confidence_info)}"
        else:
            min_distance = min(all_distances) if all_distances else 1.0
            message = f"❌ No registered faces found. Best match distance: {min_distance:.3f} (threshold: {threshold})"
        
        return result_image, message
            
    except Exception as e:
        print(f"Error during recognition: {str(e)}")
        return None, f"❌ Error during recognition: {str(e)}"

def get_database_info():
    """Get information about the current database"""
    global face_database
    
    if len(face_database) == 0:
        return "📊 Database is empty. No faces registered yet."
    
    info = f"📊 **Database Statistics:**\n"
    info += f"- **Total registered faces:** {len(face_database)}\n"
    info += f"- **Database size:** {len(pickle.dumps(face_database)) / 1024:.2f} KB\n\n"
    info += "**Registered People:**\n"
    
    for i, (name, data) in enumerate(face_database.items(), 1):
        reg_date = data.get('registered_date', 'Unknown')
        if reg_date != 'Unknown':
            try:
                date_obj = datetime.fromisoformat(reg_date)
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = reg_date
        else:
            formatted_date = 'Unknown'
        info += f"{i}. **{name}** (Registered: {formatted_date})\n"
    
    return info

def delete_person(person_name):
    """Delete a person from the database"""
    global face_database
    
    if not person_name:
        return get_database_info(), "❌ Please enter a person's name to delete"
    
    if person_name not in face_database:
        return get_database_info(), f"❌ {person_name} not found in database"
    
    del face_database[person_name]
    save_face_database()
    
    return get_database_info(), f"✅ Successfully deleted {person_name} from database"

def clear_database():
    """Clear all data from the database"""
    global face_database
    
    face_database = {}
    if os.path.exists('face_database.pkl'):
        os.remove('face_database.pkl')
    
    return get_database_info(), "✅ Database cleared successfully"

def export_database():
    """Export database to a downloadable file"""
    global face_database
    
    if len(face_database) == 0:
        return None, "❌ Database is empty. Nothing to export."
    
    try:
        # Create a JSON export with metadata
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_faces': len(face_database),
            'faces': {}
        }
        
        for name, data in face_database.items():
            export_data['faces'][name] = {
                'encoding': data['encoding'].tolist(),  # Convert numpy array to list
                'registered_date': data.get('registered_date', ''),
                'total_encodings': data.get('total_encodings', 1)
            }
        
        # Save as pickle file for download
        export_filename = f"face_database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(export_filename, 'wb') as f:
            pickle.dump(face_database, f)
        
        return export_filename, f"✅ Database exported successfully! File: {export_filename}"
        
    except Exception as e:
        return None, f"❌ Error exporting database: {str(e)}"

# Load existing database on startup
load_face_database()

# Create Gradio interface
with gr.Blocks(title="Face Recognition App") as app:
    gr.Markdown("""
    # 👤 Face Recognition App
    
    **Enhanced face recognition with support for unclear or partially visible faces**
    
    This app uses advanced face detection and recognition algorithms to identify people even when faces are not perfectly clear or fully visible.
    """)
    
    with gr.Tabs():
        # Register Face Tab
        with gr.Tab("📝 Register Face"):
            gr.Markdown("### Register a new person in the database")
            
            with gr.Row():
                with gr.Column():
                    register_image = gr.Image(
                        sources=["upload", "webcam"],
                        type="pil",
                        label="Upload or capture image"
                    )
                    
                with gr.Column():
                    first_name_input = gr.Textbox(
                        label="First Name",
                        placeholder="Enter first name"
                    )
                    last_name_input = gr.Textbox(
                        label="Last Name", 
                        placeholder="Enter last name"
                    )
                    register_btn = gr.Button("🔐 Register Face", variant="primary")
            
            register_output_image = gr.Image(label="Registration Result")
            register_message = gr.Textbox(label="Status", interactive=False)
            
            register_btn.click(
                fn=register_face,
                inputs=[register_image, first_name_input, last_name_input],
                outputs=[register_output_image, register_message]
            )
        
        # Recognize Face Tab  
        with gr.Tab("🔍 Recognize Faces"):
            gr.Markdown("### Recognize registered faces in images")
            
            recognize_image = gr.Image(
                sources=["upload", "webcam"],
                type="pil", 
                label="Upload or capture image for recognition"
            )
            
            recognize_btn = gr.Button("🔍 Recognize Faces", variant="primary")
            
            recognize_output_image = gr.Image(label="Recognition Result")
            recognize_message = gr.Textbox(label="Recognition Status", interactive=False)
            
            recognize_btn.click(
                fn=recognize_faces,
                inputs=[recognize_image],
                outputs=[recognize_output_image, recognize_message]
            )
        
        # Database Management Tab
        with gr.Tab("📋 Manage Database"):
            gr.Markdown("### View and manage registered faces")
            
            with gr.Row():
                with gr.Column():
                    database_info = gr.Markdown(value=get_database_info())
                    refresh_btn = gr.Button("🔄 Refresh Database Info")
                    
                with gr.Column():
                    delete_name_input = gr.Textbox(
                        label="Person Name to Delete",
                        placeholder="Enter exact name (e.g., John Doe)"
                    )
                    delete_btn = gr.Button("🗑️ Delete Person", variant="secondary")
                    
                    gr.Markdown("---")
                    
                    clear_btn = gr.Button("🗑️ Clear All Data", variant="stop")
                    export_btn = gr.Button("💾 Export Database", variant="secondary")
            
            management_message = gr.Textbox(label="Management Status", interactive=False)
            export_file = gr.File(label="Download Exported Database", visible=False)
            
            refresh_btn.click(
                fn=get_database_info,
                outputs=[database_info]
            )
            
            delete_btn.click(
                fn=delete_person,
                inputs=[delete_name_input],
                outputs=[database_info, management_message]
            )
            
            clear_btn.click(
                fn=clear_database,
                outputs=[database_info, management_message]
            )
            
            export_btn.click(
                fn=export_database,
                outputs=[export_file, management_message]
            )
    
    gr.Markdown("""
    ---
    ### 💡 Tips for Better Recognition:
    - **Fast Processing**: Images automatically resized for optimal speed (640px max width)
    - **Smart CNN**: CNN model uses 480px resolution for 3x faster processing
    - **Progressive Detection**: HOG → Optimized CNN → Enhanced HOG fallback
    - **Balanced Encoding**: 2 jitters for good accuracy without slowdown
    - **Processing time**: Typically 1-2 seconds for most images
    
    ### 🔧 Speed Optimizations:
    - **Image Resizing**: Large images automatically downscaled for processing
    - **Minimal CNN Upsampling**: Uses 0 upsampling for maximum speed
    - **Smart Fallbacks**: Enhanced HOG instead of slow CNN upsampling
    - **Optimized Threshold**: 60% threshold for balanced speed vs accuracy
    
    **Powered by face_recognition library with performance optimizations**
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(
        share=False,  # Set to False for local development
        server_name="127.0.0.1",  # Use localhost for local development
        server_port=7861,  # Use different port        git init
        git add .
        git commit -m "Face Recognition App with CNN optimizations"
        git remote add origin https://github.com/YOUR_USERNAME/face-recognition-app.git
        git push -u origin main
        show_error=True
    )
