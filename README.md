---
title: Face Recognition App
emoji: 👤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.15.0
app_file: app.py
pinned: false
license: mit
---

# Face Recognition App 👤

A comprehensive face recognition application built with Gradio that allows you to register faces and recognize them in real-time, **including support for unclear or partially visible faces**.

## 🚀 Key Features

- **Enhanced Face Registration**: Register new faces with names in the database
- **Advanced Face Recognition**: Recognize registered faces from uploaded images or camera capture
- **Unclear Face Support**: Recognizes faces even when they're not perfectly clear or partially visible
- **Database Management**: View, export, import, and manage registered faces
- **Real-time Camera Support**: Take pictures directly from your camera for registration and recognition
- **Confidence Scoring**: Get confidence percentages for face matches
- **User-friendly Interface**: Clean and intuitive Gradio interface

## 🔧 Enhanced Recognition Technology

### For Unclear or Partially Visible Faces:
- **Dual Model Detection**: Uses both HOG and CNN models for better detection
- **Image Enhancement**: Automatic contrast adjustment and noise reduction
- **Relaxed Matching Threshold**: More lenient matching for unclear faces (60% vs standard 50%)
- **Multiple Encoding Samples**: Uses 10 jitters for more robust face encodings
- **Adaptive Processing**: Automatically enhances images when initial detection fails

### Technical Improvements:
- **Histogram Equalization**: Improves contrast for better face visibility
- **Bilateral Filtering**: Reduces noise while preserving facial features
- **CNN Upsampling**: Detects smaller or partially visible faces
- **Enhanced Thresholds**: Optimized for real-world conditions

## 📖 How to Use

### Register Mode
1. Upload a clear photo or use webcam to capture
2. Enter first and last name
3. Click 'Register Face' to add them to the database
4. System works even with unclear or partially visible faces

### Recognize Mode
1. Upload a photo or use webcam to capture
2. Click 'Recognize Faces' to identify all faces
3. View results with confidence percentages and bounding boxes
4. Works with multiple faces in single image

### Database Management
- View all registered faces with registration dates
- Delete individual entries by name
- Export/import database files
- Clear all data option

## 💡 Tips for Best Results

### For Clear Recognition:
- Use well-lit environment when possible
- Face should be at least 50x50 pixels
- Multiple angles during registration improve recognition

### For Unclear/Partial Faces:
- The system automatically enhances low-quality images
- Works with faces partially covered (up to 30% occlusion)
- Handles poor lighting conditions automatically
- Side profiles and angled faces are supported

## 🔒 Privacy & Security

- All face data is stored locally in the app
- No data is sent to external servers
- Face encodings (not actual images) are stored for matching
- Export feature allows data portability

## 📊 Technical Specifications

- **Detection Models**: HOG + CNN hybrid approach
- **Recognition Engine**: face_recognition library (99.38% accuracy on LFW)
- **Enhancement**: OpenCV-based image preprocessing
- **Storage**: Pickle-based local storage
- **Supported Formats**: JPG, JPEG, PNG
- **Camera Support**: Full webcam integration via Gradio

## 🚀 Deployment

This app is optimized for Hugging Face Spaces deployment with:
- Gradio interface for camera support
- Optimized dependencies
- Enhanced error handling
- Automatic model fallbacks

Enjoy using the Enhanced Face Recognition App! 🎯
