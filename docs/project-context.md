# Project Context: MINA Fishcare YOLO

## Overview
MINA is a disease detection progressive web application (PWA) for fish, built with React and Vite. The core feature is predicting fish diseases and drawing bounding boxes in real-time using a device camera or gallery images.

## Core Requirements & Constraints
1. **Offline-First & On-Device**: The model must map directly to the PWA requirements by running entirely locally in the browser without a network connection.
2. **Lightweight**: Model size, memory usage, and inference speed are highly constrained because everything happens strictly on edge devices natively in the web browser (utilizing ONNX Runtime Web). 
3. **Current Workflow**:
   - The user captures/selects an image in the web app.
   - Inference runs locally via the bundled `.onnx` model (based on YOLOv8n).
   - If boxes are found, symptoms and treatments are presented.
   - Diagnoses histories are stored locally using IndexedDB.

## Current Issue
**False Positives on non-fish images:** 
When a picture that *does not contain a fish* is passed to the current model, the model predicts bounding boxes anyway, falsely categorizing random background objects (e.g., tables, keyboards) as "healthy fish" (or sometimes another disease class). 

## Objective
The application needs to recognize when there is *no fish* in the picture and gracefully abort inference or display an applicable message (e.g., "No fish detected"). 
While adding a classification step ("fish" vs "not fish") is an option, whatever solution is chosen and implemented must respect the offline-first, lightweight nature of the PWA, ensuring mobile browser support without heavy performance penalties or excessive model download sizes.
