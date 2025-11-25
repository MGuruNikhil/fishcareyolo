# Requirements Document

## Introduction

Fish Care is a mobile application that enables fish farmers and aquarium enthusiasts to detect fish diseases in real-time using their device's camera. The system runs a YOLOv8-based machine learning model entirely on-device, providing instant offline disease detection without requiring network connectivity. Users receive immediate diagnostic information and recommended actions.

## Glossary

- **Fish_Care_App**: The React Native (Expo) mobile application that provides the user interface and runs on-device inference
- **On_Device_Model**: The YOLOv8 model bundled with the app that runs inference locally on the user's device
- **YOLOv8**: You Only Look Once version 8, a real-time object detection model used for identifying fish diseases
- **Bounding_Box**: A rectangular overlay drawn around detected fish or disease regions in an image
- **Confidence_Score**: A numerical value (0.0 to 1.0) indicating the model's certainty about a detection
- **Disease_Class**: A category of fish disease that the model can identify (e.g., fin rot, ich, fungal infection)
- **Detection_Result**: A structured object containing identified diseases, confidence scores, and bounding box coordinates
- **Inference**: The process of running an image through the trained model to produce predictions

## Requirements

### Requirement 1
As a fish owner, I want to capture images of my fish using the app, so that I can have them analyzed for diseases.

#### Acceptance Criteria
1. WHEN a user opens the camera screen THEN the Fish_Care_App SHALL display a live camera preview with capture controls
2. WHEN a user taps the capture button THEN the Fish_Care_App SHALL capture the current frame and store it for analysis
3. WHEN a user selects an image from their gallery THEN the Fish_Care_App SHALL load the selected image for analysis
4. IF the camera permission is denied THEN the Fish_Care_App SHALL display a clear message explaining why camera access is needed and provide a link to settings

### Requirement 2
As a fish owner, I want disease detection to work offline, so that I can check my fish's health anywhere without internet.

#### Acceptance Criteria
1. WHEN the app is installed THEN the Fish_Care_App SHALL bundle the On_Device_Model as part of the application assets
2. WHEN a user submits an image for analysis THEN the Fish_Care_App SHALL run Inference using the On_Device_Model without network connectivity
3. WHEN the On_Device_Model performs Inference THEN the Fish_Care_App SHALL complete the analysis within 2 seconds on supported devices
4. WHEN the device lacks sufficient resources THEN the Fish_Care_App SHALL display a message indicating minimum device requirements

### Requirement 3
As a fish owner, I want to see detection results displayed clearly, so that I can understand what diseases were found.

#### Acceptance Criteria
1. WHEN the Fish_Care_App completes Inference THEN the Fish_Care_App SHALL display the analyzed image with Bounding_Boxes around detected regions
2. WHEN displaying detection results THEN the Fish_Care_App SHALL show each Disease_Class name alongside its Confidence_Score as a percentage
3. WHEN multiple diseases are detected THEN the Fish_Care_App SHALL list all detections sorted by Confidence_Score in descending order
4. WHEN no diseases are detected THEN the Fish_Care_App SHALL display a message indicating the fish appears healthy

### Requirement 4
As a fish owner, I want accurate disease detection, so that I can trust the results and take appropriate action.

#### Acceptance Criteria
1. WHEN the On_Device_Model performs Inference THEN the On_Device_Model SHALL identify common fish diseases including fin rot, ich, fungal infection, and bacterial infection
2. WHEN the On_Device_Model completes Inference THEN the On_Device_Model SHALL produce Detection_Results containing Disease_Class, Confidence_Score, and Bounding_Box coordinates for each detection
3. WHEN the Confidence_Score is below 0.3 THEN the On_Device_Model SHALL exclude that detection from the results
4. WHEN serializing Detection_Results for storage THEN the Fish_Care_App SHALL encode them as JSON with consistent field names
5. WHEN loading Detection_Results from storage THEN the Fish_Care_App SHALL parse the JSON and reconstruct the detection objects

### Requirement 5
As a fish owner, I want to view my detection history, so that I can track my fish's health over time.

#### Acceptance Criteria
1. WHEN a detection is completed THEN the Fish_Care_App SHALL persist the image and Detection_Result to local storage
2. WHEN a user opens the history screen THEN the Fish_Care_App SHALL display a list of past detections with timestamps
3. WHEN a user taps a history item THEN the Fish_Care_App SHALL display the full Detection_Result for that entry
4. WHEN displaying history THEN the Fish_Care_App SHALL sort entries by timestamp in descending order (newest first)

### Requirement 6
As a fish owner, I want to receive information about detected diseases, so that I can take appropriate action.

#### Acceptance Criteria
1. WHEN a disease is detected THEN the Fish_Care_App SHALL display a brief description of the Disease_Class
2. WHEN displaying disease information THEN the Fish_Care_App SHALL include common symptoms and recommended treatments
3. WHEN the user taps on a disease name THEN the Fish_Care_App SHALL navigate to a detailed information screen for that disease

### Requirement 7
As a fish owner, I want the app to feel quick and effortless, so that checking my fish's health becomes a natural habit.

#### Acceptance Criteria
1. WHEN the app launches THEN the Fish_Care_App SHALL display the camera screen within 2 seconds
2. WHEN a user captures an image THEN the Fish_Care_App SHALL provide immediate visual feedback (animation or haptic) within 100 milliseconds
3. WHEN transitioning between screens THEN the Fish_Care_App SHALL use smooth animations that complete within 300 milliseconds
4. WHEN displaying detection results THEN the Fish_Care_App SHALL animate the appearance of Bounding_Boxes and disease labels progressively
5. WHEN the user performs common actions THEN the Fish_Care_App SHALL require no more than 2 taps to capture and analyze an image from the home screen

### Requirement 8
As a fish owner, I want the app to look beautiful and feel intuitive, so that I enjoy using it regularly.

#### Acceptance Criteria
1. WHEN displaying any screen THEN the Fish_Care_App SHALL use a consistent color palette inspired by aquatic themes (blues, teals, soft whites)
2. WHEN presenting interactive elements THEN the Fish_Care_App SHALL use clear visual affordances with adequate touch targets (minimum 44x44 points)
3. WHEN showing detection results THEN the Fish_Care_App SHALL use color-coded Bounding_Boxes (green for healthy indicators, yellow for warnings, red for disease detections)
4. WHEN displaying text content THEN the Fish_Care_App SHALL use readable typography with sufficient contrast ratios meeting accessibility standards
5. WHEN a user first opens the app THEN the Fish_Care_App SHALL present a brief onboarding flow that explains core features in 3 screens or fewer
