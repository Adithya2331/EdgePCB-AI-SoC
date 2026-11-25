PCB Vision AI – Intelligent PCB Defect Detection & AI-Assisted Quality Inspection

<p align="center">
  <strong>AI-Powered Edge Inspection | Computer Vision | Raspberry Pi | On-Device AI | Future SoC Acceleration</strong>
</p><p align="center">
  An intelligent PCB inspection system designed to detect manufacturing defects, identify their locations, explain their possible impact, and assist quality-control decisions using Edge AI.
</p>---

1. Project Overview

Printed Circuit Boards (PCBs) are essential components of almost every electronic device, including smartphones, computers, automotive electronics, medical equipment, communication systems, and industrial control systems.

During PCB manufacturing, several defects can occur, such as short circuits, open circuits, missing pads, mouse bites, unwanted copper, and other manufacturing irregularities.

Traditional inspection methods depend heavily on manual observation or dedicated Automated Optical Inspection (AOI) systems. Manual inspection can be time-consuming, while conventional inspection systems may require expensive dedicated equipment and specialized infrastructure.

Our project, PCB Vision AI, aims to develop an intelligent, affordable, and extensible PCB inspection solution using Artificial Intelligence and Edge Computing.

The system uses a YOLO-based computer vision model to identify visible PCB defects from images. The detection results can then be processed by an AI-assisted explanation module to provide understandable information about the detected defects, their possible effects, and suggested inspection actions.

The system is designed around Raspberry Pi hardware so that image processing and AI-assisted inspection can progressively move towards local, edge-based operation instead of depending entirely on cloud computing.

Our main objective

To transform a normal PCB inspection process into an intelligent inspection workflow that can:

- Capture or receive PCB images.
- Detect visible manufacturing defects.
- Identify defect categories.
- Locate defects within the PCB image.
- Estimate detection confidence.
- Explain the possible impact of defects.
- Generate structured inspection reports.
- Assist operators in making quality-control decisions.

---

2. Problem Statement

In electronics manufacturing, even a small PCB defect can affect the performance and reliability of the final product.

Examples include:

- An unwanted copper connection can produce a short circuit.
- A broken conductor can create an open circuit.
- Missing conductive material can interrupt electrical connections.
- Excess copper may create unintended connections.
- Damage around PCB tracks can reduce manufacturing quality.

The major problem is not simply identifying whether a PCB contains a defect. A useful inspection system should also communicate:

1. What defect was detected?
2. Where is the defect located?
3. How confident is the detection?
4. What could be the possible effect?
5. What should the inspection operator do next?

Our project attempts to address these questions through computer vision and AI-assisted reporting.

---

3. Proposed Solution

We propose a Raspberry Pi-based intelligent PCB inspection system consisting of two major intelligence layers.

Layer 1: Computer Vision – Defect Detection

A YOLO-based object detection model processes PCB images and identifies defects.

It produces:

- Defect classification.
- Bounding-box coordinates.
- Confidence scores.
- Annotated PCB images.

Layer 2: AI-Assisted Inspection – Defect Explanation

The detected information is converted into structured data.

A rule-based inspection layer and an intended lightweight Llama language model integration use this information to generate human-readable explanations.

The explanation module is intended to communicate:

- The detected defect.
- Its possible electrical or manufacturing impact.
- Its severity according to configured inspection rules.
- Whether additional testing or operator review is recommended.

Important: The language model is not responsible for visually detecting PCB defects. YOLO performs visual detection; the language model explains the supplied detection information.

---

4. Overall System Architecture

             PCB / ELECTRONIC BOARD
                       |
                       v
               IMAGE ACQUISITION
                       |
             Camera / Sample Image
                       |
                       v
                RASPBERRY PI
                       |
                       v
              IMAGE PREPROCESSING
                 Python / OpenCV
                       |
                       v
                YOLO DETECTION
                       |
             +---------+---------+
             |                   |
             v                   v
        Defect Class       Bounding Boxes
             |                   |
             +---------+---------+
                       |
                       v
              STRUCTURED RESULTS
                  JSON DATA
                       |
                       v
              INSPECTION RULE ENGINE
                       |
                       v
              AI EXPLANATION MODULE
             Llama / ExecuTorch
                       |
                       v
              INSPECTION REPORT
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Defect       Possible      Suggested
       Details       Impact         Action
          |            |            |
          +------------+------------+
                       |
                       v
              OPERATOR / USER

This architecture separates visual recognition from explanation and decision support. This makes the system easier to maintain and allows individual modules to be improved independently.

---

5. Hardware Components

Component| Purpose
Raspberry Pi 4| Current prototype edge-computing platform
Camera module| Capturing PCB images
USB storage| Operating-system storage and project files
Power supply| Supplying power to Raspberry Pi
Display / SSH computer| Monitoring and controlling the system
PCB sample images| Input for inspection and testing

Why Raspberry Pi?

Raspberry Pi is a compact single-board computer capable of running Linux, Python applications, image-processing libraries, and lightweight AI inference workloads.

Its advantages include:

- Compact hardware.
- Low power consumption.
- GPIO connectivity.
- Linux-based development environment.
- Camera interfacing.
- Possibility of offline processing.
- Support for future hardware acceleration.

Current hardware note: Our working prototype uses Raspberry Pi 4. Raspberry Pi 5 is the intended upgrade target for a more powerful implementation and closer alignment with the challenge's specified hardware configuration.

The Raspberry Pi 4 prototype should not be confused with a completed Raspberry Pi 5 deployment.

---

6. Software and Technologies Used

Software / Technology| Role in the Project
Python| Main application and integration language
YOLO / Ultralytics| PCB defect object detection
OpenCV| Image processing and image handling
PyTorch| Deep-learning model development and training
Kaggle| Dataset access and model experimentation
DsPCBSD+| Real-world PCB defect dataset
Raspberry Pi OS / Linux| Edge-computing operating environment
ExecuTorch| Intended lightweight on-device model runtime
Llama 3.2 1B Instruct| Intended natural-language explanation model
Git| Version control
GitHub| Source-code management and project documentation

The exact trained YOLO checkpoint, package versions, and runtime configurations should be recorded alongside the final submitted code.

Why Python?

Python provides a convenient development environment for computer vision, deep learning, image processing, and AI integration.

Why YOLO?

YOLO (You Only Look Once) is an object-detection approach that identifies objects and their locations in an image.

Unlike a simple image classifier that only predicts a class for the entire image, an object detector can identify multiple defects and draw separate bounding boxes around them.

Why OpenCV?

OpenCV is used for image-related operations, including:

- Loading images.
- Resizing images.
- Camera-frame processing.
- Image conversion.
- Drawing bounding boxes.
- Displaying detection results.

Why ExecuTorch and Llama?

ExecuTorch is intended to provide a lightweight execution environment for running supported AI models on edge devices.

Llama 3.2 1B Instruct is the proposed language model for interpreting detection results and producing understandable inspection explanations.

The model's role is to explain the information supplied by the detection pipeline, not to replace the YOLO detection model.

---

7. Dataset – DsPCBSD+

One of the most important parts of an AI-based inspection system is its dataset.

For this project, we use PCB defect datasets, including DsPCBSD+ (Dataset of PCB Surface Defect).

Dataset reference:

- Research paper: https://www.nature.com/articles/s41597-024-03656-8
- Official dataset: https://doi.org/10.6084/m9.figshare.24970329

DsPCBSD+ Dataset Information

Parameter| Details
Dataset name| DsPCBSD+
Images| 10,259
Annotated defects| 20,276
Defect categories| 9
Annotation type| Bounding boxes
Annotation formats| YOLO and COCO
Dataset source| Real PCB manufacturing images
Published| 2024
License| CC BY 4.0

The dataset contains actual PCB surface-defect images collected from an industrial manufacturing environment.

It provides manually annotated defect locations, making it useful for training and evaluating object-detection models.

Nine defect categories in DsPCBSD+

No.| Defect| Explanation
1| Short (SH)| Unwanted conductive connection
2| Spur (SP)| Unwanted copper protruding from a conductor
3| Spurious Copper (SC)| Unwanted copper material
4| Open (OP)| Break or interruption in a conductive path
5| Mouse Bite (MB)| Irregular missing material along a conductor edge
6| Hole Breakout (HB)| Damage or breakout around a PCB hole
7| Conductor Scratch (CS)| Physical scratch affecting a conductor
8| Conductor Foreign Object (CFO)| Foreign material on a conductor
9| Base Material Foreign Object (BMFO)| Foreign material on the PCB base material

How the dataset supports our project

The dataset provides a foundation for teaching the model how different PCB defects appear visually.

Our selected detection scope focuses on six defect classes:

1. Missing Hole / Missing Pad
2. Mouse Bite
3. Open Circuit
4. Short Circuit
5. Spur
6. Spurious Copper

The five directly matching DsPCBSD+ categories are Short, Spur, Spurious Copper, Open, and Mouse Bite.

Missing Hole / Missing Pad requires a compatible additional dataset or separately validated annotation mapping. It must not automatically be treated as identical to Hole Breakout.

The remaining DsPCBSD+ classes can be considered for future model expansion.

Dataset preparation workflow

Original PCB Dataset
        |
        v
Dataset Download
        |
        v
Image and Annotation Inspection
        |
        v
Class Mapping
        |
        v
Training / Validation Split
        |
        v
YOLO-Compatible Dataset
        |
        v
Model Training
        |
        v
Model Evaluation

The dataset must be checked for:

- Incorrect annotations.
- Missing label files.
- Duplicate images.
- Incorrect class IDs.
- Inconsistent image dimensions.
- Class imbalance.

Only validated annotations should be used for model training.

---

8. YOLO Model – How Defect Detection Works

YOLO is the primary visual intelligence component of our project.

It processes the input PCB image and predicts the location and category of visible defects.

Working process

Step 1 – Input image

A PCB image is captured using a camera or supplied as a sample image.

Step 2 – Preprocessing

The image is prepared for model inference, including resizing and other required transformations.

Step 3 – YOLO inference

The trained model analyzes the image and identifies possible defect regions.

Step 4 – Classification

Each detected region is assigned a defect class.

Step 5 – Bounding boxes

The model produces coordinates identifying the approximate location of each detected defect.

Step 6 – Confidence score

The model provides a confidence score associated with each prediction.

Step 7 – Output

The application produces an annotated image and structured detection information.

Example of the intended output:

PCB INSPECTION RESULT

Image: sample_pcb.jpg

Detected Defects:
1. Open Circuit
   Confidence: [model output]
   Location: [bounding box coordinates]

2. Mouse Bite
   Confidence: [model output]
   Location: [bounding box coordinates]

Total Detected Defects: 2

The values above are illustrative. Actual results must come from model inference.

---

9. AI-Assisted Defect Explanation

A normal object-detection model identifies visual defects, but it does not necessarily explain their practical consequences.

This is where our AI-assisted explanation module becomes useful.

Proposed processing

YOLO Detection
      |
      v
Structured Detection Data
      |
      v
Rule-Based Inspection
      |
      v
Llama 3.2 1B Instruct
      |
      v
Human-Readable Explanation

Example

Suppose YOLO detects an Open Circuit.

The system can provide an explanation such as:

Detected defect: Open Circuit

Possible effect: An interruption in the conductive path may prevent the intended electrical signal or current from reaching the connected circuit.

Recommended action: Hold the board for further electrical continuity testing and operator inspection.

This makes the system more informative than a simple bounding-box detector.

Why use a rule engine?

The rule engine provides structured inspection logic.

For example:

- Critical defects can trigger HOLD or REJECT recommendations.
- Uncertain detections can trigger manual review.
- Boards without detected defects can proceed to the next inspection stage, subject to other required quality checks.

The final decision must be based on validated manufacturing rules, not unrestricted language-model output.

---

10. Raspberry Pi Deployment

The purpose of using Raspberry Pi is to move the inspection application from a development environment towards a real edge-computing platform.

Deployment workflow

MODEL DEVELOPMENT
       |
       v
YOLO TRAINING
       |
       v
TRAINED MODEL CHECKPOINT
       |
       v
MODEL DEPLOYMENT
       |
       v
RASPBERRY PI
       |
       v
IMAGE INPUT
       |
       v
LOCAL INFERENCE
       |
       v
DETECTION OUTPUT
       |
       v
INSPECTION APPLICATION

Current implementation

The following activities have been achieved in the prototype:

- Raspberry Pi hardware setup and connectivity.
- Linux-based access and project environment setup.
- Software installation and configuration.
- Running the PCB inspection workflow.
- Processing a sample PCB image.
- Obtaining sample defect-detection output.

The complete model configuration, benchmark results, and exact package versions should be documented from the final working environment.

Edge Computing Advantage

Instead of sending every PCB image to an external cloud server, an edge-based system can process images locally.

Potential benefits:

- Reduced dependence on internet connectivity.
- Better control over inspection data.
- Reduced image-upload requirements.
- Potentially lower communication latency.
- More flexible deployment in industrial environments.

Local processing does not automatically guarantee faster inference; actual speed depends on the model, hardware, image size, and runtime optimization.

---

11. User Interface and Inspection Output

A useful inspection system should not only perform detection but also present results in a format that operators can understand.

Our intended interface is divided into the following sections:

A. Image Input Interface

Allows the operator to provide:

- PCB image.
- Camera-captured image.
- Previously stored inspection image.

B. Detection Interface

Displays:

- Original PCB image.
- Annotated PCB image.
- Detected defect names.
- Bounding-box locations.
- Confidence values.

C. AI Explanation Interface

Displays:

- Defect description.
- Possible electrical impact.
- Suggested inspection action.
- Operator review recommendation.

D. Inspection Summary

A structured report can contain:

Field| Description
Inspection ID| Unique inspection reference
Input image| Inspected PCB image
Defect count| Number of detected defects
Defect classes| Identified categories
Confidence| Model prediction confidence
Inspection status| Suggested workflow status
Explanation| Human-readable defect information

Suggested future dashboard

A future web-based interface can be developed using Streamlit or Gradio.

It could contain:

- Upload PCB Image.
- Capture from Camera.
- Run Inspection.
- View Detection Results.
- AI Explanation.
- Download Inspection Report.
- Inspection History.

These are proposed interface improvements and should be marked complete only after implementation.

---

12. Present Development Status

We distinguish between completed prototype work, ongoing integration, and future development.

Module| Current status
Project concept and architecture| Established
PCB defect dataset research| Completed
YOLO-based defect detection workflow| Implemented
Raspberry Pi setup| Completed for prototype
Raspberry Pi connectivity| Working
Sample PCB image processing| Demonstrated
Sample defect detection| Demonstrated
Camera-based continuous inspection| Further validation required
Structured inspection reporting| Development / integration
Llama explanation module| Integration in progress
ExecuTorch deployment| Requires final end-to-end validation
Complete operator dashboard| Future development
Custom RTL accelerator| Future development

Note: A working sample-image detection demonstration is not equivalent to a fully validated industrial inspection system. Accuracy, inference time, model robustness, and complete offline operation must be measured before making those claims.

---

13. What Makes This Project Different?

Our project aims to go beyond conventional image classification.

Conventional detection system| Our proposed intelligent inspection system
Detects visual defects| Detects and explains defects
Displays defect labels| Displays labels and locations
Focuses on model output| Adds structured inspection information
May depend on external computing| Targets local edge inference
Limited operator explanation| AI-assisted understandable reports
Detection-focused| Detection plus quality-control assistance

The main contribution is the integration of computer vision, edge deployment, structured inspection logic, and AI-assisted explanations into one modular workflow.

---

14. Advantages

1. Automated Inspection: Reduces dependence on manual visual checking for the targeted defect categories.

2. Multiple Defect Detection: Object detection can identify multiple defects in one image.

3. Defect Localization: Bounding boxes help operators understand where a defect appears.

4. Edge Computing: Enables a path towards local inspection without requiring continuous cloud connectivity.

5. AI-Assisted Explanation: Converts technical detection results into understandable descriptions.

6. Modular Architecture: Detection, reporting, and explanation modules can be independently upgraded.

7. Scalability: Additional defect categories and inspection rules can be introduced through further training and development.

8. Hardware Flexibility: The software architecture can be adapted to more powerful edge processors and accelerators.

---

15. Limitations and Challenges

A technically honest project should also explain its limitations.

Limitation| Explanation
Small defects| Tiny defects may be missed depending on image resolution
Lighting variations| Different i
