---
layout: page
title: "ToolVisionLM: Enhancing Vision-Language Models for Industrial Safety [Ongoing]"
description: A comprehensive evaluation framework for vision-language models in technical domains with focus on industrial tool recognition and safety guidance
img: assets/img/project-7/VLM.png
importance: 7
category: work
related_publications: false
---

### 1. Overview

ToolVisionLM is an innovative research project that explores the application of Vision-Language Models (VLMs) to specialized technical domains, with a focus on industrial tool recognition, usage instruction, and safety guidance. While VLMs have demonstrated remarkable capabilities in general visual understanding tasks, their application to specialized domains remains limited. This project addresses this gap by developing a comprehensive evaluation framework for assessing VLM performance in industrial settings where proper tool handling directly impacts workplace safety and operational efficiency.

<div style="text-align: center;">
    <img src="/assets/img/project-7/VLM.png" alt="Vision-Language Model Architecture" style="width: 90%; max-width: 800px;">
    <p><em>General architecture of vision-language models for tool recognition showing the image encoding, vision-language fusion, and task-specific outputs</em></p>
</div>

---

### 2. Technical Approach

#### 2.1 Dataset Preparation

The project features a meticulously curated dataset spanning 13 core tool categories:

- **Primary Tools**: Wrenches, hammers, pliers, screwdrivers (most common industrial tools)
- **Secondary Tools**: Bolts, dynamometers, testers, tool boxes
- **Measurement Tools**: Tape measures, calipers
- **Power/Cutting Tools**: Ratchets, drills, saws

<div style="text-align: center;">
    <img src="/assets/img/project-7/data_graph.png" alt="Dataset Distribution" style="width: 90%; max-width: 800px;">
    <p><em>Distribution of industrial tool categories across the three datasets used in this project, showing the comprehensive coverage across tool types</em></p>
</div>

Data aggregation was performed across three distinct sources to ensure comprehensive coverage:
1. A specialized tool dataset with fine-grained classifications (Dataset 1)
2. A general tool dataset with broad category coverage (Dataset 2)
3. A high-quality dataset with diverse tool representations (Dataset 3)

This consolidation produced a balanced dataset with over 20 images per tool category (minimum) and more than 1,000 total images, enabling robust model training and evaluation.

#### 2.2 Model Selection

The project evaluates several state-of-the-art Vision-Language Models to benchmark their performance on specialized tool recognition tasks:

- **Qwen2-VL-7B-Instruct**: Alibaba's 7B parameter instruction-tuned multimodal model
- **Phi-3-vision-128k-instruct**: Microsoft's vision-capable model with extended context window
- **Llama-3.2-11B-Vision-Instruct**: Meta's 11B parameter multimodal model
- **SmolVLM**: A lightweight VLM optimized for efficient deployment
- **PaliGemma**: Google's multimodal model built on the Gemma architecture

<div style="text-align: center;">
    <img src="/assets/img/project-7/flowchart.png" alt="Project Pipeline" style="width: 75%; max-width: 700px;">
    <p><em>ToolVisionLM pipeline showing dataset preparation, model selection, parallel approach strategies, and evaluation methodology</em></p>
</div>

#### 2.3 Dual Enhancement Approach

The project implements two parallel enhancement strategies to optimize VLM performance for tool-related tasks:

##### Fine-tuning Strategy
- Dataset-specific model adaptation using controlled fine-tuning procedures
- Parameter-efficient fine-tuning techniques to preserve general capabilities
- Domain-specific prompt engineering optimized for tool recognition tasks
- Balanced training across all tool categories to prevent bias

##### RAG (Retrieval-Augmented Generation) Approach
- Development of a specialized knowledge base containing detailed tool specifications
- Implementation of efficient embedding and retrieval mechanisms
- Context-aware information retrieval to enhance model responses
- Hybrid retrieval approaches combining visual and textual information

---

### 3. Evaluation Framework

The evaluation methodology incorporates multiple dimensions to provide a comprehensive assessment of model performance:

#### 3.1 Recognition Accuracy
- **Precision, Recall, and F1 Scores**: Traditional metrics for identification accuracy
- **Cross-Category Confusion Analysis**: Identifying common misclassification patterns
- **Challenging Scenario Testing**: Performance under occlusion, unusual angles, and poor lighting

#### 3.2 Instruction Quality
- **Completeness**: Evaluating whether responses include all critical information
- **Correctness**: Assessing technical accuracy of usage instructions
- **Clarity**: Measuring how understandable the instructions are for end-users

#### 3.3 Safety Guidance
- **Safety-Critical Information**: Identifying presence of essential safety warnings
- **Hazard Recognition**: Evaluating model's ability to identify potential dangers
- **Protective Equipment Recommendations**: Checking for appropriate safety gear suggestions

The framework employs both automated metrics and targeted qualitative analysis to identify strengths, weaknesses, and potential improvement areas for each model.

---

### 4. Preliminary Findings

While the project is ongoing, initial explorations have yielded several promising insights:

- VLMs demonstrate varying capabilities in recognizing tool categories, with performance generally correlating with model size
- Fine-grained tool identification (distinguishing subtypes like Phillips vs. flat-head screwdrivers) remains challenging for most models
- All models show significant improvements when augmented with either fine-tuning or RAG approaches
- Safety guidance quality varies substantially, with larger models providing more comprehensive safety information
- The RAG approach shows particular promise for enhancing safety instruction accuracy without extensive model adaptation

---

### 5. Applications and Impact

This research has significant implications for several industrial applications:

- **Safety Training**: Enhanced VLMs could provide on-demand tool usage guidance in industrial settings
- **Maintenance Support**: Interactive systems could assist technicians with proper tool selection and usage
- **Quality Assurance**: Automated systems could verify correct tool usage in manufacturing processes
- **Accessibility**: Improved visual recognition systems could make technical work more accessible

By addressing the limitations of current VLMs in specialized domains, this project aims to bridge the gap between general visual understanding and domain-specific technical knowledge, ultimately enhancing workplace safety and efficiency.

---

### 6. Future Directions

As the project progresses, several promising avenues for future work have been identified:

- Expanding the tool dataset to include more specialized industrial categories and rare tools
- Incorporating multimodal feedback mechanisms to improve instruction clarity and correctness
- Developing benchmark datasets for other technical domains using similar methodologies
- Exploring lightweight deployment options for resource-constrained industrial environments
- Integrating real-time safety monitoring capabilities into the system

---

### 7. Project Repository

[ToolVisionLM on GitHub](https://github.com/Srecharan/ToolVisionLM)