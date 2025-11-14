# Machine Unlearning Project – Team Contributions

---

## Project Manager: Abdul

### Overview
As the project leader, I established and organized multiple teams, creating comprehensive roadmaps, mentorship structures, and implementation guidelines for each team.  
I defined the entire project architecture, guided the teams through technical execution, and ensured alignment across all modules.

### Technical Implementation
- Developed `unlearning.py`, enhancing the SISA unlearning implementation with a precise user mapping system to track which shards contain specific users.  
- Implemented selective retraining logic, updating only the affected shards to eliminate full retraining overhead.  
- Simplified the model pipeline by adopting a unified **Logistic Regression** model, removing dependency on `best_model.json`.  
- Reorganized the directory structure under `sisa_data/` for improved maintainability.  
- Replaced hard-coded user IDs with dynamic runtime input for flexible and efficient unlearning operations.  
- Implemented shard-level unlearning functions to update affected shards while preserving overall model consistency.  

### Impact
- Improved computational efficiency by avoiding unnecessary retraining.  
- Enhanced workflow clarity, flexibility, and SISA compliance.  
- Strengthened project organization and maintainability across all teams.

---

## Team 1: Gauri

### Project Contribution: SISA Data Preprocessing Pipeline

My role in Team 1 focused on implementing the **initial data preprocessing stage (Cell 1)**, ensuring the dataset was fully prepared before entering the SISA sharding and splitting pipeline.

### Preprocessing Functions & Structure
- **User ID Detection:**  
  Implemented a helper function to automatically detect the user-ID column by checking multiple common naming patterns, ensuring flexible dataset compatibility.
- **Dataset Preparation (Cell 1):**  
  Created the `prepare_dataset()` function to load, clean, encode, and normalize data for consistent downstream processing.
- **User ID Handling:**  
  Added a fallback mechanism to generate a new `user_id` column when missing, ensuring consistent identification across all records.
- **Index Assignment:**  
  Introduced an index column to maintain row-level reference throughout the SISA pipeline.

### Outcome
- Standardized input data for all subsequent operations.  
- Ensured dataset integrity, encoding, and uniform identification for sharding and unlearning tasks.

---

## Team 1: Dhruv

### Project Contribution: SISA Data Preparation Pipeline

My role in Team 1 centered on executing the **final stage of the data preparation pipeline (Cells 3 & 4)** and finalizing the **core user mapping structure (Cell 2)**.

### Implementation & File Delivery
- **User Mapping Refinement:**  
  Enhanced the mapping logic in Cell 2 to provide row-level traceability in `user_mapping.json`.  
- **Pipeline Execution (Cells 3 & 4):**  
  Responsible for complete execution of the `main()` function, configuring shard and split generation.  
- **File Management:**  
  Managed file-saving logic and successfully generated final split files (e.g., `shard_0_split_0.csv`) and `user_mapping.json`.  
- **Quality Check:**  
  Conducted a validation run verifying correct file generation and mapping for 20 users, ensuring accurate partitioning.

### Outcome
- Delivered a reliable, traceable dataset ready for shard-level training and unlearning.  
- Verified output integrity and ensured readiness for Team 2’s model training stage.

---
### Team 1: Aryan 
## Project Contribution Summary: SISA Sharding & User Mapping
---
## Overview

### My contribution : focused on developing the core *SISA (Sharded, Isolated, Slice-based Aggregation) splitting logic* and constructing an advanced, auditable *user-to-data mapping system*. This system is essential for enabling shard-level training and precise machine unlearning.

##  Key Implementation Highlights

## 1. Core SISA Logic Development
* Implemented the complete *SISA splitting algorithm (Cell 2)*, covering:
    * Randomized dataset shuffling.
    * Accurate *shard* and *split* creation.
    * User-level tracking throughout the partitioning process.
* Ensured accurate segmentation for downstream SISA-style training methodologies.

### 2. Auditable User Mapping System
* Designed and generated a comprehensive **user_mapping.json** file.
* This structure tracks the exact location of every user's data across *shards* and *splits*, along with their original row indices.
* *Outcome:* Provided precise data traceability, which is fundamental for future auditing and unlearning workflows.

### 3. Integration & Stability
* Achieved *complete synchronicity* with *Cell 1 (Preprocessing)* by aligning with generated user IDs and preserving index order.
* Implemented robust *error handling* to safeguard against common failure points, such as empty shards, empty splits, or uneven user distribution.

---

##  Project Outcome

A *fully functional and robust SISA splitting mechanism* was delivered, accompanied by a detailed and auditable *user-location map*. This output provides Team 2 with the reliable, well-structured data foundation required for:

1.  *Efficient Downstream Training*
2.  *Data Auditing*
3.  *Precise Machine Unlearning Phases*

## Team 2: Hardik

### Project Contribution: Machine Unlearning Implementation and Retraining Workflow

### Contributions
1. **Implemented `remove_user_id.py`**  
   - Designed a script to locate and remove specific user IDs from each data shard.  
   - Ensured shard consistency and maintained balanced distributions after ID removal.  

2. **Developed `train_all_shards.py`**  
   - Implemented training using a **Logistic Regression** model for each shard independently.  
   - Combined all trained shard models into a single unified model.  
   - Validated retraining efficiency and confirmed consistent performance after unlearning.

### Impact
- Automated the unlearning process for compliance and privacy assurance.  
- Simplified retraining through modular shard-based design.  
- Enhanced reproducibility and scalability for future data removal scenarios.

---

## Team 3: Aman 

### Project Contribution: Model Validation, Student Model Distillation & Integration Testing

### Contributions

1. **Comprehensive Testing & Validation**  
   - Conducted extensive testing on Team 2's unlearning implementation (`remove_user_id.py` and `train_all_shards.py`).  
   - Validated the integrity of the SISA ensemble after user removal, ensuring no data leakage or inconsistencies.  
   - Performed edge-case testing to verify correct handling of multi-shard user distributions and boundary conditions.

2. **Hard-Label Student Model Implementation in `unlearning.py`**  
   - Integrated a **hard-label distillation framework** using the SISA ensemble to generate hard predictions for training a compact student model.  
   - Implemented `train_student_model()` function that collects hard predictions across all shards and trains a unified Logistic Regression student model.  
   - Added `evaluate_student_model()` to measure accuracy and ensure the student model maintains performance post-unlearning.

3. **Workflow Enhancement & Performance Analysis**  
   - Preserved the original Team 2 logic while seamlessly embedding the student model training pipeline after unlearning operations.  
   - Designed comparative accuracy reporting to track the performance delta between initial SISA accuracy and post-unlearning student model accuracy.  
   - Collaborated with team members to ensure smooth integration without disrupting existing shard-level retraining mechanisms.

### Impact
- Ensured robustness and reliability of the unlearning pipeline through rigorous testing.  
- Introduced hard-label distillation to create a lightweight, deployable model while maintaining SISA's unlearning benefits.  
- Enabled transparent performance tracking, providing clear insights into the trade-offs between model efficiency and accuracy post-unlearning.

---
