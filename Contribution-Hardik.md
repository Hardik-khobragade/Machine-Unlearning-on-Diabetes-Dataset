# Project: Machine Unlearning  
**Role:** Team Member (Unofficial Team Lead - Team 2)

## Team Structure  
- **Team 1:** Responsible for data preprocessing, shard splitting, and indexing of user IDs.  
- **Team 2 (My Team):** Focused on unlearning implementation and retraining workflow.

## Contributions  

1. **Implemented `remove_user_id.py`**  
   - Developed a script to identify and remove specific user IDs from each data shard.  
   - Ensured data integrity and maintained balanced shard distribution after removal.  

2. **Developed `train_all_shards.py`**  
   - Implemented training using a **Logistic Regression** model for each shard independently.  
   - Combined all trained shard models into a single unified model.  
   - Validated retraining efficiency and ensured consistent model performance after unlearning.  

## Impact  
- Automated the user unlearning process for compliance and privacy validation.  
- Simplified retraining through a modular shard-based approach.  
- Improved reproducibility and scalability by scripting both unlearning and retraining stages.
