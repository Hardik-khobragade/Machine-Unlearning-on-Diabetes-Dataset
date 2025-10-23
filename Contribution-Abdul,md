# My Contribution

As the project leader, I established and organized multiple teams, creating comprehensive roadmaps, mentorship structures, and implementation guidelines for each team. 
I defined the entire project architecture and explained what exactly needed to be constructed while guiding the teams through the technical approach and execution strategy.

## Technical Implementation

Technically, I developed `unlearning.py`, where I improved the SISA unlearning implementation by creating a user mapping system that precisely tracks which data shards contain particular users.
I did this by only retraining affected shards, which in turn did away with the need to go through all of them during unlearning.
In this, I simplified the model pipeline to use only LogisticRegression, which in turn removed the `best_model.json` selection process, and I did this for a smooth workflow. 
I reorganized the directory structure under `sisa_data/` for better project management and maintainability.
Hard-coded user IDs I replaced with dynamic user input for runtime unlearning, which in turn made the system more flexible. 
In `unlearning.py`, I implemented unlearning function which updates only the affected shards also, I maintained proper user mapping, which in turn made the process efficient and SISA compliant. 
As a whole these improvements I made very precise and did away with retraining overhead.
