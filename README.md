# Lead Conversion Model 

This project is an end to end AI driven machine learning application designed to predict the likelihood of a sales lead converting, enabling more effective prioritization and decision-making. 
The model is built using **LightGBM** for ranking and classification and can be retrained dynamically using feedback data. MongoDB is utilised for storing the data, results and feedback. 
The project also has a a **Node.js + HTML + JavaScript** frontend to display and update lead statuses interactively.

### Project Structure 
- Lead_Conversion_Model/ : Machine learning pipeline that trains a LightGBM model on lead data, ranks leads, and pushes predictions to MongoDB.
- ranked_leads/ : Frontend + backend dashboard built with HTML, CSS, and Node.js for viewing, searching, and updating lead states.
- feedback loop : integrated within the two folders and automatically listens for state changes (“Converted” / “Rejected”) in MongoDB and retrains the
  model when ≥5% new feedback data is collected.

### Project Flow 
```text
MongoDB (mock_data_leads_new)
       ↓
Lead_Conversion_Model.ipynb → results_new.csv
       ↓
MongoDB (results_new)
       ↓
Ranked Leads UI (server.js + index.html)
       ↓ user updates state (Converted / Rejected)
MongoDB Change Stream → feedback_watcher.js
       ↓
MongoDB (leads_feedback)
       ↓
retrain_with_feedback.py (triggered automatically)
       ↓
Updated Model + ranked_predictions_all.csv
       ↓
MongoDB (ranked_results)
       ↓
UI shows updated rankings



