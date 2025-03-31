📌 Cancer Detection Using Artificial Neural Networks (ANN)
🔗 Streamlit Dashboard: 055056 ANN Dashboard
🏆 Project Details
This project aims to predict cancer malignancy using a deep learning model built with Artificial Neural Networks (ANN). The model is trained on a structured dataset containing patient tumor attributes. The primary focus is on hyperparameter tuning to enhance model performance. The dataset used is cancer_detectionANN.csv.
👥 Contributors
Sejal Raj (055041)
Vidushi Rawat (055056)________________________________________
🔑 Key Activities
•	Data Preprocessing: Handling missing values, encoding categorical variables, and normalizing numerical attributes.
•	Model Development: Constructing an ANN with tunable hyperparameters for optimization.
•	Hyperparameter Tuning: Investigating the impact of different hyperparameter values on accuracy.
•	Visualization & Insights: Analyzing model performance using loss and accuracy plots.
•	Managerial Interpretation: Extracting actionable insights for healthcare decision-making.
________________________________________
💻 Technologies Used
•	Python
•	TensorFlow & Keras (for ANN modeling)
•	Pandas & NumPy (for data manipulation)
•	Matplotlib & Seaborn (for visualization)
•	Scikit-learn (for preprocessing and evaluation)
•	Streamlit (for interactive dashboard development)
________________________________________
📊 Nature of Data
The dataset consists of structured clinical data with exclusively numerical attributes, representing tumor characteristics extracted from medical imaging.
________________________________________
📌 Variable Information
Feature	Type	Description
ID	Identifier	Unique patient ID
Diagnosis	Binary	0 = Benign, 1 = Malignant
Radius Mean	Continuous	Mean radius of the tumor
Texture Mean	Continuous	Mean texture of the tumor
Perimeter Mean	Continuous	Mean perimeter of the tumor
Area Mean	Continuous	Mean area of the tumor
Smoothness Mean	Continuous	Mean smoothness of the tumor
Compactness Mean	Continuous	Mean compactness of the tumor
Concavity Mean	Continuous	Mean concavity of the tumor
Concave Points Mean	Continuous	Mean concave points of the tumor
Symmetry Mean	Continuous	Mean symmetry of the tumor
Fractal Dimension Mean	Continuous	Mean fractal dimension of the tumor
...	...	... (More tumor characteristics)
________________________________________
🎯 Problem Statements
•	How can ANN models effectively classify tumors as benign or malignant?
•	What impact does hyperparameter tuning have on model accuracy?
•	Can the model achieve high accuracy without overfitting?
________________________________________
🏗️ Model Information
•	Input Layer: Accepts all numerical features.
•	Hidden Layers: Number of layers and neurons customizable via Streamlit UI.
•	Activation Functions: ReLU
•	Dropout Rate: Adjustable to prevent overfitting (0.2)
•	Optimizer Options: Adam
•	Loss Function Options: Binary Cross-Entropy
•	Output Layer: Single neuron with Sigmoid activation for binary classification.
________________________________________
📉 Observations from Hyperparameter Tuning
1️⃣ Number of Hidden Layers
•	1-2 layers: Moderate accuracy (~88%).
•	3-4 layers: Optimal accuracy (~94%) without overfitting.
•	5+ layers: Slight improvement but increased computational cost.
2️⃣ Neurons per Layer
•	10-50 neurons: Stable and consistent training.
•	>50 neurons: Marginal improvement, but risk of overfitting increases.
3️⃣ Activation Functions
•	ReLU: Performs best in hidden layers.
4️⃣ Optimizer Comparison
•	Adam: Best performance, balances speed and accuracy.
5️⃣ Dropout Rate
•	0-0.2: Best accuracy (~94%).
6️⃣ Epochs
•	50 epochs: Sufficient for convergence.
________________________________________
📈 Managerial Insights
🔹 Healthcare Applications
•	This ANN model can help doctors and healthcare professionals make early diagnoses of cancer.
•	Automating tumor classification reduces human error and speeds up treatment decisions.
🔹 Business Value
•	Hospitals and insurance companies can use AI-driven risk assessment for better patient outcomes.
•	Implementing ANN-based diagnostic tools can reduce costs and workloads for radiologists and oncologists.
🔹 Future Improvements
•	Feature Engineering: Extracting more relevant medical features for better accuracy.
•	Hybrid Models: Combining ANN with other machine learning techniques (e.g., XGBoost, Random Forest).
•	Explainability: Using techniques like SHAP values to understand feature importance in predictions.
________________________________________
🚀 Conclusion
This project successfully demonstrates how deep learning can be leveraged for cancer detection. The ANN model achieves ~97.37% accuracy after hyperparameter tuning, making it a powerful tool for medical diagnostics.
