# Explainable AI (XAI) - SHAP and LIME Techniques

A comprehensive implementation demonstrating Explainable AI (XAI) techniques using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) on an Artificial Neural Network (ANN) trained on the classic Iris dataset. This project provides both theoretical understanding and practical implementation of model interpretability techniques.

## What is XAI?

Explainable AI (XAI) is basically a set of tools, techniques and frameworks that help humans understand and trust the decisions made by machine learning (ML) models. As ML models (especially deep learning) become more and more complex, they often operate as 'black boxes' where the decision-making process is opaque. XAI techniques aim to make their outputs more interpretable, transparent, and trustworthy. 
SHAP and LIME are two popular model-agnostic (can be applied to any model; just observing the inputs and outputs, like a 'black box') XAI techniques used to interpret machine learning models. They help us understand 'why' a model made a certain prediction.

SHAP (SHapley Additive exPlanations), specifically Shapley values, help in assigning each feature a positive/negative importance score (SHAP value), for a particular prediction. Originated from cooperative game theory, Shapley values fairly distribute the contribution of each player (feature) to the final outcome (prediction). We can visualize SHAP values with different kinds of plots.

LIME (Local Interpretable Model-Agnostic Explanations) explains a prediction by fitting a simple, interpretable 'surrogate' model (like linear regression) around the neighborhood of the input instance. It perturbs the input slightly (such as, changing the feature values a little) and observes changes in predictions to interpret the local behaviour using the surrogate model, which helps us to understand feature importance. And just like SHAP, we use different graphs/plots to visualize this too.

## Project Flow

1. Firstly, we build and train an ANN for Iris Classification. We use a 4→8→6→3 neuron dense-layered architecture with ReLU and Softmax activation.
2. Then after successful training, we move on to the XAI analysis. We plot the following SHAP and LIME plots:

#### 1: SHAP Analysis
1. Global Feature Importance:
   - Bar plots showing overall feature importance
   - Summary plots with value distributions
2. Class-Specific Analysis:
   - Individual SHAP analysis for each Iris species
   - Comparative visualizations across classes
3. Feature Dependence and Interaction Analysis:
   - Dependence plots for all the features
   - Interaction effects between important features
4. Individual Prediction Analysis:
   - Waterfall plots for specific instances
   - Force plots showing contribution breakdown
   - Decision plots showing prediction paths

#### Part 2: LIME Analysis
1. Individual Instance Explanations:
   - Detailed explanations for multiple test instances
   - Comparison with actual vs predicted classes
2. Stability Analysis:
   - Multiple LIME runs on the same instance
   - Consistency evaluation of explanations
3. LIME vs SHAP Comparison:
   - Side-by-side feature importance comparison

## How to Run

1. Make sure Python 3.8+ is installed in your system.
2. Clone this repository on your local machine.
3. Open and run the first cell of `XAI.ipynb` Jupyter Notebook to install the required dependencies.
4. Then, run the second cell to train the ANN on the Iris dataset, and then the third cell for the SHAP and LIME analysis.

## Acknowledgements

Special thanks to the Scientific Analysis Group (SAG), Defence Research and Development Organization (DRDO), for supporting this work.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
