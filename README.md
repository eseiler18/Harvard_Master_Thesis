# Master Thesis

## Physics-Informed Neural Networks for Solving Ordinary Differential Equations in Stiffness Regimes: A Multi-Head Architecture and Transfer Learning Approach

### Author 
**Emilien Seiler**  
-  *Master's student at EPFL, Department of Mathematics, Computational Science and Engineering*  
-  *Visiting student in Harvard School Of Engineering And Applied Sciences*

### Advisors 
**Prof. Pavlos Protopapas**  
*Scientific Program Director and Lecturer, Harvard School Of Engineering And Applied Sciences*  
**Prof. Hesthaven S Jan**  
*Director of Chair of Computational Mathematics and Simulation Science, EPFL*  

## Abstract
This thesis explores the use of Physics Informed Neural Networks to solve stiff linear and non-linear ordinary differential equations. Physics Informed Neural Networks integrate governing equations into neural network structures via automatic differentiation, and stiffness introduces difficulty in encoding the solution during training. Behaviors such as rapid transient phases can be particularly challenging to encode.
We extend previous methodologies to tackle stiffness with a novel transfer-learning-based approach. The approach consists of training a multi-head architecture in a non-stiff regime and transferring it to a stiff regime without the need for retraining the model. The present approach is compared to both vanilla Physics-Informed Neural Networks and numerical methods, such as RK45 and Radau methods. Experiments were conducted on two linear examples and extended to one non-linear example using perturbation theory.
Our analysis indicates that transfer learning from a less stiff regime can be used to compute a solution in a stiffer regime, thereby reducing the complications associated with training in stiff systems. Transfer learning has been achieved on the Duffing equation, from a stiffness ratio less than 100 to a regime where the stiffness ratio is greater than 5000, outperforming the vanilla PINNs model. The average absolute error has been maintained at less than 10^-3. The approach provides competitive computational efficiency, especially when modifying initial conditions or force functions within a stiff domain. It is 70 and 7 times faster than using the Radau method for investigating linear and non-linear equations, respectively.
However, challenges persist in transferring to very stiff regimes too far from the training one and handling all forms of non-linear equations.

## Structure
There are two branch in this repository.  
The `main` one is about the content of the thesis and the `research` one is the all resherch on this project   
This is the structure of the `main` branch of the repository:

- `Deliverables`: 
  - `Thesis_Report.pdf`:
  - `Poster.pdf`
  - `Defense.pptx`
- `src`:
  - `linearODE`: linear ODE examples
    - `DHO`: Damped Harmonic Oscillator
       - `train.ipynb`:
       - `transfer_learning.ipynb`
       - `change_IC_force.ipynb`
    - `SOncFF`: System of ODE with not constant Force Function
       - `train.ipynb`
       - `transfer_learning.ipynb`
       - `change_IC_force.ipynb`
  - `nonlinearODE`: linear ODE example
    - `Duffing`: Duffing Equation
      - `train.ipynb`
      - `LBFGS_transfer_learning.ipynb`
      - `perturbation_transfer_learning.ipynb`
      - `change_IC_force.ipynb`
  - `src`: .py file with source code
    - `load_save.py`: save and load model helpers
    - `loss.py`: loss calculation
    - `model.py`: model pytorch implementations 
    - `train.py`: training code
    - `transfer_learning.py`: one shot linear transfer learning code
    - `nonlinear_transfer_learning.py`: one shot nonlinear transfer learning with perturbation code
    - `utils_plot.py`: helpers for vizualization
  - `result_history`: $MAE$ and $MaxAE$ of all methods with increasing stiffness
    - `DHO_Error_Trained.json`: DHO metrics for Learning in the Stiff regime
    - `DHO_Error_Transfer.json`: DHO metrics for Transfering into Stiff regime
    - `SOncFF_Error_Trained.json`: SOncFF metrics for Learning in the Stiff regime
    - `SOncFF_Error_Transfer.json`: SOncFF metrics for Transfering into Stiff regime
    - `Duffing_Error_Trained.json`: Duffing metrics for Learning in the Stiff regime
    - `Duffing_Error_TrainedIter.json`: Duffing metrics for LBFGS Transfering into Stiff regime
    - `Duffing_Error_Transfer.json`: Duffing metrics for Transfering with perturbation into Stiff regime
    - `plot_general.ipynb`: Vizualise $MAE$ and $MaxAE$ of all the approachs
  - `model_history`: pretrain model and traning of history (git ignore)

