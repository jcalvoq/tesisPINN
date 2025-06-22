# ThesisPINN

This repository provides resources and examples for the implementation of Physics-Informed Neural Networks (PINNs) methods. Used for solving ordinary differential equations (ODEs) and delay differential equations (DDEs). 

## 🔧 Prior Requirements 

In order to execute this project, it is necessary to have the following programs in place beforehand:

* Python 3.12.0+
* pip 25.1.0+

## 📂 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── ExpDelay           <- Results and code used for exponential delay differential equation.
│    └── Results                 <- Folder including images and weights for the DDE.
├── Logistic Delay a0.3     <- Results and code utilized for the logistic DDE with a = 0.3
│    ├── Results including the additional condition     <- Folder with the images and weights including the additional 
│    │                                                     condition to the loss function.
│    └── Results without the additional conidition      <- Folder with the images and weights without using the additional condition.
├── Logistic Delay a1.4     <- Results and code utilized for the logistic DDE with a = 1.4
│    ├── Results including the additional condition     <- Folder with the images and weights including the additional 
│    │                                                     condition to the loss function.
│    └── Results without the additional conidition      <- Folder with the images and weights without using the additional condition.
├── Logistic           <- Results and code used for logstic differential equation without delay.
│    ├── Results with additional condition              <- Folder including the results of the logistic equation
│    │                                                     with the additional condition.
│    └── Results without the additional condition       <- Folder including the results of the logistic equation
│    │                                                     without the additional condition on the loss function.
├── Lotka-Volterra Delay        <- Results and code used for the Lokta-Volterra equations with delay.
│    └── Results                        <- Folder including the images and weights for the system of delay differential equations.
├── Lotka-Volterra        <- Results and code used for the Lokta-Volterra equations without delay.
│    └── Results                        <- Folder including the images and weights for the system of differential equations.
├── Simple Example        <- Results and code used for the simple example with known solution.
│    └── Results                        <- Folder including the images and weights for the ordinary differential equation.
├── Van der Pol           <- Results and code used for the Van der Pol oscillator.
└──  └── Results                        <- Folder including the images and weights for the Van der Pol oscillator.
```

## 📥 Clone Project

To clone the project to your computer, run the following command line:

```bash
git clone https://github.com/jcalvoq/thesisPINN
```

To validate that the project has been cloned correctly, run the following commands to verify that you have the latest version of the project:

```bash
cd thesisPINN
git status
```

## 🐍 Virtual Environment Creation

To run the project, the native Python `venv` option was used. First, navigate to your project folder.

If you want to do it manually, you can create it with the following commands. It's recommended to create this environment in the project folder:

```bash
cd C:\path\to\project
python -m venv <Virtual_Environment_Name>
```

To activate the environment, use the following command:

```bash
# Windows
.\Virtual_Environment_Name\Scripts\activate

#Linux
source Virtual_Environment_Name/bin/activate
```

To deactivate the virtual environment, use the command:

```bash
deactivate
```

## 📦 Dependency Installation

To install the necessary dependencies, you can use the command:

```bash
pip install -r requirements.txt
```

## ▶️ How to Use

For the correct usage of this code it is necessary to first run the Python file named after the differential equation, which saves the weights and the values of the loss function in storage. If it is desired to plot the approximations use the plot Python file, which will compare the approximations to Euler's or RK4 method. If it is desired to plot the values of the loss function use the plot loss Python file.

**Example usage:**

For the example corresponding to the exponential equation with delay first run this Python file

```bash
python ExpDelay/ExponentialDelay.py
```

For plotting the solutions use the following file

```bash
python ExpDelay/plotDDE.py
```

For plotting the loss function values throughout the runtime use the following file

```bash
python ExpDelay/plotLoss.py
```

## 👍 Acknowledgements

The author would like to acknowledge **ACARUS** from the **Universidad de Sonora**, for providing their support and facilities during the numerical computations.