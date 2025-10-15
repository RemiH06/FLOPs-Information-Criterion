![Build with Love](http://ForTheBadge.com/images/badges/built-with-love.svg)

```ascii
███████╗██╗      ██████╗ ██████╗     ██╗ ██████╗
██╔════╝██║     ██╔═══██╗██╔══██╗    ██║██╔════╝
█████╗  ██║     ██║   ██║██████╔╝    ██║██║     
██╔══╝  ██║     ██║   ██║██╔═══╝     ██║██║     
██║     ███████╗╚██████╔╝██║         ██║╚██████╗
╚═╝     ╚══════╝ ╚═════╝ ╚═╝         ╚═╝ ╚═════╝
       by Hex (@RemiH06)          version 0.3.0
```

![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

### General Description
I don't really trust AIC or BIC for some of my models. I understand there are many other criterions out there but I rather create my own for a new kind of comparison that I conceived. It makes a count of FLOPS and uses them in a balance between likelihood, amount of params in the model and FLOPs, each of them can be prioritized with hiperparameters for the formula and with the results of the FIC, models can be chosen over others having in count computing cost.

The final version is a library made with only python implementing a not so huge math investigation.

```diff
- Lib functions work with a few libraries
- pip library has not been made yet
```

## Installation

1. Here I'll place the installation guide, will turn it into a pip package when I find the time.

## Notebooks

- For now, you can compare models through FLOPs within basic_usage.py
- Also take a look for more advanced usages within fic_usages.ipynb
- Feel free to stalk the not so beautiful refining method I slothfuly came up with
- If you care about the math and my investigation, please read concept.ipynb to know more
- My main tests were MNIST (10 different models), I dare to say that was the very trial by fire. Please go take a look to see its implementation in a project and how it's pretty helpful at deciding which model is better at saving energy (or memory)