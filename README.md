# Lab 2: Hamiltonian Monte Carlo

In this lab, you'll implement Hamiltonian Monte Carlo (HMC) from scratch using JAX and compare it to a basic random walk Metropolis sampler.

## Learning Objectives

- Understand the mechanics of the leapfrog integrator
- Implement HMC with Metropolis-Hastings-Rosenbluth correction
- Compare sampling efficiency between random walk and HMC
- Gain intuition for HMC hyperparameters (step size, trajectory length)

## Getting Started

### 1. Accept the assignment

Click the GitHub Classroom link shared by the instructor. This creates your own private copy of this repository under your GitHub account.

### 2. Clone your repository

Open a terminal and run:

```bash
git clone https://github.com/bu-ds595/lab-02-hmc-YOUR_USERNAME.git
```

Replace `YOUR_USERNAME` with your actual GitHub username.

Then navigate into the folder:

```bash
cd lab-02-hmc-YOUR_USERNAME
```

### 3. Install dependencies

From inside the lab folder, run:

```bash
pip install -r requirements.txt
```

If you get permission errors, try `pip install --user -r requirements.txt`.

### 4. Open the notebook

**Option A: VS Code**
1. Open VS Code
2. File → Open Folder → select the lab folder
3. Open `lab-02-hmc.ipynb`
4. If prompted, install the Python and Jupyter extensions

**Option B: JupyterLab**
```bash
jupyter lab
```
Then click on `lab-02-hmc.ipynb` in the file browser.

**Option C: Google Colab**

Upload the notebook to [Google Colab](https://colab.research.google.com/). You'll need to install dependencies by adding a cell at the top:
```python
!pip install jax jaxlib arviz
```

## Exercises

Complete the `TODO` sections in `hmc.py`:

1. `leapfrog(theta, rho, grad_log_prob_fn, epsilon, L)` — The leapfrog integrator
2. `hmc_step(key, theta, log_prob_fn, epsilon, L)` — Single HMC transition with Metropolis correction

**Tips:**
- The `hmc.py` file contains detailed hints for each step
- If you're unsure about JAX patterns (random keys, etc.), look at the random walk implementation in the notebook — it uses the same structure
- The notebook provides tests to verify your implementation

## Submitting Your Work

Save your notebook and `hmc.py`, then commit and push your changes. From the lab folder:

```bash
git add lab-02-hmc.ipynb hmc.py
git commit -m "Complete lab 2"
git push
```

If `git push` asks for credentials, enter your GitHub username and a [personal access token](https://github.com/settings/tokens) (not your password).

You can push multiple times—only the final version at the deadline will be graded.

## Grading

This lab is graded based on **correct implementation** of the two functions in `hmc.py`:

1. **`leapfrog`** — Must correctly implement the leapfrog integrator (half-step, full-step, half-step pattern)
2. **`hmc_step`** — Must correctly implement the full HMC transition (momentum sampling, Hamiltonian computation, leapfrog integration, and Metropolis accept/reject)

The notebook includes test cells that will help you verify your implementation before submitting.
