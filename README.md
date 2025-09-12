# USTC-fuSEP2025

## Gyrokinetic IVP Solver for Trapped Electron Modes 

### IVP-SIM 
#### Description
This code implements a numerical solution of gyrokinetic equations for investigating plasma instabilities caused by trapped electrons. The model includes:

<ul>
<li> Solution of ion and electron distribution function equations </li>

<li> Calculation of electrostatic potential </li>

<li> Analysis of growth rates and oscillation frequencies </li>

<li> Real-time visualization of results </li>

</ul>

#### Quick Start
Some Python packages are required to run simulation. To install all the requirements run in terminal:
```
pip install requirements.txt
```
To demonstrate a single run of the model with the given parameters, an example has been prepared in the ```core.py``` file.

The class can be initialized with the following parameters:

<ul>
<li> <code>ky</code>: Perpendicular wave number </li>
<li> <code>wr</code> : Real part of complex frequency</li> 
<li> <code>wi</code>: Imaginary part of complex frequency (high level)</li>
<li> <code>tau</code>: Temperature ratio Ti/Te</li>
<li> <code>f_trap</code>: Trapped electron fraction</li>
<li> <code>epsn</code>: Normalized density gradient</li>
<li> <code>kz</code>: Parallel wave number</li>
<li> <code>kapt</code>: Normalized temperature gradient</li>
<li> <code>nvx</code>: Number of grid points in parallel velocity space</li>
<li> <code>nvy</code>: Number of fonts in a specific velocity space</li>
<li> <code>dt</code>: Time step size</li>
<li> <code>vxmax</code>: Maximum parallel velocity</li>
<li> <code>vymax</code>: Maximum perpendicular velocity</li>
<li> <code>vymin</code>: Minimum perpendicular velocity</li>
<li> <code>vxmin</code>: Minimum parallel velocity</li>
</ul>

For some input parameters, the default value has been removed to ensure the program runs correctly:
<ul>
  <li>       <code>tau</code>: float = 1.0</li>
  <li>       <code>f_trap</code>: float = 1.0</li>
  <li>       <code>epsn</code>: float = 0.2</li>
  <li>       <code>kz</code>: float = 0.0</li>
  <li>       <code>kapt</code>: float = 0.5</li>
  <li>       <code>nvx</code>: int = 32</li>
  <li>       <code>nvy</code>: int = 64</li>
  <li>       <code>dt</code>: float = 0.02</li>
  <li>       <code>vxmax</code>: float = 5.0</li>
  <li>       <code>vymax</code>: float = 5.0</li>
  <li>       <code>vymin</code>: float = 0.0</li>
  <li>       <code>vxmin</code>: float = -5.0 </li>
</ul>

The run function of the <code>IVPModel</code> class can set the following parameters:
<ul>
<li> <code>nt</code>: Number of time steps (default valuev is 500) </li>
<li><code>plot_results</code>: Whether to plot results after simulation </li>
<li><code>real_time_plot</code>: Whether to show real-time plotting during simulation </li>
<li><code>rk</code>: Runge-Kutta Order (0 for Euler, 1 for RK4) </li>
</ul>

The run function of the model returns the parameters:
<ul>
<li> potential history </li>
<li> distribution functions </li>
<li> runtime </li>
<li> growing rate </li>
<li> frequency </li>
</ul>

To demonstrate the construction of the dependence of the attenuation increment on the model parameters, a test example has been prepared in the test.py file. To draw the graphs, two main(2) functions have been prepared (without input characteristics, parameterization is inside), which can and should be changed.

#### Visualization Notes

Run function supports two visualization modes:
<ol> 
  <li> <code> Real-time plotting </code>: Live plotting during simulation </li>
  <li> <code> Post-processing </code>: Comprehensive plotting after simulation completion </li>
</ol>

To start frame-by-frame rendering of the simulation results, you need to set the <code> real_time_plot = True </code> flag in the run input parameters. However, it is worth understanding that rendering in "real time" will take more time and computer resources.

#### Usage Examples

The run() call with creation of a model and setting the values ​​of flags as run() input params:
```Pyton
    data = np.array(
        [
            [0.1, 0.1277 + 2.8258j],
            [0.2, 0.2487 + 2.8211j],
            [0.5, 0.6369 + 2.6664j],
            [1.0, 1.3108 + 2.2697j],
            [1.5, 2.1182 + 1.9633j],
            [2.0, 3.1047 + 1.9143j],
        ]
    )
    id = 3
    ky = float(data[id, 0])
    wr = float(np.real(data[id, 1]))
    wi = float(np.imag(data[id, 1]))

    model = IVPModel(ky=ky, wr=wr, wi=wi, tau=1.0, epsn=0.2, kz=0.0, kapt=0.5)
    phit, gi, ge, runtime, gamma, omega_r = model.run(nt=500, plot_results=True, real_time_plot=False)
    print(f"runtime={runtime:.2f}s, gamma={gamma:.3f}, omega_r={omega_r:.3f}")
```
