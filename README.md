Take a look at `correct_hamiltonian` pluto notebook to get started, 

clone this repository and cd into it
```
git clone https://github.com/ArbitRandomUser/correction_learning_scripts
cd correction_learning_scripts
```
Init the julia environment

`julia --project=.`

```
julia> using Pkg
julia> Pkg.instantiate()
```
For good measure restart julia again this time with threading
`julia --project=. --threads=6`

choose the thread number according to the number of cpu. more threads require more ram , if you run into an oom try with lower number of threads.
memory usage for training roughly peaks at 5 to 6 gb with 6 threads.

```
julia> using Pluto
julia> Pluto.run()
```
The browser should open showing you the pluto interface, 
You should be able to find the notebook called `correct_hamiltonian` by clicking in the "Enter path or URL" text box shown below

<img width="573" height="176" alt="fig" src="https://github.com/user-attachments/assets/1591f518-794d-43dc-85ba-7f07e81d9bb9" />

