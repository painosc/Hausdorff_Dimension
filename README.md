# Hausdorff_Dimension
Computing the dimension using positive operator approach. 
It is well know that the Hausdorff dimension of Continued fractions can be analysed using the Perron-Frobenius Operators and to estimate the Haudorff dimension, one has to find a matrix A_s and B_s such that the operator satisfies $A_sw \leq (L_sv_s)(x_k) \leq B_sw$ and it will then follow that the spectral radius of $L_s$ is between then spectral raidus of matrices $A_s$ and $B_s$.
This approach was first developed by R. Nussbaum and R. Falk. As it plays a crucial role in my thesis, we implement it in python.
\newline 
Every irrational number $x$ in the unit interval has a unique representation of the form 
\[
x = [ a_1, a_2,a_3,  \ldots] = \cfrac{1}{a_1 + \cfrac{1}{a_2+  \frac{1}{a_3 + \ldots}}} ,
\]
with $a_i \in \N$ for $i \in \mathbb{N}$.
For example 
\[ 
\sqrt{2} - 1 = [ 2, 2,2,  \ldots] = \cfrac{1}{2 + \cfrac{1}{2+  \frac{1}{2 + \ldots}}}. 
\]
Note that $\sqrt{2} - 1$ is also a fixed point in $[0, 1]$ of the map $x \mapsto \frac{1}{2 + x}$.
Given $E \subset \mathbb{N}$ finite, let $J_E$ is the collection of all irrationals $x \in (0, 1)$ whose continued expansion digits belong to $E$, so
\[ J_E = \{ x = [a_1, a_2, a_3, \ldots ] \colon a_i \in E, i \in \N\} \subseteq [0, 1].\] 
These sets typically have a non-integral Hausdorff dimension, and the goal is to approximate this Hausdorff dimension.
