# Motoneuron-Driven Motor-Unit Muscle Force Simulation

**Purpose:** This note summarizes the mathematical simulation workflow for a motoneuron-driven motor-unit-resolved muscle model, based on Caillet et al. (2023), *Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy*.

The model maps:

```text
motor-neuron spike trains
    → motor-neuron action potentials
    → muscle-fiber action potentials
    → Ca²⁺ dynamics
    → Ca²⁺–troponin binding
    → motor-unit active state
    → motor-unit force
    → whole-muscle force
```

The key difference from a conventional EMG-driven Hill-type muscle model is that the muscle is represented as a **population of individual motor units (MUs)**, and each MU receives its own spike-train input.

---

## 1. Inputs and notation

### 1.1 Time grid

Let

\[
t_m = m\Delta t, \qquad m=0,1,\dots,N_t-1.
\]

The experimental paper used high-density EMG sampled at 2048 Hz, so a natural choice is

\[
\Delta t = \frac{1}{2048}\;\text{s}.
\]

### 1.2 Spike-train input

For motor unit \(k\), the decomposed spike train is a binary function:

\[
sp_k(t_m)=
\begin{cases}
1, & \text{if MU } k \text{ fires at } t_m,\\
0, & \text{otherwise.}
\end{cases}
\]

The input to the model is therefore

\[
\mathbf{sp}(t)=\left[sp_1(t),sp_2(t),\dots,sp_n(t)\right]^T,
\]

where \(n=N_r\) if only experimentally identified MUs are used, or \(n=N=400\) if the complete MU pool is reconstructed.

---

## 2. Subject-specific musculoskeletal scaling

The model needs subject-specific quantities to convert normalized muscle force into Newtons.

### 2.1 Optimal muscle length and tendon slack length

The paper scales generic musculoskeletal parameters using the subject-specific muscle-tendon length:

\[
l_0^M = \frac{l^{MT}}{l_{Raj}^{MT}}\,l_{0,Raj}^M,
\]

\[
l_s^T = \frac{l^{MT}}{l_{Raj}^{MT}}\,l_{s,Raj}^T.
\]

where:

| Symbol | Meaning |
|---|---|
| \(l^{MT}\) | subject-specific muscle-tendon length |
| \(l_{Raj}^{MT}\) | generic muscle-tendon length from the Rajagopal model |
| \(l_0^M\) | subject-specific optimal muscle length |
| \(l_s^T\) | subject-specific tendon slack length |

### 2.2 Maximum isometric muscle force

The subject-specific maximum isometric muscle force is estimated as

\[
F_0^M = \frac{V^M}{l_0^M}\sigma,
\]

where:

| Symbol | Meaning |
|---|---|
| \(V^M\) | muscle volume |
| \(l_0^M\) | optimal muscle length |
| \(\sigma\) | specific tetanic tension, e.g. \(60\;\text{N}/\text{cm}^2\) |

### 2.3 Experimental target force from joint torque

Because individual tibialis anterior force is not directly measured, it is estimated from ankle torque:

\[
F_{TA}(t)=\frac{T(t)-\Delta T(T(t))}{L_{TA}}.
\]

where:

| Symbol | Meaning |
|---|---|
| \(T(t)\) | measured ankle torque |
| \(\Delta T(T(t))\) | correction for other agonist/antagonist muscles |
| \(L_{TA}\) | tibialis anterior moment arm |

This estimated \(F_{TA}(t)\) is the validation target for the predicted muscle force.

---

## 3. Generic motor-unit pool

For the human tibialis anterior, the paper assumes:

\[
N = 400 \quad \text{motor units},
\]

\[
N_f^{tot}=200{,}000 \quad \text{muscle fibers}.
\]

Each MU is indexed by \(j\in\{1,2,\dots,N\}\), ordered from low-threshold/small to high-threshold/large.

---

## 4. Recruitment threshold distribution

The torque recruitment threshold of MU \(j\), expressed as percent MVC, is described by a linear-exponential distribution:

\[
T^{th}(j)
=
0.50\left[
58.12\frac{j}{N}
+
120^{\left(\frac{j}{N}\right)^{1.83}}
\right],
\qquad j\in\{1,\dots,N\}.
\]

This distribution maps low-index MUs to low recruitment thresholds and high-index MUs to high recruitment thresholds.

---

## 5. Mapping experimentally identified MUs into the full pool

### 5.1 Why mapping is needed

The HDEMG decomposition identifies only a subset of the real motor-unit pool. For example, the biological model assumes

\[
N = 400
\]

motor units in the tibialis anterior, but the experiment may identify only

\[
N_r \ll N
\]

motor units. Therefore, an experimentally identified MU indexed by \(i\) is **not automatically MU \(i\) in the biological pool**. It must be located inside the full recruitment-ranked pool.

The key physiological assumption is Henneman's size principle:

\[
	ext{lower recruitment threshold}
\quad \Longleftrightarrow \quad
	ext{smaller, earlier recruited MU},
\]

\[
	ext{higher recruitment threshold}
\quad \Longleftrightarrow \quad
	ext{larger, later recruited MU}.
\]

Thus, recruitment threshold is used as the coordinate that maps an experimentally observed MU into the full pool.

---

### 5.2 Full-pool recruitment-threshold curve

The full-pool model defines a monotonic recruitment-threshold curve

\[
T^{th}(j)
=
0.50\left[
58.12\frac{j}{N}
+
120^{\left(\frac{j}{N}
\right)^{1.83}}
\right],
\qquad j=1,\dots,N.
\]

Here:

| Symbol | Meaning |
|---|---|
| \(j\) | index of a motor unit in the full 400-MU pool |
| \(N\) | total number of MUs, here \(N=400\) |
| \(T^{th}(j)\) | recruitment threshold of full-pool MU \(j\), expressed as %MVC |

Because \(T^{th}(j)\) is increasing, it can be inverted: a measured threshold uniquely identifies a location in the full MU pool.

---

### 5.3 Measured recruitment threshold of an identified MU

For each experimentally identified MU \(i\), the first discharge time is detected from the binary spike train:

\[
t_i^{rec} = \min \{t_m : sp_i(t_m)=1\}.
\]

The measured torque at that time gives the recruitment threshold. In practice, the paper averages torque over a small window around the first discharge time:

\[
T_i^{th}
=
\frac{1}{|\mathcal{W}_i|}
\sum_{t_m\in \mathcal{W}_i} T(t_m),
\]

where \(\mathcal{W}_i\) is a short time window centered at \(t_i^{rec}\). If torque is normalized by MVC, then \(T_i^{th}\) is expressed in %MVC.

---

### 5.4 Inverting the recruitment curve

The mapped full-pool location \(N_i\) is found by solving

\[
T_i^{th}=T^{th}(N_i).
\]

Equivalently, define

\[
g(j)=T^{th}(j)-T_i^{th}.
\]

Then solve

\[
g(N_i)=0.
\]

Since \(j\) is an MU index, the solution is converted to an integer index:

\[
N_i = \operatorname*{round}\left[
\left(T^{th}
\right)^{-1}(T_i^{th})
\right].
\]

In implementation, it is often easier and more stable to use nearest-neighbor matching on the precomputed full-pool threshold curve:

\[
N_i
=
\operatorname*{argmin}_{j\in\{1,\dots,N\}}
\left|T^{th}(j)-T_i^{th}
\right|.
\]

So the mapping is

\[
i\in\{1,\dots,N_r\}
\quad\longrightarrow\quad
N_i\in\{1,\dots,N\}.
\]

This means:

\[
	ext{experimental MU } i
\quad\mapsto\quad
	ext{full-pool MU location } N_i.
\]

---

### 5.5 Numerical intuition

For \(N=400\), the recruitment-threshold curve gives approximate locations such as:

| Measured threshold \(T_i^{th}\) | Mapped full-pool location \(N_i\) | Interpretation |
|---:|---:|---|
| 1% MVC | \(\approx 7\) | very low-threshold MU |
| 5% MVC | \(\approx 61\) | low-threshold MU |
| 10% MVC | \(\approx 125\) | low-to-mid threshold MU |
| 20% MVC | \(\approx 234\) | mid-threshold MU |
| 30% MVC | \(\approx 299\) | high recruited range for 30% MVC task |
| 50% MVC | \(\approx 356\) | high-threshold MU |

Thus, if an experimentally identified MU first fires when the subject reaches 5% MVC, the model treats this MU as being located near MU \(61\) in the full 400-MU pool, not as MU \(1\) or MU \(5\) simply because it was the first or fifth unit identified by the algorithm.

---

### 5.6 Why mapping changes force assignment

The full pool has a force-capacity distribution

\[
f_0^{MU}(j), \qquad j=1,\dots,N.
\]

Low-index MUs have smaller force capacity; high-index MUs have larger force capacity. Therefore, after mapping, the identified MU receives a force capacity consistent with its physiological location.

There are two different cases.

#### Case A: full reconstructed pool is used

If the model uses all \(N=400\) MUs, then each MU directly receives its own force capacity:

\[
f_{0,k}^{MU}=f_0^{MU}(k),
\qquad k=1,\dots,N.
\]

No aggregation is needed.

#### Case B: only \(N_r\) experimentally identified MUs are used

If only the identified MUs are simulated, each identified MU must represent not only itself, but also nearby unobserved MUs. In that case, the mapped MU \(N_i\) is assigned a **representative force capacity** equal to the sum of the force capacities of its surrounding full-pool MUs.

First, sort the identified MUs by recruitment threshold:

\[
T_1^{th}<T_2^{th}<\cdots<T_{N_r}^{th},
\]

which gives mapped full-pool locations

\[
N_1<N_2<\cdots<N_{N_r}.
\]

Let \(N_a\) be the highest recruited MU index expected at the peak contraction. For example:

\[
T^{th}(N_a)=30
\quad\Rightarrow\quad
N_a\approx 299
\]

for a 30% MVC task, and

\[
T^{th}(N_a)=50
\quad\Rightarrow\quad
N_a\approx 356
\]

for a 50% MVC task.

Define the representative domain of identified MU \(i\) by midpoint boundaries:

\[
B_{i-1/2}
=
\left\lfloor \frac{N_{i-1}+N_i}{2}
\right\rfloor+1,
\]

\[
B_{i+1/2}
=
\left\lfloor \frac{N_i+N_{i+1}}{2}
\right\rfloor.
\]

For the first and last identified MUs, use boundary values such as

\[
N_0=0,
\qquad
N_{N_r+1}=N_a.
\]

Then the representative maximum force assigned to identified MU \(i\) is

\[
f_{0,i}^{MU,rep}
=
\sum_{j=B_{i-1/2}}^{B_{i+1/2}} f_0^{MU}(j).
\]

This means that an identified MU located in a sparse region of the experimental sample represents a larger block of unobserved MUs and therefore receives a larger representative force capacity.

---

### 5.7 Example of the representative-domain idea

Suppose three identified MUs are mapped into the full pool at

\[
N_9=90,
\qquad
N_{10}=120,
\qquad
N_{11}=180.
\]

Then the representative domain of the 10th identified MU is bounded by the midpoints between its neighbors:

\[
B_{10-1/2}
=
\left\lfloor \frac{90+120}{2}
\right\rfloor+1
=106,
\]

\[
B_{10+1/2}
=
\left\lfloor \frac{120+180}{2}
\right\rfloor
=150.
\]

So identified MU 10 represents full-pool MUs

\[
j=106,107,\dots,150.
\]

Its representative maximum force is therefore

\[
f_{0,10}^{MU,rep}
=
\sum_{j=106}^{150} f_0^{MU}(j).
\]

This is exactly why mapping matters: if the experimentally identified units are not uniformly distributed across the real MU pool, the model compensates by letting each identified MU represent the appropriate local region of missing MUs.

---

### 5.8 Algorithmic summary

```text
Input:
    Full-pool threshold curve Tth(j), j = 1,...,400
    Full-pool force-capacity curve f0_MU(j)
    Experimental spike trains sp_i(t)
    Experimental torque trace T(t)

For each identified MU i:
    1. Find first discharge time t_i_rec
    2. Compute measured threshold T_i_th from torque around t_i_rec
    3. Find mapped full-pool index:
           N_i = argmin_j |Tth(j) - T_i_th|

If using full reconstructed pool:
    Use f0_MU(j) directly for each full-pool MU.

If using only identified MUs:
    1. Sort identified MUs by T_i_th
    2. Use midpoint boundaries between mapped indices N_i
    3. Assign each identified MU a representative force:
           f0_i_rep = sum of f0_MU(j) over its local domain
```

The conceptual goal is:

\[
\boxed{
	ext{measured recruitment threshold}
\;\Rightarrow\;
	ext{physiological location in full MU pool}
\;\Rightarrow\;
	ext{appropriate MU force capacity}
}
\]

---

## 6. MU twitch force, maximum force, and innervation ratio

### 6.1 Normalized twitch-force distribution

The normalized twitch force is approximated as

\[
f^{tw}(j)
=
6.07\left[
4.52\frac{j}{N}
+
11.96^{\left(\frac{j}{N}\right)^{4.66}}
\right],
\qquad j\in\{1,…,N\}.
\]

### 6.2 Normalized maximum isometric MU force

The normalized maximum isometric force of MU \(j\) is

\[
f_0^{MU}(j)
=
7.86\times 10^{-4}
\left[
3.00\frac{j}{N}
+
8.20^{\left(\frac{j}{N}\right)^{5.29}}
\right],
\qquad j\in\{1,…,N\}.
\]

The distribution is normalized so that

\[
\sum_{j=1}^{N} f_0^{MU}(j) \approx 1.
\]

To obtain MU maximum force in Newtons:

\[
F_0^{MU}(j)=F_0^M f_0^{MU}(j).
\]

### 6.3 Innervation ratio

The innervation ratio is assigned proportional to twitch force:

\[
IR(j)=\frac{f^{tw}(j)}{\sum_{k=1}^{400}f^{tw}(k)}N_f^{tot}.
\]

The paper assigns the smallest 359 MUs as slow-type MUs because they account for approximately 72% of the total fibers.

---

## 7. Model structure for one motor unit

For motor unit \(k\), the force generator contains:

```text
Neuromechanical Element (NE):
    spike train → AP → Ca²⁺ → CaTn → active state

Contractile Element (CE):
    active state + force-length scaling → MU force
```

The final normalized force from MU \(k\) is

\[
f_k^{MU}(t)
=
f_{k,0}^{MU}\,a_k(t,l)\,f_{FL}(a_k,l),
\]

where:

| Symbol | Meaning |
|---|---|
| \(f_{k,0}^{MU}\) | normalized maximum force capacity of MU \(k\) |
| \(a_k(t,l)\) | active state of MU \(k\) |
| \(f_{FL}\) | force-length scaling factor |
| \(l\) | normalized MU length |

In the isometric tibialis anterior simulation, the model assumes

\[
l=1.16.
\]

---

## 8. Contractile element: force-length scaling

The normalized force-length relationship is

\[
f_{FL}(l,a)=
\exp\left[-\left(\frac{l-l_0(a)}{0.45}\right)^2\right],
\]

with activation-dependent apparent optimal length

\[
l_0(a)=0.15(1-a)+1.
\]

At submaximal activation, this shifts the apparent optimal length.

---

## 9. Excitation dynamics

### 9.1 Spike train to motoneuron action potential

For a spike at time \(t_i\), the motoneuron action potential is modeled as a half-sine pulse:

\[
sp(t_i)=1,
\]

\[
e(t)=V_e\sin\left(\frac{2\pi}{T}(t-t_i)\right),
\qquad t_i\leq t\leq t_i+\frac{T}{2},
\]

\[
e(t)=0,\qquad \text{otherwise.}
\]

Typical values:

\[
V_e=90\;\text{mV},
\qquad
T=1.4\;\text{ms}.
\]

### 9.2 Motoneuron AP to muscle-fiber AP

The muscle-fiber action potential \(u(t)\) is the second-order response to \(e(t)\):

\[
\frac{d^2u}{dt^2}
=
a_1e(t)-\left(a_2u+a_3\frac{du}{dt}\right).
\]

Parameter values:

| Parameter | Slow MU | Fast MU | Unit |
|---|---:|---:|---|
| \(a_1\) | \(9\times10^7\) | \(9\times10^7\) | \(s^{-2}\) |
| \(a_2\) | \(5\times10^7\) | \(5\times10^7\) | \(s^{-2}\) |
| \(a_3\) | \(2\times10^4\) | \(2\times10^4\) | \(s^{-1}\) |

---

## 10. Free calcium dynamics

The fiber action potential drives free calcium concentration \(c(t)\):

\[
\frac{d^2c}{dt^2}
=
b_1u(t)
-
\frac{1}{f_1(l)}
\left[
b_2f_2(l)c+b_3\frac{dc}{dt}
\right].
\]

### 10.1 Length-dependent calcium amplitude scaling

\[
f_1(l)=
\begin{cases}
0.8, & l\leq 1.0,\\
0.8+1.33(l-1.0), & l\leq 1.15,\\
1.0, & l\leq 1.30,\\
1.0-0.6(l-1.3), & l>1.30.
\end{cases}
\]

### 10.2 Length-dependent calcium decay scaling

\[
f_2(l)=
\begin{cases}
1.0, & l\leq 1.15,\\
1.0-0.4(l-1.15), & l>1.15.
\end{cases}
\]

### 10.3 Calcium-dynamics parameters

| Parameter | Slow MU | Fast MU | Unit |
|---|---:|---:|---|
| \(b_1\) | 0.4 | 0.9 | \(M\,V^{-1}s^{-2}\) |
| \(b_2\) | \(1.5\times10^5\) | \(4.3\times10^5\) | \(s^{-2}\) |
| \(b_3\) | \(2.5\times10^3\) | \(2.4\times10^3\) | \(s^{-1}\) |

---

## 11. Calcium-troponin binding dynamics

Free calcium \(c(t)\) drives calcium-troponin concentration \(P(t)\):

\[
\frac{dP}{dt}
=
\frac{c_1}{f_3(l)}
\left(\frac{P_0}{f_4(l)}-P\right)c^2
-
\frac{c_2}{f_5(l)}P.
\]

where:

| Symbol | Meaning |
|---|---|
| \(P_0\) | total troponin concentration |
| \(c_1\) | calcium-troponin binding rate |
| \(c_2\) | calcium-troponin unbinding rate |
| \(f_3(l),f_4(l),f_5(l)\) | length-dependent scaling functions |

If the supplementary length functions are not implemented, a fixed-length simplified version can set

\[
f_3(l)=f_4(l)=f_5(l)=1.
\]

### 11.1 Calcium-troponin parameters

| Parameter | Slow MU | Fast MU | Unit |
|---|---:|---:|---|
| \(c_1\) | \(6\times10^{12}\) | \(1\times10^{12}\) | \(M^{-2}s^{-1}\) |
| \(c_2\) | 21 | 41 | \(s^{-1}\) |
| \(P_0\) | \(1.7\times10^{-4}\) | \(3.8\times10^{-4}\) | \(M\) |

---

## 12. Active-state dynamics

The MU active state \(a(t)\) is driven by calcium-troponin concentration:

\[
\frac{da}{dt}
=
d_1P
-
\frac{a}{d_2+d_3P}.
\]

This equation means:

```text
activation growth ∝ CaTn concentration
activation decay depends on a P-dependent time constant
```

Parameter values:

| Parameter | Slow MU | Fast MU | Unit / role |
|---|---:|---:|---|
| \(d_1\) | \(1.0\times10^5\) | \(1.0\times10^5\) | activation gain |
| \(d_2\) | 0.024 | 0.024 | time-constant term |
| \(d_3\) | 270 | 270 | P-dependent time-constant term |

---

## 13. State-space form for numerical simulation

For each MU \(k\), define the state vector

\[
\mathbf{x}_k(t)=
\begin{bmatrix}
u_k(t)\\
\dot u_k(t)\\
c_k(t)\\
\dot c_k(t)\\
P_k(t)\\
a_k(t)
\end{bmatrix}.
\]

Let

\[
v_k(t)=\dot u_k(t),
\qquad
w_k(t)=\dot c_k(t).
\]

Then the ODE system is

\[
\dot u_k=v_k,
\]

\[
\dot v_k=a_1e_k(t)-a_2u_k-a_3v_k,
\]

\[
\dot c_k=w_k,
\]

\[
\dot w_k=b_1u_k-\frac{1}{f_1(l)}\left[b_2f_2(l)c_k+b_3w_k\right],
\]

\[
\dot P_k=
\frac{c_1}{f_3(l)}
\left(\frac{P_0}{f_4(l)}-P_k\right)c_k^2
-
\frac{c_2}{f_5(l)}P_k,
\]

\[
\dot a_k=d_1P_k-\frac{a_k}{d_2+d_3P_k}.
\]

The spike train enters through \(e_k(t)\), the motoneuron action-potential pulse train.

---

## 14. Motor-unit force calculation

The raw normalized force of MU \(k\) is

\[
f_k^{MU}(t)
=
f_{k,0}^{MU}
\left[a_k(t,l)f_{FL}(a_k,l)\right].
\]

Because fibers within the same MU do not generate force at exactly the same instant, the paper averages over small random fiber delays:

\[
F_k^{MU}(t)
=
\frac{1}{IR_k}
\sum_{c=1}^{IR_k}
f_k^{MU}(t+\alpha_c),
\qquad
\alpha_c\in\left[-\frac{\Delta t}{2},\frac{\Delta t}{2}\right].
\]

This acts like a small low-pass filter on each MU force.

---

## 15. Whole-muscle force calculation

The normalized whole-muscle force is the sum of MU forces:

\[
\overline{F}^M(t)=\sum_{k=1}^{n}F_k^{MU}(t).
\]

The dimensional whole-muscle force is

\[
F^M(t)=F_0^M\overline{F}^M(t).
\]

---

## 16. Three neural-control modes

### Mode 1: Experimental MUs with blind force assignment

Use only the experimentally identified \(N_r\) spike trains and assume they are evenly spread across the recruited MU pool.

This produces:

\[
F_{N_r,1}^{M}(t).
\]

### Mode 2: Experimental MUs with recruitment-threshold-informed assignment

Use only the experimentally identified \(N_r\) spike trains, but map each identified MU into the full recruitment-ranked MU pool using \(T_i^{th}=T^{th}(N_i)\).

For identified MU \(i\), define the represented interval:

\[
N_{i,1}=\left\lfloor\frac{N_{i-1}+N_i}{2}\right\rfloor+1,
\]

\[
N_{i,2}=\left\lfloor\frac{N_i+N_{i+1}}{2}\right\rfloor.
\]

Then the representative maximum force is

\[
f_{0,i}^{MU}=
\sum_{j=N_{i,1}}^{N_{i,2}}f_0^{MU}(j).
\]

This produces:

\[
F_{N_r,2}^{M}(t).
\]

### Mode 3: Reconstructed full MU pool

Use the identified spike trains plus reconstructed spike trains for missing MUs, giving a complete pool:

\[
n=N=400.
\]

Each MU receives its own \(f_0^{MU}(j)\) directly from the full distribution.

This produces:

\[
F_N^M(t).
\]

---

## 17. Neural-drive validation

Before validating muscle force, the quality of the neural input can be checked.

The cumulative spike train is

\[
CST(t)=\sum_{k=1}^{n}sp_k(t).
\]

The effective neural drive is obtained by low-pass filtering the cumulative spike train in the force-relevant bandwidth:

\[
D(t)=\operatorname{LPF}_{0-4\,Hz}\left[CST(t)\right].
\]

Then compare normalized \(D(t)\) with normalized experimental force \(F_{TA}(t)\).

---

## 18. Force-validation metrics

Let \(F^M(t_m)\) be predicted muscle force and \(F_{TA}(t_m)\) be estimated experimental force.

### 18.1 Maximum error

\[
ME=\max_m\left|F^M(t_m)-F_{TA}(t_m)\right|.
\]

### 18.2 Root-mean-square error

\[
RMSE=
\sqrt{
\frac{1}{N_t}
\sum_{m=1}^{N_t}
\left(F^M(t_m)-F_{TA}(t_m)\right)^2
}.
\]

### 18.3 Normalized RMSE

A common normalization is

\[
nRMSE=100\times\frac{RMSE}{\max(F_{TA})-\min(F_{TA})}.
\]

The paper reports nRMSE over:

```text
whole contraction
ascending ramp
plateau
descending ramp
```

### 18.4 Coefficient of determination

\[
r^2
=
1-
\frac{\sum_m\left(F_{TA}(t_m)-F^M(t_m)\right)^2}
{\sum_m\left(F_{TA}(t_m)-\overline{F}_{TA}\right)^2}.
\]

### 18.5 Force-onset delay

Define force onset as the time when force reaches 2% of its maximum:

\[
t_{on}^{pred}=\min\left\{t:F^M(t)\geq0.02\max_tF^M(t)\right\},
\]

\[
t_{on}^{exp}=\min\left\{t:F_{TA}(t)\geq0.02\max_tF_{TA}(t)\right\}.
\]

Then

\[
\Delta_1=t_{on}^{pred}-t_{on}^{exp}.
\]

---

## 19. Numerical implementation pseudocode

```text
Given:
    spike trains sp_k(t_m)
    MU pool parameters f0_MU(k), IR(k), MU type(k)
    subject-specific F0_M
    fixed normalized length l = 1.16

For each MU k:
    1. Generate e_k(t) from spike train using half-sine AP pulses.
    2. Initialize state:
           u = 0, du/dt = 0
           c = 0, dc/dt = 0
           P = 0
           a = 0
    3. For each time step:
           solve u dynamics
           solve c dynamics
           solve P dynamics
           solve a dynamics
           compute f_FL(l,a)
           compute raw MU force f_k_MU(t)
    4. Apply intra-MU fiber-delay averaging to obtain F_k_MU(t).

After all MUs:
    Fbar_M(t) = sum_k F_k_MU(t)
    F_M(t) = F0_M * Fbar_M(t)

Validation:
    compare F_M(t) with estimated experimental F_TA(t)
```

---

## 20. Compact mathematical summary

For each motor unit \(k\):

\[
sp_k(t)\xrightarrow{\text{Eq. AP}}e_k(t)
\xrightarrow{\text{fiber AP ODE}}u_k(t)
\xrightarrow{\text{Ca}^{2+}\text{ ODE}}c_k(t)
\xrightarrow{\text{CaTn ODE}}P_k(t)
\xrightarrow{\text{activation ODE}}a_k(t)
\xrightarrow{\text{FL scaling}}f_k^{MU}(t).
\]

Then:

\[
F^M(t)=F_0^M\sum_{k=1}^{n}
\left[
\frac{1}{IR_k}
\sum_{c=1}^{IR_k}
 f_{k,0}^{MU}a_k(t+\alpha_c)f_{FL}(l,a_k(t+\alpha_c))
\right].
\]

This is the central forward simulation model.

---

## 21. Interpretation of each mathematical block

| Model block | Biological meaning | Main output |
|---|---|---|
| \(sp_k(t)\to e_k(t)\) | motoneuron firing event | MN action potential |
| \(e_k(t)\to u_k(t)\) | excitation of muscle fiber membrane | fiber AP |
| \(u_k(t)\to c_k(t)\) | sarcoplasmic Ca²⁺ release | free Ca²⁺ concentration |
| \(c_k(t)\to P_k(t)\) | Ca²⁺ binding to troponin | CaTn concentration |
| \(P_k(t)\to a_k(t)\) | cross-bridge activation readiness | active state |
| \(a_k(t),l\to f_k^{MU}(t)\) | Hill-type contractile force | MU force |
| \(\sum_k f_k^{MU}(t)\) | parallel MU force summation | whole-muscle force |

---

## 22. How this can be extended for clinical or aging studies

The model is modular. Different disease or aging mechanisms can be inserted into different equations:

| Mechanism | Mathematical modification |
|---|---|
| motor-unit loss | reduce \(N\), remove selected MUs |
| reinnervation | increase \(IR(j)\) for surviving MUs |
| altered recruitment | modify \(T^{th}(j)\) distribution |
| reduced firing rate | modify \(sp_k(t)\) statistics |
| neuromuscular junction failure | probabilistically drop spikes before \(e_k(t)\) or \(u_k(t)\) |
| impaired calcium handling | modify \(b_1,b_2,b_3,f_1,f_2\) |
| altered troponin sensitivity | modify \(c_1,c_2,P_0\) |
| muscle atrophy | reduce \(F_0^M\) or \(f_0^{MU}(j)\) |
| fiber-type transition | change slow/fast MU assignment |
| tendon compliance | replace fixed \(l=1.16\) with muscle-tendon equilibrium |

This makes the model useful not only for force prediction, but also for mechanistic interpretation of weakness.

