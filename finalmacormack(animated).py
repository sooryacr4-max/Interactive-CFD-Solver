import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
gamma = 1.4
gas_constant = 287

# Grid setup
nx = int(input('Enter number of grid points\n'))
delta_x = 3 / (nx - 1)
x = np.linspace(0, 3, nx)

# Find throat index (x = 1.5)
throat_idx = np.argmin(np.abs(x - 1.5))
print(f"Throat located at index {throat_idx}, x = {x[throat_idx]:.3f}")

# Area variation
A = 1 + 2.2 * (x - 1.5)**2

# Initial conditions
rho = 1 - 0.3146 * x
T = 1 - 0.2314 * x
V = (0.1 + 1.09 * x) * (T**0.5)

# Apply boundary conditions at inlet
rho[0] = 1
T[0] = 1
V[0] = 2 * V[1] - V[2]

# Apply boundary conditions at exit
V[nx-1] = 2 * V[nx-2] - V[nx-3]
rho[nx-1] = 2 * rho[nx-2] - rho[nx-3]
T[nx-1] = 2 * T[nx-2] - T[nx-3]

# Time steps
nt = int(input('Enter number of time steps you want= '))

# Precompute initial speed of sound (added temperature floor to avoid sqrt of negative)
a = np.zeros(nx)
for sos in range(nx):
    T[sos] = max(T[sos], 1e-8)  # Fix 1: avoid sqrt of negative T
    a[sos] = (gamma * gas_constant * T[sos])**0.5

def predictor_corrrector(rho, V, T, nx, nt, gamma, gas_constant, throat_idx):
    # Allocate arrays
    d_rho_dt_n = np.zeros(nx)
    d_T_dt_n = np.zeros(nx)
    d_V_dt_n = np.zeros(nx)
    d_rho_dt_n1 = np.zeros(nx)
    d_T_dt_n1 = np.zeros(nx)
    d_V_dt_n1 = np.zeros(nx)
    delta_t = np.zeros(nx)
    a = np.zeros(nx)

    rho_p = np.copy(rho)
    T_p = np.copy(T)
    V_p = np.copy(V)

    d_rho_dt_av = np.zeros(nx)
    d_T_dt_av = np.zeros(nx)
    d_V_dt_av = np.zeros(nx)

    rho_n1 = np.copy(rho)
    T_n1 = np.copy(T)
    V_n1 = np.copy(V)

    # Arrays to store time evolution data at throat
    rho_history = []
    T_history = []
    V_history = []
    
    # Arrays to store spatial distribution evolution
    rho_spatial_history = []
    T_spatial_history = []
    V_spatial_history = []
    
    # Store initial values at throat
    rho_history.append(rho[throat_idx])
    T_history.append(T[throat_idx])
    V_history.append(V[throat_idx])
    
    # Store initial spatial distributions
    rho_spatial_history.append(rho.copy())
    T_spatial_history.append(T.copy())
    V_spatial_history.append(V.copy())

    for n in range(nt):
        # Update a with safe T
        for sos in range(nx):
            T[sos] = max(T[sos], 1e-8)  # Fix 2: maintain positivity
            a[sos] = (gamma * gas_constant * T[sos])**0.5

        # Time step calculation
        CFL = 0.5  # Fix 3: reduce CFL for stability
        for i_t in range(nx):
            delta_t[i_t] = CFL * delta_x / (abs(V[i_t]) + a[i_t])

        delta_t_min = min(delta_t)

        for i in range(1, nx - 1):
            rho[i] = max(rho[i], 1e-8)  # Fix 4: floor for rho to prevent div by zero

            # Predictor step
            term_1 = -rho[i] * ((V[i+1] - V[i]) / delta_x)
            term_2 = rho[i] * V[i] * ((np.log(A[i+1]) - np.log(A[i])) / delta_x)
            term_3 = V[i] * ((rho[i+1] - rho[i]) / delta_x)
            term_4 = -V[i] * ((V[i+1] - V[i]) / delta_x)
            term_5 = (1 / gamma) * (((T[i+1] - T[i]) / delta_x) + ((T[i] / rho[i]) * ((rho[i+1] - rho[i]) / delta_x)))
            term_6 = -V[i] * ((T[i+1] - T[i]) / delta_x)
            term_7 = ((gamma - 1) * T[i]) * (((V[i+1] - V[i]) / delta_x) + (V[i] * ((np.log(A[i+1]) - np.log(A[i])) / delta_x)))

            d_rho_dt_n[i] = term_1 - term_2 - term_3
            d_V_dt_n[i] = term_4 - term_5
            d_T_dt_n[i] = term_6 - term_7

            # Predict values
            rho_p[i] = rho[i] + d_rho_dt_n[i] * delta_t_min
            T_p[i] = T[i] + d_T_dt_n[i] * delta_t_min
            V_p[i] = V[i] + d_V_dt_n[i] * delta_t_min

            rho_p[i] = max(rho_p[i], 1e-8)  # Fix 4 repeated
            T_p[i] = max(T_p[i], 1e-8)

        # Corrector step
        for i in range(1, nx - 1):
            term_8 = -rho_p[i] * ((V_p[i] - V_p[i-1]) / delta_x)
            term_9 = rho_p[i] * V_p[i] * ((np.log(A[i]) - np.log(A[i-1])) / delta_x)
            term_10 = V_p[i] * ((rho_p[i] - rho_p[i-1]) / delta_x)
            term_11 = -V_p[i] * ((V_p[i] - V_p[i-1]) / delta_x)
            term_12 = (1 / gamma) * (((T_p[i] - T_p[i-1]) / delta_x) + ((T_p[i] / rho_p[i]) * ((rho_p[i] - rho_p[i-1]) / delta_x)))
            term_13 = -V_p[i] * ((T_p[i] - T_p[i-1]) / delta_x)
            term_14 = ((gamma - 1) * T_p[i]) * (((V_p[i] - V_p[i-1]) / delta_x) + (V_p[i] * ((np.log(A[i]) - np.log(A[i-1])) / delta_x)))

            d_rho_dt_n1[i] = term_8 - term_9 - term_10
            d_V_dt_n1[i] = term_11 - term_12
            d_T_dt_n1[i] = term_13 - term_14

            d_rho_dt_av[i] = 0.5 * (d_rho_dt_n1[i] + d_rho_dt_n[i])
            d_V_dt_av[i] = 0.5 * (d_V_dt_n1[i] + d_V_dt_n[i])
            d_T_dt_av[i] = 0.5 * (d_T_dt_n1[i] + d_T_dt_n[i])

            rho_n1[i] = rho[i] + delta_t_min * d_rho_dt_av[i]
            T_n1[i] = T[i] + delta_t_min * d_T_dt_av[i]
            V_n1[i] = V[i] + delta_t_min * d_V_dt_av[i]

        # Boundary conditions
        rho_n1[0] = 1
        T_n1[0] = 1
        V_n1[0] = 2 * V_n1[1] - V_n1[2]
        V_n1[nx-1] = 2 * V_n1[nx-2] - V_n1[nx-3]
        rho_n1[nx-1] = 2 * rho_n1[nx-2] - rho_n1[nx-3]
        T_n1[nx-1] = 2 * T_n1[nx-2] - T_n1[nx-3]

        # Overwrite values for next iteration
        rho = rho_n1.copy()
        T = T_n1.copy()
        V = V_n1.copy()

        # Store values at throat for time evolution plot
        rho_history.append(rho[throat_idx])
        T_history.append(T[throat_idx])
        V_history.append(V[throat_idx])
        
        # Store spatial distributions every few time steps to avoid memory issues
        if n % max(1, nt//200) == 0:  # Store up to 200 frames
            rho_spatial_history.append(rho.copy())
            T_spatial_history.append(T.copy())
            V_spatial_history.append(V.copy())

        # Fix 5: Check for nan or overflow
        if np.any(np.isnan(rho)) or np.any(np.isnan(T)) or np.any(np.isnan(V)):
            print(f"NaN or Inf encountered at time step {n}. Aborting simulation.")
            break

    return rho, V, T, rho_history, T_history, V_history, rho_spatial_history, T_spatial_history, V_spatial_history

# Run the simulation
rho_final, V_final, T_final, rho_history, T_history, V_history, rho_spatial_history, T_spatial_history, V_spatial_history = predictor_corrrector(rho, V, T, nx, nt, gamma, gas_constant, throat_idx)

print(f"Simulation completed with {len(rho_spatial_history)} spatial snapshots")

# Create animations
def animate_throat_evolution():
    """Animation of flow properties evolution at the throat"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_steps = np.arange(len(rho_history))
    
    # Set up the plot
    ax.set_xlim(0, len(rho_history))
    ax.set_ylim(min(min(rho_history), min(T_history), min(V_history)) * 0.9,
                max(max(rho_history), max(T_history), max(V_history)) * 1.1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Flow Variables')
    ax.set_title(f'Evolution of Flow Variables at Throat (x = {x[throat_idx]:.3f})')
    ax.grid(True)
    
    # Initialize empty lines
    line_rho, = ax.plot([], [], 'b-', label='Density (ρ)', linewidth=2)
    line_T, = ax.plot([], [], 'r-', label='Temperature (T)', linewidth=2)
    line_V, = ax.plot([], [], 'g-', label='Velocity (V)', linewidth=2)
    ax.legend()
    
    def animate(frame):
        # Update data up to current frame
        line_rho.set_data(time_steps[:frame+1], rho_history[:frame+1])
        line_T.set_data(time_steps[:frame+1], T_history[:frame+1])
        line_V.set_data(time_steps[:frame+1], V_history[:frame+1])
        return line_rho, line_T, line_V
    
    ani = animation.FuncAnimation(fig, animate, frames=len(rho_history), 
                                  interval=50, blit=True, repeat=True)
    plt.show()
    return ani

def animate_spatial_distribution():
    """Animation of density distribution throughout the nozzle"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the plot
    ax.set_xlim(0, 3)
    ax.set_ylim(min([min(rho_snap) for rho_snap in rho_spatial_history]) * 0.9,
                max([max(rho_snap) for rho_snap in rho_spatial_history]) * 1.1)
    ax.set_xlabel('x (Nozzle Length)')
    ax.set_ylabel('Density (ρ)')
    ax.set_title('Density Distribution Evolution Throughout Nozzle')
    ax.grid(True)
    
    # Add nozzle area visualization (secondary y-axis)
    ax2 = ax.twinx()
    ax2.plot(x, A, 'k--', alpha=0.3, label='Nozzle Area')
    ax2.set_ylabel('Nozzle Area', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Mark the throat
    ax.axvline(x=1.5, color='red', linestyle=':', alpha=0.5, label='Throat')
    
    # Initialize density line
    line, = ax.plot([], [], 'b-', linewidth=2, label='Density')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.legend(loc='upper left')
    
    def animate(frame):
        # Update density distribution
        line.set_data(x, rho_spatial_history[frame])
        time_text.set_text(f'Time Step: {frame * max(1, nt//200)}')
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, frames=len(rho_spatial_history), 
                                  interval=100, blit=True, repeat=True)
    plt.show()
    return ani

# Show static final result first
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Final density distribution
ax1.plot(x, rho_final, 'b-', linewidth=2, label='Final Density')
ax1.axvline(x=1.5, color='red', linestyle=':', alpha=0.5, label='Throat')
ax1.set_xlabel('x (Nozzle Length)')
ax1.set_ylabel('Density (ρ)')
ax1.set_title('Final Density Distribution')
ax1.grid(True)
ax1.legend()

# Final evolution at throat
time_steps = np.arange(len(rho_history))
ax2.plot(time_steps, rho_history, 'b-', label='Density (ρ)', linewidth=2)
ax2.plot(time_steps, T_history, 'r-', label='Temperature (T)', linewidth=2)
ax2.plot(time_steps, V_history, 'g-', label='Velocity (V)', linewidth=2)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Flow Variables')
ax2.set_title(f'Final Evolution at Throat (x = {x[throat_idx]:.3f})')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Create and show animations
print("\nCreating animations...")
print("1. Throat evolution animation:")
ani1 = animate_throat_evolution()

print("2. Spatial density distribution animation:")
ani2 = animate_spatial_distribution()

# Optional: Save animations as GIF files
save_animations = input("\nDo you want to save animations as GIF files? (y/n): ").lower().strip()
if save_animations == 'y':
    print("Saving throat evolution animation...")
    ani1.save('throat_evolution.gif', writer='pillow', fps=20)
    print("Saving spatial distribution animation...")
    ani2.save('density_distribution.gif', writer='pillow', fps=10)
    print("Animations saved successfully!")