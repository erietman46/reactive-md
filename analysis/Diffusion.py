import numpy as np
import matplotlib.pyplot as plt


def read_lammps_dump(filename):
    """
    Read a LAMMPS custom trajectory containing:

        id type xu yu zu

    Returns
    -------
    timesteps : numpy array
    positions : numpy array with shape
                (n_frames, n_atoms, 3)
    """

    timesteps = []
    frames = []

    with open(filename, "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            if line.startswith("ITEM: TIMESTEP"):

                # Read timestep
                timestep = int(f.readline())
                timesteps.append(timestep)

                # NUMBER OF ATOMS
                f.readline()
                n_atoms = int(f.readline())

                # BOX BOUNDS
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # ATOMS header
                f.readline()

                atoms = []

                for _ in range(n_atoms):

                    data = f.readline().split()

                    atom_id = int(data[0])

                    xu = float(data[2])
                    yu = float(data[3])
                    zu = float(data[4])

                    atoms.append(
                        [atom_id, xu, yu, zu]
                    )

                # Sort by atom ID
                atoms.sort(key=lambda x: x[0])

                positions = [
                    [a[1], a[2], a[3]]
                    for a in atoms
                ]

                frames.append(positions)

    return np.array(timesteps), np.array(frames)


# --------------------------------------------------
# Read trajectory
# --------------------------------------------------

filename = "../lammps/01_lj_basics/trajectory.lammpstrj"

timesteps, positions = read_lammps_dump(filename)


# --------------------------------------------------
# Convert timestep numbers to LJ time
# --------------------------------------------------

dt = 0.005

time = timesteps * dt


# --------------------------------------------------
# Reference positions
# --------------------------------------------------

r0 = positions[0]


# --------------------------------------------------
# Displacements
# --------------------------------------------------

displacement = positions - r0


# squared displacement of every atom
squared_displacement = np.sum(
    displacement**2,
    axis=2
)


# average over atoms
msd = np.mean(
    squared_displacement,
    axis=1
)


# --------------------------------------------------
# Fit diffusive region
# --------------------------------------------------

# Example:
# ignore first 20% of trajectory
start = int(0.2 * len(time))

slope, intercept = np.polyfit(
    time[start:],
    msd[start:],
    1
)

D = slope / 6.0


print(f"Slope of MSD = {slope:.6f}")
print(f"Diffusion coefficient D = {D:.6f} (LJ units)")


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.plot(time, msd, label="MSD")

plt.plot(
    time[start:],
    slope * time[start:] + intercept,
    "--",
    label="Linear fit"
)

plt.xlabel("Time (LJ units)")
plt.ylabel("MSD")
plt.legend()

plt.tight_layout()
plt.show()