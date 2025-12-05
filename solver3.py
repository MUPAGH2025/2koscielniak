import numpy as np
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions


class ShallowWaterEquationsIntegrator:
    def __init__(self, *, h_initial: np.ndarray, omega_per_iter: float, amplitude: float, options: Options = None):
        """ initializes the solvers for a given initial condition of `h` assuming zero momenta at t=0 """
        options = options or Options(nonoscillatory=True, infinite_gauge=True, third_order_terms=True)
        index_x, index_y, solver_grid = 0, 1, h_initial.shape
        stepper = Stepper(options=options, grid=solver_grid)
        kwargs = {
            'boundary_conditions': [boundary_conditions.Extrapolated()] * len(solver_grid),
            # 'boundary_conditions': (boundary_conditions.Extrapolated(), boundary_conditions.Constant(0)),
            'halo': options.n_halo,
        }
        advectees = {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(np.zeros(solver_grid), **kwargs),
            "vh": ScalarField(np.zeros(solver_grid), **kwargs),
        }
        self.advector = VectorField((
            np.zeros((solver_grid[index_x] + 1, solver_grid[index_y])),
            np.zeros((solver_grid[index_x], solver_grid[index_y] + 1))
        ), **kwargs
        )
        self.amp = amplitude
        self.omega_per_iter = omega_per_iter
        self.h_0 = np.max(h_initial)
        self.solvers = {k: Solver(stepper, v, self.advector) for k, v in advectees.items()}

    def __getitem__(self, key):
        """ returns `key` advectee field of the current solver state """
        return self.solvers[key].advectee.get()

    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        """ applies half of the source term in the given direction """
        self[key][:] -= .5 * g_times_dt_over_dxy * self['h'] * np.gradient(self['h'], axis=axis)

    def _apply_half_sources(self, *, key, nt):
        """ applies half of the source term in the given direction """
        grid_sources = self[key].shape

        iy = 20
        y_slice = slice(iy - 2, iy + 3)
        x_half = grid_sources[0] // 2
        half_width = 20
        # ix_1 = int(grid_sources[0] * 0.35)
        ix_1 = x_half - half_width
        x1_slice = slice(ix_1 - 2, ix_1 + 3)
        ix_2 = x_half + half_width
        # ix_2 = int(grid_sources[0] * 0.65)
        x2_slice = slice(ix_2 - 2, ix_2 + 3)

        kernel = np.array([
            [0.01, 0.02, 0.03, 0.02, 0.01],
            [0.02, 0.06, 0.08, 0.06, 0.02],
            [0.03, 0.08, 0.10, 0.08, 0.03],
            [0.02, 0.06, 0.08, 0.06, 0.02],
            [0.01, 0.02, 0.03, 0.02, 0.01],
        ])
        kernel = kernel / np.sum(kernel)

        phase = np.cos(self.omega_per_iter * nt)

        self[key][x1_slice, y_slice] += .5 * self.amp * self.h_0 * kernel * phase
        self[key][x2_slice, y_slice] += .5 * self.amp * self.h_0 * kernel * phase

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        """ computes the Courant number component from fluid column height and momenta fields """
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)

        # using slices to ensure views (over copies)
        all_indexes = slice(None, None)
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = velocity[(
            (all_but_last, all_indexes),
            (all_indexes, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, all_indexes),
            (all_indexes, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float = 1e-7):
        """ integrates `nt` timesteps and returns a dictionary of solver states recorded every `outfreq` step[s] """
        solver_output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1):
            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self._apply_half_sources(key='h', nt=it - 1)
                self.solvers["h"].advance(n_steps=1)
                self._apply_half_sources(key='h', nt=it - 1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    solver_output[key].append(self[key].copy())
        return solver_output

grid = (150, 75)

h_0 = np.ones(grid, dtype=float)
dt = 0.25
omega = 1.0
omega_it = omega * dt
amp = 0.2
grav = 10

solver3 = ShallowWaterEquationsIntegrator(h_initial=h_0, omega_per_iter=omega_it, amplitude=amp)
output3 = solver3(nt=175, g=grav, dt_over_dxy=(dt, dt), outfreq=1)

np.save("output3.npy", output3)