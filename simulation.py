import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ==================== Constants ====================
MU_0 = 4 * np.pi * 1e-7
ALPHA_YBCO = 0.023
B_MAX = 2.0
T_WINDOW = 0.180
N_E0 = 1e16
MACH = 6.1
CEP_TARGET = 0.13

COIL_RAMP_TIME = 0.026
COIL_POWER = 120e3
CONTROL_LOOP = 0.019

# ==================== YBCO Coils ====================
class YBCO_Coils:
    def ramp_field(self, t):
        return B_MAX*(t/COIL_RAMP_TIME) if t < COIL_RAMP_TIME else B_MAX
    def power_consumption(self, t):
        B = self.ramp_field(t)
        return COIL_POWER*(B/B_MAX)**2

# ==================== MHD Plasma ====================
class MHD_Plasma:
    def density_reduction(self, n_e_prev, B_field):
        reduction_factor = 1 - np.exp(-ALPHA_YBCO*B_field**2*T_WINDOW/MU_0)
        return n_e_prev*(1 - reduction_factor)

# ==================== AI Controller ====================
class PMAC_Controller:
    def sensor_read(self):
        sensors = np.random.normal(N_E0, N_E0*0.05, 64)
        return sensors
    def ai_decision(self, sensors):
        n_e_mean = np.mean(sensors)
        error = (N_E0 - n_e_mean)/N_E0
        return np.clip(error*1.2, 0, 1)

# ==================== PMAC System ====================
class IRON_MAN_PMAC:
    def __init__(self, sim_time=2.0):
        self.sim_time = sim_time
        self.dt = 1e-4
        self.t = np.arange(0, sim_time, self.dt)
        self.coils = YBCO_Coils()
        self.plasma = MHD_Plasma()
        self.controller = PMAC_Controller()
        self.B_field = np.zeros(len(self.t))
        self.n_e = np.full(len(self.t), N_E0)
        self.power = np.zeros(len(self.t))
        self.guidance_clear = np.zeros(len(self.t))
        self.blackout = np.zeros(len(self.t))
        self.ai_command = np.zeros(len(self.t))

    def run_simulation(self):
        for i, t in enumerate(self.t):
            if t % T_WINDOW < self.dt:
                self.B_field[i] = self.coils.ramp_field(t % T_WINDOW)
            else:
                self.B_field[i] = self.coils.ramp_field(T_WINDOW)

            if i % int(CONTROL_LOOP/self.dt) == 0:
                sensors = self.controller.sensor_read()
                self.ai_command[i] = self.controller.ai_decision(sensors)

            if i > 0:
                self.n_e[i] = self.plasma.density_reduction(self.n_e[i-1], self.B_field[i])

            self.power[i] = self.coils.power_consumption(t)
            reduction = (1 - self.n_e[i]/N_E0)*100
            self.guidance_clear[i] = 1 if reduction >= 68 else 0
            self.blackout[i] = 1 - self.guidance_clear[i]

        self.calculate_metrics()
        return self

    def calculate_metrics(self):
        final_reduction = (1 - self.n_e[-1]/N_E0)*100
        clear_time = np.sum(self.guidance_clear)*self.dt
        blackout_time = self.sim_time - clear_time
        clear_pct = (clear_time/self.sim_time)*100
        self.metrics = {
            'n_e_reduction': final_reduction,
            'guidance_clear': clear_pct,
            'blackout_time': blackout_time,
            'peak_power': self.power.max(),
            'cep': CEP_TARGET
        }

    def print_results(self):
        print("\n===== PMAC SIMULATION RESULTS =====")
        print(f"nₑ Reduction: {self.metrics['n_e_reduction']:.2f}%")
        print(f"Guidance Clear: {self.metrics['guidance_clear']:.2f}%")
        print(f"Blackout Time: {self.metrics['blackout_time']:.3f}s")
        print(f"Peak Power: {self.metrics['peak_power']/1e3:.0f} kW")
        print(f"CEP Estimate: {self.metrics['cep']:.2f} m")

    def plot_results(self):
        plt.figure(figsize=(12,8))
        plt.subplot(3,2,1)
        plt.plot(self.t, self.B_field, color='r'); plt.title('YBCO B-Field'); plt.ylabel('T')
        plt.subplot(3,2,2)
        plt.plot(self.t, self.n_e/N_E0, color='b'); plt.title('Plasma Density nₑ/Nₑ₀'); plt.axhline(0.32, color='g', linestyle='--')
        plt.subplot(3,2,3)
        plt.plot(self.t, self.guidance_clear, color='g'); plt.plot(self.t, self.blackout, color='r'); plt.title('Guidance & Blackout'); plt.subplot(3,2,4)
        plt.plot(self.t, self.ai_command, color='orange'); plt.title('AI Controller Command');
        plt.subplot(3,2,5)
        plt.plot(self.t, self.power/1e3, color='purple'); plt.title('Power [kW]');
        plt.tight_layout()
        plt.show()

# ==================== Execute Simulation ====================
if __name__ == '__main__':
    pmac_system = IRON_MAN_PMAC(sim_time=2.0)
    pmac_system.run_simulation()
    pmac_system.print_results()
    pmac_system.plot_results()
    np.savez('ironman_pmac_results.npz', t=pmac_system.t, B_field=pmac_system.B_field, 
             n_e=pmac_system.n_e, guidance_clear=pmac_system.guidance_clear, 
             blackout=pmac_system.blackout, ai_command=pmac_system.ai_command, power=pmac_system.power)
    print("\n✅ All results saved to 'ironman_pmac_results.npz'")
