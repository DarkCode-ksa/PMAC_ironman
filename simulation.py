import numpy as np
from parameters import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class YBCO_Coils:
    def __init__(self):
        self.n_coils = 6
    def ramp_field(self, t):
        return B_MAX * min(t/COIL_RAMP_TIME,1.0)
    def coil_current(self,B):
        return (B/(MU_0*12))*1e-3
    def power_consumption(self,t):
        B_t = self.ramp_field(t)
        return COIL_POWER*(B_t/B_MAX)**2

class MHD_Plasma:
    def density_reduction(self, t, n_e_prev, B_field):
        reduction_factor = 1 - np.exp(-ALPHA_YBCO * B_field**2 * T_WINDOW / MU_0)
        return n_e_prev*(1 - reduction_factor)
    def blackout_condition(self,n_e):
        return n_e/N_E0 > 0.32

class PMAC_Controller:
    def __init__(self):
        self.sensor_array = np.random.normal(N_E0,N_E0*0.05,64)
    def sensor_read(self,t):
        noise = np.random.normal(0,N_E0*0.02,64)
        return self.sensor_array + noise*np.sin(2*np.pi*t)
    def ai_decision(self,sensors):
        n_e_mean = np.mean(sensors)
        error = (N_E0-n_e_mean)/N_E0
        return np.clip(error*1.2,0,1)
    def control_loop(self,t):
        sensors = self.sensor_read(t)
        return self.ai_decision(sensors)

class IRON_MAN_PMAC:
    def __init__(self,sim_time=SIM_TIME):
        self.sim_time = sim_time
        self.dt = 1e-4
        self.t = np.arange(0,sim_time,self.dt)
        self.n_points = len(self.t)
        self.coils = YBCO_Coils()
        self.plasma = MHD_Plasma()
        self.controller = PMAC_Controller()
        self.B_field = np.zeros(self.n_points)
        self.n_e = np.full(self.n_points,N_E0)
        self.power = np.zeros(self.n_points)
        self.guidance_clear = np.zeros(self.n_points)
        self.blackout = np.zeros(self.n_points)
        self.ai_command = np.zeros(self.n_points)

    def run_simulation(self):
        for i,t in enumerate(self.t):
            if t%T_WINDOW<self.dt:
                self.B_field[i]=self.coils.ramp_field(t%T_WINDOW)
            else:
                self.B_field[i]=self.coils.ramp_field(T_WINDOW)
            if i%int(CONTROL_LOOP/self.dt)==0:
                self.ai_command[i]=self.controller.control_loop(t)
            if i>0:
                self.n_e[i]=self.plasma.density_reduction(t,self.n_e[i-1],self.B_field[i])
            self.power[i]=self.coils.power_consumption(t)
            reduction=(1-self.n_e[i]/N_E0)*100
            self.guidance_clear[i]=1 if reduction>=68 else 0
            self.blackout[i]=1-self.guidance_clear[i]
        self.calculate_metrics()
        return self

    def calculate_metrics(self):
        final_reduction=(1-self.n_e[-1]/N_E0)*100
        clear_time=np.sum(self.guidance_clear)*self.dt
        blackout_time=self.sim_time-clear_time
        clear_pct=(clear_time/self.sim_time)*100
        self.metrics={'n_e_reduction':final_reduction,'guidance_clear':clear_pct,
                      'blackout_time':blackout_time,'peak_power':self.power.max(),
                      'cep':CEP_TARGET}
        print(f"nₑ Reduction: {final_reduction:.1f}% | Guidance Clear: {clear_pct:.1f}% | Blackout: {blackout_time:.3f}s | Peak Power: {self.power.max()/1e3:.0f} kW | CEP: {CEP_TARGET} m")

    def plot_results(self):
        fig=make_subplots(rows=3,cols=2,subplot_titles=('B-Field','Plasma nₑ','PMAC Windows','Guidance','AI Command','Power'),vertical_spacing=0.1,horizontal_spacing=0.08)
        fig.add_trace(go.Scatter(x=self.t,y=self.B_field,name='B-Field',line=dict(color='#FF4444',width=4)),row=1,col=1)
        fig.add_trace(go.Scatter(x=self.t,y=self.n_e/N_E0,name='nₑ/Nₑ₀',line=dict(color='#4444FF',width=4)),row=1,col=2)
        window_times=np.arange(0,self.sim_time,T_WINDOW)
        for t_start in window_times:
            mask=(self.t>=t_start)&(self.t<t_start+T_WINDOW)
            fig.add_trace(go.Scatter(x=self.t[mask],y=np.ones(sum(mask))*0.9,mode='lines',line=dict(color='#FFD700',width=6),showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=self.t,y=self.guidance_clear,name='Guidance Clear',line=dict(color='#44FF44',width=4)),row=2,col=2)
        fig.add_trace(go.Scatter(x=self.t,y=self.blackout,name='Blackout',line=dict(color='#FF4444',width=4)),row=2,col=2)
        fig.add_trace(go.Scatter(x=self.t,y=self.ai_command,name='AI Command',line=dict(color='#FFAA00',width=3)),row=3,col=1)
        fig.add_trace(go.Scatter(x=self.t,y=self.power/1e3,name='Power',line=dict(color='#AA44FF',width=3)),row=3,col=2)
        fig.update_layout(title_text=f'IRON MAN PMAC Simulation | nₑ Reduction: {self.metrics["n_e_reduction"]:.1f}% | CEP {CEP_TARGET} m',height=1000,template='plotly_white')
        fig.show()
