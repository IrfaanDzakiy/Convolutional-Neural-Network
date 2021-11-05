from utils import *
import numpy as np
import random


class LSTMLayer:

    # Input Matrix is U, timestep matrix is W

    def __init__(self, num_units, return_sequence=False):
        super().__init__()
        self.layer_unit = "LSTM"
        self.num_units = num_units

        # Input is based on timestep T, Xt
        self.input = None
        self.output = None

        # U weights (from prev layer / input to LSTM)
        self.u_i = None
        self.u_f = None
        self.u_c = None
        self.u_o = None

        # W weights (LSTM Timestep Weights)
        self.w_i = None
        self.w_f = None
        self.w_c = None
        self.w_o = None

        # B bias (All bias for each gate / state)
        self.b_i = None
        self.b_f = None
        self.b_c = None
        self.b_o = None

        # Ht-1 and Ct-1
        self.ht_prev = None
        self.ct_prev = None

        # Cell States and Gates (Maybe won't be used)
        self.cell_state = None
        self.forget_gate = None
        self.input_gate = None
        self.output_gate = None

        self.return_sequence = return_sequence

    def getName(self):
        return 'lstm'

    def getOutputShape(self):
        if (self.return_sequence):
            return (self.num_units, self.num_units)
        else:
            return (1, self.num_units)

    def getParamCount(self):
        n = self.num_units
        m = len(self.input)
        return (m + n + 1) * 4 * n

    def generate_weight_U(self):
        return np.array([np.random.random(size=len(self.input)) for i in range(self.num_units)])

    def generate_weight_W(self):
        return np.array([np.random.random(size=(self.num_units)) for i in range(self.num_units)])

    def generate_bias_B(self):
        rand_res = random.random()
        return np.array([rand_res for i in range(self.num_units)]).reshape(self.num_units, 1)

    def generate_Ht_prev(self):
        return np.array([0 for i in range(self.num_units)]).reshape(self.num_units, 1)

    def set_input(self, input):
        # Self input is flattened without bias
        self.input = np.array(input).flatten()

        # Set all U here
        if (self.u_i is None):
            self.u_i: np.array = np.array(self.generate_weight_U())

        if (self.u_f is None):
            self.u_f: np.array = np.array(self.generate_weight_U())

        if (self.u_c is None):
            self.u_c: np.array = np.array(self.generate_weight_U())

        if (self.u_o is None):
            self.u_o: np.array = np.array(self.generate_weight_U())

        # Set all W here
        if (self.w_i is None):
            self.w_i: np.array = np.array(self.generate_weight_W())

        if (self.w_f is None):
            self.w_f: np.array = np.array(self.generate_weight_W())

        if (self.w_c is None):
            self.w_c: np.array = np.array(self.generate_weight_W())

        if (self.w_o is None):
            self.w_o: np.array = np.array(self.generate_weight_W())

        # Set all B here
        if (self.b_i is None):
            self.b_i: np.array = np.array(self.generate_bias_B())

        if (self.b_f is None):
            self.b_f: np.array = np.array(self.generate_bias_B())

        if (self.b_c is None):
            self.b_c: np.array = np.array(self.generate_bias_B())

        if (self.b_o is None):
            self.b_o: np.array = np.array(self.generate_bias_B())

        # Set Htprev here
        if (self.ht_prev is None):
            self.ht_prev: np.array = np.array(self.generate_Ht_prev())

        # Set Ctprev here, Ctprev is similar to Ht_prev
        if (self.ct_prev is None):
            self.ct_prev: np.array = np.array(self.generate_Ht_prev())

        # self.params = len(self.flattened_input) * self.unit

    def calculate(self, inputs):
        output = []
        activated_output = []

        # Inputs would be input for all timesteps

        last_output = None

        for curr_input in inputs:

            Xt = np.array(curr_input).reshape(len(curr_input), 1)
            self.set_input(Xt)

            # print(Xt.shape)

            # TODO : Calculate Forget Gate
            UfXt = mmult(self.u_f, Xt)
            WfHtprev = mmult(self.w_f, self.ht_prev)
            total = UfXt + WfHtprev + self.b_f

            # Activate Here (Sigmoid)
            for i in range(len(total)):
                total[i] = sigmoid(total[i])

            f_t = total

            # TODO : Calculate Input Gate
            UiXt = mmult(self.u_i, Xt)
            WiHtprev = mmult(self.w_i, self.ht_prev)
            total = UiXt + WiHtprev + self.b_i

            # Activate Here (Sigmoid)
            for i in range(len(total)):
                total[i] = sigmoid(total[i])

            i_t = total

            # TODO : Calculate Cell State
            UcXt = mmult(self.u_c, Xt)
            WcHtprev = mmult(self.w_c, self.ht_prev)
            total = UcXt + WcHtprev + self.b_c

            # Activate Here (Tanh)
            for i in range(len(total)):
                total[i] = tanh(total[i])

            C_t_flag = total

            # TODO : Calculate Output Gate
            UoXt = mmult(self.u_o, Xt)
            WoHtprev = mmult(self.w_o, self.ht_prev)
            total = UoXt + WoHtprev + self.b_o

            # Activate Here (Sigmoid)
            for i in range(len(total)):
                total[i] = sigmoid(total[i])

            o_t = total

            # Flowing Informations for Future use
            C_t = f_t * self.ct_prev + i_t * C_t_flag

            tanh_c_t = C_t
            for i in range(len(tanh_c_t)):
                tanh_c_t[i] = tanh(tanh_c_t[i])

            h_t = o_t * tanh_c_t

            # Update prev h_t
            self.ht_prev = h_t

            # Update last output
            last_output = o_t

        print(f"Output : \n{last_output}")
        self.output = last_output
        return last_output
