from src.plotter import temperature_plotter
from src.solver import solver_first_order
from input import gen_input

if __name__ == "__main__": 

    input_dict = gen_input()
    temperature = solver_first_order(input_dict)
    temperature_plotter(temperature, input_dict)