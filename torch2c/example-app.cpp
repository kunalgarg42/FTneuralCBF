#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>
using namespace std;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  float user_in[9] = {};
  for (int i = 0; i<9; i++){
    cin >> user_in[i];
  }
  // inputs.push_back(torch::user_in);
  inputs.push_back(torch::from_blob(&user_in, {1, 9}, torch::kFloat32));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << '\n';

  std::cout << "ok\n";
}

// Notes:
/*
Input: a 12-D state with structure [x, y, z, psi, theta, phi, u, v, w, r, q, p]
Output: a 13-D vector [h grad_h]

To read:
For creation and compilation of this example-app.cpp file, refer to:
https://pytorch.org/tutorials/advanced/cpp_export.html

To be done:
Get continuous state information for the input
Call the function in line 35 and get the output
Define a new function that creates f(x) and g(x) (refer to Crazyflie.py file in the dynamics folder)
- This function takes same state vector and some params values:
params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}
- and generates f(x) and g(x)
Create another function that takes (u_nominal, fx, gx, h, grad_h) as input
- setup Q, F, A and B using the above data
- setup a QPsolver (refer to https://github.com/google/osqp-cpp)
- get the output u 
*/
