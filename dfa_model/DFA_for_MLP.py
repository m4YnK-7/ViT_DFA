import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

dfa_global_error = None

def set_dfa_error(error_tensor: torch.Tensor):
    global dfa_global_error
    dfa_global_error = error_tensor.detach()

class LinearDFAFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, B, is_output_layer: bool):
        # Save input shape & metadata
        ctx.input_shape = input.shape
        ctx.use_bias = bias is not None
        ctx.is_output = is_output_layer

        # Normalize feedback matrix
        # B_sign_symmetric = B.sign() * weight.t().sign()  # Enforce sign symmetry between B and weight.T
        B_norm = F.normalize(B, dim=1)
        
        ctx.save_for_backward(input, weight, bias if bias is not None else torch.tensor([], device=input.device), B_norm)

        # Standard forward
        out = input @ weight.t()
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        global dfa_global_error
        input, weight, bias, B_norm = ctx.saved_tensors
        in_features = input.shape[-1]
        out_features = weight.shape[0]

        if ctx.is_output:
            # Capture global error and use local gradient for this layer
            err = grad_output.detach()
            set_dfa_error(err)
            local_error = grad_output
        else:
            # Ensure global error exists
            if dfa_global_error is None:
                raise RuntimeError("Global DFA error not set. Ensure output layer has is_output_layer=True.")

            # Project global error through normalized feedback
            # dfa_global_error: [batch, num_classes]
            # B_norm: [out_features, num_classes]
            local_error = dfa_global_error @ B_norm.t()

            # Broadcast if sequence dims
            if input.dim() > 2:
                expand_shape = list(input.shape[:-1]) + [out_features]
                local_error = local_error.unsqueeze(1).expand(*expand_shape)

        # Compute weight & bias grads
        flat_err = local_error.reshape(-1, out_features)
        flat_inp = input.reshape(-1, in_features)
        grad_weight = flat_err.t().matmul(flat_inp)
        grad_bias = flat_err.sum(0) if ctx.use_bias else None

        # DFA does not propagate to input
        grad_input = torch.zeros_like(input)

        # No gradient for B
        grad_B = None

        return grad_input, grad_weight, grad_bias, grad_B, None

class LinearDFA(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_classes: int, bias: bool = True, is_output_layer: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_output = is_output_layer

        # Learnable params
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2 / in_features) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Fixed feedback matrix B
        self.B = nn.Parameter(torch.randn(out_features, num_classes), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.B = nn.Parameter(self.B / self.B.norm(dim=1, keepdim=True), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearDFAFunction.apply(x, self.weight, self.bias, self.B, self.is_output)

# Example:
# head = LinearDFA(in_features=192, out_features=2, num_classes=2, bias=True, is_output_layer=True)
# mlp = nn.Sequential(
#     LinearDFA(in_f, hidden_f, num_classes),
#     nn.GELU(),
#     LinearDFA(hidden_f, out_f, num_classes)
# )
