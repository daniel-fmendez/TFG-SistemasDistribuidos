import torch

class Compressor:
    def __init__(self, strategy="none", k_ratio=0.1, quantization_bits=16):
        self.strategy = strategy
        self.k_ratio = k_ratio
        self.quantization_bits = quantization_bits

    def compress(self, state_dict):
        if self.strategy == "none":
            return state_dict, {}
        elif self.strategy == "quantization":
            return self._quantize(state_dict)
        elif self.strategy == "top_k":
            return self._top_k(state_dict)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")

    def decompress(self, state_dict, metadata):
        if self.strategy == "none":
            return state_dict
        elif self.strategy == "quantization":
            return self._dequantize(state_dict, metadata)
        elif self.strategy == "top_k":
            return self._reconstruct_top_k(state_dict, metadata)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
    
        
    def _quantize(self, state_dict):
        compressed = {}
        metadata = {}

        for key, tensor in state_dict.items():
            if self.quantization_bits == 16:
                compressed[key] = tensor.half()
                metadata[key] = {"bits": 16}
            elif self.quantization_bits == 8:
                scale = tensor.abs().max() / 127
                compressed[key] = (tensor / scale).to(torch.int8)
                metadata[key] = {"bits": 8, "scale": scale.item()}
        return compressed, metadata
    
    def _dequantize(self, state_dict, metadata):
        restored = {}

        for key, tensor in state_dict.items():
            bits = metadata[key]["bits"]

            if bits == 16:
                restored[key] = tensor.float()
            elif bits == 8:
                scale = metadata[key]["scale"]
                restored[key] = tensor.float() * scale
        return restored
    
    def _top_k(self, state_dict):
        compressed = {}
        metadata = {}

        for key, tensor in state_dict.items():
            flat = tensor.flatten()
            k = max(1, int(len(flat) * self.k_ratio))

            values, indices = torch.topk(flat.abs(), k)
            actual_values = flat[indices]

            compressed[key] = actual_values
            metadata[key] = {
                "indices": indices,
                "shape": tensor.shape
            }

        return compressed, metadata

    def _reconstruct_top_k(self, state_dict, metadata):
        restored = {}

        for key, values in state_dict.items():
            shape = metadata[key]["shape"]
            indices = metadata[key]["indices"]

            flat = torch.zeros(shape.numel())
            flat[indices] = values.float()
            restored[key] = flat.reshape(shape)

        return restored