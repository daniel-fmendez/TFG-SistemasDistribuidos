import os
import torch

class Aggregator:
    def __init__(self, registry, strategy, compressor):
        self.registry = registry
        self.strategy = strategy
        self.compressor = compressor

    def aggregate(self, step_dir, alive_list):
        if self.strategy == "fed_avg":
            return self.fed_average(step_dir, alive_list)
        elif self.strategy == "fed_median":
            return self.fed_median(step_dir, alive_list)
        elif self.strategy == "fed_trimmed_mean":
            return self.fed_trimmed_mean(step_dir, alive_list)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
        
    def fed_average(self, step_dir, alive_list):
        aggregated = None
        loaded = 0
        alive_count = len(alive_list)

        for worker_id in alive_list:
            worker_index = self.registry.get_worker_index(worker_id) 
            weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")
            
            if not os.path.exists(weights_path):
                print(f"No se encontró {weights_path}")
                continue
            
            worker_data = torch.load(weights_path, weights_only=True)
            compressed = worker_data["weights"]
            metadata = worker_data["metadata"]
            worker_weights = self.compressor.decompress(compressed, metadata)
            if aggregated is None:
                aggregated = {k: v.clone() for k, v in worker_weights.items()}
            else:
                for key in aggregated.keys():
                    aggregated[key] += worker_weights[key]
            del worker_weights
            loaded += 1

        if loaded < alive_count:
            raise RuntimeError(f"Solo se encontraron {loaded}/{alive_count} workers en step")

        for key in aggregated.keys():
            aggregated[key] /= loaded

        return aggregated

    def fed_median(self, step_dir, alive_list):
        alive_count = len(alive_list)
        all_weights = []
        loaded = 0

        for worker_id in alive_list:
            worker_index = self.registry.get_worker_index(worker_id) 
            weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")
            
            if not os.path.exists(weights_path):
                print(f"No se encontró {weights_path}")
                continue

            worker_data = torch.load(weights_path, weights_only=True)
            compressed = worker_data["weights"]
            metadata = worker_data["metadata"]
            worker_weights = self.compressor.decompress(compressed, metadata)
            all_weights.append(worker_weights)
            loaded += 1

        if loaded < alive_count:
            raise RuntimeError(f"Solo se encontraron {loaded}/{alive_count} workers en step")

        aggregated = {}
        for key in all_weights[0].keys():
            stacked = torch.stack([w[key].float() for w in all_weights], dim=0)
            aggregated[key] = torch.median(stacked, dim=0).values

        return aggregated
    
    def fed_trimmed_mean(self, step_dir, alive_list, k=1):
        alive_count = len(alive_list)
        all_weights = []
        loaded = 0

        for worker_id in alive_list:
            worker_index = self.registry.get_worker_index(worker_id) 
            weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")
            
            if not os.path.exists(weights_path):
                print(f"No se encontró {weights_path}")
                continue

            worker_weights = torch.load(weights_path, weights_only=True)
            all_weights.append(worker_weights)
            loaded += 1

        if loaded < alive_count:
            raise RuntimeError(f"Solo se encontraron {loaded}/{alive_count} workers en step")

        aggregated = {}
        for key in all_weights[0].keys():
            stacked = torch.stack([w[key].float() for w in all_weights], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k: len(all_weights) - k]
            aggregated[key] = trimmed.mean(dim=0)
        return aggregated
        
    # def fed_adam(self, step_dir, alive_list, global_weights):
