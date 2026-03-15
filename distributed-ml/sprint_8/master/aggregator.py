import os
import torch

class Aggregator:
    def __init__(self, registry):
        self.registry = registry

    def fed_average(self, step_dir, alive_list,):
        aggregated = None
        loaded = 0
        alive_count = len(alive_list)

        for worker_id in alive_list:
            worker_index = self.registry.get_worker_index(worker_id) 
            weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")
            
            if not os.path.exists(weights_path):
                print(f"No se encontró {weights_path}")
                continue
            
            worker_weights = torch.load(weights_path, weights_only=True)
            if aggregated is None:
                aggregated = {k: v.clone() for k, v in worker_weights.items()}
            else:
                for key in aggregated.keys():
                    aggregated[key] += worker_weights[key]
            del worker_weights
            loaded += 1

        if loaded < alive_count:
            raise RuntimeError(f"Solo se encontraron {loaded}/{alive_count} workers en step {step}")

        for key in aggregated.keys():
            aggregated[key] /= loaded

        return aggregated

    def fed_prox(self):
        pass

    def fed_median(self):
        pass