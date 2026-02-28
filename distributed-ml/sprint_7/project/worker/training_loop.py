import datetime
import torch
import torch.nn.functional as F
import training_pb2
from torch.utils.data import DataLoader
from torch.optim import AdamW

class ModelOutput:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss

class TrainingLoop:
    def __init__(self, config, device, grpc_client, synchronizer):
        self.cfg = config
        self.device = device
        self.client = grpc_client
        self.synchronizer = synchronizer

    def run(self,  worker_id, model, dataset, dataset_info, signal):
        start = int(signal.start)
        end = int(signal.end)
        report_step = int(signal.report_step)

        shard_len = self.cfg.total_samples // self.cfg.num_workers
        worker_index = start // shard_len

        train_loader, optim, accumulation_steps = self._setup(model, dataset)

        batch_counter = 0
        for epoch in range(self.cfg.epochs):
            sync_every = self.cfg.sync_every_early if epoch == 0 else self.cfg.sync_every
            correct_predictions = 0
            total_samples = 0
            pending_gradients = False

            for step, batch in enumerate(train_loader):
                outputs, labels = self._forward(model, batch, dataset_info)
                outputs.loss.backward()
                pending_gradients = True

                if (step + 1) % accumulation_steps == 0:
                    optim.step()
                    optim.zero_grad()
                    pending_gradients = False
                    batch_counter += 1

                    if batch_counter % sync_every == 0:
                        self.synchronizer.sync(model, batch_counter)

                if step % report_step == 0:
                    correct_predictions, total_samples = self._report_metrics(
                        worker_id, epoch, step, outputs, labels,  # <- labels en vez de batch
                        correct_predictions, total_samples
                    )

            if pending_gradients:
                optim.step()
                optim.zero_grad()
                self.synchronizer.sync(model, batch_counter)
                
    def _setup(self, model, dataset):
        train_loader = DataLoader(dataset, batch_size=self.cfg.micro_batch_size, shuffle=True)
        optim = AdamW(model.parameters(), lr=self.cfg.learning_rate)
        accumulation_steps = self.cfg.batch_size // self.cfg.micro_batch_size
        return train_loader, optim, accumulation_steps

    def _forward(self, model, batch, dataset_info):
        labels = batch["label"].to(self.device)

        if dataset_info["type"] == "text_classification":
            outputs = model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=labels
            )
            return outputs, labels
        
        elif dataset_info["type"] == "image_classification":
            pixel_values = batch["pixel_values"].to(self.device)
            raw_output = model(pixel_values)
            logits = raw_output.logits if hasattr(raw_output, 'logits') else raw_output
            loss = F.cross_entropy(logits, labels)
            return ModelOutput(logits=logits, loss=loss), labels
        
    def _report_metrics(self, worker_id, epoch, step, outputs, labels, correct, total):
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0

        try:
            self.client.report_metrics(training_pb2.MetricData(
                worker_id=worker_id,
                epoch=epoch,
                step=step,
                loss=outputs.loss.item(),
                accuracy=accuracy,
                timestamp=datetime.datetime.now().isoformat(),
            ))
            print(f"Epoch {epoch}, Step {step}: Loss={outputs.loss:.4f}, Acc={accuracy:.4f}")

        except Exception as e:
            print(f"Error reportando métricas: {e}")

        return correct, total