import json
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
import os
import shutil
import time
import numpy as np

from model_running.dataset.custom_dataset import JointCapPuncDataset
from model_running.base_models import PretrainedModel
from model_running.model_utils.evaluation_errors_utils import error


IGNORE_INDEX = -100

class ModelExecutor:
    def __init__(
        self,
        batch_size: int = 16,
        model_name: str = "TwoheartsTechnologies/GoiBERT-S",
        checkpoint_path: str = None,
        num_cap: int = 0,
        num_punc: int = 4,
        cache_dir: str = None,
        freeze_flag: bool = False,
        dropout_rate: float = 0.1,
        num_epochs: int = 100,
        learning_rate: int = 2e-5,
        weight_decay: float = 0.01,
        device: str = "cuda"
    ):
        super(ModelExecutor, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.num_cap = num_cap
        self.num_punc = num_punc

        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.model = PretrainedModel(
            num_punc=self.num_punc,
            num_cap=self.num_cap,
            model_name=model_name,
            cache_dir=cache_dir,
            freeze_model=freeze_flag,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.cap_loss_fn = CrossEntropyLoss()
        self.punc_loss_fn = CrossEntropyLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.epoch = 1
        self.patience = 0
        self.num_epochs = num_epochs

    def create_loader(
        self,
        train_dataset: JointCapPuncDataset = None,
        validation_dataset: JointCapPuncDataset = None,
        test_dataset: JointCapPuncDataset = None,
        data_type: str = "train"
    ):
        """Create data loader."""
        if data_type == "train":
            self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=train_dataset.collate_fn)
            self.validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=1, collate_fn=validation_dataset.collate_fn)
            self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate_fn)
        else:
            self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate_fn)

    def run(
        self,
        train_dataset: JointCapPuncDataset,
        validation_dataset: JointCapPuncDataset,
        test_dataset: JointCapPuncDataset,
        patience_threshold: int = 10
    ):
        """Run executor."""
        self.create_loader(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset
        )

        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            if os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
                best_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
                best_mean_f1_score = best_checkpoint["valid_mean_f1_score"]
            else:
                best_mean_f1_score = -1
            
            last_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            self.epoch = last_checkpoint["epoch"] + 1
            self.model.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
            training_time = last_checkpoint["training_time"]
            
            print(f"Resuming from epoch {self.epoch}")
        else:
            training_time = 0
            best_mean_f1_score = -1
        
        print("Start training!")
        while True:
            start_time = time.time()
            train_loss = self.train()
            end_time = time.time()

            training_time += (end_time - start_time)

            best = False
            valid_result = self.evaluate(type="validation")
            valid_f1 = valid_result["mean_f1_score"]
            
            self.save_checkpoint(
                train_loss=train_loss,
                valid_loss=valid_result["loss"],
                valid_metrics=valid_result["metrics"],
                valid_mean_f1_score=valid_result["mean_f1_score"],
                training_time=training_time
            )

            if valid_f1 > best_mean_f1_score:
                best = True
                best_mean_f1_score = valid_f1
                self.patience = 0
            else:
                self.patience += 1
            
            if best:
                shutil.copyfile(
                    os.path.join(self.checkpoint_path, "last_model.pth"), 
                    os.path.join(self.checkpoint_path, "best_model.pth")
                )

                with open(os.path.join(self.checkpoint_path, "best_dev_results.json"), "w", encoding="utf-8") as file:
                    json.dump(valid_result, file, ensure_ascii=False, indent=4)

            if self.epoch == self.num_epochs or self.patience == patience_threshold:
                break

            self.epoch += 1

        print("Finish training!")

        print("Start testing!")
        checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_time = time.time()
        test_result = self.evaluate(type="test")
        end_time = time.time()

        elapsed = end_time - start_time

        with open(os.path.join(self.checkpoint_path, "test_results.json"), "w", encoding="utf-8") as file:
            json.dump(test_result, file, ensure_ascii=False, indent=4)

        print(f"Thời gian tính toán (test dataset): {elapsed:.2f} giây ({elapsed / 60:.2f} phút)")

    def train(self):
        """Train model."""
        self.model.train()

        running_loss = 0.0
        with tqdm(desc="Epoch %d - Training" % self.epoch, unit="it", total=len(self.train_loader)) as pbar:
            for i, batch in enumerate(self.train_loader, start=1):
                cap_labels = batch["cap_labels"].to(self.device)
                punc_labels = batch["punc_labels"].to(self.device)

                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                punc_logits, cap_logits, _ = self.model(**inputs)
               
                loss = self._compute_loss(punc_logits, cap_logits, punc_labels, cap_labels)

                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_postfix(loss=f"{running_loss / i}")
                pbar.update()
        
        return running_loss / len(self.train_loader)
    
    def evaluate(
        self,
        type: str
    ):
        """Evaluate model."""
        loader = self.validation_loader if type == "validation" else self.test_loader

        self.model.eval()

        all_cap_preds, all_cap_labels = [], []
        all_punc_preds, all_punc_labels = [], []
        results = []

        running_loss = 0.0
        with tqdm(desc="Epoch %d - Evaluation" % self.epoch, unit="it", total=len(loader)) as pbar:
            for i, batch in enumerate(loader, start=1):
                with torch.no_grad():
                    cap_labels = batch["cap_labels"].to(self.device)
                    punc_labels = batch["punc_labels"].to(self.device)
                    
                    inputs = {
                        "input_ids": batch["input_ids"].to(self.device),
                        "attention_mask": batch["attention_mask"].to(self.device)
                    }
                    punc_logits, cap_logits, _ = self.model(**inputs)

                    loss = self._compute_loss(punc_logits, cap_logits, punc_labels, cap_labels)
                    running_loss += loss.item()

                cap_pred = torch.argmax(cap_logits, dim=-1)
                punc_pred = torch.argmax(punc_logits, dim=-1)

                valid = cap_labels != IGNORE_INDEX
                cp = cap_pred[valid].tolist()
                cl = cap_labels[valid].tolist()
                pp = punc_pred[valid].tolist()
                pl = punc_labels[valid].tolist()

                all_cap_preds.extend(cp)
                all_cap_labels.extend(cl)
                all_punc_preds.extend(pp)
                all_punc_labels.extend(pl)

                results.append({
                    "cap_prediction": cp,
                    "cap_label": cl,
                    "punc_prediction": pp,
                    "punc_label": pl,
                })

                pbar.set_postfix(loss=f"{running_loss / i:.4f}")
                pbar.update()

        cap_acc, cap_f1, cap_precision, cap_recall = error(
            np.array(all_cap_labels),
            np.array(all_cap_preds),
            self.num_cap
        )
        punc_acc, punc_f1, punc_precision, punc_recall = error(
            np.array(all_punc_labels),
            np.array(all_punc_preds),
            self.num_punc
        )

        metrics = {
            "cap": {"accuracy": cap_acc, "f1_score": cap_f1, "precision": cap_precision, "recall": cap_recall},
            "punc": {"accuracy": punc_acc, "f1_score": punc_f1, "precision": punc_precision, "recall": punc_recall},
        }

        metrics["mean_f1_score"] = (metrics["cap"]["f1_score"] + metrics["punc"]["f1_score"]) / 2

        print(
            f"[Cap]  Acc {cap_acc:.4f} | F1 {cap_f1:.4f} | P {cap_precision:.4f} | R {cap_recall:.4f}\n"
            f"[Punc] Acc {punc_acc:.4f} | F1 {punc_f1:.4f} | P {punc_precision:.4f} | R {punc_recall:.4f}\n"
        )

        return {
            "results": results,
            "loss": running_loss / len(loader),
            "predictions": {"cap": all_cap_preds, "punc": all_punc_preds},
            "targets": {"cap": all_cap_labels, "punc": all_punc_labels},
            "metrics": metrics,
            "mean_f1_score": metrics["mean_f1_score"]
        }

    def save_checkpoint(
        self,
        train_loss: float,
        valid_loss: float,
        valid_metrics: dict,
        valid_mean_f1_score: float,
        training_time: float
    ):
        """Save checkpoint."""
        dict_for_saving = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_metrics": valid_metrics,
            "valid_mean_f1_score": valid_mean_f1_score,
            "training_time": training_time
        }

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def load_checkpoint(self, file_path: str):
        """Load checkpoint."""
        if not os.path.exists(file_path):
            return None
        print("Loading checkpoint from ", file_path)
        checkpoint = torch.load(file_path)
        return checkpoint

    def _compute_loss(self, punc_logits, cap_logits, punc_labels, cap_labels):
        loss_punc = self.punc_loss_fn(punc_logits.reshape(-1, punc_logits.size(-1)), punc_labels.reshape(-1))
        loss_cap = self.cap_loss_fn(cap_logits.reshape(-1, cap_logits.size(-1)), cap_labels.reshape(-1))
        return loss_punc + loss_cap

    
    # def run_evaluation(
    #     self,
    #     test_dataset: JointCapPuncDataset,
    #     dataset_name: str
    # ):
    #     """Run executor."""
    #     self.create_loader(
    #         test_dataset=test_dataset,
    #         data_type="test"
    #     )

    #     if os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
    #         best_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
    #     else:
    #         raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path!")

    #     print("Start testing!")
    #     self.epoch = best_checkpoint["epoch"]
    #     self.model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)

    #     start_time = time.time()
    #     (
    #         test_results,
    #         test_predictions,
    #         test_targets,
    #         _,
    #         test_accuracy,
    #         test_f1_score,
    #         test_precision,
    #         test_recall
    #     ) = self.evaluate(type="test")
    #     end_time = time.time()

    #     elapsed = end_time - start_time

    #     save_json(
    #         {
    #             "time": elapsed,
    #             "predictions": test_predictions.tolist(),
    #             "targets": test_targets.tolist(),
    #             "results": test_results,
    #             "accuracy": test_accuracy,
    #             "f1_score": test_f1_score,
    #             "precision": test_precision,
    #             "recall": test_recall
    #         },
    #         os.path.join(self.checkpoint_path, f"{dataset_name}_test_results.json")
    #     )

    #     print(f"Thời gian tính toán (test dataset): {elapsed:.2f} giây ({elapsed / 60:.2f} phút)")
