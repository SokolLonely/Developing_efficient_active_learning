"""

This script contains all models:

    - MLP: a simple feed forward multi-layer perceptron. Supports weight anchoring - Pearce et al. (2018)
    - GCN: a simple graph convolutional NN - Kipf & Welling (2016). Supports weight anchoring - Pearce et al. (2018)
    - Model: A wrapper class that contains a train(), and predict() loop
    - Ensemble: Class that ensembles n Model classes. Contains a train() method and an predict() method that outputs
        logits_N_K_C, defined as [N, num_inference_samples, num_classes]. Also has an optimize_hyperparameters() method.

    Author: Simon Ryabinkin, University of Calgary, 2025-2026, modified from Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm, GATConv, GINConv
from tqdm.auto import trange
#from sklearn.ensemble import RandomForestClassifier
#import datamol as dm
#from active_learning.hyperopt import optimize_hyperparameters
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
import pandas as pd
class MLP(torch.nn.Module): 
    def __init__(self, in_feats: int = 1024, n_hidden: int = 1024,function = 'relu', n_out: int = 2, n_layers: int = 3, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.gelu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
class SmilesMLP(torch.nn.Module): 
    def __init__(self, in_feats: int = 1024, n_hidden: int = 2048, n_out: int = 2, n_layers: int = 4, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.gelu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
class AttMLP(torch.nn.Module):
    def __init__(self, in_feats: int = 1024, n_hidden: int = 1024,function = 'relu', n_out: int = 2, n_layers: int = 3, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)
        if function == 'relu':
            self.func = F.relu
        elif function == 'gelu':
            self.func = F.gelu
        elif function == 'tanh':
            self.func = F.tanh
        elif function == 'leakyrelu':
            self.func = F.leaky_relu
        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        self.fc.append(torch.nn.MultiheadAttention(embed_dim=in_feats, num_heads=16, batch_first=True))
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        attn = self.fc[0]
        x, _ = attn(x, x, x)
        for lin, norm in zip(self.fc[1:], self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = self.func(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 5, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GCNConv(n_hidden, n_hidden))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class GAT(torch.nn.Module):#AFTER  DONE TESTING, CHANGE EPOCHS BACK TO 50
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GATConv(n_hidden, n_hidden, add_self_loops=True, negative_slope=0.2,
                                      heads=8, concat=False))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        SimpleMLP = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden))

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GINConv(nn=SimpleMLP))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
class Chemberta(torch.nn.Module):
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", epochs = 50, seed = 42, weight_decay = 0.01, **kwargs):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.epochs = epochs
        self.seed = seed
        self.weight_decay = weight_decay
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)    # def train(self):
    def compute_metrics(self, eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=1)
                acc = accuracy_score(labels, predictions)
                return {"accuracy": acc}
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f'best loss= {self.best_loss}')
        else:
            self.counter += 1
            print(f"{val_loss}> {self.best_loss - self.min_delta}")
            if self.counter >= self.patience:
                self.early_stop = True
from transformers import TrainingArguments, Trainer
class Model(torch.nn.Module):
    def __init__(self, architecture: str, n_hidden, n_layers=3, function = 'relu', epochs = 50, lr = 3e-4, **kwargs):
        super().__init__()
        #assert architecture in ['gcn', 'mlp', 'gat', 'gin',  'chemberta', 'morfeus_mlp', 'only_morfeus', 'mlp2048', 'robert768', 'chemgpt', 'acsf', 'maccs']
        self.architecture = architecture

# Default kwargs
        common_kwargs = dict(
        n_layers=n_layers,
        function=function,
        epochs=epochs,
        n_hidden = n_hidden,
        lr = lr,
        )

        ARCH_PARAMS = {
            "mlp":       (MLP, common_kwargs),
            "gcn":       (GCN, {}),
            "gin":       (GIN, {}),
            
            "chembert":  (Chemberta, {}),  # see if statement below
            "morfeus_mlp": (MLP, {**common_kwargs, "in_feats": 2850}),
            "only_morfeus": (MLP, {**common_kwargs, "in_feats": 1826}),
            "mlp2048":   (MLP, {**common_kwargs, "n_hidden": 2048, "in_feats": 2048}),
            "robert768": (MLP, {**common_kwargs, "in_feats": 768}),
            "chemgpt":   (MLP, {**common_kwargs, "n_hidden": 2048, "in_feats": 2048}),
            "maccs":     (MLP, {**common_kwargs, "n_layers": 2, "in_feats": 167}),
            "acsf":      (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 1200}),
            "mm":        (AttMLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 1200}),
            "mmnoatt":   (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 1200}),
            "amlp":      (AttMLP, common_kwargs),
            "x_512":     (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 512}),
            "x_256":     (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 256}),
            "x_128":     (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 128}),
            "mm_768":    (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 768}),
            "mm_512":    (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 512}),
            "mm_256":    (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 256}),
            "x4":        (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 2048}),
            "x4_1024":   (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 1024}),
            "x4_768":    (MLP, {**common_kwargs, "n_hidden": n_hidden, "in_feats": 768}),

}

        if architecture == "chemberta":
                self.model = Chemberta(**kwargs)
                self.training_args = TrainingArguments(
                    output_dir="./chemberta-finetuned",
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    num_train_epochs=epochs,
                    weight_decay=self.model.weight_decay,
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    report_to="none",
                    )
        elif architecture in ARCH_PARAMS:
                model_class, params = ARCH_PARAMS[architecture]
                self.model = model_class(**params, **kwargs)
        else:
    # Default fallback
             self.model = GAT(**kwargs)

        
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.loss_fn = torch.nn.NLLLoss()
        if self.architecture != 'chemberta':
        # Move the whole model to the gpu
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model.lr,
                                          weight_decay=self.model.weight_decay)
        # Save initial weights in the model for the anchored regularization and move them to the gpu
            if self.model.anchored:
                self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
                self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:
      if self.architecture != 'chemberta':
        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()
        print()
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        early_stopping = EarlyStopping(patience=6, min_delta=0)
        for i in bar:#epoch loop 
            running_loss = 0
            items = 0
            print(i, end = ' ')
            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()
                
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                        y_hat = self.model(x)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    loss = self.loss_fn(y_hat, y.squeeze())

                    if self.model.anchored:
                        # Calculate the total anchored L2 loss
                        l2_loss = 0
                        for param_name, params in self.model.named_parameters():
                            anchored_param = self.model.anchor_weights[param_name]

                            l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param,
                                                                                   params - anchored_param).sum()

                        # Add anchored loss to regular loss according to Pearce et al. (2018)
                        loss = loss + l2_loss

                    scaler.scale(loss).backward()
                    # loss.backward()
                    scaler.step(self.optimizer)
                    # self.optimizer.step()
                    scaler.update()
                    scheduler.step(loss)
                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            self.epoch += 1
            early_stopping(epoch_loss)
            if early_stopping.early_stop:
               #print("Early stopping triggered")
               #break
               pass
        
      else:
            
            trainer = Trainer(
                model=self.model.model,
                args=self.training_args,
                train_dataset=dataloader,
                eval_dataset=dataloader,
                tokenizer=self.model.tokenizer,
                compute_metrics=self.model.compute_metrics,
                data_collator=self.model.data_collator,
    #compute_metrics=compute_metrics
                    )
            print(type(dataloader))
            trainer.train()
      print('train function finished')
    def predict(self, dataloader: DataLoader, architecture) -> Tensor:
      """ Predict
        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
      if architecture != 'chemberta':
        y_hats = torch.tensor([]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats
      else:
          trainer = Trainer(
                model=self.model.model,
                args=self.training_args,
                train_dataset=dataloader,
                eval_dataset=dataloader,
                tokenizer=self.model.tokenizer,
                compute_metrics=self.model.compute_metrics,
                data_collator=self.model.data_collator,
    #compute_metrics=compute_metrics
                    )
          results = trainer.predict(dataloader)
          return results


           
class Ensemble(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, ensemble_size: int = 10, seed: int = 0,n_layers =3, n_hidden = 1024, function = 'relu',epochs = 50, architecture: str = 'mlp', lr =3e-4, **kwargs) -> None:
        self.ensemble_size = ensemble_size
        if architecture == 'chemberta':
            self.ensemble_size = 1
        self.architecture = architecture
        self.seed = seed
        self.epochs = epochs
        self.n_layers = n_layers
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, self.ensemble_size)
        self.models = {i: Model(seed=s, architecture=architecture, n_layers =n_layers, n_hidden = n_hidden, epochs = epochs,lr = lr, function = function, **kwargs) for i, s in enumerate(self.seeds)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> None:
        for i, m in self.models.items():
            m.train(dataloader, **kwargs)

    def predict(self, dataloader, architecture, **kwargs) -> Tensor:
       # """ logits_N_K_C = [N, num_inference_samples, num_classes] """
       # if architecture != 'chemberta':
            raw_logits_N_K_C = [m.predict(dataloader, architecture) for m in self.models.values()]
            if architecture != 'chemberta':
                logits_N_K_C = torch.stack(raw_logits_N_K_C, 1)  # [N, num_inference_samples, num_classes]
                return logits_N_K_C#64, 10, 2
            else:
                arrays = []
                for el in raw_logits_N_K_C:
                    arrays.append( el.predictions)
                    #print(prediction.shape)
                result = np.stack(arrays)
                result = np.transpose(result, (1, 0, 2))
                #all_preds = all_preds[np.newaxis, ...]
                return torch.tensor(result, dtype=torch.float32)
       # else:
           # return self.models.values()[0].predict(dataloader, architecture)

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"

class RfEnsemble():
    """ Ensemble of RFs"""
    def __init__(self, ensemble_size: int = 10, seed: int = 0, **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, ensemble_size)
        self.models = {i: RandomForestClassifier(random_state=s, class_weight="balanced", **kwargs) for i, s in enumerate(self.seeds)}

    def train(self, x, y, **kwargs) -> None:
        for i, m in self.models.items():
            m.fit(x, y)

    def predict(self, x, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        # logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)
        eps = 1e-10  # we need to add this so we don't get divide by zero errors in our log function

        y_hats = []
        for m in self.models.values():

            y_hat = torch.tensor(m.predict_proba(x) + eps)
            if y_hat.shape[1] == 1:  # if only one class if predicted with the RF model, add a column of zeros
                y_hat = torch.cat((y_hat, torch.zeros((y_hat.shape[0], 1))), dim=1)
            y_hats.append(y_hat)

        logits_N_K_C = torch.stack(y_hats, 1)

        logits_N_K_C = torch.log(logits_N_K_C)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} RF Classifiers"
