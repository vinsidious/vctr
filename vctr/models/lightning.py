import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress import rich_progress
from torch import nn, optim, torch  # type: ignore
from torchmetrics import F1Score
from vctr.data.data_loader import get_data_with_features_and_labels
from vctr.data.lstm_preprocessor import preprocess_data


class LSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=3, learning_rate=1e-3):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.model = nn.Sequential(self.lstm, self.gru, nn.Flatten(start_dim=1), self.fc)

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        output = self.fc(gru_out[:, -1, :])
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_labels = torch.argmax(y_hat, dim=1)

        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True, enable_graph=True)

        f1 = F1Score(task='multiclass', num_classes=3, average='macro')
        f1 = f1(y_hat_labels.cpu(), y.cpu())

        self.log('train_f1', f1, prog_bar=True, on_epoch=True, on_step=True, logger=True, enable_graph=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_labels = torch.argmax(y_hat, dim=1)

        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, enable_graph=True)

        f1 = F1Score(task='multiclass', num_classes=3, average='macro')
        f1 = f1(y_hat_labels.cpu(), y.cpu())

        self.log('val_f1', f1, prog_bar=True, on_epoch=True, logger=True, enable_graph=True)


def get_data_loader(X, y, batch_size=128, num_workers=12):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_train_val_loaders(*args, test_pct=0.25, batch_size=128, sequence_length=24, **kwargs):
    X_train, y_train, X_val, y_val = preprocess_data(
        *get_data_with_features_and_labels(*args, **kwargs),
        lookback=sequence_length,
        test_pct=test_pct,
        torch=True,
    )
    return (
        get_data_loader(X_train, y_train, batch_size=batch_size),
        get_data_loader(X_val, y_val, batch_size=batch_size),
    )


def train_model(model, *args, max_epochs=50, **kwargs):
    progress_bar = RichProgressBar(
        theme=rich_progress.RichProgressBarTheme(
            progress_bar_finished='#27AE60',
            batch_progress='#F1C40F',
            progress_bar_pulse='#8E44AD',
            description='#F1C40F',
            processing_speed='#263238',
            progress_bar='#27AE60',
            metrics='#263238',
            time='#263238',
        )
    )

    train_loader, val_loader = get_train_val_loaders(*args, **kwargs)
    trainer = pl.Trainer(accelerator='mps', max_epochs=max_epochs, callbacks=[progress_bar])
    trainer.fit(model, train_loader, val_loader)
