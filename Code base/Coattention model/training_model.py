import sys 
sys.path.append('E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Trainer')

import torch
from torch import nn
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset import get_data_loaders
from model import CoattentionModel



def main():


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    model = CoattentionModel()
    

    lmdb_path = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Final VMD/VMD.lmdb"
    seed = 42
    train_loader, val_loader = get_data_loaders(lmdb_path=lmdb_path, batch_size=5, num_workers=12, prefetch_factor=2, seed=seed)


    learning_rate = 2e-5
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


    trainer = ModularTrainer(
    model=model,
    train_loader=train_loader,  
    val_loader=val_loader,
    criterion=loss_fn,
    optimizer=optimizer,
    device=device,
    config={'epochs': 50,
            'save_dir' : 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints vmd_attention_aa_imf',
            'log_dir' : 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Logs',
            'log_file' : 'vmd_attention_aa_imf.log',
            'verbose' : True
        }
    )


    #history = trainer.train(resume_from="E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Checkpoints vmd_coattention_ba_mel_imf/model_epoch_50.pth")
    history = trainer.train(resume_from=None)

    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.savefig('E:/Amrita/Subjects/Sem 5/BMSP paper work/Train data/Graphs/vmd_attention_aa_imf.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()
