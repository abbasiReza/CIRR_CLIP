# train.py

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from models.text_encoder import CLIPTextEncoder
from models.image_encoder import CLIPImageEncoder
from models.contrastive_model import ContrastiveLearningModel
from fusion_models.fusion_fc import FusionFC
from fusion_models.fusion_resnet import FusionResNet
from fusion_models.fusion_transformer import FusionTransformer
from fusion_models.fusion_attention import FusionAttention
from fusion_models.fusion_gnn import FusionGNN
from utils.utils import initialize_writer
from data.custom_dataset import CustomImageCaptionDataset
import open_clip
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a contrastive model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--run_name', type=str, default='simple', help='Name for the TensorBoard run.')
    parser.add_argument('--model_name', type=str, default='ViT-L-14', help='CLIP model name.')
    parser.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k', help='Pretrained weights for the CLIP model.')
    parser.add_argument('--lr_text', type=float, default=3e-5, help='Learning rate for the text encoder.')
    parser.add_argument('--lr_image', type=float, default=3e-4, help='Learning rate for the image encoder.')
    parser.add_argument('--lr_fusion', type=float, default=3e-4, help='Learning rate for the fusion model.')
    parser.add_argument('--save_path', type=str, default='../weights/', help='Path to save the model weights.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize the models
    clip_model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    text_encoder = CLIPTextEncoder(embed_dim=768, context_length=77, vocab_size=49408, transformer_width=768,
                                   transformer_heads=12, transformer_layers=12)
    image_encoder = CLIPImageEncoder(embed_dim=768, image_resolution=224, vision_layers=24, vision_width=1024,
                                     vision_patch_size=14, vision_heads=16)
    text_encoder.load_state_dict(clip_model.state_dict(), strict=False)
    image_encoder.load_state_dict(clip_model.state_dict(), strict=False)

    # Load the dataset
    df = pd.read_csv(args.data_path)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = CustomImageCaptionDataset(dataframe=df, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the Fusion Model
    fusion_model = FusionFC(input_dim=2 * 768, output_dim=768, dropout=0.1).cuda()
    contrastive_model = ContrastiveLearningModel(text_encoder, image_encoder, fusion_model).cuda()

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on GPU." if torch.cuda.is_available() else "Training on CPU.")

    # Initialize TensorBoard writer
    writer = initialize_writer(args.run_name)

    # Move model to device
    contrastive_model.to(device)

    # Create parameter groups
    optimizer = optim.Adam([
        {'params': contrastive_model.text_encoder.parameters(), 'lr': args.lr_text},
        {'params': contrastive_model.image_encoder.parameters(), 'lr': args.lr_image},
        {'params': contrastive_model.fusion_model.parameters(), 'lr': args.lr_fusion}
    ])

    # Initialize the scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, cooldown=0, verbose=True)

    # Training Loop
    for epoch in range(args.num_epochs):
        contrastive_model.train()
        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            target_images = batch['target_image'].to(device)
            ref_images = batch['ref_image'].to(device)
            tokenized_captions = batch['tokenized_caption'].to(device)

            optimizer.zero_grad()
            fused_features, image_only_features = contrastive_model(tokenized_captions, ref_images, target_images)
            loss = contrastive_model.contrastive_loss_clip(fused_features, image_only_features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                writer.add_scalar('Training Loss', running_loss / 10, epoch * len(dataloader) + i)
                running_loss = 0.0

        average_loss = running_loss / len(dataloader)
        scheduler.step(average_loss)
        if epoch % 5 == 0:
            torch.save(contrastive_model.state_dict(), os.path.join(args.save_path, f'contrastive_model_{epoch}_simple.pt'))
        print(f'Epoch {epoch + 1} Average Loss: {average_loss}')
        writer.add_scalar('Average Loss', average_loss, epoch)

    # Close the TensorBoard writer
    writer.close()
    torch.save(contrastive_model.state_dict(), os.path.join(args.save_path, 'contrastive_model_final_simple.pt'))

    print('Finished Training')

if __name__ == '__main__':
    main()
