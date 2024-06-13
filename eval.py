# eval.py

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from models.text_encoder import CLIPTextEncoder
from models.image_encoder import CLIPImageEncoder
from models.contrastive_model import ContrastiveLearningModel
from fusion_models.fusion_fc import FusionFC
from data.custom_dataset import CustomImageCaptionDataset
import open_clip
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a trained contrastive model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the model weights file.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--model_name', type=str, default='ViT-L-14', help='CLIP model name.')
    parser.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k', help='Pretrained weights for the CLIP model.')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K accuracy to evaluate.')
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the Fusion Model
    fusion_model = FusionFC(input_dim=2 * 768, output_dim=768, dropout=0.1).cuda()
    contrastive_model = ContrastiveLearningModel(text_encoder, image_encoder, fusion_model).cuda()

    # Load the trained model weights
    contrastive_model.load_state_dict(torch.load(args.weights_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contrastive_model.to(device)

    # Evaluation
    all_queries = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            target_images = batch['target_image'].to(device)
            ref_images = batch['ref_image'].to(device)
            tokenized_captions = batch['tokenized_caption'].to(device)
            query, target = contrastive_model(tokenized_captions, ref_images, target_images)
            all_queries.append(query)
            all_targets.append(target)

    # Concatenate all queries and targets
    all_queries = torch.cat(all_queries, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Normalize the embeddings to compute cosine similarity
    all_queries = torch.nn.functional.normalize(all_queries, dim=1)
    all_targets = torch.nn.functional.normalize(all_targets, dim=1)

    # Compute cosine similarity
    similarity_matrix = torch.matmul(all_queries, all_targets.T)

    # Calculate top-K accuracy
    _, top_k_indices = torch.topk(similarity_matrix, args.top_k, dim=1)

    # Assuming your dataset has a mapping from index to the correct target index
    correct_targets = torch.arange(len(all_queries)).to(device)
    top_k_accuracy = (top_k_indices == correct_targets.unsqueeze(1)).sum().item() / len(all_queries)

    print(f'Top-{args.top_k} accuracy: {top_k_accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
