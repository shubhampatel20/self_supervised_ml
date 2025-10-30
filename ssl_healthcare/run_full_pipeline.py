import os
import argparse
from pathlib import Path

from data_loader import create_sample_dataset
from train_ssl import train_ssl
from fine_tune import fine_tune
from evaluate import evaluate_model
from utils import set_seed, create_directory_structure, save_training_plots, save_finetuning_plots


def run_complete_pipeline(
    data_dir=None,
    create_data=True,
    num_images=500,
    ssl_epochs=50,
    finetune_epochs=30,
    batch_size_ssl=64,
    batch_size_ft=32,
    backbone='resnet18',
    num_classes=3,
    seed=42
):
    print("="*70)
    print("Self-Supervised Learning for Healthcare Images")
    print("Complete End-to-End Pipeline")
    print("="*70)

    set_seed(seed)
    print(f"\nRandom seed set to {seed} for reproducibility")

    base_dir = create_directory_structure('./ssl_project')

    if create_data:
        print("\n" + "="*70)
        print("STEP 1: Creating Synthetic Medical Image Dataset")
        print("="*70)
        data_dir, label_file = create_sample_dataset(
            output_dir=os.path.join(base_dir, 'data/medical_images'),
            num_images=num_images
        )
    else:
        if data_dir is None:
            raise ValueError("data_dir must be provided if create_data=False")
        label_file = os.path.join(data_dir, 'labels.csv')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

    print("\n" + "="*70)
    print("STEP 2: Self-Supervised Pretraining (SimCLR)")
    print("="*70)
    print("\nThis phase learns meaningful representations from UNLABELED images")
    print("using contrastive learning (SimCLR).\n")

    ssl_output_dir = os.path.join(base_dir, 'checkpoints/ssl')
    model_ssl, loss_history = train_ssl(
        data_dir=data_dir,
        output_dir=ssl_output_dir,
        backbone=backbone,
        projection_dim=128,
        batch_size=batch_size_ssl,
        num_epochs=ssl_epochs,
        lr=0.0003,
        temperature=0.5,
        num_workers=4,
        image_size=224
    )

    ssl_checkpoint = os.path.join(ssl_output_dir, 'best_ssl_model.pth')
    plot_path = os.path.join(base_dir, 'plots/ssl_training_loss.png')
    save_training_plots(loss_history, plot_path)

    print("\n" + "="*70)
    print("STEP 3: Fine-Tuning with SSL Pretraining")
    print("="*70)
    print("\nFine-tuning the pretrained encoder on LABELED data")
    print("for supervised classification.\n")

    ft_output_dir = os.path.join(base_dir, 'checkpoints/finetuned')
    model_ft_ssl, train_hist_ssl, val_hist_ssl = fine_tune(
        data_dir=data_dir,
        label_file=label_file,
        ssl_checkpoint=ssl_checkpoint,
        output_dir=ft_output_dir,
        backbone=backbone,
        num_classes=num_classes,
        batch_size=batch_size_ft,
        num_epochs=finetune_epochs,
        lr=0.001,
        freeze_encoder=True,
        unfreeze_after_epoch=int(finetune_epochs * 0.5),
        train_split=0.8,
        num_workers=4,
        image_size=224
    )

    save_finetuning_plots(
        train_hist_ssl,
        val_hist_ssl,
        os.path.join(base_dir, 'plots')
    )

    print("\n" + "="*70)
    print("STEP 4: Baseline (Training from Scratch)")
    print("="*70)
    print("\nTraining classifier from scratch WITHOUT SSL pretraining")
    print("for comparison.\n")

    scratch_output_dir = os.path.join(base_dir, 'checkpoints/scratch')
    model_scratch, train_hist_scratch, val_hist_scratch = fine_tune(
        data_dir=data_dir,
        label_file=label_file,
        ssl_checkpoint=None,
        output_dir=scratch_output_dir,
        backbone=backbone,
        num_classes=num_classes,
        batch_size=batch_size_ft,
        num_epochs=finetune_epochs,
        lr=0.001,
        freeze_encoder=False,
        train_split=0.8,
        num_workers=4,
        image_size=224
    )

    print("\n" + "="*70)
    print("STEP 5: Evaluation - SSL Pretrained Model")
    print("="*70)

    eval_dir_ssl = os.path.join(base_dir, 'evaluation_results/with_ssl')
    checkpoint_ssl = os.path.join(ft_output_dir, 'best_finetuned_model.pth')
    metrics_ssl = evaluate_model(
        checkpoint_path=checkpoint_ssl,
        data_dir=data_dir,
        label_file=label_file,
        output_dir=eval_dir_ssl,
        num_classes=num_classes,
        batch_size=batch_size_ft,
        num_workers=4,
        image_size=224
    )

    print("\n" + "="*70)
    print("STEP 6: Evaluation - Baseline Model (From Scratch)")
    print("="*70)

    eval_dir_scratch = os.path.join(base_dir, 'evaluation_results/from_scratch')
    checkpoint_scratch = os.path.join(scratch_output_dir, 'best_finetuned_model.pth')
    metrics_scratch = evaluate_model(
        checkpoint_path=checkpoint_scratch,
        data_dir=data_dir,
        label_file=label_file,
        output_dir=eval_dir_scratch,
        num_classes=num_classes,
        batch_size=batch_size_ft,
        num_workers=4,
        image_size=224
    )

    print("\n" + "="*70)
    print("STEP 7: Final Comparison")
    print("="*70)

    print("\nPerformance Comparison:")
    print("-" * 70)
    print(f"{'Metric':<20} {'With SSL Pretraining':<25} {'From Scratch':<25}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {metrics_ssl['accuracy']:>20.4f}    {metrics_scratch['accuracy']:>20.4f}")
    print(f"{'F1 Score':<20} {metrics_ssl['f1_score']:>20.4f}    {metrics_scratch['f1_score']:>20.4f}")
    print(f"{'ROC-AUC':<20} {metrics_ssl['roc_auc']:>20.4f}    {metrics_scratch['roc_auc']:>20.4f}")
    print("-" * 70)

    improvement_acc = (metrics_ssl['accuracy'] - metrics_scratch['accuracy']) * 100
    improvement_f1 = (metrics_ssl['f1_score'] - metrics_scratch['f1_score']) * 100
    improvement_auc = (metrics_ssl['roc_auc'] - metrics_scratch['roc_auc']) * 100

    print(f"\nImprovement with SSL:")
    print(f"  Accuracy: {improvement_acc:+.2f}%")
    print(f"  F1 Score: {improvement_f1:+.2f}%")
    print(f"  ROC-AUC:  {improvement_auc:+.2f}%")

    from utils import compare_models
    compare_models(
        ssl_results=metrics_ssl,
        scratch_results=metrics_scratch,
        output_dir=os.path.join(base_dir, 'plots')
    )

    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print(f"\nAll results saved to: {base_dir}")
    print(f"\nCheckpoints:")
    print(f"  - SSL Model: {ssl_checkpoint}")
    print(f"  - Fine-tuned (SSL): {checkpoint_ssl}")
    print(f"  - Fine-tuned (Scratch): {checkpoint_scratch}")
    print(f"\nEvaluation Results:")
    print(f"  - With SSL: {eval_dir_ssl}")
    print(f"  - From Scratch: {eval_dir_scratch}")
    print(f"\nPlots: {os.path.join(base_dir, 'plots')}")

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("1. Self-supervised learning leverages unlabeled data effectively")
    print("2. SSL pretraining improves performance with limited labeled data")
    print("3. The learned representations capture meaningful anatomical features")
    print("4. Fine-tuning strategy (freeze → unfreeze) is crucial for success")
    print("="*70)

    return {
        'ssl_metrics': metrics_ssl,
        'scratch_metrics': metrics_scratch,
        'ssl_checkpoint': checkpoint_ssl,
        'scratch_checkpoint': checkpoint_scratch,
        'base_dir': base_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run complete SSL pipeline for healthcare images'
    )
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to existing data directory (optional)')
    parser.add_argument('--create_data', action='store_true', default=True,
                        help='Create synthetic dataset')
    parser.add_argument('--num_images', type=int, default=500,
                        help='Number of synthetic images to generate')
    parser.add_argument('--ssl_epochs', type=int, default=50,
                        help='SSL pretraining epochs (use 100+ for real data)')
    parser.add_argument('--finetune_epochs', type=int, default=30,
                        help='Fine-tuning epochs')
    parser.add_argument('--batch_size_ssl', type=int, default=64,
                        help='Batch size for SSL training')
    parser.add_argument('--batch_size_ft', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    run_complete_pipeline(
        data_dir=args.data_dir,
        create_data=args.create_data,
        num_images=args.num_images,
        ssl_epochs=args.ssl_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size_ssl=args.batch_size_ssl,
        batch_size_ft=args.batch_size_ft,
        backbone=args.backbone,
        num_classes=args.num_classes,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
