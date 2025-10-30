import sys

def test_imports():
    print("Testing imports...")
    print("-" * 50)

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'umap': 'UMAP'
    }

    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            failed.append(name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True


def test_pytorch():
    print("\n" + "=" * 50)
    print("Testing PyTorch Setup")
    print("=" * 50)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  Running on CPU (slower training)")

    print("\nTesting tensor operations...")
    x = torch.randn(10, 3, 224, 224)
    print(f"Created test tensor: {x.shape}")
    print("✅ PyTorch working correctly!")


def test_modules():
    print("\n" + "=" * 50)
    print("Testing Project Modules")
    print("=" * 50)

    modules = [
        'data_loader',
        'model_ssl',
        'train_ssl',
        'fine_tune',
        'evaluate',
        'utils'
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except Exception as e:
            print(f"✗ {module}.py - ERROR: {str(e)[:50]}")
            failed.append(module)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All modules loaded successfully!")
        return True


def test_model_creation():
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)

    import torch
    from model_ssl import create_simclr_model, NTXentLoss, create_classifier

    try:
        print("Creating SimCLR model...")
        model = create_simclr_model(backbone='resnet18', projection_dim=128)
        x = torch.randn(2, 3, 224, 224)
        h, z = model(x)
        print(f"  Input: {x.shape}")
        print(f"  Features: {h.shape}")
        print(f"  Projection: {z.shape}")
        print("  ✓ SimCLR model works!")

        print("\nTesting NTXent loss...")
        criterion = NTXentLoss(temperature=0.5)
        z1 = torch.randn(4, 128)
        z2 = torch.randn(4, 128)
        loss = criterion(z1, z2)
        print(f"  Loss value: {loss.item():.4f}")
        print("  ✓ NTXent loss works!")

        print("\nCreating classifier...")
        classifier = create_classifier(model.encoder, num_classes=3)
        logits = classifier(x)
        print(f"  Output: {logits.shape}")
        print("  ✓ Classifier works!")

        print("\n✅ All models working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    print("\n" + "=" * 50)
    print("Testing Data Loading")
    print("=" * 50)

    import os
    import shutil
    from data_loader import create_sample_dataset, create_ssl_dataloader

    try:
        test_dir = './test_data_temp'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        print("Creating small test dataset...")
        data_dir, label_file = create_sample_dataset(
            output_dir=test_dir,
            num_images=10
        )
        print(f"  ✓ Created 10 test images")

        print("\nTesting SSL dataloader...")
        loader = create_ssl_dataloader(data_dir, batch_size=4, num_workers=0)
        for batch in loader:
            view1, view2 = batch
            print(f"  View 1: {view1.shape}")
            print(f"  View 2: {view2.shape}")
            break
        print("  ✓ SSL dataloader works!")

        shutil.rmtree(test_dir)
        print(f"\nCleaned up test data")

        print("\n✅ Data loading working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 50)
    print("SSL HEALTHCARE - INSTALLATION TEST")
    print("=" * 50 + "\n")

    results = []

    results.append(("Package imports", test_imports()))

    if results[-1][1]:
        results.append(("PyTorch setup", test_pytorch()))
        results.append(("Project modules", test_modules()))

        if results[-1][1]:
            results.append(("Model creation", test_model_creation()))
            results.append(("Data loading", test_data_loading()))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Run full pipeline: python run_full_pipeline.py")
        print("  2. Or follow step-by-step guide in USAGE_GUIDE.md")
        print("  3. Check README.md for detailed documentation")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check Python version (requires 3.8+)")
        print("  - Verify file permissions")
        return 1


if __name__ == '__main__':
    sys.exit(main())
