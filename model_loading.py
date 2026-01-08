def try_to_load_model_from_checkpoint_path(
        checkpoint_path: str,
) -> torch.nn.Module:
    from graph_learning.utils import load_checkpoint as load_model_checkpoint
    from graph_learning.utils.packaging import load_model_from_bundle
    from scripts.arch_study import ArchTestingModule

    # try multiple loading methods
    attempts = {
        "ArchTestingModule": lambda: ArchTestingModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=device
        ).model,
        "Bare Torch Load": lambda: torch.load(checkpoint_path, map_location="cpu"),
        "Cloudpickle Custom Loader": lambda: load_model_checkpoint(checkpoint_path, device="cpu"),
        "torch package bundle": lambda: load_model_from_bundle(checkpoint_path, map_location="cpu"),
    }

    last_exc = None
    for attempt_name, attempt in attempts.items():
        try:
            print(f"Trying to load model via: {attempt_name}")
            result = attempt()
            break
        except Exception as e:
            print(e, file=os.sys.stderr)
            last_exc = e
    else:
        raise last_exc

    model: torch.nn.Module = result
    return model

def get_model(output_dir) -> Optional[torch.nn.Module]:
    # try model_bundle.ptpkg
    try:
        return try_to_load_model_from_checkpoint_path(os.path.join(output_dir, 'model_bundle.ptpkg'))
    except Exception:
        pass
    # try best.ckpt
    try:
        return try_to_load_model_from_checkpoint_path(os.path.join(output_dir, 'best.ckpt'))
    except Exception:
        pass
    # 2. try last.ckpt
    try:
        return (
            try_to_load_model_from_checkpoint_path(os.path.join(output_dir, 'last.ckpt')))
    except Exception:
        pass
    # 3. try other files
    extensions = {'.pt', '.ckpt', '.ckp', '.ptpkg'}
    files = [
        p.resolve() for p in Path('.').rglob('*')
        if p.is_file() and p.suffix in extensions
    ]
    for file in files:
        try:
            return try_to_load_model_from_checkpoint_path(str(file))
        except Exception:
            pass

    return None


def fix_model(model: torch.nn.Module) -> torch.nn.Module:
    # normalize model architecture ... add needed fields
    # basically a stupid monkey path
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, GINConv):
                module.in_channels = module.nn[0].in_features
                module.out_channels = module.nn[-1].out_features
    return model
