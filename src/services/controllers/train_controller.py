import os
import time

import torch
import torch.nn as nn

def train_models(train_ds, train_loader, test_loader, config, model_registry):
    
    tuning_schedule = [
        {"epochs": config.epochs, "lr": config.lr},
        {"epochs": config.epochs + 10, "lr": config.lr * 0.1},
        {"epochs": config.epochs + 20, "lr": config.lr * 0.01},
    ]

    with config.logger:

        log = config.logger.log if config.logger else print

        for name, fns in model_registry.items():

            # ── Treino padrão ──────────────────────────────────────────────────
#            current_model = None
#
#            for attempt, hp in enumerate(tuning_schedule, 1):
#                _print_header(f"{name} | Tentativa {attempt} | epochs={hp['epochs']} | lr={hp['lr']}", log)
#
#                start = time.perf_counter()
#                model, accuracy = fns["train_fn"](
#                    train_loader,
#                    test_loader, 
#                    config,
#                    hp["epochs"], 
#                    hp["lr"], 
#                    current_model, 
#                )
#                current_model = model
#                _print_elapsed(start, log)
#
#                log(f"  → Acurácia: {accuracy*100:.2f}%")
#
#                if accuracy >= config.min_accuracy:
#                    log("  ✓ Acurácia mínima atingida!")
#                    break
#            else:
#                log(f"  ✗ Acurácia mínima não atingida após {len(tuning_schedule)} tentativas.")

#            _save(model, config.classes, f"generated/{name.lower()}.pth", log)

            # ── Treino com K-Fold ──────────────────────────────────────────────
            current_model = None

            for attempt, hp in enumerate(tuning_schedule, 1):
                _print_header(f"{name} K-Fold | Tentativa {attempt} | epochs={hp['epochs']} | lr={hp['lr']}", log)

                start = time.perf_counter()
                model_kfold, accuracy_kfold = fns["train_kfold_fn"](
                    train_ds,
                    test_loader, 
                    config,
                    hp["epochs"], 
                    hp["lr"], 
                    current_model, 
                )
                current_model = model_kfold
                _print_elapsed(start, log)

                log(f"  → Acurácia: {accuracy_kfold*100:.2f}%")

                if accuracy_kfold >= config.min_accuracy:
                    log("  ✓ Acurácia mínima atingida!")
                    break
            else:
                log(f"  ✗ Acurácia mínima não atingida após {len(tuning_schedule)} tentativas.")

            _save(model_kfold, config.classes, f"generated/{name.lower()}-with-kfold.pth", log)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_header(title: str, log=print):
    log(f"\n{'='*55}")
    log(f"  {title}")
    log(f"{'='*55}")


def _print_elapsed(start: float, log=print):
    elapsed = time.perf_counter() - start
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)
    log(f"  Tempo: {minutes}min {seconds}s")


def _save(model, classes, path: str, log=print):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state=model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {"model_state_dict": state, "classes": classes},
        path,
    )
    log(f"  Modelo salvo em: {path}")