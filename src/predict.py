import argparse
from pathlib import Path

import torch

from src.model import SMSClassifier
from src.model_io import load_final_model
from src.sequences import encode_and_pad

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.pt"


def predict_sms(message: str, model: SMSClassifier, checkpoint: dict) -> dict[str, str | float]:
    input_ids, padding_mask = encode_and_pad(
        message,
        checkpoint["vocabulary"],
        checkpoint["max_length"],
    )
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.tensor([padding_mask], dtype=torch.bool, device=device)

    model.eval()
    with torch.inference_mode():
        logits = model(ids, mask)
        scores = torch.softmax(logits, dim=1)[0]

    label_to_id = checkpoint["label_to_id"]
    id_to_label = {class_id: label for label, class_id in label_to_id.items()}
    return {
        "label": id_to_label[int(scores.argmax().item())],
        "ham_score": float(scores[label_to_id["ham"]].item()),
        "spam_score": float(scores[label_to_id["spam"]].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Klasifikuj jednu SMS poruku")
    parser.add_argument("message", help="Tekst SMS poruke")
    args = parser.parse_args()

    model, checkpoint = load_final_model(MODEL_PATH)
    prediction = predict_sms(args.message, model, checkpoint)
    print(f"Klasa: {prediction['label']}")
    print(f"Ham skor: {prediction['ham_score']:.3f}")
    print(f"Spam skor: {prediction['spam_score']:.3f}")


if __name__ == "__main__":
    main()
