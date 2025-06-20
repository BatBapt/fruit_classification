import os
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

import custom_dataset as custom_dataset
from model import get_model
import configuration as cfg


if __name__ == "__main__":
    dataset_test = custom_dataset.CustomDataset(
        csv_path=os.path.join(cfg.BASE_PATH, "test.csv"),
        transforms=custom_dataset.get_transform(train=False)
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False,
    )

    class_name = custom_dataset.get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))

    list_image_test = list(dataset_test.csv["image:FILE"])  # not very good way
    true_labels = []
    for image_name in list_image_test:
        image_name_split = image_name.split("/")
        label = int(image_name_split[1])
        true_labels.append(label)

    device = cfg.DEVICE

    best_model = get_model(len(class_name))
    best_model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
    model = best_model.to(device).eval()

    predictions = []

    # Boucle de test
    with torch.no_grad():
        test_loop = tqdm(test_data_loader)
        for inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().tolist())

    print(classification_report(true_labels, predictions, target_names=class_name))

    output_res = pd.DataFrame({
        "id": list_image_test,
        "predicted_label": predictions,
        "true_label": true_labels
    })
    output_res["predicted_name"] = output_res["predicted_label"].apply(lambda idx: class_name[idx])
    true_names = [class_name[idx] for idx in true_labels]
    output_res["true_name"] = true_names

    output_res.to_csv("submissions.csv", index=False)
    print("Saved at submissions.csv")