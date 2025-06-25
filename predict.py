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

    train_classnames = custom_dataset.get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))
    list_image_test = list(dataset_test.csv["image:FILE"])  # not very good way

    device = cfg.DEVICE
    best_model = get_model(len(train_classnames))
    best_model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
    model = best_model.to(device).eval()

    pred_labels = []

    # Boucle de test
    with torch.no_grad():
        test_loop = tqdm(test_data_loader)
        for inputs, _ in test_loop:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            preds = outputs.argmax(dim=1)

            pred_labels.extend(preds.cpu().tolist())


    # Save the predictions and try to "compute" real label
    output_res = pd.DataFrame({
        "id": list_image_test,
        "predicted_label": pred_labels,
    })
    output_res["predicted_name"] = output_res["predicted_label"].apply(lambda idx: train_classnames[idx])
    output_res["true_class_id"] = output_res["id"].apply(lambda path: path.split("/")[1])


    # Dictionnary containing all the maximum occurence of predicted label for a class for a given true class
    # exemple: dico_class[28] = 0
    dico_class = output_res.groupby('true_class_id')['predicted_label'].agg(lambda x: x.mode()[0]).to_dict()

    # New column to map the majority vote for each row
    output_res['majority_vote'] = output_res['true_class_id'].map(dico_class)
    output_res['true_class_name'] = output_res['majority_vote'].map(lambda x: train_classnames[x])

    print(classification_report(output_res['majority_vote'], output_res["predicted_label"], target_names=train_classnames))

    output_res.to_csv("submissions.csv", index=False)
    print("Saved at submissions.csv")