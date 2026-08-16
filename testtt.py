import torch
import matplotlib.pyplot as plt
import numpy as np

from model_unetpp import UnetPlusPlus
from model_attention_unet import AttentionUnet


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)



def load_model(model_type):


    if model_type == "unetpp":

        MODEL_PATH = "./unetplusplus_polyp_60epoch.pth"

        model = UnetPlusPlus(
            in_chann=3,
            num_class=1
        ).to(device)


    elif model_type == "attention":

        MODEL_PATH = "./attention_unet_polyp_60epoch.pth"

        model = AttentionUnet(
            in_chann=3,
            num_class=1
        ).to(device)


    else:

        raise ValueError(
            "model_type 'unetpp' veya 'attention' olmalı."
        )



    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )


    model.eval()


    return model




def calculate_metrics(
    prediction,
    mask
):


    prediction = prediction.view(-1)
    mask = mask.view(-1)


    true_positive = (
        (prediction == 1) &
        (mask == 1)
    ).sum().item()


    false_positive = (
        (prediction == 1) &
        (mask == 0)
    ).sum().item()


    false_negative = (
        (prediction == 0) &
        (mask == 1)
    ).sum().item()


    epsilon = 1e-8



    dice = (
        2 * true_positive
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )



    iou = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )



    precision = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + epsilon
    )



    recall = (
        true_positive
    ) / (
        true_positive
        + false_negative
        + epsilon
    )


    return (
        dice,
        iou,
        precision,
        recall
    )




def test_model(
    model,
    test_loader
):


    model.eval()


    total_dice = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0

    total_images = 0



    with torch.no_grad():


        for images, masks in test_loader:


            images = images.to(device)
            masks = masks.to(device)


            outputs = model(images)


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities > 0.5
            ).float()



            for i in range(images.size(0)):


                dice, iou, precision, recall = calculate_metrics(
                    predictions[i],
                    masks[i]
                )


                total_dice += dice
                total_iou += iou
                total_precision += precision
                total_recall += recall

                total_images += 1



    average_dice = (
        total_dice / total_images
    )


    average_iou = (
        total_iou / total_images
    )


    average_precision = (
        total_precision / total_images
    )


    average_recall = (
        total_recall / total_images
    )



    print(
        f"Average Dice: {average_dice:.4f}"
    )

    print(
        f"Average IoU: {average_iou:.4f}"
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print(
        f"Average Recall: {average_recall:.4f}"
    )



    return (
        average_dice,
        average_iou,
        average_precision,
        average_recall
    )




def denormalize_image(image):


    image = image / 2 + 0.5


    return image.clamp(
        0,
        1
    )




def visualize_predictions(
    images,
    masks,
    predictions,
    number_of_images=2
):


    number_of_images = min(
        number_of_images,
        images.size(0)
    )


    plt.figure(
        figsize=(12, number_of_images * 3)
    )



    for i in range(number_of_images):


        image = denormalize_image(
            images[i]
        )


        image = image.permute(
            1,
            2,
            0
        ).numpy()


        real_mask = masks[i][0].numpy()


        predicted_mask = predictions[i][0].numpy()



        dice, iou, precision, recall = calculate_metrics(
            predictions[i],
            masks[i]
        )



        plt.subplot(
            number_of_images,
            4,
            4 * i + 1
        )


        plt.imshow(
            image
        )


        plt.title(
            "Polyp Image"
        )


        plt.axis(
            "off"
        )




        plt.subplot(
            number_of_images,
            4,
            4 * i + 2
        )


        plt.imshow(
            real_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        plt.title(
            "Ground Truth"
        )


        plt.axis(
            "off"
        )




        plt.subplot(
            number_of_images,
            4,
            4 * i + 3
        )


        plt.imshow(
            predicted_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        plt.title(
            f"Prediction\n"
            f"Dice: {dice:.4f} | IoU: {iou:.4f}\n"
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )


        plt.axis(
            "off"
        )




        plt.subplot(
            number_of_images,
            4,
            4 * i + 4
        )


        plt.imshow(
            image
        )


        polyp_mask = np.ma.masked_where(
            predicted_mask == 0,
            predicted_mask
        )


        plt.imshow(
            polyp_mask,
            alpha=0.5,
            cmap="autumn",
            vmin=0,
            vmax=1
        )


        plt.title(
            f"Prediction Overlay\n"
            f"Dice: {dice:.4f} | IoU: {iou:.4f}\n"
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )


        plt.axis(
            "off"
        )



    plt.tight_layout()

    plt.show()




def continuous_test(
    model,
    test_loader
):


    model.eval()


    with torch.no_grad():


        for images, masks in test_loader:


            images = images.to(device)
            masks = masks.to(device)


            outputs = model(images)


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities > 0.5
            ).float()



            images = images.cpu()
            masks = masks.cpu()
            predictions = predictions.cpu()



            visualize_predictions(
                images=images,
                masks=masks,
                predictions=predictions,
                number_of_images=2
            )