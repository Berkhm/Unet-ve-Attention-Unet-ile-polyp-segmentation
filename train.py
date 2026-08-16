import torch
import torch.nn as nn
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

def define_loss_and_optimizer(model):

    criterion = nn.BCEWithLogitsLoss() # çıktığımız var yada yok olduğu için(0 veya 1) CrossEntropyLoss yerine bu

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )


    return criterion, optimizer



def train_model(model,train_loader,criterion,optimizer,epochs=60,model_name="Model"):


    model.train()

    train_losses = []

    for epoch in range(epochs):

        total_loss = 0

        for batch_index, (images, masks) in enumerate(train_loader):


            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)



            loss = criterion(outputs, masks)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


            if (batch_index + 1) % 50 == 0:
                print(
                    f"Epoch: {epoch + 1}/{epochs}, "
                    f"Batch: {batch_index + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.5f}"
                )

        average_loss = total_loss / len(train_loader)

        train_losses.append(average_loss)

        print(
            f"\nEpoch: {epoch + 1}/{epochs}, "
            f"Ortalama Loss: {average_loss:.5f}\n"
        )


    plt.figure()

    plt.plot(
        range(1, epochs + 1),
        train_losses,
        marker="o",
        linestyle="-",
        label="Train Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training Loss")
    plt.legend()
    plt.show()


    return train_losses