from dataset import get_data_loaders
from testtt import load_model, test_model, continuous_test


MODEL_TYPE = "unetpp"


train_loader, test_loader = get_data_loaders(
    batch_size=8,
    image_size=256
)


model = load_model(
    model_type=MODEL_TYPE
)


test_model(
    model=model,
    test_loader=test_loader
)


continuous_test(
    model=model,
    test_loader=test_loader
)