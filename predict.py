import json
import sys

MODEL_FILE = "model.json"

def load_model(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: model.json not found. Train the model first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: model.json is corrupted.")
        sys.exit(1)


def predict_price(mileage, model):
    scaled_km = (mileage - model["km_mean"]) / model["km_std"]
    price_scaled = model["theta0"] + model["theta1"] * scaled_km
    price = price_scaled * model["price_std"] + model["price_mean"]
    return price


def main():
    model = load_model(MODEL_FILE)

    user_input = input("Enter car mileage (in km): ").strip()
    if not user_input:
        print("Error: Mileage input is empty.")
        sys.exit(1)

    try:
        mileage = float(user_input)
        if mileage < 0:
            raise ValueError
    except ValueError:
        print("Error: Mileage must be a positive number.")
        sys.exit(1)

    estimated_price = predict_price(mileage, model)
    print(f"Estimated price: €{estimated_price:.2f}")


if __name__ == "__main__":
    main()
