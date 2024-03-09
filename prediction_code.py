from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def predict(image_file):
    model = load_model("C:/Users/dulsh/Desktop/Assesment/model.h5", compile=False)
    class_names = [line.strip() for line in open("C:/Users/dulsh/Desktop/Assesment/labels.txt", "r").readlines()]
    class_counts = {class_name: 0 for class_name in class_names}

    image = Image.open(image_file).convert("RGB")
    image_size = (224, 224)
    image = ImageOps.fit(image, image_size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(data)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    class_counts[predicted_class_name] += 1

    print("Predicted Class:", predicted_class_name)
    print("Confidence Score:", prediction[0][predicted_class_index])
    print("\nClass Counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    return predicted_class_name
